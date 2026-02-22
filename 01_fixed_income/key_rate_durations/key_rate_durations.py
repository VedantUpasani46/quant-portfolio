"""
Key Rate Duration (KRD) Analysis
==================================
Key rate duration measures a bond or portfolio's sensitivity to
a change in yield at a SPECIFIC point on the curve, holding all
other tenors constant.

Unlike DV01 (which aggregates all rate sensitivity into one number),
KRD gives a BUCKETED view — essential for:
  - Understanding where curve risk is concentrated
  - Building hedges that match the shape, not just the level, of risk
  - Regulatory reporting (Basel III, FRTB, CCAR)

Intuition:
  DV01 tells you: "I lose $50K if the whole curve shifts up 1bp."
  KRD tells you: "I lose $12K from the 2Y bucket, $23K from the 5Y,
                   and $15K from the 10Y — so hedge the 5Y belly."

Construction:
  For each key tenor k ∈ {0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30}:
    1. Shift the yield at tenor k up by 1bp
    2. Linearly interpolate the shift to adjacent key tenors
       (so the shift is 1bp at k, tapering to 0 at neighbouring key rates)
    3. KRD_k = −(ΔP / (P × 0.0001))  [like modified duration, per bp]
    4. Dollar KRD (DV01_k) = KRD_k × P × 0.0001  [$ loss per 1bp]

The key insight: Σₖ DV01_k ≈ Total DV01  (up to interpolation error)

Partial DV01 (PV01) — same concept, used for swap books:
  PV01_k = ΔPV / Δswap_rate_k

Bucket hedging:
  To hedge a bond portfolio, match KRDs with a combination of:
    - Treasury futures (pinned at 2Y, 5Y, 10Y, 30Y)
    - On-the-run Treasuries
    - Interest rate swaps

References:
  - Ho, T.S.Y. (1992). Key Rate Durations: Measures of Interest Rate Risks.
    Journal of Fixed Income 2(2), 29–44.
  - Tuckman, B. & Serrat, A. (2012). Fixed Income Securities, 3rd ed. Wiley.
  - Fabozzi, F.J. (2012). Fixed Income Mathematics, 4th ed. McGraw-Hill.
  - PIMCO Fixed Income Primer (2019). Understanding Duration.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Yield curve with perturbation support
# ---------------------------------------------------------------------------

class PerturbableYieldCurve:
    """
    A yield curve that supports bucket-specific perturbations.
    Used to compute key rate durations via numerical differentiation.
    """

    def __init__(self, tenors: np.ndarray, zero_rates: np.ndarray):
        self.base_tenors = np.array(tenors, dtype=float)
        self.base_rates  = np.array(zero_rates, dtype=float)

    def zero_rate(self, T: float,
                  perturb_tenor: Optional[float] = None,
                  perturb_bp: float = 1.0,
                  interp_method: str = "linear") -> float:
        """
        Zero rate at maturity T, with optional key-rate perturbation.

        The perturbation uses a tent function centred at perturb_tenor:
          - Full +perturb_bp at perturb_tenor
          - Linearly decays to 0 at the adjacent key tenors
          - 0 everywhere else
        """
        rates = self.base_rates.copy()

        if perturb_tenor is not None:
            shift = self._tent_shift(self.base_tenors, perturb_tenor, perturb_bp / 10000)
            rates = rates + shift

        if interp_method == "linear":
            interp = interp1d(self.base_tenors, rates,
                              bounds_error=False, fill_value=(rates[0], rates[-1]))
        else:
            from scipy.interpolate import CubicSpline
            interp = CubicSpline(self.base_tenors, rates, extrapolate=True)

        return float(interp(T))

    def discount_factor(self, T: float, **kwargs) -> float:
        z = self.zero_rate(T, **kwargs)
        return math.exp(-z * T)

    @staticmethod
    def _tent_shift(tenors: np.ndarray, centre: float, magnitude: float) -> np.ndarray:
        """
        Tent (piecewise-linear) perturbation:
          = magnitude at tenors[i] == centre
          = linearly interpolates to 0 at neighbouring key tenors
          = 0 elsewhere
        """
        shifts = np.zeros_like(tenors)
        idx = np.searchsorted(tenors, centre)

        # Find exact index or nearest
        if idx < len(tenors) and np.isclose(tenors[idx], centre):
            shifts[idx] = magnitude
            # Taper left
            if idx > 0:
                # Not needed: tent already 0 at adjacent key tenors
                pass
        else:
            # Centre is between two key tenors — interpolate peak
            idx = np.argmin(np.abs(tenors - centre))
            shifts[idx] = magnitude

        return shifts


# ---------------------------------------------------------------------------
# Bond repricing with perturbed curve
# ---------------------------------------------------------------------------

def bond_price_from_curve(
    coupon: float,       # annual coupon rate
    face: float,         # face value
    maturity: float,     # years to maturity
    freq: int,           # coupon frequency (2 = semiannual)
    curve: PerturbableYieldCurve,
    **curve_kwargs,
) -> float:
    """
    Price bond by discounting all cash flows using a yield curve.
    """
    dt = 1.0 / freq
    coupon_payment = coupon * face / freq
    pv = 0.0
    t = dt
    while t <= maturity + 1e-6:
        cf = coupon_payment + (face if abs(t - maturity) < 1e-6 else 0)
        df = curve.discount_factor(t, **curve_kwargs)
        pv += cf * df
        t += dt
    return pv


# ---------------------------------------------------------------------------
# Key Rate Duration computation
# ---------------------------------------------------------------------------

@dataclass
class KRDResult:
    key_tenors: np.ndarray
    dollar_krd: np.ndarray     # DV01 in $ per 1bp shift at each tenor
    pct_krd: np.ndarray        # % of total DV01 from each tenor
    total_dv01: float          # full parallel-shift DV01 ($)
    sum_krd: float             # sum of KRDs (should ≈ total DV01)

    def summary(self, name: str = "Bond") -> str:
        lines = [
            f"\n  Key Rate Duration: {name}",
            f"  {'Tenor':>8} {'Dollar KRD ($)':>16} {'% of Total':>12}",
            "  " + "─" * 40,
        ]
        for t, d, p in zip(self.key_tenors, self.dollar_krd, self.pct_krd):
            bar = "█" * max(0, int(abs(p) / 2))
            lines.append(f"  {t:>8.2f}Y  ${d:>13,.0f}  {p:>10.1f}%  {bar}")
        lines.append("  " + "─" * 40)
        lines.append(f"  {'Total DV01':>8}  ${self.total_dv01:>13,.0f}  100.0%")
        lines.append(f"  {'KRD sum':>8}  ${self.sum_krd:>13,.0f}  "
                     f"({self.sum_krd/self.total_dv01:.3f}× total)")
        return "\n".join(lines)


def key_rate_durations(
    coupon: float,
    face: float,
    maturity: float,
    freq: int,
    curve: PerturbableYieldCurve,
    key_tenors: Optional[np.ndarray] = None,
    perturb_bp: float = 1.0,
) -> KRDResult:
    """
    Compute key rate durations for a single bond.

    For each key tenor, shifts that point by perturb_bp and reprices.
    Dollar KRD_k = P_base − P_shifted   ($ loss per 1bp up at tenor k)
    """
    if key_tenors is None:
        key_tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

    # Filter to tenors ≤ bond maturity + buffer
    key_tenors = key_tenors[key_tenors <= maturity + 0.26]

    # Base price
    P_base = bond_price_from_curve(coupon, face, maturity, freq, curve)

    dollar_krds = []
    for k in key_tenors:
        P_up = bond_price_from_curve(coupon, face, maturity, freq, curve,
                                     perturb_tenor=k, perturb_bp=perturb_bp)
        dv01_k = P_base - P_up  # positive = price falls when rate rises
        dollar_krds.append(dv01_k)

    dollar_krd = np.array(dollar_krds)
    total_dv01_parallel = sum(dollar_krd)

    # Also compute true parallel DV01 for comparison
    P_par_up = bond_price_from_curve(coupon, face, maturity, freq, curve,
                                      perturb_tenor=None)
    # Shift ALL rates by 1bp
    shifted = PerturbableYieldCurve(curve.base_tenors,
                                     curve.base_rates + perturb_bp / 10000)
    P_par_up = bond_price_from_curve(coupon, face, maturity, freq, shifted)
    true_dv01 = P_base - P_par_up

    pct = dollar_krd / true_dv01 * 100 if true_dv01 != 0 else dollar_krd * 0

    return KRDResult(
        key_tenors=key_tenors,
        dollar_krd=dollar_krd,
        pct_krd=pct,
        total_dv01=true_dv01,
        sum_krd=float(dollar_krd.sum()),
    )


# ---------------------------------------------------------------------------
# Portfolio KRD aggregation
# ---------------------------------------------------------------------------

def portfolio_krd(
    positions: list[tuple],    # list of (coupon, face_value, maturity, freq, quantity)
    curve: PerturbableYieldCurve,
    key_tenors: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Aggregate KRDs across a bond portfolio.
    Returns DataFrame: key tenor × (dollar_krd, pct_total).
    """
    if key_tenors is None:
        key_tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

    # Aggregate dollar KRDs
    portfolio_dollar_krd = np.zeros(len(key_tenors))
    total_port_dv01 = 0.0

    for coupon, face, maturity, freq, qty in positions:
        krd = key_rate_durations(coupon, face * qty, maturity, freq, curve, key_tenors)
        for i, k in enumerate(key_tenors):
            if k in krd.key_tenors:
                idx = np.where(krd.key_tenors == k)[0]
                if len(idx) > 0:
                    portfolio_dollar_krd[i] += krd.dollar_krd[idx[0]]
        total_port_dv01 += krd.total_dv01

    rows = []
    for k, dkrd in zip(key_tenors, portfolio_dollar_krd):
        rows.append({
            "tenor": k,
            "dollar_krd": dkrd,
            "pct_total": dkrd / total_port_dv01 * 100 if total_port_dv01 != 0 else 0.0,
        })
    return pd.DataFrame(rows), total_port_dv01


def hedge_with_key_rates(
    portfolio_krd: np.ndarray,   # portfolio dollar KRDs
    key_tenors: np.ndarray,
    hedge_instruments: list[tuple],  # (name, coupon, maturity, face=1, freq=2)
    curve: PerturbableYieldCurve,
) -> dict:
    """
    Find hedge notionals for a set of instruments to flatten portfolio KRD profile.
    Uses least-squares: minimise ||H·x + krd||² where H is the KRD matrix.
    """
    n_hedge = len(hedge_instruments)
    n_buckets = len(key_tenors)

    # Build KRD matrix for hedge instruments
    H = np.zeros((n_buckets, n_hedge))
    for j, (name, coupon, maturity, freq) in enumerate(hedge_instruments):
        krd_h = key_rate_durations(coupon, 1.0, maturity, freq, curve, key_tenors)
        for i, k in enumerate(key_tenors):
            if k in krd_h.key_tenors:
                idx = np.where(krd_h.key_tenors == k)[0]
                if len(idx) > 0:
                    H[i, j] = krd_h.dollar_krd[idx[0]]

    # Least-squares: H·x = -portfolio_krd
    x, residuals, rank, sv = np.linalg.lstsq(H, -portfolio_krd, rcond=None)
    residual_krd = H @ x + portfolio_krd

    hedge_result = {}
    for j, (name, *_) in enumerate(hedge_instruments):
        hedge_result[name] = {"notional": x[j], "direction": "short" if x[j] < 0 else "long"}

    return {
        "hedge_notionals": hedge_result,
        "residual_krd": residual_krd,
        "residual_dv01": float(residual_krd.sum()),
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 64)
    print("  Key Rate Duration Analysis")
    print("  Bucketed rate sensitivity for bonds and portfolios")
    print("═" * 64)

    # Market yield curve
    tenors     = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    zero_rates = np.array([0.040, 0.042, 0.044, 0.047, 0.049, 0.051,
                            0.052, 0.053, 0.054, 0.056])
    curve = PerturbableYieldCurve(tenors, zero_rates)

    KEY_TENORS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

    # ── Single bond KRD ───────────────────────────────────────────
    print(f"\n── 10-Year 5% Coupon Bond (semiannual, $1M face) ──")
    krd_10y = key_rate_durations(
        coupon=0.05, face=1_000_000, maturity=10.0, freq=2, curve=curve,
        key_tenors=KEY_TENORS
    )
    print(krd_10y.summary("10Y 5% Bond"))

    print(f"\n── 30-Year 4.5% Coupon Bond (semiannual, $1M face) ──")
    krd_30y = key_rate_durations(
        coupon=0.045, face=1_000_000, maturity=30.0, freq=2, curve=curve,
        key_tenors=KEY_TENORS
    )
    print(krd_30y.summary("30Y 4.5% Bond"))

    # ── Zero coupon bond — pure single-bucket exposure ────────────
    print(f"\n── 5-Year Zero-Coupon Bond ($1M face) ──")
    krd_zcb = key_rate_durations(
        coupon=0.0, face=1_000_000, maturity=5.0, freq=2, curve=curve,
        key_tenors=KEY_TENORS
    )
    print(krd_zcb.summary("5Y ZCB"))
    print(f"\n  ZCB has ALL risk concentrated at the 5Y tenor ✓")
    print(f"  (This is the defining property of zero-coupon bonds)")

    # ── Portfolio KRD ─────────────────────────────────────────────
    print(f"\n── Bond Portfolio KRD Aggregation ──")
    portfolio = [
        # (coupon, face, maturity, freq, quantity)
        (0.050, 1_000_000, 10.0, 2, 5),   # 5 × 10Y 5% bonds
        (0.045, 1_000_000, 30.0, 2, 3),   # 3 × 30Y 4.5% bonds
        (0.030, 1_000_000,  2.0, 2, 10),  # 10 × 2Y 3% bonds
        (0.000, 1_000_000,  5.0, 2, 2),   # 2 × 5Y zeros
    ]
    port_krd_df, total_dv01 = portfolio_krd(portfolio, curve, KEY_TENORS)

    print(f"\n  Total portfolio DV01: ${total_dv01:,.0f} per 1bp parallel shift")
    print(f"\n  {'Tenor':>8} {'Dollar KRD ($)':>16} {'% Total':>10}")
    print("  " + "─" * 38)
    for _, row in port_krd_df.iterrows():
        bar = "█" * max(0, int(row["pct_total"] / 3))
        print(f"  {row['tenor']:>8.2f}Y  ${row['dollar_krd']:>13,.0f}  "
              f"{row['pct_total']:>8.1f}%  {bar}")

    # ── Hedging with Treasury futures ─────────────────────────────
    print(f"\n── Bucket Hedging with Treasury Instruments ──")
    # Hedge instruments: 2Y, 5Y, 10Y, 30Y on-the-run Treasuries
    hedge_instrs = [
        ("2Y Treasury",  0.045, 2.0,  2),
        ("5Y Treasury",  0.048, 5.0,  2),
        ("10Y Treasury", 0.050, 10.0, 2),
        ("30Y Treasury", 0.053, 30.0, 2),
    ]
    port_krd_arr = port_krd_df["dollar_krd"].values

    hedge = hedge_with_key_rates(port_krd_arr, KEY_TENORS, hedge_instrs, curve)
    print(f"\n  Hedge notionals ($ face value, negative = short):")
    for name, info in hedge["hedge_notionals"].items():
        direction = "SHORT" if info["notional"] < 0 else "LONG"
        print(f"    {name:<20} {direction}: ${abs(info['notional']):>12,.0f}")
    print(f"\n  Residual DV01 after hedge: ${hedge['residual_dv01']:,.0f}")
    print(f"  Residual KRD profile:")
    for t, r in zip(KEY_TENORS, hedge["residual_krd"]):
        if abs(r) > 1:
            print(f"    {t:.1f}Y: ${r:,.0f}")

    # ── KRD vs DV01: why bucketed matters ─────────────────────────
    print(f"""
── Why KRD vs DV01 Matters ──

  Two portfolios with identical total DV01 = $50,000 can have
  very different risk profiles:

  Portfolio A (bullet):  All DV01 in the 10Y bucket
    → Loses $50K if 10Y rises 1bp, immune to 2Y/30Y moves

  Portfolio B (barbell): $25K in 2Y, $25K in 30Y
    → Losses depend on CURVE SHAPE, not just level
    → Gains if yield curve flattens (30Y falls, 2Y rises)
    → Losses if curve steepens

  KRD reveals this: DV01 is blind to it.
  Portfolio managers at PIMCO, Goldman Strats, JPM use KRD
  to manage curve exposure, not just overall rate risk.
    """)
