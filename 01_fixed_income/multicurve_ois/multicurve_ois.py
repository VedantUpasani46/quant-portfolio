"""
Multi-Curve Framework: OIS Discounting and LIBOR-OIS Spread
=============================================================
Pre-2008, practitioners used a single yield curve for both:
  (1) Discounting cash flows
  (2) Projecting future LIBOR/floating rates

Post-2008, this assumption broke down because:
  - LIBOR (unsecured bank lending) carried significant credit/liquidity risk
  - OIS (overnight index swaps, e.g. Fed Funds / SOFR) is nearly risk-free
  - The LIBOR-OIS spread spiked from ~10bp to >350bp during the GFC
  - Discounting at LIBOR inflates the PV of derivatives vs their true economic value

The modern framework (Bianchetti 2010, Mercurio 2010):
  - DISCOUNT curve:  OIS (risk-free, collateral-consistent)
  - PROJECTION curve: LIBOR/EURIBOR (for each tenor: 1M, 3M, 6M, 1Y)
  - FRA and swap pricing: project floating rates from LIBOR curve,
    discount from OIS curve

Key instruments:
  - OIS: swap where floating = daily overnight rate (Fed Funds, SOFR, EONIA)
    Nearly risk-free; used as the discount curve
  - IRS (Interest Rate Swap): fixed vs LIBOR
    In dual-curve: discount on OIS, project on LIBOR
  - FRA (Forward Rate Agreement): lock in a borrowing rate today
  - Basis swap: exchange one floating rate for another (e.g. 3M LIBOR vs 1M LIBOR)

Single-curve vs dual-curve pricing difference:
  In the pre-2008 single-curve world:
    P(0,T1,T2)_implied_fwd = [P(0,T1)/P(0,T2) - 1] / τ  [from one curve]
  In the dual-curve world:
    Discount: P_OIS(0,T)
    Forward LIBOR: L(0,T1,T2) = [P_LIBOR(0,T1)/P_LIBOR(0,T2) - 1] / τ
  These differ by the LIBOR-OIS spread, which can be large in stress periods.

References:
  - Bianchetti, M. (2010). Two Curves, One Price. Risk Magazine.
  - Mercurio, F. (2010). A LIBOR Market Model with a Stochastic Basis.
    Bloomberg Education & Quantitative Research.
  - Hull, J.C. & White, A. (2013). LIBOR vs OIS: The Derivatives Discounting Dilemma.
    Journal of Investment Management 11(3).
  - Tuckman, B. & Serrat, A. (2012). Fixed Income Securities, Ch. 2.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Curve objects
# ---------------------------------------------------------------------------

class DiscountCurve:
    """
    A zero-rate discount curve for a single currency.
    Supports linear interpolation in zero rates (or log-DF for monotonicity).
    """

    def __init__(self, tenors: np.ndarray, zero_rates: np.ndarray,
                 label: str = "Curve"):
        self.tenors = np.array(tenors, dtype=float)
        self.zero_rates = np.array(zero_rates, dtype=float)
        self.label = label
        self._interp = interp1d(tenors, zero_rates, kind="linear",
                                bounds_error=False,
                                fill_value=(zero_rates[0], zero_rates[-1]))

    def zero_rate(self, T: float) -> float:
        return float(self._interp(T))

    def df(self, T: float) -> float:
        """Discount factor P(0,T) = exp(-r(T)·T)."""
        return math.exp(-self.zero_rate(T) * T) if T > 0 else 1.0

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Continuously compounded forward rate from T1 to T2.
        f(T1,T2) = [ln P(0,T1) - ln P(0,T2)] / (T2 - T1)
        """
        return (math.log(self.df(T1)) - math.log(self.df(T2))) / (T2 - T1)

    def simple_forward_rate(self, T1: float, T2: float) -> float:
        """
        Simply-compounded LIBOR forward rate (the standard convention):
        L(T1,T2) = [P(0,T1)/P(0,T2) - 1] / (T2 - T1)
        """
        tau = T2 - T1
        return (self.df(T1) / self.df(T2) - 1) / tau


# ---------------------------------------------------------------------------
# LIBOR-OIS spread model
# ---------------------------------------------------------------------------

@dataclass
class CurvePair:
    """
    The dual-curve pair used in modern swap pricing.
    discount_curve: OIS (risk-free)
    projection_curve: LIBOR or SOFR term rate (for floating legs)
    """
    discount_curve: DiscountCurve
    projection_curve: DiscountCurve

    def libor_ois_spread(self, T1: float, T2: float) -> float:
        """
        LIBOR-OIS spread for period [T1, T2]:
        = LIBOR forward rate (from projection curve)
          - OIS forward rate (from discount curve)
        Both in simple compounding.
        """
        libor_fwd = self.projection_curve.simple_forward_rate(T1, T2)
        ois_fwd   = self.discount_curve.simple_forward_rate(T1, T2)
        return libor_fwd - ois_fwd


# ---------------------------------------------------------------------------
# FRA (Forward Rate Agreement)
# ---------------------------------------------------------------------------

class FRAPricer:
    """
    Forward Rate Agreement pricing in single-curve and dual-curve frameworks.

    A FRA is an agreement to pay/receive a fixed rate K on notional N
    for the period [T1, T2]:
      At settlement T1: PV = τ·N·(L(T1,T2) - K) / (1 + τ·L(T1,T2))
      At maturity  T2: PV = τ·N·(L(T1,T2) - K)

    Under the dual-curve framework:
      The forward LIBOR is projected from the LIBOR curve.
      The PV is discounted using the OIS curve.
    """

    def price(
        self,
        K: float,           # FRA fixed rate (strike)
        T1: float,          # FRA start date (years)
        T2: float,          # FRA end date (years)
        notional: float,
        curves: CurvePair,
        payer: bool = True,  # payer receives L, pays K
    ) -> dict:
        """
        Price a FRA under the dual-curve framework.

        FRA cash flow at T2: N·τ·(L - K)  where L is the realised LIBOR
        PV₀ = N·τ·(F - K)·P_OIS(0,T2)
        where F = L(0,T1,T2) is the forward LIBOR from the projection curve.

        At T1 settlement: the payment is discounted for period [T1,T2]:
          PV_T1 = τ·N·(F-K) / (1 + τ·F)
        """
        tau = T2 - T1
        F = curves.projection_curve.simple_forward_rate(T1, T2)
        K_ois = curves.discount_curve.simple_forward_rate(T1, T2)
        df_T2 = curves.discount_curve.df(T2)
        df_T1 = curves.discount_curve.df(T1)

        sign = 1 if payer else -1
        # PV at T2 payment convention
        pv_t2 = sign * notional * tau * (F - K) * df_T2
        # PV at T1 settlement convention (standard for FRAs)
        pv_t1_settle = sign * notional * tau * (F - K) / (1 + tau * F) * df_T1

        # Single-curve price (using OIS for everything — for comparison)
        F_single = curves.discount_curve.simple_forward_rate(T1, T2)
        pv_single = sign * notional * tau * (F_single - K) * df_T2

        return {
            "forward_libor": F,
            "forward_ois":   K_ois,
            "libor_ois_spread_bp": (F - K_ois) * 10000,
            "pv_dual_curve": pv_t2,
            "pv_t1_settle":  pv_t1_settle,
            "pv_single_curve": pv_single,
            "dual_vs_single_bp": (pv_t2 - pv_single) / notional * 10000,
        }


# ---------------------------------------------------------------------------
# Interest Rate Swap (dual-curve)
# ---------------------------------------------------------------------------

class DualCurveSwapPricer:
    """
    Fixed-for-floating interest rate swap under the dual-curve framework.

    Fixed leg: pays fixed rate K at intervals dt_fixed (typically 6M or annual)
    Float leg: pays LIBOR at intervals dt_float (typically 3M or 6M)

    Par swap rate S (dual-curve):
      S = [1 - P_OIS(0,T)] / Σᵢ τᵢ · P_OIS(0,Tᵢ)     [OIS annuity]
      This is DIFFERENT from:
      S_single = [1 - P_single(0,T)] / Σᵢ τᵢ · P_single(0,Tᵢ)
    """

    def par_swap_rate(
        self,
        maturity: float,
        curves: CurvePair,
        float_freq: int = 4,  # quarterly floating
        fixed_freq: int = 2,  # semiannual fixed
    ) -> dict:
        """
        Par swap rate: fixed rate S s.t. PV(swap) = 0 at inception.

        Dual-curve formula:
          Float leg PV = Σⱼ τⱼ · F(Tⱼ₋₁, Tⱼ) · P_OIS(0,Tⱼ)
          Fixed leg PV = S · Σᵢ τᵢ · P_OIS(0,Tᵢ)  [annuity]
          S_dual = Float PV / Fixed annuity

        Single-curve (for comparison):
          S_single = [1 - P(0,T)] / annuity  (closed form)
        """
        # Floating leg PV (project from LIBOR, discount from OIS)
        dt_f = 1.0 / float_freq
        float_pv = 0.0
        t = dt_f
        while t <= maturity + 1e-8:
            T1, T2 = t - dt_f, t
            F = curves.projection_curve.simple_forward_rate(T1, T2) if T1 > 0 else \
                curves.projection_curve.simple_forward_rate(dt_f * 0.01, t)
            tau = T2 - T1
            df = curves.discount_curve.df(T2)
            float_pv += tau * F * df
            t += dt_f

        # Fixed leg annuity (discount from OIS)
        dt_x = 1.0 / fixed_freq
        annuity = 0.0
        t = dt_x
        while t <= maturity + 1e-8:
            annuity += dt_x * curves.discount_curve.df(t)
            t += dt_x

        S_dual   = float_pv / annuity if annuity > 0 else 0.0

        # Single-curve par rate (closed form, using OIS only)
        df_T = curves.discount_curve.df(maturity)
        S_single = (1 - df_T) / annuity if annuity > 0 else 0.0

        return {
            "par_rate_dual_curve":   S_dual,
            "par_rate_single_curve": S_single,
            "spread_bp":             (S_dual - S_single) * 10000,
            "float_leg_pv":          float_pv,
            "annuity":               annuity,
        }

    def swap_pv(
        self,
        fixed_rate: float,
        maturity: float,
        notional: float,
        curves: CurvePair,
        payer: bool = True,     # payer: pay fixed, receive float
        float_freq: int = 4,
        fixed_freq: int = 2,
    ) -> dict:
        """
        Mark-to-market PV of an off-market swap under dual-curve framework.
        sign: +1 = payer (pay fixed, receive float).
        """
        par = self.par_swap_rate(maturity, curves, float_freq, fixed_freq)
        S = par["par_rate_dual_curve"]
        annuity = par["annuity"]
        sign = 1 if payer else -1

        pv = sign * (S - fixed_rate) * annuity * notional

        return {
            "pv":                pv,
            "par_rate":          S,
            "dv01":              annuity * notional / 10000,
            "payer":             payer,
            "fixed_rate":        fixed_rate,
            "dual_vs_single_pv": (S - par["par_rate_single_curve"]) * annuity * notional,
        }


# ---------------------------------------------------------------------------
# Basis swap (3M vs 6M LIBOR, or LIBOR vs SOFR)
# ---------------------------------------------------------------------------

class BasisSwapPricer:
    """
    Tenor basis swap: exchange 3M LIBOR flat vs 6M LIBOR flat + spread.
    The spread (basis) compensates for the difference in credit/liquidity risk
    between 3M and 6M tenors.

    Post-2008: USD basis swaps became significant (peaked at ~50bp during GFC).
    """

    def basis_swap_spread(
        self,
        maturity: float,
        curve_3m: DiscountCurve,   # 3M LIBOR projection curve
        curve_6m: DiscountCurve,   # 6M LIBOR projection curve
        ois_curve: DiscountCurve,  # OIS discount curve
    ) -> float:
        """
        Fair basis spread: 3M LIBOR + spread vs 6M LIBOR flat.
        Found by equating PVs of both legs.
        """
        # 6M floating leg PV
        dt_6m = 0.5
        pv_6m = 0.0
        t = dt_6m
        while t <= maturity + 1e-8:
            T1, T2 = max(t - dt_6m, 0), t
            if T1 == 0:
                T1 = 0.01
            F6 = curve_6m.simple_forward_rate(T1, T2)
            pv_6m += dt_6m * F6 * ois_curve.df(t)
            t += dt_6m

        # 3M floating leg PV (annuity in 3M payment terms)
        dt_3m = 0.25
        pv_3m = 0.0
        annuity_3m = 0.0
        t = dt_3m
        while t <= maturity + 1e-8:
            T1, T2 = max(t - dt_3m, 0), t
            if T1 == 0:
                T1 = 0.01
            F3 = curve_3m.simple_forward_rate(T1, T2)
            df = ois_curve.df(t)
            pv_3m += dt_3m * F3 * df
            annuity_3m += dt_3m * df
            t += dt_3m

        # Spread = (PV_6m - PV_3m) / annuity_3m
        spread = (pv_6m - pv_3m) / annuity_3m if annuity_3m > 0 else 0.0
        return spread


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  Multi-Curve Framework: OIS Discounting")
    print("  Dual-curve FRA and swap pricing, LIBOR-OIS spread")
    print("═" * 66)

    tenors = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

    # OIS curve (risk-free, SOFR/Fed Funds)
    ois_rates = np.array([0.0430, 0.0435, 0.0438, 0.0440, 0.0445, 0.0450,
                           0.0455, 0.0460, 0.0465, 0.0475, 0.0480])

    # 3M LIBOR curve (higher — credit/liquidity premium over OIS)
    libor_spread_3m = np.array([0.0020, 0.0022, 0.0025, 0.0028, 0.0026, 0.0024,
                                  0.0020, 0.0018, 0.0016, 0.0012, 0.0010])
    libor_3m_rates = ois_rates + libor_spread_3m

    # 6M LIBOR curve (slightly more than 3M due to tenor basis)
    libor_spread_6m = libor_spread_3m + 0.0003  # 3bp additional tenor basis
    libor_6m_rates = ois_rates + libor_spread_6m

    ois_curve     = DiscountCurve(tenors, ois_rates, "OIS (SOFR)")
    libor_3m_curve = DiscountCurve(tenors, libor_3m_rates, "3M LIBOR")
    libor_6m_curve = DiscountCurve(tenors, libor_6m_rates, "6M LIBOR")

    curves_3m = CurvePair(discount_curve=ois_curve, projection_curve=libor_3m_curve)
    curves_6m = CurvePair(discount_curve=ois_curve, projection_curve=libor_6m_curve)

    # ── LIBOR-OIS spread profile ──────────────────────────────────
    print(f"\n── LIBOR-OIS Spread Profile ──")
    print(f"\n  {'Tenor':>8} {'OIS Rate':>10} {'3M LIBOR':>10} {'Spread (bp)':>12}")
    print("  " + "─" * 44)
    for T1, T2 in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]:
        if T1 == 0:
            T1 = 0.01
        ois_fwd   = ois_curve.simple_forward_rate(T1, T2)
        libor_fwd = libor_3m_curve.simple_forward_rate(T1, T2)
        spread_bp = (libor_fwd - ois_fwd) * 10000
        print(f"  {T1:.2f}-{T2:.2f}Y {ois_fwd:>10.4%} {libor_fwd:>10.4%} {spread_bp:>12.2f}bp")

    # ── FRA pricing ───────────────────────────────────────────────
    print(f"\n── FRA Pricing: Dual-Curve vs Single-Curve ──")
    fra = FRAPricer()
    notional = 10_000_000
    print(f"\n  FRA 3×6 (locks in 3M LIBOR rate between months 3 and 6)")
    print(f"  Notional: ${notional:,}, Payer (receives LIBOR, pays fixed)")
    print(f"\n  {'K (fixed rate)':>16} {'Dual-curve PV':>16} {'Single-curve PV':>18} {'Diff (bp)':>12}")
    print("  " + "─" * 66)
    for K in [0.044, 0.046, 0.048, 0.050, 0.052]:
        result = fra.price(K, T1=0.25, T2=0.5, notional=notional,
                           curves=curves_3m, payer=True)
        print(f"  {K:>16.3%} ${result['pv_dual_curve']:>14,.0f} ${result['pv_single_curve']:>16,.0f} "
              f"{result['dual_vs_single_bp']:>12.3f}bp")

    fwd_libor = curves_3m.projection_curve.simple_forward_rate(0.25, 0.5)
    print(f"\n  ATM forward LIBOR (3×6): {fwd_libor:.4%}")
    print(f"  Forward OIS (3×6):       {curves_3m.discount_curve.simple_forward_rate(0.25, 0.5):.4%}")
    print(f"  LIBOR-OIS spread:        {(fwd_libor - curves_3m.discount_curve.simple_forward_rate(0.25,0.5))*10000:.2f}bp")

    # ── Swap par rates ────────────────────────────────────────────
    print(f"\n── IRS Par Rates: Dual-Curve vs Single-Curve ──")
    swap = DualCurveSwapPricer()
    print(f"\n  {'Maturity':>10} {'Dual-curve':>14} {'Single-curve':>14} {'Spread (bp)':>13}")
    print("  " + "─" * 55)
    for mat in [1, 2, 3, 5, 7, 10, 20, 30]:
        result = swap.par_swap_rate(float(mat), curves_3m)
        print(f"  {mat:>10}Y {result['par_rate_dual_curve']:>14.4%} "
              f"{result['par_rate_single_curve']:>14.4%} {result['spread_bp']:>13.2f}bp")

    print(f"\n  The spread = LIBOR-OIS basis embedded in swap rates. ✓")

    # ── Swap PV / MTM ─────────────────────────────────────────────
    print(f"\n── Off-Market Swap Mark-to-Market ──")
    print(f"\n  $10M 5Y payer swap, struck at 4.40% when par is {swap.par_swap_rate(5.0, curves_3m)['par_rate_dual_curve']:.4%}")
    result = swap.swap_pv(
        fixed_rate=0.0440, maturity=5.0, notional=10_000_000,
        curves=curves_3m, payer=True
    )
    print(f"  PV (dual-curve):  ${result['pv']:>10,.0f}")
    print(f"  DV01:             ${result['dv01']:>10,.0f} per 1bp")
    print(f"  Dual vs single:   ${result['dual_vs_single_pv']:>10,.0f} difference")

    # ── Basis swap ────────────────────────────────────────────────
    print(f"\n── Tenor Basis Swap (3M LIBOR vs 6M LIBOR) ──")
    basis = BasisSwapPricer()
    print(f"\n  {'Maturity':>10} {'Basis spread (bp)':>20}")
    print("  " + "─" * 34)
    for mat in [1, 2, 3, 5, 10]:
        spread = basis.basis_swap_spread(float(mat), libor_3m_curve, libor_6m_curve, ois_curve)
        print(f"  {mat:>10}Y {spread*10000:>20.2f}bp")

    print(f"""
── Why Multi-Curve Matters ──

  Pre-2008 (single-curve):
    One LIBOR curve used for both projection AND discounting
    → Simple, consistent, but wrong when credit risk matters

  2008 GFC: LIBOR-OIS spread spiked to 350bp
    → Discounting at LIBOR undervalues collateralised swaps by
      the full LIBOR-OIS spread (hundreds of millions on large books)
    → Banks began mis-stating derivative PVs by material amounts

  Post-2008 (dual-curve, now universal):
    Discount: OIS  (collateral-consistent, nearly risk-free)
    Project:  LIBOR/SOFR term rates (tenor-specific)
    → Correct pricing for collateralised derivatives
    → Separate basis swaps to exchange between tenors

  SOFR transition (2022):
    USD LIBOR replaced by SOFR (Secured Overnight Financing Rate)
    Same dual-curve logic: discount on SOFR OIS, project SOFR term rates
    """)
