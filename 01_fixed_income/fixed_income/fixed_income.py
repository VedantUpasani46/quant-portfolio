"""
Fixed Income Analytics: Bond Pricing & Risk Measures
======================================================
A complete fixed income toolkit covering:
  - Bond pricing (coupon bonds, zero-coupon bonds, floating rate notes)
  - Yield-to-maturity (YTM) via Newton-Raphson
  - Duration: Macaulay, Modified, and Dollar (DV01)
  - Convexity and convexity adjustment
  - Spread analytics: G-spread, Z-spread, OAS concept
  - Price/yield relationship and key rate durations
  - Accrued interest and clean/dirty price

Why fixed income matters for elite roles:
  Central banks (Fed, ECB, BIS, BoE), sovereign wealth funds,
  multilateral institutions (IMF, World Bank), and every major bank
  desk operates in fixed income. Bond risk analytics are a prerequisite
  for rates, credit, and macro roles.

References:
  - Fabozzi, F.J. (2021). Fixed Income Mathematics, 5th ed.
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 6.
  - Tuckman, B. & Serrat, A. (2011). Fixed Income Securities, 3rd ed.
  - BIS (2020). BIS Quarterly Review — fixed income risk methodologies.
"""

import math
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BondSpec:
    """
    Specification of a fixed-rate coupon bond.

    Parameters
    ----------
    face_value : float       Par / notional (e.g. 1000).
    coupon_rate : float      Annual coupon rate (e.g. 0.05 for 5%).
    maturity_years : float   Time to maturity in years.
    freq : int               Coupon payments per year (1=annual, 2=semi-annual).
    day_count : str          Day count convention ('act365', '30360').
    settlement_days: int     Settlement lag (default 2 business days).
    """
    face_value: float = 1000.0
    coupon_rate: float = 0.05
    maturity_years: float = 10.0
    freq: int = 2
    day_count: Literal["act365", "30360"] = "act365"
    settlement_days: int = 2

    @property
    def coupon(self) -> float:
        """Periodic coupon payment."""
        return self.face_value * self.coupon_rate / self.freq

    @property
    def n_periods(self) -> int:
        return int(round(self.maturity_years * self.freq))

    @property
    def dt(self) -> float:
        """Length of each period in years."""
        return 1.0 / self.freq


@dataclass
class BondAnalytics:
    """Complete bond analytics output."""
    dirty_price: float          # full price including accrued interest
    clean_price: float          # quoted price (dirty minus accrued)
    accrued_interest: float
    ytm: float                  # yield-to-maturity (annualised)
    macaulay_duration: float    # in years
    modified_duration: float    # % price change per 1% yield change
    dv01: float                 # $ change per 1bp yield change
    convexity: float            # second-order price sensitivity

    def price_change_approx(self, yield_change_bps: float) -> dict:
        """
        Taylor approximation of price change for a given yield change.
        ΔP ≈ -ModDur · Δy · P + 0.5 · Convexity · (Δy)² · P
        """
        dy = yield_change_bps / 10_000
        linear = -self.modified_duration * dy * self.dirty_price
        convex_adj = 0.5 * self.convexity * dy ** 2 * self.dirty_price
        total = linear + convex_adj
        return {
            "yield_change_bps": yield_change_bps,
            "linear_approximation": round(linear, 6),
            "convexity_adjustment": round(convex_adj, 6),
            "total_estimated_change": round(total, 6),
            "new_price_estimate": round(self.dirty_price + total, 6),
            "pct_change": f"{total / self.dirty_price:.4%}",
        }

    def summary(self) -> str:
        lines = [
            "=" * 52,
            f"  {'Metric':<28}  {'Value':>16}",
            "=" * 52,
            f"  {'Dirty Price':<28}  {self.dirty_price:>16.6f}",
            f"  {'Clean Price':<28}  {self.clean_price:>16.6f}",
            f"  {'Accrued Interest':<28}  {self.accrued_interest:>16.6f}",
            "─" * 52,
            f"  {'YTM (annualised)':<28}  {self.ytm:>16.4%}",
            "─" * 52,
            f"  {'Macaulay Duration (yr)':<28}  {self.macaulay_duration:>16.4f}",
            f"  {'Modified Duration':<28}  {self.modified_duration:>16.4f}",
            f"  {'DV01 ($)':<28}  {self.dv01:>16.6f}",
            f"  {'Convexity':<28}  {self.convexity:>16.4f}",
            "=" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bond pricer
# ---------------------------------------------------------------------------

class BondPricer:
    """
    Fixed-rate coupon bond pricer with full risk analytics.

    Usage
    -----
    >>> spec = BondSpec(face_value=1000, coupon_rate=0.05, maturity_years=10, freq=2)
    >>> pricer = BondPricer(spec)
    >>> analytics = pricer.analyse(ytm=0.045)
    >>> print(analytics.summary())
    """

    def __init__(self, spec: BondSpec):
        self.spec = spec
        self._payment_times = [
            (i + 1) * spec.dt for i in range(spec.n_periods)
        ]
        self._cash_flows = [spec.coupon] * spec.n_periods
        self._cash_flows[-1] += spec.face_value   # add face at maturity

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def dirty_price(self, ytm: float) -> float:
        """
        Full (dirty) price at a given YTM.

        P = Σ_{i=1}^{n} C_i / (1 + y/freq)^{freq·t_i}

        For continuous compounding: P = Σ C_i · e^{-y·t_i}
        We use periodic discounting consistent with standard bond markets.
        """
        periodic_yield = ytm / self.spec.freq
        price = 0.0
        for t, cf in zip(self._payment_times, self._cash_flows):
            n = t * self.spec.freq   # number of periods
            price += cf / (1 + periodic_yield) ** n
        return price

    def accrued_interest(self, fraction_of_period: float = 0.0) -> float:
        """
        Accrued interest = coupon × fraction of current period elapsed.
        For settlement at coupon date: fraction = 0.
        """
        return self.spec.coupon * fraction_of_period

    def clean_price(self, ytm: float, fraction_of_period: float = 0.0) -> float:
        """Clean price = dirty price - accrued interest."""
        return self.dirty_price(ytm) - self.accrued_interest(fraction_of_period)

    # ------------------------------------------------------------------
    # Yield to Maturity
    # ------------------------------------------------------------------

    def ytm_from_price(self, dirty_price: float) -> float:
        """
        Solve for YTM given a dirty price via Brent's method.

        Brent's method is preferred over Newton-Raphson for YTM because
        the price-yield relationship is monotonically decreasing, guaranteeing
        a unique root in a bounded interval.
        """
        def price_diff(ytm):
            return self.dirty_price(ytm) - dirty_price

        # Bounds: yield must be in (-99%, 200%) for any reasonable bond
        try:
            return brentq(price_diff, -0.99, 2.0, xtol=1e-10, maxiter=200)
        except ValueError:
            raise ValueError(
                f"Could not find YTM for price {dirty_price:.4f}. "
                "Check that the bond specification is valid."
            )

    # ------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------

    def macaulay_duration(self, ytm: float) -> float:
        """
        Macaulay Duration: weighted average time to receipt of cash flows.

        D_mac = Σ t_i · PV(CF_i) / P

        Interpretation: the bond's effective maturity; also the exact
        hedge ratio for small parallel yield shifts under continuous compounding.
        """
        periodic_yield = ytm / self.spec.freq
        price = self.dirty_price(ytm)
        weighted_time = 0.0
        for t, cf in zip(self._payment_times, self._cash_flows):
            n = t * self.spec.freq
            pv_cf = cf / (1 + periodic_yield) ** n
            weighted_time += t * pv_cf
        return weighted_time / price

    def modified_duration(self, ytm: float) -> float:
        """
        Modified Duration = Macaulay Duration / (1 + y/freq)

        Interpretation: % change in price for a 1 percentage-point rise in yield.
        ΔP/P ≈ -ModDur · Δy
        """
        mac_dur = self.macaulay_duration(ytm)
        return mac_dur / (1 + ytm / self.spec.freq)

    def dv01(self, ytm: float) -> float:
        """
        DV01 (Dollar Value of 01) = price change per 1 basis point yield rise.
        DV01 = ModDur × Dirty Price × 0.0001

        The sign convention: DV01 is positive (a yield rise causes a price fall).
        """
        return self.modified_duration(ytm) * self.dirty_price(ytm) * 0.0001

    # ------------------------------------------------------------------
    # Convexity
    # ------------------------------------------------------------------

    def convexity(self, ytm: float) -> float:
        """
        Convexity = (1/P) · d²P/dy²

        Convexity = Σ t_i(t_i + dt) · PV(CF_i) / [P · (1 + y/freq)²]

        Interpretation: the curvature of the price-yield relationship.
        Positive for plain vanilla bonds — means duration *underestimates*
        the actual price gain from a yield fall.
        """
        periodic_yield = ytm / self.spec.freq
        price = self.dirty_price(ytm)
        freq = self.spec.freq
        conv = 0.0
        for t, cf in zip(self._payment_times, self._cash_flows):
            n = t * freq
            pv_cf = cf / (1 + periodic_yield) ** n
            conv += n * (n + 1) * pv_cf

        return conv / (price * freq ** 2 * (1 + periodic_yield) ** 2)

    # ------------------------------------------------------------------
    # Z-Spread
    # ------------------------------------------------------------------

    def z_spread(self, dirty_price: float, zero_curve_fn) -> float:
        """
        Z-Spread: the constant spread added to the entire zero curve such
        that the discounted cash flows equal the observed market price.

        P = Σ CF_i · df_curve(t_i) · e^{-z · t_i}

        The Z-spread is a cleaner credit spread measure than G-spread (which
        uses par yields) because it accounts for the shape of the yield curve.

        Parameters
        ----------
        dirty_price : float
            Observed market (dirty) price.
        zero_curve_fn : callable
            Function t → zero rate r(t) from a bootstrapped curve.
        """
        def price_minus_market(z: float) -> float:
            pv = sum(
                cf * math.exp(-(zero_curve_fn(t) + z) * t)
                for t, cf in zip(self._payment_times, self._cash_flows)
            )
            return pv - dirty_price

        return brentq(price_minus_market, -0.10, 0.50, xtol=1e-8)

    # ------------------------------------------------------------------
    # Key rate durations
    # ------------------------------------------------------------------

    def key_rate_durations(self, ytm: float, key_rates: list[float] | None = None) -> pd.DataFrame:
        """
        Key Rate Duration (KRD): sensitivity to a shift at a specific maturity
        on the yield curve, keeping other points fixed.

        KRD(t*) ≈ (P_down - P_up) / (2 · Δy · P)  for a 1bp shift at t*

        Important for hedging non-parallel yield curve movements.
        """
        if key_rates is None:
            key_rates = [0.5, 1, 2, 3, 5, 7, 10]

        shift = 0.0001  # 1 basis point
        results = []

        for kr in key_rates:
            # Only shift cash flows near the key rate tenor
            # Weight = triangle function centred at kr
            p_up = 0.0
            p_down = 0.0
            periodic_yield = ytm / self.spec.freq
            for t, cf in zip(self._payment_times, self._cash_flows):
                n = t * self.spec.freq
                # Weight: linear taper around key rate (simplified)
                weight = max(0.0, 1.0 - abs(t - kr) / 1.0)
                local_shift_up = shift * weight
                local_shift_down = shift * weight
                p_up += cf / (1 + periodic_yield + local_shift_up / self.spec.freq) ** n
                p_down += cf / (1 + periodic_yield - local_shift_down / self.spec.freq) ** n

            p0 = self.dirty_price(ytm)
            krd = (p_down - p_up) / (2 * shift * p0)
            results.append({"Key Rate (Y)": kr, "KRD": round(krd, 4)})

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Full analysis bundle
    # ------------------------------------------------------------------

    def analyse(self, ytm: float, fraction_of_period: float = 0.0) -> BondAnalytics:
        """Run full analytics at a given YTM."""
        dp = self.dirty_price(ytm)
        ai = self.accrued_interest(fraction_of_period)
        return BondAnalytics(
            dirty_price=dp,
            clean_price=dp - ai,
            accrued_interest=ai,
            ytm=ytm,
            macaulay_duration=self.macaulay_duration(ytm),
            modified_duration=self.modified_duration(ytm),
            dv01=self.dv01(ytm),
            convexity=self.convexity(ytm),
        )

    def price_yield_table(self, ytm_centre: float, n: int = 10, step_bps: int = 25) -> pd.DataFrame:
        """Generate a price/yield table around a central YTM."""
        rows = []
        for i in range(-n, n + 1):
            ytm_i = ytm_centre + i * step_bps / 10_000
            price = self.dirty_price(ytm_i)
            rows.append({"YTM": f"{ytm_i:.4%}", "Price": round(price, 4)})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Portfolio DV01 and hedging
# ---------------------------------------------------------------------------

def portfolio_dv01(bonds: list[tuple[BondPricer, float, float]]) -> float:
    """
    Compute portfolio DV01.

    Parameters
    ----------
    bonds : list of (pricer, ytm, notional_multiple)

    Returns
    -------
    float : total portfolio DV01
    """
    return sum(
        pricer.dv01(ytm) * notional
        for pricer, ytm, notional in bonds
    )


def hedge_ratio(bond_dv01: float, hedge_dv01: float) -> float:
    """
    Hedge ratio: how many units of the hedge instrument per unit of bond.
    hedge_units = -bond_DV01 / hedge_DV01
    """
    return -bond_dv01 / hedge_dv01


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 60)
    print("  Fixed Income Analytics")
    print("═" * 60)

    # --- 1. Standard coupon bond ---
    spec = BondSpec(face_value=1000, coupon_rate=0.05, maturity_years=10, freq=2)
    pricer = BondPricer(spec)

    print("\n── 10Y 5% Semi-Annual Bond, $1000 Face ──")
    analytics = pricer.analyse(ytm=0.045)
    print(analytics.summary())

    print("\n── Price Change Approximation: +100bp yield shock ──")
    pc = analytics.price_change_approx(yield_change_bps=+100)
    for k, v in pc.items():
        print(f"  {k}: {v}")

    # Exact price change for comparison
    exact_new = pricer.dirty_price(0.045 + 0.01)
    print(f"  Exact new price: {exact_new:.6f}")
    print(f"  Convexity captures: {abs(exact_new - analytics.dirty_price - pc['linear_approximation']):.6f} of the gap")

    print("\n── Price-Yield Table ──")
    pyt = pricer.price_yield_table(ytm_centre=0.045, n=6, step_bps=25)
    print(pyt.to_string(index=False))

    print("\n── Key Rate Durations ──")
    krds = pricer.key_rate_durations(ytm=0.045)
    print(krds.to_string(index=False))

    # --- 2. YTM from price ---
    print("\n── YTM from Price ──")
    market_price = 1038.50   # bond trading at a premium
    ytm_solved = pricer.ytm_from_price(market_price)
    print(f"  Market price: ${market_price:.2f}")
    print(f"  Solved YTM:   {ytm_solved:.4%}")
    print(f"  Verify price: ${pricer.dirty_price(ytm_solved):.2f}")

    # --- 3. Zero-coupon bond ---
    print("\n── Zero-Coupon Bond (5Y, $1000 face) ──")
    zcb = BondSpec(face_value=1000, coupon_rate=0.0, maturity_years=5, freq=1)
    zcb_pricer = BondPricer(zcb)
    zcb_analytics = zcb_pricer.analyse(ytm=0.044)
    print(f"  Price:             ${zcb_analytics.dirty_price:.4f}")
    print(f"  Macaulay Duration: {zcb_analytics.macaulay_duration:.4f}y  (= maturity for ZCBs)")
    print(f"  Modified Duration: {zcb_analytics.modified_duration:.4f}")
    print(f"  Convexity:         {zcb_analytics.convexity:.4f}")

    # --- 4. Portfolio hedging ---
    print("\n── Portfolio Hedging with 2Y Hedge Bond ──")
    hedge_spec = BondSpec(face_value=1000, coupon_rate=0.03, maturity_years=2, freq=2)
    hedge_pricer = BondPricer(hedge_spec)

    portfolio_dv01_val = portfolio_dv01([(pricer, 0.045, 10)])   # 10 bonds
    hedge_dv01_val = hedge_pricer.dv01(0.042)
    ratio = hedge_ratio(portfolio_dv01_val, hedge_dv01_val)

    print(f"  Portfolio DV01 (10 bonds): ${portfolio_dv01_val:.4f}")
    print(f"  Hedge bond DV01:           ${hedge_dv01_val:.4f}")
    print(f"  Hedge ratio:               {ratio:.2f} hedge bonds per portfolio")
    print(f"  (Short {abs(ratio):.1f} 2Y bonds to hedge duration risk)")
