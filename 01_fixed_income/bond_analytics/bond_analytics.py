"""
Fixed Income Bond Analytics
=============================
Full suite of bond pricing and risk metrics:
  - Clean / dirty price from yield
  - Yield-to-maturity (Newton-Raphson solver)
  - Macaulay duration, Modified duration, Effective duration
  - Dollar duration (DV01/PV01)
  - Convexity
  - Z-spread (OAS proxy) over a benchmark curve
  - Price sensitivity to parallel and non-parallel shifts

Why this matters:
  Duration and convexity are the first-order and second-order approximations
  of price sensitivity to yield changes — the bedrock of fixed income risk
  management. DV01 is how traders and risk managers quote rate sensitivity.

Taylor expansion approximation:
  ΔP/P ≈ -D_mod · Δy + ½ · C · (Δy)²

References:
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 4.
  - Fabozzi, F.J. (2012). The Handbook of Fixed Income Securities, 8th ed.
  - Tuckman, B. & Serrat, A. (2022). Fixed Income Securities, 4th ed.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Bond data container
# ---------------------------------------------------------------------------

@dataclass
class Bond:
    """
    A fixed-rate bullet bond.

    Parameters
    ----------
    face_value : float      Par / face amount (typically 100 or 1000).
    coupon_rate : float     Annual coupon rate as decimal (e.g. 0.05 for 5%).
    maturity : float        Years to maturity from today.
    freq : int              Coupon payments per year (2 = semi-annual US, 1 = annual Europe).
    """
    face_value: float
    coupon_rate: float
    maturity: float
    freq: int = 2           # semi-annual by default (US convention)
    accrued_days: float = 0 # days since last coupon (for dirty price)

    @property
    def coupon(self) -> float:
        """Periodic coupon payment."""
        return self.face_value * self.coupon_rate / self.freq

    @property
    def n_periods(self) -> int:
        """Total number of coupon periods remaining."""
        return round(self.maturity * self.freq)

    def cash_flows(self) -> list[tuple[float, float]]:
        """
        Return list of (time_in_years, cash_flow) tuples.
        The final cash flow includes the face value redemption.
        """
        dt = 1.0 / self.freq
        cfs = []
        for i in range(1, self.n_periods + 1):
            t = i * dt
            cf = self.coupon + (self.face_value if i == self.n_periods else 0.0)
            cfs.append((t, cf))
        return cfs


# ---------------------------------------------------------------------------
# Bond Pricer
# ---------------------------------------------------------------------------

class BondPricer:
    """
    Prices a fixed-rate bond and computes full risk analytics.

    Usage
    -----
    >>> bond = Bond(face_value=1000, coupon_rate=0.05, maturity=10, freq=2)
    >>> pricer = BondPricer(bond)
    >>> price = pricer.dirty_price(ytm=0.045)
    >>> print(pricer.risk_report(ytm=0.045))
    """

    def __init__(self, bond: Bond):
        self.bond = bond

    # ------------------------------------------------------------------
    # Price from yield
    # ------------------------------------------------------------------

    def dirty_price(self, ytm: float) -> float:
        """
        Full (dirty) price: present value of all future cash flows
        discounted at the yield-to-maturity.

        P = sum_i [ CF_i / (1 + y/m)^(i) ]

        where m is the payment frequency and i indexes coupon periods.
        Dirty price includes accrued interest.
        """
        b = self.bond
        y_period = ytm / b.freq
        price = 0.0
        for i, (t, cf) in enumerate(b.cash_flows(), start=1):
            price += cf / (1 + y_period) ** i
        return price

    def clean_price(self, ytm: float) -> float:
        """
        Clean (flat) price = dirty price - accrued interest.
        Accrued interest = coupon * (days_since_last_coupon / days_in_period)
        """
        accrued = self.bond.coupon * (self.bond.accrued_days / (365 / self.bond.freq))
        return self.dirty_price(ytm) - accrued

    # ------------------------------------------------------------------
    # Yield from price
    # ------------------------------------------------------------------

    def yield_to_maturity(self, dirty_price: float, tol: float = 1e-10) -> float:
        """
        Solve for YTM given a dirty price using Brent's method.

        YTM is the internal rate of return that makes PV(cash flows) = price.
        The yield is expressed as an annual rate (not per-period).
        """
        b = self.bond

        def price_diff(ytm: float) -> float:
            return self.dirty_price(ytm) - dirty_price

        # Bracket: try to find sign change
        lo, hi = 0.0001, 0.9999
        try:
            return brentq(price_diff, lo, hi, xtol=tol, maxiter=500)
        except ValueError:
            raise ValueError(
                f"Could not bracket YTM for price={dirty_price:.4f}. "
                f"Check that the bond parameters are consistent."
            )

    # ------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------

    def macaulay_duration(self, ytm: float) -> float:
        """
        Macaulay Duration: time-weighted present value of cash flows.

        D_mac = sum_i [t_i * PV(CF_i)] / P

        Units: years. Interpretation: the weighted-average time to receive
        the bond's cash flows — the 'effective maturity' of the bond.
        """
        b = self.bond
        y_period = ytm / b.freq
        P = self.dirty_price(ytm)
        weighted_t = 0.0
        for i, (t, cf) in enumerate(b.cash_flows(), start=1):
            pv_cf = cf / (1 + y_period) ** i
            weighted_t += (i / b.freq) * pv_cf  # time in years
        return weighted_t / P

    def modified_duration(self, ytm: float) -> float:
        """
        Modified Duration: D_mac / (1 + y/m)

        Interpretation: the percentage price change per unit change in yield.
        ΔP/P ≈ -D_mod · Δy

        For a bond with D_mod = 7 and a 100bp yield rise:
        ΔP/P ≈ -7 × 0.01 = -7%
        """
        D_mac = self.macaulay_duration(ytm)
        return D_mac / (1 + ytm / self.bond.freq)

    def effective_duration(self, ytm: float, dy: float = 0.0001) -> float:
        """
        Effective Duration via finite difference (numerical first derivative).
        Preferred for bonds with embedded options (callable, puttable).

        D_eff = (P(y - dy) - P(y + dy)) / (2 * P * dy)
        """
        P = self.dirty_price(ytm)
        P_up = self.dirty_price(ytm + dy)
        P_down = self.dirty_price(ytm - dy)
        return (P_down - P_up) / (2 * P * dy)

    def dollar_duration(self, ytm: float) -> float:
        """
        Dollar Duration (DD) = D_mod * P / 100

        This is the dollar price change per 1% (100bp) change in yield.
        The related DV01 = DD / 100 is the dollar change per 1bp.
        """
        return self.modified_duration(ytm) * self.dirty_price(ytm) / 100

    def dv01(self, ytm: float) -> float:
        """
        DV01 (Dollar Value of 1 Basis Point):
        Dollar price change for a 1bp (0.01%) yield move.

        DV01 = D_mod * P / 10,000  = dollar_duration / 100
        """
        P_up = self.dirty_price(ytm + 0.0001)
        P_down = self.dirty_price(ytm - 0.0001)
        return (P_down - P_up) / 2  # sign convention: positive for long bonds

    # ------------------------------------------------------------------
    # Convexity
    # ------------------------------------------------------------------

    def convexity(self, ytm: float) -> float:
        """
        Convexity: second-order price sensitivity to yield.

        C = sum_i [ t_i * (t_i + 1/m) * PV(CF_i) ] / (P * (1 + y/m)²)

        Interpretation: bonds with higher convexity gain more when yields fall
        and lose less when yields rise — convexity is always beneficial for long bonds.
        The second-order price approximation:
        ΔP/P ≈ -D_mod·Δy + ½·C·(Δy)²
        """
        b = self.bond
        y_period = ytm / b.freq
        P = self.dirty_price(ytm)
        conv_sum = 0.0
        for i, (t, cf) in enumerate(b.cash_flows(), start=1):
            pv_cf = cf / (1 + y_period) ** i
            conv_sum += i * (i + 1) * pv_cf
        return conv_sum / (P * (1 + y_period) ** 2 * b.freq ** 2)

    def effective_convexity(self, ytm: float, dy: float = 0.0001) -> float:
        """Numerical convexity via finite difference (handles option-embedded bonds)."""
        P = self.dirty_price(ytm)
        P_up = self.dirty_price(ytm + dy)
        P_down = self.dirty_price(ytm - dy)
        return (P_up + P_down - 2 * P) / (P * dy ** 2)

    # ------------------------------------------------------------------
    # Z-spread
    # ------------------------------------------------------------------

    def z_spread(self, dirty_price: float, discount_fn: Callable[[float], float]) -> float:
        """
        Z-spread (zero-volatility spread): the constant spread added to every
        point on the risk-free zero curve to price the bond at par.

        Z-spread answers: "How much extra yield does this bond offer above
        the risk-free curve at every maturity?"

        Solve: P = sum_i [ CF_i * exp(-(z(t_i) + Z) * t_i) ]
        where z(t_i) is the benchmark zero rate and Z is the Z-spread.

        Parameters
        ----------
        dirty_price : float
            Observed market price.
        discount_fn : Callable
            A function t → P(0,t) from the risk-free curve.
        """
        b = self.bond

        def price_diff(Z: float) -> float:
            pv = 0.0
            for t, cf in b.cash_flows():
                # Adjust risk-free discount factor by Z-spread
                p_rf = discount_fn(t)
                z_rf = -math.log(p_rf) / t if p_rf > 0 else 0
                pv += cf * math.exp(-(z_rf + Z) * t)
            return pv - dirty_price

        return brentq(price_diff, -0.10, 0.50, xtol=1e-8)

    # ------------------------------------------------------------------
    # Price sensitivity scenarios
    # ------------------------------------------------------------------

    def price_change_approximation(self, ytm: float, dy: float) -> dict:
        """
        First- and second-order approximation of price change for a yield shift dy.

        ΔP ≈ -D_mod · P · Δy          (duration only)
        ΔP ≈ -D_mod · P · Δy + ½ · C · P · (Δy)²  (duration + convexity)

        Also compute the exact price change for comparison.
        """
        P = self.dirty_price(ytm)
        D_mod = self.modified_duration(ytm)
        C = self.convexity(ytm)
        P_new = self.dirty_price(ytm + dy)

        delta_P_exact = P_new - P
        delta_P_dur = -D_mod * P * dy
        delta_P_dur_conv = -D_mod * P * dy + 0.5 * C * P * dy ** 2

        return {
            "yield_shift_bps": dy * 10000,
            "price_initial": round(P, 4),
            "price_final_exact": round(P_new, 4),
            "change_exact": round(delta_P_exact, 4),
            "change_duration_only": round(delta_P_dur, 4),
            "change_dur_plus_convex": round(delta_P_dur_conv, 4),
            "convexity_benefit": round(delta_P_dur_conv - delta_P_dur, 4),
        }

    # ------------------------------------------------------------------
    # Full risk report
    # ------------------------------------------------------------------

    def risk_report(self, ytm: float) -> str:
        b = self.bond
        P = self.dirty_price(ytm)
        lines = [
            "=" * 52,
            f"  Bond Risk Analytics",
            f"  Face: {b.face_value} | Coupon: {b.coupon_rate:.2%} | "
            f"Maturity: {b.maturity}Y | Freq: {b.freq}x/yr",
            "=" * 52,
            f"  {'YTM':<28} {ytm:.4%}",
            f"  {'Dirty Price':<28} {P:.4f}",
            f"  {'Clean Price':<28} {self.clean_price(ytm):.4f}",
            "─" * 52,
            f"  {'Macaulay Duration':<28} {self.macaulay_duration(ytm):.4f} yrs",
            f"  {'Modified Duration':<28} {self.modified_duration(ytm):.4f}",
            f"  {'Effective Duration':<28} {self.effective_duration(ytm):.4f}",
            f"  {'Dollar Duration (per 1%)':<28} ${self.dollar_duration(ytm):.4f}",
            f"  {'DV01 (per 1bp)':<28} ${self.dv01(ytm):.4f}",
            "─" * 52,
            f"  {'Convexity':<28} {self.convexity(ytm):.4f}",
            f"  {'Effective Convexity':<28} {self.effective_convexity(ytm):.4f}",
            "=" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 10-year 5% semi-annual bond
    bond = Bond(face_value=1000, coupon_rate=0.05, maturity=10, freq=2)
    pricer = BondPricer(bond)
    ytm = 0.045  # 4.5% YTM

    print(pricer.risk_report(ytm))

    print("\n── Yield-to-Maturity Recovery ──")
    price = pricer.dirty_price(ytm)
    solved_ytm = pricer.yield_to_maturity(price)
    print(f"  Input YTM:  {ytm:.6%}")
    print(f"  Solved YTM: {solved_ytm:.6%}  (difference: {abs(ytm - solved_ytm):.2e})")

    print("\n── Price Change Scenarios (±100bp, ±200bp) ──")
    print(f"  {'Shift':<12} {'Exact ΔP':>12} {'Dur Only':>12} {'Dur+Conv':>12} {'Conv Benefit':>14}")
    print("  " + "─" * 64)
    for shift_bps in [-200, -100, -50, 50, 100, 200]:
        sc = pricer.price_change_approximation(ytm, shift_bps / 10000)
        print(
            f"  {shift_bps:>+5}bps    "
            f"{sc['change_exact']:>12.4f} "
            f"{sc['change_duration_only']:>12.4f} "
            f"{sc['change_dur_plus_convex']:>12.4f} "
            f"{sc['convexity_benefit']:>14.4f}"
        )

    print("\n── Zero Coupon vs Coupon Bond Duration Comparison ──")
    for maturity in [2, 5, 10, 20, 30]:
        coupon_bond = Bond(face_value=1000, coupon_rate=0.05, maturity=maturity, freq=2)
        zero_bond = Bond(face_value=1000, coupon_rate=0.0001, maturity=maturity, freq=2)
        cp = BondPricer(coupon_bond)
        zp = BondPricer(zero_bond)
        print(
            f"  {maturity:>2}Y: coupon bond D_mod = {cp.modified_duration(0.05):.2f}  "
            f"zero coupon D_mod = {zp.modified_duration(0.05):.2f}  "
            f"(zero → D_mod ≈ T)"
        )
