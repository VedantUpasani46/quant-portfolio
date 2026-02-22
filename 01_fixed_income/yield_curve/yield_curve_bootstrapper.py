"""
Yield Curve Bootstrapper
=========================
Constructs a zero-coupon yield curve from market instrument quotes:
  Tier 1  — Deposit rates (overnight to 1Y)
  Tier 2  — Forward Rate Agreements / Eurodollar futures (1M–2Y)
  Tier 3  — Interest Rate Swaps (2Y–30Y)

Why bootstrapping?
  Market instruments don't directly give us zero rates for arbitrary maturities.
  Bootstrapping strips out the zero curve iteratively: each instrument prices
  exactly at par given the already-known zero rates, allowing us to solve for
  the next unknown discount factor.

Applications:
  - Pricing any fixed income instrument (bonds, swaps, swaptions)
  - Computing forward rates for FRAs and floating legs
  - Building discount curves for derivatives (OIS discounting post-2008)
  - PV01 / DV01 calculation for rate risk

References:
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 4, 6–7.
  - Hagan, P. & West, G. (2006). Interpolation Methods for Curve Construction.
    Applied Mathematical Finance, 13(2), 89–129.
  - Ametrano, F. & Bianchetti, M. (2009). Bootstrapping the Illiquidity.
    SSRN Working Paper.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DayCountConvention(Enum):
    ACT_360 = "ACT/360"       # money market, LIBOR
    ACT_365 = "ACT/365"       # sterling, some govt bonds
    THIRTY_360 = "30/360"     # US corp bonds, fixed swap legs
    ACT_ACT = "ACT/ACT"       # government bonds (ISDA)


class CompoundingFrequency(Enum):
    CONTINUOUS = "continuous"
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4


# ---------------------------------------------------------------------------
# Day count helpers
# ---------------------------------------------------------------------------

def year_fraction(
    t1: float, t2: float,
    convention: DayCountConvention = DayCountConvention.ACT_360
) -> float:
    """
    Compute year fraction between two times (expressed in days from today).
    For simplicity, we treat t1/t2 as fractional years directly.
    In production, these would be actual dates.
    """
    delta = t2 - t1
    if convention == DayCountConvention.ACT_360:
        return delta
    elif convention == DayCountConvention.ACT_365:
        return delta * (360 / 365)
    elif convention == DayCountConvention.THIRTY_360:
        return delta
    else:
        return delta


# ---------------------------------------------------------------------------
# Market instrument data containers
# ---------------------------------------------------------------------------

@dataclass
class DepositRate:
    """
    A cash deposit rate (money market instrument).

    Pricing: P(0,T) = 1 / (1 + r * T)  under ACT/360 simple compounding.
    """
    maturity: float       # in years (e.g. 0.25 for 3 months)
    rate: float           # decimal (e.g. 0.053 for 5.3%)
    day_count: DayCountConvention = DayCountConvention.ACT_360


@dataclass
class SwapRate:
    """
    A par interest rate swap quote.

    The fixed rate R that makes the swap have zero NPV at inception:
    R * sum(tau_i * P(0, t_i)) = (P(0, t_0) - P(0, t_N)) / sum(tau_i * P(0, t_i))

    where tau_i is the day fraction of the i-th fixed coupon period.
    """
    maturity: float       # in years (e.g. 5.0 for 5-year swap)
    rate: float           # par swap rate as decimal
    fixed_freq: int = 2   # payment frequency per year (2 = semi-annual)
    float_freq: int = 4   # floating leg reset frequency


@dataclass
class FRARate:
    """
    Forward Rate Agreement: locks in a rate between start and end.
    """
    start: float          # in years (e.g. 0.25)
    end: float            # in years (e.g. 0.50)
    rate: float           # FRA rate as decimal


# ---------------------------------------------------------------------------
# Interpolation methods
# ---------------------------------------------------------------------------

class Interpolator:
    """
    Wraps SciPy interpolation for the zero curve with multiple methods.
    Supports: linear on log-discount, cubic spline, flat forward.
    """
    def __init__(self, method: str = "log_linear"):
        self.method = method
        self._pillars: list[float] = []
        self._log_df: list[float] = []
        self._spline: CubicSpline | None = None

    def add_point(self, t: float, discount_factor: float):
        """Add a (maturity, discount_factor) pillar to the curve."""
        self._pillars.append(t)
        self._log_df.append(math.log(discount_factor))

    def build(self):
        """Fit the interpolant after all pillars are added."""
        t = np.array(self._pillars)
        log_df = np.array(self._log_df)
        if self.method == "cubic_spline":
            self._spline = CubicSpline(t, log_df, bc_type="not-a-knot")

    def discount_factor(self, t: float) -> float:
        """Return P(0, t) by interpolation."""
        if t <= 0:
            return 1.0
        pillars = np.array(self._pillars)
        log_dfs = np.array(self._log_df)

        if t <= pillars[0]:
            # Linear extrapolation back to t=0
            slope = self._log_df[0] / self._pillars[0]
            return math.exp(slope * t)

        if t >= pillars[-1]:
            # Flat forward extrapolation beyond last pillar
            if len(pillars) >= 2:
                fwd = (log_dfs[-1] - log_dfs[-2]) / (pillars[-1] - pillars[-2])
                return math.exp(log_dfs[-1] + fwd * (t - pillars[-1]))
            return math.exp(log_dfs[-1])

        if self.method == "cubic_spline" and self._spline is not None:
            return math.exp(float(self._spline(t)))
        else:
            # Log-linear (default): linear interpolation on log(P)
            log_df = np.interp(t, pillars, log_dfs)
            return math.exp(log_df)

    def zero_rate(self, t: float,
                  freq: CompoundingFrequency = CompoundingFrequency.CONTINUOUS) -> float:
        """
        Convert discount factor to zero rate with specified compounding.

        Continuous:  P = e^(-r*T)    → r = -ln(P)/T
        Semi-annual: P = (1+r/2)^(-2T) → r = 2*(P^{-1/2T} - 1)
        Annual:      P = (1+r)^(-T)  → r = P^{-1/T} - 1
        """
        P = self.discount_factor(t)
        if P <= 0 or t <= 0:
            return 0.0
        if freq == CompoundingFrequency.CONTINUOUS:
            return -math.log(P) / t
        elif freq == CompoundingFrequency.ANNUAL:
            return P ** (-1 / t) - 1
        elif freq == CompoundingFrequency.SEMI_ANNUAL:
            return 2 * (P ** (-1 / (2 * t)) - 1)
        elif freq == CompoundingFrequency.QUARTERLY:
            return 4 * (P ** (-1 / (4 * t)) - 1)
        return -math.log(P) / t

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Instantaneous forward rate between t1 and t2 (continuously compounded).

        f(t1, t2) = [ln P(0,t1) - ln P(0,t2)] / (t2 - t1)
        """
        P1 = self.discount_factor(t1)
        P2 = self.discount_factor(t2)
        if t2 <= t1 or P2 <= 0:
            return 0.0
        return (math.log(P1) - math.log(P2)) / (t2 - t1)


# ---------------------------------------------------------------------------
# Core bootstrapper
# ---------------------------------------------------------------------------

class YieldCurveBootstrapper:
    """
    Strips a zero-coupon yield curve from deposit rates, FRAs, and par swap rates.

    The bootstrapping algorithm:
    ──────────────────────────────────────────────────────────────────────────
    1. Deposits → direct inversion:
       P(0, T) = 1 / (1 + r * T)

    2. FRAs → given P(0, t_start) is known, solve for P(0, t_end):
       P(0, t_end) = P(0, t_start) / (1 + r_FRA * (t_end - t_start))

    3. Swaps → solve numerically for P(0, T_N) given known P(0, t_i) for i < N:
       R * sum_i [tau_i * P(0, t_i)] = P(0, t_0) - P(0, T_N)
       → P(0, T_N) = P(0, t_0) - R * sum_i [tau_i * P(0, t_i)]
         where all P(0, t_i) for i < N are already bootstrapped.

    Usage
    -----
    >>> bootstrapper = YieldCurveBootstrapper(interpolation='log_linear')
    >>> bootstrapper.add_deposits([DepositRate(0.25, 0.053), ...])
    >>> bootstrapper.add_swaps([SwapRate(2.0, 0.047), ...])
    >>> curve = bootstrapper.build()
    >>> print(f"5Y zero rate: {curve.zero_rate(5.0):.4%}")
    """

    def __init__(self, interpolation: str = "log_linear"):
        self.interp = Interpolator(method=interpolation)
        self._deposits: list[DepositRate] = []
        self._fras: list[FRARate] = []
        self._swaps: list[SwapRate] = []
        self._built = False

        # Anchor: P(0, 0) = 1
        self.interp.add_point(0.0, 1.0)

    def add_deposits(self, deposits: list[DepositRate]) -> "YieldCurveBootstrapper":
        self._deposits.extend(sorted(deposits, key=lambda d: d.maturity))
        return self

    def add_fras(self, fras: list[FRARate]) -> "YieldCurveBootstrapper":
        self._fras.extend(sorted(fras, key=lambda f: f.end))
        return self

    def add_swaps(self, swaps: list[SwapRate]) -> "YieldCurveBootstrapper":
        self._swaps.extend(sorted(swaps, key=lambda s: s.maturity))
        return self

    # ------------------------------------------------------------------
    # Instrument bootstrapping routines
    # ------------------------------------------------------------------

    def _bootstrap_deposit(self, dep: DepositRate):
        """Direct inversion: P(0,T) = 1 / (1 + r*T)"""
        P = 1.0 / (1.0 + dep.rate * dep.maturity)
        self.interp.add_point(dep.maturity, P)

    def _bootstrap_fra(self, fra: FRARate):
        """
        FRA implies forward rate between start and end.
        P(0, end) = P(0, start) / (1 + r_fra * (end - start))
        """
        P_start = self.interp.discount_factor(fra.start)
        tau = fra.end - fra.start
        P_end = P_start / (1.0 + fra.rate * tau)
        self.interp.add_point(fra.end, P_end)

    def _bootstrap_swap(self, swap: SwapRate):
        """
        Par swap pricing equation → solve for the last (unknown) discount factor.

        For a swap with fixed coupon R, fixed leg cash flows at t_1, ..., t_N:
        PV(fixed) = R * sum_i [tau_i * P(0, t_i)] = P(0, t_0) - P(0, t_N)
        where P(0, t_0) = 1 (funded from spot).

        Rearranging: P(0, t_N) = 1 - R * sum_{i=1}^{N-1} [tau_i * P(0, t_i)]
                                   / (1 + R * tau_N)

        All P(0, t_i) for i < N are known; this gives P(0, t_N) analytically.
        """
        dt = 1.0 / swap.fixed_freq
        coupon = swap.rate / swap.fixed_freq  # coupon per period

        # Sum of known annuity payments (all intermediate periods)
        annuity = 0.0
        t = dt
        while t < swap.maturity - 1e-9:
            annuity += coupon * self.interp.discount_factor(t)
            t = round(t + dt, 10)

        # Solve for last discount factor
        tau_N = dt
        P_N = (1.0 - annuity) / (1.0 + coupon)
        self.interp.add_point(swap.maturity, P_N)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Interpolator:
        """
        Execute the full bootstrap sequence.
        Order: deposits → FRAs → swaps (by maturity within each tier).
        """
        for dep in self._deposits:
            self._bootstrap_deposit(dep)
        for fra in self._fras:
            self._bootstrap_fra(fra)
        for swap in self._swaps:
            self._bootstrap_swap(swap)

        self.interp.build()
        self._built = True
        return self.interp

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_curve(
        self,
        maturities: list[float] | None = None,
        freq: CompoundingFrequency = CompoundingFrequency.CONTINUOUS
    ):
        """Print zero rates, discount factors, and forward rates at key maturities."""
        if not self._built:
            self.build()

        if maturities is None:
            maturities = [0.08, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

        print(f"\n{'Maturity':>10} {'Zero Rate':>12} {'Discount F.':>14} {'Fwd Rate (1Y)':>16}")
        print("─" * 56)
        for T in maturities:
            z = self.interp.zero_rate(T, freq)
            P = self.interp.discount_factor(T)
            fwd = self.interp.forward_rate(T, T + 1.0)
            print(f"{T:>10.2f} {z:>12.4%} {P:>14.6f} {fwd:>16.4%}")

    def par_rate_check(self, swap_maturity: float, fixed_freq: int = 2) -> float:
        """
        Verify bootstrapping quality: compute the implied par swap rate from
        the bootstrapped curve and compare to the input market quote.

        A good bootstrap will exactly reprice the input instruments.
        """
        dt = 1.0 / fixed_freq
        annuity = 0.0
        t = dt
        while t <= swap_maturity + 1e-9:
            annuity += dt * self.interp.discount_factor(t)
            t = round(t + dt, 10)
        par = (1.0 - self.interp.discount_factor(swap_maturity)) / annuity
        return par


# ---------------------------------------------------------------------------
# Sample market data (representative USD swap curve, stylised)
# ---------------------------------------------------------------------------

def build_sample_usd_curve() -> tuple[YieldCurveBootstrapper, Interpolator]:
    """
    Build a representative USD yield curve from stylised market data.
    Rates approximate a typical USD swap curve.
    """
    deposits = [
        DepositRate(maturity=1/12,  rate=0.0530),   # 1M
        DepositRate(maturity=3/12,  rate=0.0533),   # 3M
        DepositRate(maturity=6/12,  rate=0.0527),   # 6M
        DepositRate(maturity=1.0,   rate=0.0512),   # 1Y
    ]

    swaps = [
        SwapRate(maturity=2.0,   rate=0.0475),
        SwapRate(maturity=3.0,   rate=0.0455),
        SwapRate(maturity=5.0,   rate=0.0435),
        SwapRate(maturity=7.0,   rate=0.0428),
        SwapRate(maturity=10.0,  rate=0.0425),
        SwapRate(maturity=15.0,  rate=0.0420),
        SwapRate(maturity=20.0,  rate=0.0415),
        SwapRate(maturity=30.0,  rate=0.0408),
    ]

    bootstrapper = YieldCurveBootstrapper(interpolation="log_linear")
    bootstrapper.add_deposits(deposits).add_swaps(swaps)
    curve = bootstrapper.build()
    return bootstrapper, curve


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 60)
    print("  USD Yield Curve Bootstrapper")
    print("  Deposits (1M–1Y) + Swaps (2Y–30Y)")
    print("═" * 60)

    bs, curve = build_sample_usd_curve()
    bs.print_curve(freq=CompoundingFrequency.CONTINUOUS)

    print("\n── Par Rate Verification (bootstrapped curve reprices inputs) ──")
    for T in [2, 3, 5, 7, 10, 15, 20, 30]:
        implied = bs.par_rate_check(float(T))
        print(f"  {T:>2}Y swap: implied par rate = {implied:.4%}")

    print("\n── Forward Rate Structure (implied 1Y rates) ──")
    print(f"  {'Period':<12} {'Forward Rate':>14}")
    print("  " + "─" * 28)
    for t in [1, 2, 3, 4, 5, 7, 9]:
        fwd = curve.forward_rate(float(t), float(t + 1))
        print(f"  {t}Y–{t+1}Y      {fwd:>14.4%}")
