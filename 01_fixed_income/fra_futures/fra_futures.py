"""
FRA, Bond Futures, Cheapest-to-Deliver, and Convexity Adjustment
==================================================================
Three classical fixed-income instruments asked in every rates interview:

1. FRA (Forward Rate Agreement)
   Lock in a borrowing rate for a future period [T₁, T₂].
   FRA rate = simply-compounded forward rate extracted from the yield curve.
   Cash settlement at T₁:
     Payoff = N·τ·(LIBOR − K) / (1 + τ·LIBOR)   [discounted back from T₂]

2. Bond Futures
   Exchange-traded futures on a notional Treasury bond (e.g. T-Note, T-Bond).
   The seller delivers the cheapest bond from a basket of eligible bonds.
   Futures price quoted as a "standard" 6% coupon bond; conversion factors
   normalise actual coupons.

3. Cheapest-to-Deliver (CTD)
   The seller of a bond futures contract profits by delivering the bond
   with the highest: invoice price − futures price × conversion factor.
   This is the bond where: (market price / conversion factor) is LOWEST.

4. Convexity Adjustment (Futures vs Forwards)
   Futures price ≠ Forward bond price.
   Futures are daily-settled (marked-to-market); forwards are settled at maturity.
   Daily settlement creates a correlation between P&L and discounting →
   the futures price is LOWER than the forward price by a convexity adjustment.

   Convexity Adjustment = −½ · σ_r² · T_futures · T_bond · B(T_bond)
   where B = bond duration, σ_r = rate volatility.

   In practice: futures price = forward price × exp(−½·σ_r²·T₁·T₂)
   (exact formula from Jarrow-Oldfield or Hull-White)

Why this matters:
  - Every rate desk trades bond futures (most liquid rate instrument in the world)
  - The CTD bond determines which bond hedges the futures position (conversion factor)
  - Misunderstanding the convexity adjustment leads to P&L errors on large books
  - Goldman Strats, JPM, Citi, PIMCO, Pimco all test this in interviews

References:
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 6.
  - Tuckman, B. & Serrat, A. (2012). Fixed Income Securities, Ch. 4–5.
  - Burghardt, G. et al. (1994). The Treasury Bond Basis. McGraw-Hill.
  - Jarrow, R. & Oldfield, G. (1981). Forward Contracts and Futures Contracts.
    Journal of Financial Economics 9(4), 373–382.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Yield curve
# ---------------------------------------------------------------------------

class YieldCurve:
    def __init__(self, tenors: np.ndarray, zero_rates: np.ndarray):
        self.tenors = np.array(tenors)
        self.zero_rates = np.array(zero_rates)
        self._interp = interp1d(tenors, zero_rates, bounds_error=False,
                                fill_value=(zero_rates[0], zero_rates[-1]))

    def zero_rate(self, T: float) -> float:
        return float(self._interp(max(T, 1e-6)))

    def df(self, T: float) -> float:
        return math.exp(-self.zero_rate(T) * T) if T > 0 else 1.0

    def simple_forward(self, T1: float, T2: float) -> float:
        """Simply-compounded forward rate for period [T1, T2]."""
        tau = T2 - T1
        return (self.df(T1) / self.df(T2) - 1) / tau

    def cont_forward(self, T1: float, T2: float) -> float:
        return (math.log(self.df(T1)) - math.log(self.df(T2))) / (T2 - T1)


# ---------------------------------------------------------------------------
# 1. FRA (Forward Rate Agreement)
# ---------------------------------------------------------------------------

@dataclass
class FRAResult:
    T1: float              # start date (years)
    T2: float              # end date (years)
    fra_rate: float        # fair FRA rate (= forward LIBOR)
    fixed_rate: float      # actual fixed rate in the contract
    notional: float
    pv: float              # PV of the FRA (to payer of fixed rate)
    settlement_amount: float  # cash at T1

    def summary(self) -> str:
        tau = self.T2 - self.T1
        return (
            f"  FRA {self.T1:.2f}×{self.T2:.2f}"
            f"  | Fair rate: {self.fra_rate:.4%}"
            f"  | Fixed: {self.fixed_rate:.4%}"
            f"  | τ={tau:.4f}Y"
            f"  | PV: ${self.pv:,.0f}"
            f"  | Settlement @ T1: ${self.settlement_amount:,.0f}"
        )


def price_fra(
    T1: float,
    T2: float,
    fixed_rate: float,
    notional: float,
    curve: YieldCurve,
    payer: bool = True,   # payer receives LIBOR, pays fixed
) -> FRAResult:
    """
    FRA price in single-curve framework.

    Fair FRA rate = simply-compounded forward rate:
      F = [P(0,T1)/P(0,T2) − 1] / τ

    PV at T2: N·τ·(F − K)·P(0,T2)
    Settlement at T1: N·τ·(F − K) / (1 + τ·F)·P(0,T1)
    """
    tau = T2 - T1
    F = curve.simple_forward(T1, T2)
    sign = 1 if payer else -1

    pv_t2 = sign * notional * tau * (F - fixed_rate) * curve.df(T2)
    settlement = sign * notional * tau * (F - fixed_rate) / (1 + tau * F)

    return FRAResult(T1=T1, T2=T2, fra_rate=F, fixed_rate=fixed_rate,
                     notional=notional, pv=pv_t2,
                     settlement_amount=settlement * curve.df(T1))


# ---------------------------------------------------------------------------
# 2. Bond analytics (needed for futures)
# ---------------------------------------------------------------------------

def bond_price(coupon: float, face: float, maturity: float,
               freq: int, ytm: float) -> float:
    """Bond price given YTM (flat yield curve)."""
    dt = 1.0 / freq
    cf = coupon * face / freq
    pv = 0.0
    t = dt
    while t <= maturity + 1e-8:
        payment = cf + (face if abs(t - maturity) < 1e-8 else 0)
        pv += payment / (1 + ytm / freq) ** (t * freq)
        t += dt
    return pv


def bond_price_curve(coupon: float, face: float, maturity: float,
                     freq: int, curve: YieldCurve) -> float:
    """Bond price discounted using a full yield curve."""
    dt = 1.0 / freq
    cf = coupon * face / freq
    pv = 0.0
    t = dt
    while t <= maturity + 1e-8:
        payment = cf + (face if abs(t - maturity) < 1e-8 else 0)
        pv += payment * curve.df(t)
        t += dt
    return pv


def bond_duration(coupon: float, face: float, maturity: float,
                  freq: int, ytm: float) -> dict:
    """Macaulay and modified duration."""
    dt = 1.0 / freq
    cf = coupon * face / freq
    price = bond_price(coupon, face, maturity, freq, ytm)
    mac_dur = 0.0
    t = dt
    while t <= maturity + 1e-8:
        payment = cf + (face if abs(t - maturity) < 1e-8 else 0)
        pv_cf = payment / (1 + ytm / freq) ** (t * freq)
        mac_dur += t * pv_cf
        t += dt
    mac_dur /= price
    mod_dur = mac_dur / (1 + ytm / freq)
    dv01 = mod_dur * price / 10000
    return {"macaulay": mac_dur, "modified": mod_dur, "dv01": dv01, "price": price}


# ---------------------------------------------------------------------------
# 3. Bond Futures: Conversion Factors and CTD
# ---------------------------------------------------------------------------

def conversion_factor(coupon: float, maturity_years: float,
                      freq: int = 2,
                      standard_coupon: float = 0.06) -> float:
    """
    CBOT/CME conversion factor: price of the bond assuming YTM = 6%,
    normalised to face value 1.0.

    The CF is designed to make all deliverable bonds roughly equivalent
    at the standard 6% coupon rate.
    """
    return bond_price(coupon, 1.0, maturity_years, freq, standard_coupon)


def futures_invoice_price(
    futures_price: float,    # quoted futures price (per $100 face)
    cf: float,               # conversion factor
    accrued: float = 0.0,   # accrued interest on delivery date
) -> float:
    """
    Invoice price = futures_price × conversion_factor + accrued interest.
    This is what the futures seller receives when delivering the bond.
    """
    return futures_price * cf / 100 + accrued


def find_ctd(
    bonds: list[dict],       # list of deliverable bonds
    futures_price: float,    # current quoted futures price
    curve: YieldCurve,
    delivery_date: float,    # years to delivery
) -> pd.DataFrame:
    """
    Find the Cheapest-to-Deliver bond.
    CTD maximises: invoice_price − market_price (net profit to seller)
    Equivalently: minimise market_price / conversion_factor.

    For each bond:
      Invoice price  = futures_price × CF + accrued
      Market price   = full (dirty) price today
      Net basis      = clean_price − futures_price × CF  (should be ≈ 0 for CTD)
      Delivery option = max(0, futures_price × CF − clean_price)
    """
    records = []
    for b in bonds:
        # Current market price (using yield curve)
        mkt_price = bond_price_curve(b["coupon"], 100, b["maturity"], 2, curve)

        # Conversion factor
        cf = conversion_factor(b["coupon"], b["maturity"] - delivery_date)

        # Invoice price per $100 face
        invoice = futures_price * cf

        # Gross basis (market − invoice): positive = expensive to deliver
        gross_basis = mkt_price - invoice

        # Net basis = gross basis − carry (simplified: zero carry here)
        carry = (b["coupon"] * 100 / 2 - mkt_price * curve.simple_forward(0, delivery_date) * 0.5)
        net_basis = gross_basis - carry * delivery_date

        # CTD score: lower price/CF ratio = cheaper to deliver
        ctd_score = mkt_price / cf

        records.append({
            "bond":        b["name"],
            "coupon":      b["coupon"],
            "maturity":    b["maturity"],
            "mkt_price":   mkt_price,
            "conv_factor": cf,
            "invoice":     invoice,
            "gross_basis": gross_basis,
            "net_basis":   net_basis,
            "ctd_score":   ctd_score,
            "is_ctd":      False,
        })

    df = pd.DataFrame(records)
    ctd_idx = df["ctd_score"].idxmin()
    df.loc[ctd_idx, "is_ctd"] = True
    return df


# ---------------------------------------------------------------------------
# 4. Convexity Adjustment
# ---------------------------------------------------------------------------

def convexity_adjustment(
    T_futures: float,        # futures expiry (years)
    T_bond: float,           # bond maturity (years)
    rate_vol: float,         # annualised short-rate vol
    mean_reversion: float = 0.10,  # Hull-White mean reversion speed
) -> float:
    """
    Convexity adjustment: futures price - forward price.

    Under Hull-White one-factor model (Brigo-Mercurio, Ch. 2):
      Adj = −½ · σ² · B(T₁, T₂)² · (e^{−2a·T₁} − e^{−2a·T₁}) / (2a)
            where B(T₁,T₂) = (1 − e^{−a(T₂−T₁)}) / a

    Simplified (a→0, flat rate):
      Adj ≈ −½ · σ_r² · T_futures · T_bond · Duration_bond

    Sign: futures price = forward price + adj (adj < 0, so futures < forward)
    Equivalently: futures yield > forward yield by convexity adjustment.
    """
    a = mean_reversion
    # Hull-White exact formula for the yield convexity adjustment
    B12 = (1 - math.exp(-a * T_bond)) / a
    term = (1 - math.exp(-2 * a * T_futures)) / (2 * a)
    adj = -0.5 * rate_vol**2 * B12**2 * term

    # Also compute simplified approximation
    adj_simple = -0.5 * rate_vol**2 * T_futures * T_bond

    return {"exact_adj": adj, "simple_adj": adj_simple}


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 68)
    print("  FRA, Bond Futures, CTD, and Convexity Adjustment")
    print("  Interest rate desk essentials")
    print("═" * 68)

    tenors     = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    zero_rates = np.array([0.0430, 0.0435, 0.0438, 0.044, 0.0445, 0.045,
                            0.046, 0.047, 0.048, 0.050, 0.051])
    curve = YieldCurve(tenors, zero_rates)

    # ── FRA pricing ───────────────────────────────────────────────
    print(f"\n── Forward Rate Agreements ──")
    print(f"\n  Standard FRAs on 3-month LIBOR (N×M = start in N months):")
    print(f"\n  {'FRA':>8} {'FRA Rate':>10} {'Payer PV':>12} {'Rcvr PV':>12} {'Settlement':>12}")
    print("  " + "─" * 58)

    fras = [
        (0.25, 0.50, "3×6"),
        (0.50, 0.75, "6×9"),
        (0.75, 1.00, "9×12"),
        (1.00, 1.25, "12×15"),
        (2.00, 2.25, "24×27"),
    ]
    notional = 10_000_000
    for T1, T2, label in fras:
        payer = price_fra(T1, T2, 0.046, notional, curve, payer=True)
        recvr = price_fra(T1, T2, 0.046, notional, curve, payer=False)
        print(f"  {label:>8} {payer.fra_rate:>10.4%} ${payer.pv:>10,.0f} "
              f"${recvr.pv:>10,.0f} ${payer.settlement_amount:>10,.0f}")

    # Show at-the-money FRA (fair rate = contract rate → PV = 0)
    atm_fra = price_fra(0.25, 0.50, curve.simple_forward(0.25, 0.5), notional, curve)
    print(f"\n  At-money FRA (K = {atm_fra.fra_rate:.4%}): PV = ${atm_fra.pv:,.2f} ≈ 0 ✓")

    # ── Bond futures: CTD analysis ────────────────────────────────
    print(f"\n── Bond Futures: Cheapest-to-Deliver Analysis ──")
    print(f"  (Jun 10Y T-Note futures, delivery in 3 months)")

    deliverable_bonds = [
        {"name": "2.875% Nov 2030", "coupon": 0.02875, "maturity": 7.5},
        {"name": "3.125% Nov 2031", "coupon": 0.03125, "maturity": 8.5},
        {"name": "4.000% Nov 2032", "coupon": 0.04000, "maturity": 9.5},
        {"name": "4.500% Aug 2033", "coupon": 0.04500, "maturity": 10.5},
        {"name": "5.000% Nov 2033", "coupon": 0.05000, "maturity": 11.0},
    ]
    futures_px = 109.25   # quoted futures price
    delivery = 0.25

    ctd_df = find_ctd(deliverable_bonds, futures_px, curve, delivery)

    print(f"\n  Futures price: {futures_px}")
    print(f"\n  {'Bond':>25} {'Mkt Px':>8} {'CF':>8} {'Invoice':>8} "
          f"{'Gross Basis':>12} {'CTD?':>6}")
    print("  " + "─" * 72)
    for _, row in ctd_df.iterrows():
        ctd_flag = "← CTD" if row["is_ctd"] else ""
        print(f"  {row['bond']:>25} {row['mkt_price']:>8.4f} {row['conv_factor']:>8.6f} "
              f"{row['invoice']:>8.4f} {row['gross_basis']:>12.4f} {ctd_flag}")

    ctd = ctd_df[ctd_df["is_ctd"]].iloc[0]
    print(f"\n  CTD: {ctd['bond']}")
    print(f"  Market price: {ctd['mkt_price']:.4f}")
    print(f"  Invoice price: {ctd['invoice']:.4f}")
    print(f"  Gross basis: {ctd['gross_basis']:.4f} (lower = more attractive to deliver)")
    print(f"\n  DV01 hedge ratio (futures vs CTD):")
    dur = bond_duration(ctd["coupon"], 100, ctd["maturity"], 2,
                        ytm=curve.zero_rate(ctd["maturity"]))
    dv01_ctd  = dur["dv01"]
    dv01_fut  = dv01_ctd * ctd["conv_factor"] / 100
    contracts = 1 / dv01_fut
    print(f"  CTD DV01:     ${dv01_ctd:.4f} per $100 face")
    print(f"  Futures DV01: ${dv01_fut:.4f} per $100 notional")
    print(f"  To hedge $1M DV01: short {1_000_000 / (dv01_fut * 100_000):.1f} contracts")

    # ── Convexity adjustment ──────────────────────────────────────
    print(f"\n── Futures-Forward Convexity Adjustment ──")
    print(f"\n  Futures yield > forward yield by the convexity adjustment.")
    print(f"  (Futures = daily-settled; forwards = settled at maturity)")
    print(f"  Under Hull-White: σ_r = 1%, mean reversion a = 0.10")
    print(f"\n  {'Futures Expiry':>15} {'Bond Tenor':>12} {'Exact Adj (bp)':>16} {'Simple Adj (bp)':>17}")
    print("  " + "─" * 62)
    for T_fut, T_bond in [(0.25, 1), (0.25, 5), (0.25, 10), (0.5, 5), (0.5, 10), (1.0, 10), (2.0, 10)]:
        adj = convexity_adjustment(T_fut, T_bond, rate_vol=0.01)
        print(f"  {T_fut:>15.2f}Y {T_bond:>12.0f}Y "
              f"{adj['exact_adj']*10000:>16.4f}bp {adj['simple_adj']*10000:>17.4f}bp")

    print(f"""
── Why Convexity Adjustment Matters ──

  A 10-year Eurodollar futures contract expiring in 2 years:
  Convexity adj ≈ {convexity_adjustment(2, 10, 0.01)['exact_adj']*10000:.1f}bp

  If you ignore this and use futures yield as a proxy for forward yield:
    → You overestimate the forward rate by {abs(convexity_adjustment(2,10,0.01)['exact_adj']*10000):.1f}bp
    → On a $1B book of 10Y swaps, this is ~$10M mispricing

  Goldman Strats interview classic:
  Q: "Why is the Eurodollar futures yield above the FRA rate?"
  A: "Daily mark-to-market creates a negative correlation between P&L
      and discounting. When rates rise, futures P&L is received immediately
      (can be re-invested at higher rates); when rates fall, P&L is paid
      when discounting is higher. This asymmetry makes futures cheaper →
      futures yield is higher by the convexity adjustment."
    """)
