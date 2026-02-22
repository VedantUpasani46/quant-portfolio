"""
Constant Maturity Swap (CMS) Pricing
======================================
A Constant Maturity Swap is an interest rate swap where the floating leg
pays the rate of a FIXED-MATURITY SWAP rather than a short-term LIBOR rate.

Example: "5-year CMS vs 3-month LIBOR"
  - On each payment date, the floating payer pays the THEN-CURRENT 5-year swap rate
  - The other leg pays 3-month LIBOR (or a fixed rate)
  - Used to express views on the SLOPE and SHAPE of the yield curve

Why CMS is not trivial to price:
  1. The 5-year swap rate S₅(t) is a non-trivial function of the yield curve
  2. The payment is at time T (the reset date), but the rate S₅(T) is a
     rate for the period [T, T+5] — the payment is "in the wrong measure"
  3. This creates a CONVEXITY ADJUSTMENT: E^Q[S(T)] ≠ Forward Swap Rate

The convexity adjustment (in annuity measure):
  CMS fixing = Forward swap rate + Convexity Adjustment
  
  Under Hull-White / lognormal:
  Convexity adj ≈ S · Var(S) / (dAnnuity/dS)
  
  More precisely (Hagan 2003):
  CMS(T) = Forward Swap Rate · [1 + CA(T)]
  CA(T) ≈ (σ_S)² · T · g'(S₀) · S₀ / g(S₀)
  where g(S) = 1 - (1+S/freq)^(-n·freq) is the annuity function.

CMS products:
  1. CMS swap: pay fixed K, receive CMS₅(T) on each date
  2. CMS cap: payoff = max(CMS₅(T) - K, 0)
  3. CMS floor: payoff = max(K - CMS₅(T), 0)
  4. CMS spread option: payoff = max(CMS₁₀(T) - CMS₂(T) - spread, 0)
     Used to trade curve steepener/flattener

Pricing methods:
  1. Hagan (2003) analytical formula — the industry standard
  2. Replication: CMS caplet = portfolio of swaptions (exact but expensive)
  3. Numerical (PDE, MC under CIR/HW) — most flexible

Applications:
  - Curve steepener/flattener trades
  - Liability-driven investing (pension funds matching long-duration)
  - Structured products (reverse floaters, range accruals)
  - Rates desks express views on curve shape

References:
  - Hagan, P.S. (2003). Convexity Conundrums: Pricing CMS Swaps, Caps, and Floors.
    Wilmott Magazine, 38–44.
  - Mercurio, F. (2005). Pricing CMS Spreads in the LFM.
  - Brigo, D. & Mercurio, F. (2006). Interest Rate Models, 2nd ed. Springer, Ch. 13.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Yield curve and forward swap rate
# ---------------------------------------------------------------------------

class YieldCurve:
    def __init__(self, tenors, rates):
        self.tenors = np.array(tenors)
        self.rates  = np.array(rates)
        self._interp = interp1d(tenors, rates, bounds_error=False,
                                fill_value=(rates[0], rates[-1]))
    
    def zero_rate(self, T):
        return float(self._interp(max(T, 1e-6)))
    
    def df(self, T):
        return math.exp(-self.zero_rate(T) * T) if T > 0 else 1.0


def annuity(
    T_start: float,
    swap_tenor: float,      # CMS swap tenor in years (e.g. 5 for CMS5)
    freq: int,              # payment frequency (2 = semi-annual)
    curve: YieldCurve,
) -> float:
    """
    Annuity value for a swap starting at T_start with given tenor.
    A(T_start, T_start + swap_tenor) = Σᵢ (1/freq) · P(0, T_start + i/freq)
    """
    dt = 1.0 / freq
    t = T_start + dt
    ann = 0.0
    while t <= T_start + swap_tenor + 1e-8:
        ann += dt * curve.df(t)
        t += dt
    return ann


def forward_swap_rate(
    T_start: float,         # swap start date
    swap_tenor: float,      # e.g. 5.0 for 5-year swap rate
    freq: int,
    curve: YieldCurve,
) -> float:
    """
    Forward swap rate S(T_start, T_start + tenor) as seen today.
    S = [P(0,T_start) - P(0,T_end)] / A(T_start, T_end)
    """
    T_end = T_start + swap_tenor
    ann = annuity(T_start, swap_tenor, freq, curve)
    return (curve.df(T_start) - curve.df(T_end)) / (ann + 1e-12)


# ---------------------------------------------------------------------------
# Convexity adjustment (Hagan 2003)
# ---------------------------------------------------------------------------

def cms_convexity_adjustment(
    T_reset: float,         # CMS reset/payment date
    swap_tenor: float,      # tenor of the CMS rate (e.g. 5 for CMS5)
    freq: int,
    vol_swaption: float,    # ATMF swaption vol (lognormal)
    curve: YieldCurve,
) -> float:
    """
    CMS convexity adjustment using Hagan (2003) approximation.
    
    The CMS convexity adjustment accounts for the Jensen's inequality effect:
    E^Q[S(T)] > S₀  because of the non-linear mapping from swap rate to payment.
    
    Hagan (2003) formula:
      CA ≈ S₀ · σ²_S · T_reset · g'(S₀)/g(S₀)
    where:
      g(S) = annuity as function of S (at flat rate S)
      g'(S) = dg/dS
      
    Flat-rate approximation:
      g(S) = [1 - (1 + S/n)^(-n·T)] / (S/n)   for payment freq n
      g'(S) = dg/dS
    """
    S0 = forward_swap_rate(T_reset, swap_tenor, freq, curve)
    n  = freq  # payments per year
    nT = int(swap_tenor * freq)  # total payment dates
    
    if S0 < 1e-6:
        return 0.0
    
    # Annuity as function of flat rate s: g(s) = (1 - (1+s/n)^-N) / (s/n)
    def g(s):
        s_n = s / n
        if abs(s_n) < 1e-8:
            return nT / n  # L'Hopital limit
        return (1 - (1 + s_n)**(-nT)) / s_n
    
    # Numerical derivative g'(S0)
    ds = 1e-4  # absolute step for better numerics
    g_prime = (g(S0 + ds) - g(S0 - ds)) / (2 * ds)
    
    # Convexity adjustment
    ca = -S0**2 * vol_swaption**2 * T_reset * g_prime / (g(S0) + 1e-12)
    
    return ca


# ---------------------------------------------------------------------------
# CMS caplet/floorlet (Black formula for CMS rate)
# ---------------------------------------------------------------------------

def cms_caplet(
    T_reset: float,
    swap_tenor: float,
    strike: float,
    vol_swaption: float,
    freq: int,
    curve: YieldCurve,
    is_cap: bool = True,
) -> dict:
    """
    CMS caplet/floorlet using Black formula.
    
    The CMS rate with convexity adjustment:
      E^Q[S(T)] = S₀ + CA
    
    Then apply Black formula treating the CMS rate as a forward:
      Caplet = P(0,T) · [F·N(d₁) - K·N(d₂)]
      where F = S₀ + CA  (CMS rate + convexity adjustment)
    
    This is an approximation — full pricing requires replication with swaptions.
    """
    S0 = forward_swap_rate(T_reset, swap_tenor, freq, curve)
    ca = cms_convexity_adjustment(T_reset, swap_tenor, freq, vol_swaption, curve)
    F  = S0 + ca  # CMS rate with convexity adjustment
    
    df_T = curve.df(T_reset)
    sigma_T = vol_swaption * math.sqrt(T_reset)
    
    if sigma_T < 1e-8 or F < 1e-8:
        payoff = max(F - strike, 0) if is_cap else max(strike - F, 0)
        return {'price': df_T * payoff, 'cms_rate': F, 'ca': ca, 'forward': S0}
    
    d1 = (math.log(F / strike) + 0.5 * sigma_T**2) / sigma_T
    d2 = d1 - sigma_T
    
    sign = 1 if is_cap else -1
    price = df_T * sign * (F * norm.cdf(sign * d1) - strike * norm.cdf(sign * d2))
    
    return {
        'price': price,
        'cms_rate': F,
        'forward_swap_rate': S0,
        'convexity_adj': ca,
        'convexity_adj_bps': ca * 10000,
        'd1': d1,
        'd2': d2,
    }


# ---------------------------------------------------------------------------
# CMS swap pricer
# ---------------------------------------------------------------------------

def cms_swap_price(
    T_start: float,
    T_end: float,
    cms_tenor: float,       # e.g. 5 for CMS5
    fixed_rate: float,      # fixed leg rate K
    notional: float,
    payment_freq: int,
    vol_swaption: float,
    curve: YieldCurve,
    pay_cms: bool = True,   # True = receive fixed, pay CMS
) -> dict:
    """
    CMS swap: pay CMS₅(T) quarterly vs receive fixed K.
    Price = Σₜ P(0,t) · [CMS_rate(t) - K] · τ · N
    where CMS_rate(t) = forward_swap_rate + convexity_adj
    """
    dt = 1.0 / payment_freq
    t  = T_start + dt
    
    total_cms_pv = 0.0
    total_fixed_pv = 0.0
    payment_dates = []
    
    while t <= T_end + 1e-8:
        T_reset = t - dt  # reset at start of period
        
        S0 = forward_swap_rate(T_reset, cms_tenor, payment_freq, curve)
        ca = cms_convexity_adjustment(T_reset, cms_tenor, payment_freq,
                                       vol_swaption, curve)
        cms_rate = S0 + ca
        
        df = curve.df(t)
        
        cms_pv   = cms_rate * dt * notional * df
        fixed_pv = fixed_rate * dt * notional * df
        
        total_cms_pv   += cms_pv
        total_fixed_pv += fixed_pv
        
        payment_dates.append({
            'date': t,
            'cms_rate': cms_rate,
            'forward_swap_rate': S0,
            'ca_bps': ca * 10000,
            'df': df,
            'cms_pv': cms_pv,
            'fixed_pv': fixed_pv,
        })
        
        t += dt
    
    sign = 1 if pay_cms else -1
    total_pv = sign * (total_fixed_pv - total_cms_pv)
    
    return {
        'pv': total_pv,
        'cms_leg_pv': total_cms_pv,
        'fixed_leg_pv': total_fixed_pv,
        'payments': pd.DataFrame(payment_dates),
        'pay_cms': pay_cms,
    }


# ---------------------------------------------------------------------------
# CMS spread option
# ---------------------------------------------------------------------------

def cms_spread_option(
    T_reset: float,
    cms_long_tenor: float,   # e.g. 10Y (long rate)
    cms_short_tenor: float,  # e.g. 2Y (short rate)
    strike_spread: float,    # strike on the spread
    vol_long: float,
    vol_short: float,
    corr: float,
    freq: int,
    curve: YieldCurve,
    is_call: bool = True,    # call = benefits from steepening
) -> dict:
    """
    CMS spread option: payoff = max(CMS_long - CMS_short - K, 0)
    Used to trade curve steepeners/flatteners.
    
    Kirk's approximation for spread options:
    F_spread = F_long - F_short  (adjusted CMS rates)
    σ_spread ≈ √(σ_L² + (F_S/F_L)²·σ_S² - 2·ρ·(F_S/F_L)·σ_L·σ_S)
    """
    # CMS rates with convexity adjustment
    S_long  = forward_swap_rate(T_reset, cms_long_tenor, freq, curve)
    ca_long = cms_convexity_adjustment(T_reset, cms_long_tenor, freq, vol_long, curve)
    F_long  = S_long + ca_long
    
    S_short  = forward_swap_rate(T_reset, cms_short_tenor, freq, curve)
    ca_short = cms_convexity_adjustment(T_reset, cms_short_tenor, freq, vol_short, curve)
    F_short  = S_short + ca_short
    
    F_spread = F_long - F_short
    
    # Kirk's approximation for spread vol
    ratio = F_short / (F_long + 1e-10)
    vol_spread = math.sqrt(max(0, vol_long**2 + ratio**2 * vol_short**2
                                - 2 * corr * ratio * vol_long * vol_short))
    
    df_T = curve.df(T_reset)
    sigma_T = vol_spread * math.sqrt(T_reset)
    
    if sigma_T < 1e-8:
        payoff = max(F_spread - strike_spread, 0) if is_call else max(strike_spread - F_spread, 0)
        return {'price': df_T * payoff * 10000, 'spread': F_spread}  # in bps
    
    d1 = (math.log((F_spread + 1e-6) / (strike_spread + 1e-6)) + 0.5 * sigma_T**2) / sigma_T
    d2 = d1 - sigma_T
    
    sign = 1 if is_call else -1
    price_bps = df_T * sign * (F_spread * norm.cdf(sign * d1) - strike_spread * norm.cdf(sign * d2)) * 10000
    
    return {
        'price_bps': price_bps,
        'F_long': F_long,
        'F_short': F_short,
        'forward_spread': F_spread * 10000,
        'vol_spread': vol_spread,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 68)
    print("  CMS (Constant Maturity Swap) Pricing")
    print("  Convexity adjustment, CMS caps/floors, spread options")
    print("═" * 68)
    
    # USD-style yield curve (upward sloping)
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    rates  = np.array([0.044, 0.045, 0.047, 0.049, 0.050, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057])
    curve  = YieldCurve(tenors, rates)
    
    freq = 2  # semi-annual
    vol_cms5 = 0.20   # CMS5 swaption vol
    vol_cms10 = 0.18  # CMS10 swaption vol
    
    # ── 1. Forward swap rates and convexity adjustments ──────────
    print(f"\n── 1. CMS Forward Rates and Convexity Adjustments ──")
    print(f"\n  {'Reset T':>8} {'CMS5 Fwd':>12} {'CA (bps)':>12} {'CA+Fwd':>12}")
    print("  " + "─" * 48)
    
    for T_reset in [0.25, 0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        S0 = forward_swap_rate(T_reset, 5.0, freq, curve)
        ca = cms_convexity_adjustment(T_reset, 5.0, freq, vol_cms5, curve)
        print(f"  {T_reset:>8.2f}Y {S0:>12.4%} {ca*10000:>12.3f}bp {(S0+ca):>12.4%}")
    
    print(f"\n  CA grows with reset time — longer-dated CMS payments are more convex")
    
    # ── 2. CMS cap/floor pricing ──────────────────────────────────
    print(f"\n── 2. CMS Caplets and Floorlets ──")
    print(f"\n  CMS5 caplets, K=5.5%, N=$10M")
    print(f"\n  {'Reset T':>8} {'Fwd Rate':>12} {'CA (bps)':>10} {'Cap PV':>12} {'Floor PV':>12}")
    print("  " + "─" * 58)
    
    K_strike = 0.055
    notional = 10_000_000
    
    for T_reset in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]:
        cap = cms_caplet(T_reset, 5.0, K_strike, vol_cms5, freq, curve, is_cap=True)
        flo = cms_caplet(T_reset, 5.0, K_strike, vol_cms5, freq, curve, is_cap=False)
        print(f"  {T_reset:>8.2f}Y {cap['forward_swap_rate']:>12.4%} "
              f"{cap['convexity_adj_bps']:>10.2f}bp "
              f"${cap['price']*notional:>10,.0f} ${flo['price']*notional:>10,.0f}")
    
    # ── 3. Full CMS swap ──────────────────────────────────────────
    print(f"\n── 3. CMS Swap: Pay CMS5 vs Receive Fixed 5.20% ──")
    
    cms_result = cms_swap_price(
        T_start=0.0, T_end=5.0, cms_tenor=5.0,
        fixed_rate=0.0520, notional=10_000_000,
        payment_freq=2, vol_swaption=vol_cms5,
        curve=curve, pay_cms=True
    )
    
    print(f"\n  Notional: $10M, 5-year swap, semi-annual payments")
    print(f"  CMS5 leg PV:   ${cms_result['cms_leg_pv']:>10,.0f}")
    print(f"  Fixed leg PV:  ${cms_result['fixed_leg_pv']:>10,.0f}")
    print(f"  Swap PV:       ${cms_result['pv']:>10,.0f}  (pay CMS, receive fixed)")
    
    print(f"\n  First 5 payment dates:")
    print(f"  {'Date':>8} {'CMS Rate':>12} {'Fwd Swap':>12} {'CA (bps)':>10} {'P(0,t)':>8}")
    print("  " + "─" * 54)
    for _, row in cms_result['payments'].head(5).iterrows():
        print(f"  {row['date']:>8.2f}Y {row['cms_rate']:>12.4%} "
              f"{row['forward_swap_rate']:>12.4%} {row['ca_bps']:>10.2f}bp "
              f"{row['df']:>8.4f}")
    
    # ── 4. CMS spread option (10Y-2Y steepener) ──────────────────
    print(f"\n── 4. CMS Spread Option (10Y - 2Y Curve Steepener) ──")
    print(f"\n  Steepener call: payoff = max(CMS10 - CMS2 - K, 0)")
    print(f"  Correlation(CMS10, CMS2) = 0.85")
    
    S10 = forward_swap_rate(0.25, 10.0, freq, curve)
    S2  = forward_swap_rate(0.25, 2.0, freq, curve)
    fwd_spread = (S10 - S2) * 10000
    print(f"\n  Current CMS10: {S10:.4%},  CMS2: {S2:.4%}")
    print(f"  Forward 10-2 spread: {fwd_spread:.1f}bps")
    
    print(f"\n  {'Strike (bps)':>14} {'Call Price (bps)':>18} {'Put Price (bps)':>18}")
    print("  " + "─" * 52)
    
    for K_spread_bps in [0, 25, 50, 75, 100]:
        K_spread = K_spread_bps / 10000
        call = cms_spread_option(0.25, 10.0, 2.0, K_spread, vol_cms10, 0.22, 0.85, freq, curve)
        put  = cms_spread_option(0.25, 10.0, 2.0, K_spread, vol_cms10, 0.22, 0.85, freq, curve, is_call=False)
        print(f"  {K_spread_bps:>14}bps {call['price_bps']:>18.2f}bps {put['price_bps']:>18.2f}bps")
    
    print(f"\n  Forward spread = {fwd_spread:.1f}bps.  ATM call priced at ~0bp strike.")
    
    print(f"""
── Why CMS Matters ──

  CMS vs plain IRS:
    IRS: floating = LIBOR (short end of curve)
    CMS: floating = swap rate (a point ON the curve)
    CMS expresses a view on WHERE the curve will be, not just the level

  Why convexity adjustment exists:
    The CMS payment at time T is proportional to S(T), the THEN-CURRENT
    5-year swap rate. But the LIBOR forward measure prices payments at T
    using P(0,T), which creates a "timing mismatch" vs the annuity measure.
    Jensen's inequality: E^Q[S(T)] > S_forward because S appears in both
    the numerator (payment) and denominator (discounting) of the bond price.

  CMS spread trades:
    CMS₁₀ - CMS₂ = slope of the 2-10 part of the curve
    Steepener (buy spread): profits if curve steepens (10Y >> 2Y)
    Flattener (sell spread): profits if curve flattens

  Interview question (Goldman Rates Desk, JPM):
  Q: "Why does the CMS5 rate differ from the forward 5-year swap rate?"
  A: "The convexity adjustment. When rates are high, the annuity factor
      is small → each CMS payment is worth more. This creates a positive
      covariance between the payment size and its discount factor.
      By Jensen's inequality, E[S(T)] > S₀, so the CMS fixing exceeds the
      forward swap rate by a convexity adjustment that grows with reset time."
    """)
