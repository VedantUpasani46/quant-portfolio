"""
Liquidity Risk Measurement
============================
Liquidity risk = the risk that a position cannot be exited at the
current market price without significant adverse price impact.

Two dimensions of illiquidity:
  1. Bid-ask spread: the cost of an IMMEDIATE small trade
  2. Market depth / price impact: the cost of a LARGE trade over time

Key measures:

1. Amihud (2002) Illiquidity Ratio:
   ILLIQ_i = (1/T) Σₜ |r_{i,t}| / Volume_{i,t}   [in $ millions]
   Interpretation: price impact per $1M of trading volume.
   Higher ILLIQ = more illiquid. Amihud shows ILLIQ predicts returns
   (illiquidity premium): E[r] ≈ α + β·ILLIQ + ...

2. Roll (1984) Spread Estimator:
   s = 2 · √(−Cov(ΔP_t, ΔP_{t-1}))
   Based on: bid-ask bounce creates negative serial correlation in prices.
   Works even when bid-ask data is unavailable (only needs trade prices).

3. Turnover ratio:
   Turnover = Volume / Shares_outstanding
   Low turnover → illiquid. Used in Fama-French extended factors.

4. Liquidation risk (LiquidityVaR):
   Standard VaR ignores liquidation cost.
   LiqVaR = VaR + Exogenous Liquidity Cost + Endogenous Liquidity Cost
   ELC = ½ · spread · position_size
   ILC = price_impact × position_size²   [from market impact model]

5. Liquidity-adjusted VaR (LVaR) [Bangia et al. 1999]:
   LVaR = VaR + ELC + ILC
   = σ·z_α·P + ½·spread·P + κ·P²

6. Time-to-liquidate:
   How many days does it take to liquidate a position?
   Constraint: trade at most X% of ADV per day (e.g. 20%)
   Days = position / (ADV × participation_rate)

References:
  - Amihud, Y. (2002). Illiquidity and Stock Returns. Journal of Financial
    Markets 5(1), 31–56.
  - Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread.
    Journal of Finance 39(4), 1127–1139.
  - Bangia, A. et al. (1999). Ratings Migration and the Business Cycle, with Application
    to Credit Portfolio Stress Testing. Journal of Banking & Finance.
  - Brunnermeier, M. & Pedersen, L.H. (2009). Market Liquidity and Funding Liquidity.
    Review of Financial Studies 22(6), 2201–2238.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Amihud illiquidity ratio
# ---------------------------------------------------------------------------

def amihud_illiquidity(
    returns: np.ndarray,           # daily returns (T,)
    dollar_volume: np.ndarray,     # daily $ volume in millions (T,)
    window: int = None,
) -> float | np.ndarray:
    """
    Amihud (2002) illiquidity ratio.
    ILLIQ = mean(|r_t| / DollarVolume_t)   [$ impact per $1M volume]

    If window is given, returns rolling ILLIQ series.
    """
    illiq = np.abs(returns) / (dollar_volume + 1e-10)
    if window is not None:
        return pd.Series(illiq).rolling(window).mean().values
    return illiq.mean()


def amihud_illiquidity_panel(
    returns_df: pd.DataFrame,
    volume_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Amihud ratio for each stock in a panel.
    Returns DataFrame with same columns as returns_df.
    """
    result = {}
    for col in returns_df.columns:
        r = returns_df[col].dropna().values
        v = volume_df[col].dropna().values
        n = min(len(r), len(v))
        result[col] = amihud_illiquidity(r[:n], v[:n])
    return pd.Series(result).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Roll (1984) bid-ask spread estimator
# ---------------------------------------------------------------------------

def roll_spread(prices: np.ndarray) -> float:
    """
    Roll (1984) effective bid-ask spread from price data.
    s = 2 · √max(0, −Cov(ΔP_t, ΔP_{t-1}))

    Based on the bid-ask bounce mechanism:
      Price_t = M_t + q_t · s/2
      where q_t = ±1 (buyer or seller initiated)
      ΔP_t = ΔM_t + (q_t - q_{t-1}) · s/2
      Cov(ΔP_t, ΔP_{t-1}) = −s²/4   (since q_t independent of M)
      → s = 2√(−Cov)
    """
    dp = np.diff(prices)
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    if cov >= 0:
        return 0.0  # no negative autocorrelation → spread estimate = 0
    return 2 * np.sqrt(-cov)


def roll_spread_pct(prices: np.ndarray) -> float:
    """Roll spread as a % of price."""
    s = roll_spread(prices)
    return s / np.mean(prices)


def roll_spread_rolling(prices: np.ndarray, window: int = 63) -> pd.Series:
    """Rolling Roll spread estimate."""
    spreads = np.full(len(prices), np.nan)
    for t in range(window, len(prices)):
        px_window = prices[t - window:t]
        spreads[t] = roll_spread_pct(px_window)
    return pd.Series(spreads)


# ---------------------------------------------------------------------------
# Liquidity-adjusted VaR (LVaR)
# ---------------------------------------------------------------------------

@dataclass
class LVaRResult:
    var_base: float           # standard VaR (no liquidity adjustment)
    elc: float                # exogenous liquidity cost (half spread × position)
    ilc: float                # endogenous liquidity cost (market impact)
    lvar: float               # LVaR = VaR + ELC + ILC
    lvar_bps: float           # LVaR as bps of position value
    time_to_liquidate: float  # days to exit position

    def summary(self) -> str:
        return (
            f"  VaR (base):         ${self.var_base:>12,.0f}\n"
            f"  Exogenous LC (½s):  ${self.elc:>12,.0f}  (bid-ask spread)\n"
            f"  Endogenous LC:      ${self.ilc:>12,.0f}  (market impact)\n"
            f"  ─────────────────────────────────────\n"
            f"  LVaR:               ${self.lvar:>12,.0f}  ({self.lvar_bps:.1f}bps)\n"
            f"  Time to liquidate:  {self.time_to_liquidate:>12.1f} days"
        )


def liquidity_adjusted_var(
    position_value: float,     # $ value of position
    daily_vol: float,          # daily return volatility (e.g. 0.015)
    spread_pct: float,         # bid-ask spread as % of price (e.g. 0.001 = 10bps)
    adv: float,                # average daily $ volume
    impact_coeff: float = 0.1, # market impact coefficient (price moves by this × participation)
    confidence: float = 0.99,  # VaR confidence level
    holding_days: int = 1,
    max_participation: float = 0.20,  # max % of ADV per day
) -> LVaRResult:
    """
    Bangia et al. (1999) Liquidity-Adjusted VaR.

    LVaR = VaR + ELC + ILC
      VaR = σ · z_α · √T · V           (standard market risk VaR)
      ELC = ½ · spread · V             (exogenous: always pay half-spread)
      ILC = impact_coeff · (V/ADV)·σ·V (endogenous: price moves with order)
    """
    z = norm.ppf(confidence)
    
    # Base VaR (market risk only)
    var_base = position_value * daily_vol * z * np.sqrt(holding_days)
    
    # Exogenous liquidity cost (pay half the spread to exit)
    elc = 0.5 * spread_pct * position_value
    
    # Endogenous liquidity cost (price impact from selling the position)
    # Participation rate = position / ADV
    participation = position_value / (adv * holding_days)
    ilc = impact_coeff * participation * daily_vol * position_value
    
    # Total LVaR
    lvar = var_base + elc + ilc
    lvar_bps = lvar / position_value * 10000
    
    # Time to liquidate at max_participation
    ttl = position_value / (adv * max_participation) if adv > 0 else np.inf
    
    return LVaRResult(
        var_base=var_base, elc=elc, ilc=ilc, lvar=lvar,
        lvar_bps=lvar_bps, time_to_liquidate=ttl
    )


# ---------------------------------------------------------------------------
# Portfolio liquidation schedule
# ---------------------------------------------------------------------------

def liquidation_schedule(
    position: float,              # initial position size ($)
    adv: float,                   # average daily $ volume
    vol: float,                   # daily vol
    max_participation: float = 0.20,
    risk_aversion_liq: float = 1e-4,
) -> pd.DataFrame:
    """
    Optimal liquidation schedule: how much to trade each day.
    Simple uniform TWAP + risk-adjusted tilt.

    Risk-adjusted: trade faster early when temporary impact is less than
    variance of holding. Simplified Almgren-Chriss result.
    """
    daily_capacity = adv * max_participation
    T = int(np.ceil(position / daily_capacity))  # minimum days
    
    rows = []
    remaining = position
    cumulative_impact = 0.0
    
    for day in range(1, T + 1):
        trade_today = min(daily_capacity, remaining)
        remaining -= trade_today
        
        # Market impact for today's trade (sqrt model)
        impact_bps = 5.0 * np.sqrt(trade_today / adv) * 100  # rough calibration
        
        # Variance cost of holding remaining position one more day
        variance_cost = vol**2 * remaining**2 * risk_aversion_liq
        
        cumulative_impact += impact_bps * trade_today / 10000
        
        rows.append({
            'day': day,
            'trade_size': trade_today,
            'remaining': remaining,
            'participation': trade_today / adv,
            'impact_bps': impact_bps,
            'cumulative_impact': cumulative_impact,
        })
    
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  Liquidity Risk Measurement")
    print("  Amihud, Roll spread, LVaR, liquidation schedule")
    print("═" * 66)
    
    np.random.seed(42)
    
    # Simulate 5 stocks with different liquidity profiles
    T = 252
    stocks = {
        'AAPL':  {'vol': 0.015, 'adv_m': 8000, 'spread_bps': 1.0},
        'MSFT':  {'vol': 0.014, 'adv_m': 5000, 'spread_bps': 1.2},
        'SMID':  {'vol': 0.022, 'adv_m':  200, 'spread_bps': 8.0},
        'SMALL': {'vol': 0.030, 'adv_m':   30, 'spread_bps': 25.0},
        'MICRO': {'vol': 0.045, 'adv_m':    5, 'spread_bps': 80.0},
    }
    
    # ── 1. Amihud illiquidity ratio ───────────────────────────────
    print(f"\n── 1. Amihud Illiquidity Ratio ──")
    print(f"\n  {'Stock':>8} {'ADV ($M)':>10} {'Spread':>8} {'Amihud ILLIQ':>14} {'Illiq Rank':>12}")
    print("  " + "─" * 56)
    
    amihud_vals = {}
    for name, params in stocks.items():
        # Simulate returns and volume
        r = np.random.normal(0.0005, params['vol'], T)
        # Volume ~ lognormal with some autocorrelation
        v = np.random.lognormal(np.log(params['adv_m'] / 252), 0.3, T)
        amihud_vals[name] = amihud_illiquidity(r, v)
    
    sorted_stocks = sorted(amihud_vals, key=lambda k: amihud_vals[k])
    ranks = {name: i+1 for i, name in enumerate(sorted_stocks)}
    
    for name, params in stocks.items():
        illiq = amihud_vals[name]
        print(f"  {name:>8} {params['adv_m']:>10,} {params['spread_bps']:>7.1f}bps "
              f"{illiq:>14.6f} {ranks[name]:>12}/5")
    
    print(f"\n  Higher ILLIQ = more illiquid = higher expected return (illiquidity premium)")
    
    # ── 2. Roll spread estimator ──────────────────────────────────
    print(f"\n── 2. Roll Bid-Ask Spread Estimator ──")
    print(f"\n  Estimating spread from price data alone (no L1 bid-ask data needed)")
    print(f"\n  {'Stock':>8} {'True Spread':>12} {'Roll Estimate':>14} {'Ratio':>8}")
    print("  " + "─" * 46)
    
    for name, params in stocks.items():
        # Simulate prices with bid-ask bounce
        true_spread = params['spread_bps'] / 10000  # as fraction
        mid_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, params['vol'], T)))
        # Add bid-ask bounce
        q = np.random.choice([-1, 1], T)  # buy/sell indicator
        prices = mid_prices * (1 + q * true_spread / 2)
        
        estimated = roll_spread_pct(prices)
        ratio = estimated / true_spread if true_spread > 0 else np.nan
        print(f"  {name:>8} {true_spread*10000:>10.1f}bps {estimated*10000:>12.1f}bps "
              f"{ratio:>8.2f}×")
    
    print(f"\n  Roll estimates the spread from negative return autocorrelation.")
    print(f"  Slight noise in small samples — works well over 250+ days.")
    
    # ── 3. Liquidity-adjusted VaR ─────────────────────────────────
    print(f"\n── 3. Liquidity-Adjusted VaR (Bangia et al. 1999) ──")
    print(f"\n  $50M position in each stock (99% 1-day VaR)")
    print(f"\n  {'Stock':>8} {'Base VaR':>12} {'ELC':>10} {'ILC':>10} {'LVaR':>12} {'Adj%':>8}")
    print("  " + "─" * 60)
    
    position = 50_000_000  # $50M
    
    for name, params in stocks.items():
        adv = params['adv_m'] * 1_000_000  # convert to $
        result = liquidity_adjusted_var(
            position_value=position,
            daily_vol=params['vol'],
            spread_pct=params['spread_bps'] / 10000,
            adv=adv,
            confidence=0.99,
            holding_days=1,
        )
        adj_pct = (result.lvar / result.var_base - 1) * 100
        print(f"  {name:>8} ${result.var_base/1e6:>9.2f}M ${result.elc/1e6:>7.3f}M "
              f"${result.ilc/1e6:>7.3f}M ${result.lvar/1e6:>9.2f}M "
              f"{adj_pct:>7.1f}%")
    
    print(f"\n  For MICRO-cap: LVaR is dramatically larger than base VaR")
    print(f"  → Standard VaR MASSIVELY underestimates risk for illiquid positions")
    
    # ── 4. LVaR detail for illiquid position ─────────────────────
    print(f"\n── LVaR Detail: $50M in MICRO-cap ──")
    adv_micro = stocks['MICRO']['adv_m'] * 1_000_000
    result = liquidity_adjusted_var(
        position_value=position,
        daily_vol=stocks['MICRO']['vol'],
        spread_pct=stocks['MICRO']['spread_bps'] / 10000,
        adv=adv_micro,
        confidence=0.99,
        holding_days=1,
    )
    print(f"\n{result.summary()}")
    
    # ── 5. Liquidation schedule ───────────────────────────────────
    print(f"\n── 5. Liquidation Schedule: $50M in SMALL-cap (ADV=$30M) ──")
    sched = liquidation_schedule(
        position=50_000_000,
        adv=30_000_000,
        vol=stocks['SMALL']['vol'],
        max_participation=0.20,
    )
    
    print(f"\n  {'Day':>6} {'Trade ($M)':>12} {'Remaining ($M)':>16} "
          f"{'Partic.':>10} {'Impact (bps)':>14}")
    print("  " + "─" * 62)
    for _, row in sched.iterrows():
        print(f"  {row['day']:>6.0f} ${row['trade_size']/1e6:>9.1f}M "
              f"${row['remaining']/1e6:>13.1f}M "
              f"{row['participation']:>10.1%} {row['impact_bps']:>14.1f}bps")
    
    print(f"\n  Total liquidation: {len(sched)} days")
    total_ic = sched['cumulative_impact'].iloc[-1]
    print(f"  Total impact cost: ${total_ic:,.0f}  ({total_ic/position*10000:.1f}bps of position)")
    
    print(f"""
── Liquidity Risk in Practice ──

  Position sizing rule (liquidity budget):
    Max position = ADV × max_days_to_exit × participation_rate
    E.g.: ADV=$100M, 5 days exit, 20% participation = $100M position max

  Amihud illiquidity premium:
    High-ILLIQ stocks earn ~0.4% more monthly than low-ILLIQ stocks
    → Size of illiquidity premium: ~3-5% annually (Amihud 2002)
    → Worth harvesting if you can tolerate the liquidation risk

  Funding liquidity (Brunnermeier-Pedersen 2009):
    Market liquidity ↔ Funding liquidity spiral:
    Falling prices → margin calls → forced selling → prices fall further
    This is why correlations spike in crises (2008, 2020)

  Interview question (Goldman Strats, JPM Risk):
  Q: "You have $200M in a stock with $50M ADV. Calculate your LVaR."
  A: "Market impact limits me to 20% ADV × 5 days = $50M/day.
      Takes 4 days to exit. Multi-day VaR = σ×z×√4×$200M plus:
      ELC = ½×spread×$200M, ILC = impact×(participation)×σ×$200M.
      Total LVaR is likely 2-3× the single-day base VaR."
    """)
