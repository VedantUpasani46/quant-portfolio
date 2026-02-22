"""
Almgren-Chriss Optimal Execution Model
=========================================
Computes the optimal trade trajectory that minimises the expected
cost plus risk of executing a large order over time.

The problem:
  A trader must buy X shares over [0, T].
  Trading too fast → large market impact cost.
  Trading too slow → large timing risk (price moves against you).
  The optimal schedule balances these two forces.

Almgren-Chriss (2001) model:
  Price dynamics with market impact:
    S_k = S_{k-1} + σ·√τ·Z_k - g(v_k)·τ   [temporary impact]
    S̃_k = S̃_{k-1} - h(x_k)                 [permanent impact]

  where:
    v_k = n_k/τ = trade rate (shares/time)
    n_k = shares traded in period k
    x_k = inventory remaining at time k

  Linear impact functions:
    Temporary: g(v) = η·v   (cost only while trading)
    Permanent:  h(x) = γ·x  (shifts the mid-price permanently)

  Utility function:
    U = E[cost] + λ·Var[cost]
    = implementation shortfall + risk aversion × variance

  Optimal solution (closed form):
    x*(t) = X · sinh(κ(T-t)) / sinh(κT)

    where κ² = λσ²/η   (trade-off between risk and impact)

  The parameter κ:
    κ → 0: TWAP (uniform execution, ignore risk)
    κ → ∞: very risk-averse, trade immediately

Benchmarks:
  TWAP: time-weighted average price → uniform trajectory
  VWAP: volume-weighted average price → trade proportional to ADV
  IS:   implementation shortfall → optimal (Almgren-Chriss)

References:
  - Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions.
    Journal of Risk 3(2), 5–39.
  - Almgren, R. (2003). Optimal Execution with Nonlinear Impact Functions and
    Trading-Enhanced Risk. Applied Mathematical Finance 10(1), 1–18.
  - Kissell, R. & Glantz, M. (2003). Optimal Trading Strategies. AMACOM.
  - Bertsimas, D. & Lo, A. (1998). Optimal Control of Execution Costs.
    Journal of Financial Markets 1(1), 1–50.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Market impact parameters
# ---------------------------------------------------------------------------

@dataclass
class MarketParams:
    """
    Market microstructure parameters for a single stock.

    Attributes
    ----------
    S0 : float       Current mid price ($)
    sigma : float    Daily volatility (annualised: daily = sigma / sqrt(252))
    ADV : float      Average daily volume (shares)
    spread_bps : float  Bid-ask spread in basis points
    eta : float      Temporary impact coefficient. Typical: 0.1 × sigma / sqrt(ADV)
    gamma : float    Permanent impact coefficient. Typical: 0.5 × eta
    """
    S0: float
    sigma_annual: float
    ADV: float
    spread_bps: float = 5.0
    eta: float = None       # temporary impact (will be inferred if None)
    gamma: float = None     # permanent impact (will be inferred if None)

    def __post_init__(self):
        sigma_daily = self.sigma_annual / math.sqrt(252)
        if self.eta is None:
            # Almgren (2003): η ≈ 0.1 × σ_daily / √ADV
            self.eta = 0.1 * sigma_daily / math.sqrt(self.ADV)
        if self.gamma is None:
            self.gamma = 0.5 * self.eta

    @property
    def sigma_daily(self) -> float:
        return self.sigma_annual / math.sqrt(252)


# ---------------------------------------------------------------------------
# Execution trajectories
# ---------------------------------------------------------------------------

def twap_trajectory(X: float, N: int) -> np.ndarray:
    """
    Time-Weighted Average Price: uniform schedule.
    Trade X/N shares in each of N intervals.
    """
    return np.full(N, X / N)


def vwap_trajectory(X: float, N: int, volume_profile: np.ndarray = None) -> np.ndarray:
    """
    Volume-Weighted Average Price: trade proportional to intraday volume.
    If no volume profile given, use a U-shaped intraday pattern (stylised fact).
    """
    if volume_profile is None:
        # Stylised U-shape: high volume at open and close, low at midday
        t = np.linspace(0, 1, N)
        volume_profile = 0.5 + 1.5 * (t ** 2 + (1 - t) ** 2)
    volume_profile = volume_profile / volume_profile.sum()
    return X * volume_profile


def almgren_chriss_trajectory(
    X: float,        # total shares to trade
    T: float,        # total execution horizon (days)
    N: int,          # number of trading intervals
    params: MarketParams,
    lam: float = 1e-6,   # risk aversion parameter λ
) -> np.ndarray:
    """
    Almgren-Chriss optimal execution trajectory.

    Closed-form solution:
      x*(t) = X · sinh(κ(T-t)) / sinh(κT)
      n_k = x*(t_k) - x*(t_{k+1})  (trades each period)

    Parameters
    ----------
    X   : Total shares to trade.
    T   : Total time horizon in days.
    N   : Number of intervals (e.g. 390 for minute-by-minute over a day).
    params : MarketParams
    lam : Risk aversion (λ). Higher → more front-loaded execution.
           Typical range: 1e-7 (passive) to 1e-5 (aggressive).

    Returns
    -------
    trades : np.ndarray of shape (N,) — shares to trade in each interval.
    """
    sigma = params.sigma_daily
    eta = params.eta

    # κ² = λ·σ²/η  (characteristic decay rate of the optimal schedule)
    kappa_sq = lam * sigma ** 2 / eta
    kappa = math.sqrt(max(kappa_sq, 1e-12))

    dt = T / N
    # Remaining inventory at each time step
    t_arr = np.linspace(0, T, N + 1)
    x_arr = X * np.sinh(kappa * (T - t_arr)) / math.sinh(kappa * T + 1e-12)
    x_arr[-1] = 0.0  # ensure full liquidation

    trades = np.diff(-x_arr)  # positive = buying
    return np.maximum(trades, 0)  # ensure non-negative (buying trajectory)


# ---------------------------------------------------------------------------
# Cost simulation
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    Result of simulating an execution trajectory.

    Attributes
    ----------
    trajectory_name : str
    trades : np.ndarray         Shares traded each period
    prices : np.ndarray         Execution price each period
    implementation_shortfall : float  Total IS cost in $ (vs arrival price)
    is_bps : float              IS in basis points of notional
    market_impact_cost : float  Total market impact ($)
    timing_cost : float         Adverse price drift cost ($)
    spread_cost : float         Bid-ask spread cost ($)
    realised_variance : float   Risk: variance of total cost
    """
    trajectory_name: str
    trades: np.ndarray
    prices: np.ndarray
    implementation_shortfall: float
    is_bps: float
    market_impact_cost: float
    timing_cost: float
    spread_cost: float
    realised_variance: float

    def summary(self) -> str:
        return (
            f"  {'Trajectory':<30} {self.trajectory_name}\n"
            f"  {'IS Cost ($)':>30} ${self.implementation_shortfall:,.2f}\n"
            f"  {'IS (bps)':>30} {self.is_bps:.2f} bps\n"
            f"  {'Market Impact ($)':>30} ${self.market_impact_cost:,.2f}\n"
            f"  {'Timing Risk ($)':>30} ${self.timing_cost:,.2f}\n"
            f"  {'Spread Cost ($)':>30} ${self.spread_cost:,.2f}\n"
            f"  {'Cost Variance ($²)':>30} ${self.realised_variance:,.2f}"
        )


def simulate_execution(
    trades: np.ndarray,
    params: MarketParams,
    trajectory_name: str = "Custom",
    seed: int = 42,
    n_paths: int = 1000,
) -> ExecutionResult:
    """
    Simulate the execution of a given trade schedule with market impact.

    Model (Almgren-Chriss):
      Price at step k:
        S_k = S_{k-1} - γ·n_{k-1} + σ·√dt·Z_k   [permanent impact + noise]
      Execution price:
        P_k = S_k - η·(n_k/dt)                   [temporary impact]

    IS = Σ n_k · P_k - X · S_0   (negative = buying above arrival price)

    Runs n_paths Monte Carlo paths and averages.
    """
    rng = np.random.default_rng(seed)
    N = len(trades)
    X = trades.sum()
    dt = 1.0 / N   # fraction of day per interval
    sigma = params.sigma_daily
    notional = X * params.S0

    total_is = []
    total_impact = []
    total_timing = []
    total_spread = []

    for _ in range(n_paths):
        S = params.S0
        is_cost = 0.0
        impact_cost = 0.0
        timing_drift = 0.0

        for k, n_k in enumerate(trades):
            if n_k <= 0:
                continue
            # Noise
            dS = sigma * math.sqrt(dt) * rng.standard_normal()
            # Permanent impact of previous trades
            S += dS - params.gamma * n_k
            # Execution price = mid + temporary impact + half spread
            temp_impact = params.eta * (n_k / max(dt, 1e-8))
            half_spread = params.S0 * params.spread_bps / 20000  # bps/2 in $ terms
            exec_price = S + temp_impact + half_spread

            trade_is = n_k * (exec_price - params.S0)
            is_cost += trade_is
            impact_cost += n_k * temp_impact
            timing_drift += n_k * dS
            total_spread.append(n_k * half_spread)

        total_is.append(is_cost)
        total_impact.append(impact_cost)
        total_timing.append(timing_drift)

    mean_is = float(np.mean(total_is))
    mean_impact = float(np.mean(total_impact))
    mean_timing = float(np.mean(total_timing))
    mean_spread = float(np.mean([sum(total_spread)] * n_paths))
    var_is = float(np.var(total_is))

    # Representative execution prices (single path for display)
    S_path = params.S0
    prices = []
    for n_k in trades:
        dS = sigma * math.sqrt(dt) * rng.standard_normal()
        S_path += dS - params.gamma * n_k
        if n_k > 0:
            exec_p = S_path + params.eta * (n_k / max(dt, 1e-8))
            prices.append(exec_p)
        else:
            prices.append(S_path)

    is_bps = mean_is / notional * 10000 if notional > 0 else 0.0

    return ExecutionResult(
        trajectory_name=trajectory_name,
        trades=trades,
        prices=np.array(prices),
        implementation_shortfall=mean_is,
        is_bps=is_bps,
        market_impact_cost=mean_impact,
        timing_cost=mean_timing,
        spread_cost=mean_spread / n_paths,
        realised_variance=var_is,
    )


# ---------------------------------------------------------------------------
# Efficient frontier of execution strategies
# ---------------------------------------------------------------------------

def execution_frontier(
    X: float, T: float, N: int,
    params: MarketParams,
    lambda_range: np.ndarray = None,
) -> pd.DataFrame:
    """
    The execution efficient frontier: E[IS] vs Var[IS] across risk aversion λ.

    Each λ gives a different optimal trajectory. The frontier shows the
    best achievable trade-off between expected cost and cost uncertainty.
    Lower λ → TWAP-like (lower variance, higher expected cost)
    Higher λ → front-loaded (lower expected cost if drift favourable, higher variance)
    """
    if lambda_range is None:
        lambda_range = np.logspace(-8, -4, 15)

    records = []
    for lam in lambda_range:
        trades = almgren_chriss_trajectory(X, T, N, params, lam)
        result = simulate_execution(trades, params, n_paths=500)
        records.append({
            "lambda":     lam,
            "mean_IS_bps": result.is_bps,
            "var_IS":     result.realised_variance,
            "front_loading": trades[:N//4].sum() / X,  # fraction in first quarter
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  Almgren-Chriss Optimal Execution")
    print("  Optimal trade schedule minimising IS + risk")
    print("═" * 66)

    # Large-cap liquid stock example (e.g. S&P 500 component)
    params = MarketParams(
        S0=100.0,
        sigma_annual=0.25,      # 25% annual vol → ~1.6% daily
        ADV=5_000_000,          # 5M shares/day ADV
        spread_bps=2.0,
        # η and γ auto-computed from ADV and vol
    )

    X = 100_000  # 100K shares = 2% of ADV (moderately large order)
    T = 1.0      # execute over 1 full trading day
    N = 390      # minute-by-minute (6.5 hours × 60 min)

    print(f"\n  Order: {X:,} shares  ({100*X/params.ADV:.1f}% of ADV)")
    print(f"  Notional: ${X*params.S0/1e6:.1f}M  |  Stock vol: {params.sigma_annual:.0%}/yr")
    print(f"  η (temp impact): {params.eta:.2e}  |  γ (perm impact): {params.gamma:.2e}")

    # ── Three strategies ──────────────────────────────────────────
    print(f"\n── Strategy Comparison (Monte Carlo, 1000 paths) ──")

    twap    = twap_trajectory(X, N)
    vwap    = vwap_trajectory(X, N)
    ac_mid  = almgren_chriss_trajectory(X, T, N, params, lam=1e-6)
    ac_aggr = almgren_chriss_trajectory(X, T, N, params, lam=1e-5)

    strategies = [
        ("TWAP",           twap),
        ("VWAP",           vwap),
        ("AC (λ=1e-6)",   ac_mid),
        ("AC (λ=1e-5)",   ac_aggr),
    ]

    print(f"\n  {'Strategy':<20} {'IS (bps)':>10} {'IS ($K)':>10} {'Var ($²)':>14} {'Front25%':>10}")
    print("  " + "─" * 68)
    results = {}
    for name, traj in strategies:
        res = simulate_execution(traj, params, name, n_paths=1000)
        results[name] = res
        front_load = traj[:N//4].sum() / X
        print(f"  {name:<20} {res.is_bps:>10.2f} "
              f"${res.implementation_shortfall/1000:>8.2f}K "
              f"${res.realised_variance:>12,.0f} {front_load:>10.2%}")

    # ── Execution frontier ────────────────────────────────────────
    print(f"\n── Execution Frontier (E[IS] vs Var[IS] across λ) ──")
    frontier_df = execution_frontier(X, T, N, params,
                                     lambda_range=np.logspace(-8, -4, 8))
    print(f"\n  {'λ':>12} {'E[IS] (bps)':>14} {'Var(IS)':>14} {'Front25%':>10}")
    print("  " + "─" * 54)
    for _, row in frontier_df.iterrows():
        print(f"  {row['lambda']:>12.0e} {row['mean_IS_bps']:>14.2f} "
              f"${row['var_IS']:>12,.0f} {row['front_loading']:>10.2%}")

    # ── Trade rate schedule ────────────────────────────────────────
    print(f"\n── Trade Schedule: shares per 30-min bucket ──")
    print(f"\n  {'Time':>6}", end="")
    for name, _ in strategies[:3]:
        print(f"  {name:>16}", end="")
    print()
    print("  " + "─" * 58)
    bucket_size = N // 13  # ~13 half-hour buckets in a trading day
    for b in range(0, N, bucket_size):
        bucket_end = min(b + bucket_size, N)
        t_min = b * (390 // N) if N == 390 else b
        row = f"  {b*60//N:>4}m"
        for name, traj in strategies[:3]:
            bucket_vol = traj[b:bucket_end].sum()
            row += f"  {bucket_vol:>16,.0f}"
        print(row)

    print(f"""
  Key insights:
    1. TWAP spreads trading uniformly — simple but ignores timing risk
    2. VWAP front/back-loads to match intraday volume pattern (U-shape)
    3. AC aggressive (λ=1e-5): front-loads heavily, lower IS if price trends up
    4. AC passive (λ=1e-6): near-TWAP, lower variance, higher expected cost
    5. The frontier shows no free lunch: reducing cost ↔ increasing risk
    """)
