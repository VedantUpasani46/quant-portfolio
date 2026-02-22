"""
Transaction Cost–Aware Portfolio Optimization
===============================================
Standard Markowitz ignores transaction costs. But in practice:
  - Trading costs 5-50 bps for large-cap equities
  - Optimal rebalancing is NOT daily — you trade off alpha decay vs TC
  - The optimal portfolio lies BETWEEN today's holdings and the Markowitz target

Garleanu-Pedersen (2013) "Dynamic Trading with Predictable Returns":
  Optimal policy: trade a FRACTION of the gap to the Markowitz target.
  
  Markowitz (no TC):         x* = Σ⁻¹ · α / γ   (target portfolio)
  With TC (GP solution):     Δx = η · (x* - x)   (trade fraction η of the gap)
  
  where η = (γ · Σ + λ · I)⁻¹ · λ   [simplified for quadratic TC]
  λ = transaction cost parameter
  
  Interpretation: when TC is HIGH, η is SMALL → trade slowly toward target.
  When TC is LOW, η ≈ I → trade all the way to Markowitz target immediately.

Alternative formulation (no-trade region):
  For linear TC (bid-ask spread):
    Don't trade when ||x - x*|| < threshold
    The no-trade region is a "slab" around the target
  For quadratic TC (market impact):
    Always trade, but only a fraction of the gap

Transaction cost models:
  1. Fixed/linear (bid-ask spread): TC = c · |Δx|
     Best for retail-size trades below impact threshold
  2. Quadratic (Almgren-Chriss market impact): TC = κ · Δx²
     For large orders that move the market
  3. Square-root (empirical): TC = σ · √(|Δx|/ADV)
     Best empirical fit for institutional trading

Alpha decay:
  If your alpha signal decays with half-life τ:
    α(t) = α₀ · e^{-t/τ}
  Then the optimal policy depends on the ratio TC/α_decay_speed.
  Fast-decaying alpha + high TC = don't trade (signal is gone before you profit).

The turnover-return tradeoff:
  Expected net return = Gross alpha - TC × Turnover
  Optimal turnover minimises total cost over the holding period.

References:
  - Garleanu, N. & Pedersen, L.H. (2013). Dynamic Trading with Predictable Returns
    and Transaction Costs. Journal of Finance 68(6), 2309–2340.
  - Grinold, R.C. (2006). A Dynamic Model of Portfolio Management. Journal of
    Investment Management 4(2), 5–22.
  - Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions.
    Journal of Risk 3(2), 5–39.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Transaction cost models
# ---------------------------------------------------------------------------

def tc_linear(delta_x: np.ndarray, cost_bps: float, price: np.ndarray) -> float:
    """
    Linear (bid-ask spread) transaction cost.
    TC = Σᵢ cost_bps × |Δxᵢ| × price_i
    """
    return cost_bps / 10000 * np.sum(np.abs(delta_x) * price)


def tc_quadratic(delta_x: np.ndarray, kappa: np.ndarray) -> float:
    """
    Quadratic (market impact) transaction cost.
    TC = Σᵢ κᵢ · (Δxᵢ)²
    κᵢ = impact coefficient (higher for illiquid stocks)
    """
    return np.sum(kappa * delta_x ** 2)


def tc_sqrt(delta_x: np.ndarray, sigma: np.ndarray,
            adv: np.ndarray, price: np.ndarray) -> float:
    """
    Square-root market impact model (empirical).
    TC ≈ σᵢ · √(|Δxᵢ| / ADV_i) × price_i
    Most accurate for institutional trading.
    """
    participation = np.abs(delta_x) / adv
    return np.sum(sigma * np.sqrt(participation) * np.abs(delta_x) * price)


# ---------------------------------------------------------------------------
# TC-aware portfolio optimization
# ---------------------------------------------------------------------------

@dataclass
class TCOptResult:
    weights_target: np.ndarray      # Markowitz target (no TC)
    weights_tc_opt: np.ndarray      # TC-aware optimal weights
    weights_current: np.ndarray     # current portfolio
    
    gross_alpha_target: float       # expected return at Markowitz target
    net_alpha_target: float         # net of TC to get to Markowitz target
    net_alpha_tc_opt: float         # net of TC at TC-aware optimum
    
    tc_to_target: float             # TC of trading to Markowitz target
    tc_to_tc_opt: float             # TC of trading to TC-aware optimum
    
    turnover_to_target: float
    turnover_to_tc_opt: float


def markowitz_target(
    alpha: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
    constraints: dict = None,
) -> np.ndarray:
    """
    Markowitz optimal portfolio ignoring TC.
    max α'w - γ/2 · w'Σw
    """
    n = len(alpha)
    
    def neg_utility(w):
        return -(alpha @ w - 0.5 * risk_aversion * w @ cov @ w)
    
    def neg_utility_grad(w):
        return -(alpha - risk_aversion * cov @ w)
    
    w0 = np.ones(n) / n
    bounds = [(-1.0, 2.0)] * n  # allow shorts and leverage
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    result = minimize(neg_utility, w0, jac=neg_utility_grad,
                      method='SLSQP', bounds=bounds, constraints=cons)
    return result.x


def tc_aware_optimize(
    alpha: np.ndarray,
    cov: np.ndarray,
    current_weights: np.ndarray,
    kappa: np.ndarray,          # per-asset quadratic TC coefficient
    risk_aversion: float = 1.0,
    holding_periods: int = 20,  # expected holding period before rebalancing
) -> np.ndarray:
    """
    One-period TC-aware portfolio optimization.
    
    Objective: max α'(w+Δw) - γ/2·(w+Δw)'Σ(w+Δw) - Σᵢ κᵢ·(Δwᵢ)²
    
    Trade-off: the gain from moving closer to Markowitz target
    vs the cost of getting there.
    """
    n = len(alpha)
    w_current = current_weights
    
    def neg_utility(w):
        delta_w = w - w_current
        # Expected return
        ret = alpha @ w
        # Risk
        risk = 0.5 * risk_aversion * w @ cov @ w
        # TC (quadratic impact, amortised over holding period)
        tc = tc_quadratic(delta_w, kappa) / holding_periods
        return -(ret - risk - tc)
    
    w0 = w_current.copy()
    bounds = [(-1.0, 2.0)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    result = minimize(neg_utility, w0, method='SLSQP',
                      bounds=bounds, constraints=cons)
    return result.x


def garleanu_pedersen_policy(
    alpha: np.ndarray,
    cov: np.ndarray,
    current_weights: np.ndarray,
    lambda_tc: float,           # TC parameter (higher = slower trading)
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Garleanu-Pedersen (2013) closed-form policy (quadratic TC).
    
    Optimal trade: Δx = Aim · (x* - x)
    where:
      x* = Markowitz target = Σ⁻¹α/γ
      Aim = λ(γΣ + λI)⁻¹  = fraction of gap to trade
    
    When λ→0 (no TC): Aim→I → trade all the way to x*
    When λ→∞ (high TC): Aim→0 → don't trade
    """
    n = len(alpha)
    cov_inv = np.linalg.inv(cov)
    x_star = cov_inv @ alpha / risk_aversion  # Markowitz target (unnormalised)
    
    # Normalise to sum to 1
    x_star = x_star / x_star.sum() if x_star.sum() != 0 else x_star
    
    # Aim matrix: how fast to trade toward target
    gamma_sigma = risk_aversion * cov
    aim_matrix = lambda_tc * np.linalg.inv(gamma_sigma + lambda_tc * np.eye(n))
    
    # Trade
    gap = x_star - current_weights
    delta_x = aim_matrix @ gap
    new_weights = current_weights + delta_x
    
    return new_weights, x_star, aim_matrix


# ---------------------------------------------------------------------------
# No-trade region (for linear TC)
# ---------------------------------------------------------------------------

def no_trade_region(
    alpha: np.ndarray,
    cov: np.ndarray,
    cost_per_unit: np.ndarray,  # linear TC per unit traded
    risk_aversion: float,
) -> tuple:
    """
    Compute the no-trade region for linear transaction costs.
    
    For linear TC: it is optimal NOT to trade when the gradient of 
    utility at current weights lies within [-c, +c] componentwise.
    
    Gradient of utility = α - γΣw
    No-trade condition: |αᵢ - (γΣw)ᵢ| ≤ cᵢ for all i
    """
    def utility_gradient(w):
        return alpha - risk_aversion * cov @ w
    
    def is_in_no_trade_region(w):
        grad = utility_gradient(w)
        return np.all(np.abs(grad) <= cost_per_unit)
    
    return is_in_no_trade_region, utility_gradient


# ---------------------------------------------------------------------------
# Backtest: compare TC-naive vs TC-aware rebalancing
# ---------------------------------------------------------------------------

def backtest_rebalancing(
    returns: np.ndarray,       # T×N return matrix
    alpha: np.ndarray,         # alpha signals (assumed constant)
    cov: np.ndarray,           # covariance (assumed constant)
    kappa: np.ndarray,         # per-asset quadratic TC
    risk_aversion: float = 1.0,
    rebal_frequency: int = 21, # rebalance every 21 days (monthly)
) -> pd.DataFrame:
    """
    Backtest: Markowitz (naive, rebalances monthly ignoring TC)
    vs TC-aware (incorporates quadratic impact cost).
    """
    T, N = returns.shape
    w_markowitz = np.ones(N) / N
    w_tc_aware  = np.ones(N) / N
    
    pnl_markowitz = []
    pnl_tc_aware  = []
    tc_markowitz  = []
    tc_tc_aware   = []
    
    for t in range(T):
        # Daily P&L
        ret = returns[t]
        pnl_markowitz.append(w_markowitz @ ret)
        pnl_tc_aware.append(w_tc_aware @ ret)
        
        # Rebalance
        if t % rebal_frequency == 0:
            # Markowitz: jump to target
            w_target = markowitz_target(alpha, cov, risk_aversion)
            delta_mko = w_target - w_markowitz
            tc_m = tc_quadratic(delta_mko, kappa)
            tc_markowitz.append(tc_m)
            w_markowitz = w_target
            
            # TC-aware: trade partially toward target
            w_new, _, _ = garleanu_pedersen_policy(
                alpha, cov, w_tc_aware, lambda_tc=50.0, risk_aversion=risk_aversion
            )
            delta_tc = w_new - w_tc_aware
            tc_t = tc_quadratic(delta_tc, kappa)
            tc_tc_aware.append(tc_t)
            w_tc_aware = w_new
        
        # Update weights by returns (no rebalancing between scheduled dates)
        w_markowitz = w_markowitz * (1 + returns[t])
        w_markowitz /= w_markowitz.sum()
        w_tc_aware = w_tc_aware * (1 + returns[t])
        w_tc_aware /= w_tc_aware.sum()
    
    return pd.DataFrame({
        'pnl_markowitz': pnl_markowitz,
        'pnl_tc_aware':  pnl_tc_aware,
    }), np.sum(tc_markowitz), np.sum(tc_tc_aware)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  Transaction Cost–Aware Portfolio Optimization")
    print("  Garleanu-Pedersen (2013) framework")
    print("═" * 66)
    
    np.random.seed(42)
    N = 5
    assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Alpha signals (annualised)
    alpha = np.array([0.08, 0.06, 0.10, 0.05, 0.09])
    
    # Covariance matrix
    vol = np.array([0.28, 0.25, 0.32, 0.27, 0.35])
    corr = np.array([
        [1.00, 0.75, 0.65, 0.70, 0.72],
        [0.75, 1.00, 0.62, 0.68, 0.70],
        [0.65, 0.62, 1.00, 0.60, 0.65],
        [0.70, 0.68, 0.60, 1.00, 0.68],
        [0.72, 0.70, 0.65, 0.68, 1.00],
    ])
    cov = np.outer(vol, vol) * corr / 252  # daily covariance
    
    # Current portfolio (equal-weight)
    w_current = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    
    # TC parameters: quadratic impact (higher for less liquid stocks)
    adv_millions = np.array([8000, 6000, 4000, 5000, 3000])  # $M avg daily volume
    price = np.array([185, 340, 175, 135, 300])
    # κᵢ ≈ σᵢ²/(ADV_i · price) — Almgren-Chriss calibration
    kappa = (vol / np.sqrt(252))**2 / (adv_millions * 1e6 / price) * 1e6
    
    # ── Markowitz target (no TC) ───────────────────────────────────
    print(f"\n── Markowitz Target (No TC) vs TC-Aware Optimal ──")
    
    gamma = 2.0  # risk aversion
    alpha_daily = alpha / 252
    
    w_markowitz = markowitz_target(alpha_daily, cov, gamma)
    
    # TC-aware optimum
    w_tc_opt = tc_aware_optimize(alpha_daily, cov, w_current, kappa, gamma, holding_periods=20)
    
    # TC costs
    tc_to_markowitz = tc_quadratic(w_markowitz - w_current, kappa)
    tc_to_tc_opt    = tc_quadratic(w_tc_opt - w_current, kappa)
    
    print(f"\n  {'Asset':>8} {'Current':>10} {'Markowitz':>12} {'TC-Aware':>10}")
    print("  " + "─" * 44)
    for i, name in enumerate(assets):
        print(f"  {name:>8} {w_current[i]:>10.2%} {w_markowitz[i]:>12.2%} {w_tc_opt[i]:>10.2%}")
    
    turnover_mko = np.abs(w_markowitz - w_current).sum() / 2
    turnover_tc  = np.abs(w_tc_opt - w_current).sum() / 2
    
    print(f"\n  Turnover to Markowitz: {turnover_mko:.2%}")
    print(f"  Turnover to TC-aware:  {turnover_tc:.2%}")
    print(f"  TC to Markowitz:       {tc_to_markowitz*10000:.2f}bps")
    print(f"  TC to TC-aware:        {tc_to_tc_opt*10000:.2f}bps")
    
    # Net alpha
    exp_ret_mko = alpha_daily @ w_markowitz * 252
    exp_ret_tc  = alpha_daily @ w_tc_opt * 252
    
    print(f"\n  Expected gross alpha (Markowitz): {exp_ret_mko:.2%}/yr")
    print(f"  TC cost annualised (Markowitz):   {tc_to_markowitz*20*252/252*10000:.2f}bps")
    print(f"  Expected gross alpha (TC-aware):  {exp_ret_tc:.2%}/yr")
    print(f"  TC cost annualised (TC-aware):    {tc_to_tc_opt*20*252/252*10000:.2f}bps")
    
    # ── GP policy: trading speed vs TC ────────────────────────────
    print(f"\n── Garleanu-Pedersen Policy: How Fast to Trade ──")
    print(f"\n  λ = TC parameter. λ=0: trade all the way, λ=∞: don't trade")
    print(f"\n  {'λ':>8} {'Trade%':>10} {'Turnover':>12} {'Est. TC (bps)':>16}")
    print("  " + "─" * 50)
    
    for lam in [1, 10, 50, 200, 1000]:
        w_new, x_star, aim = garleanu_pedersen_policy(
            alpha_daily, cov, w_current, lambda_tc=lam, risk_aversion=gamma)
        trade_pct = np.abs(w_new - w_current).sum() / np.abs(x_star - w_current).sum()
        turnover = np.abs(w_new - w_current).sum() / 2
        tc_bps = tc_quadratic(w_new - w_current, kappa) * 10000
        print(f"  {lam:>8} {trade_pct:>10.1%} {turnover:>12.2%} {tc_bps:>16.2f}bps")
    
    # ── No-trade region (linear TC) ───────────────────────────────
    print(f"\n── No-Trade Region (Linear TC = 10bps per asset) ──")
    
    cost_per_unit = np.full(N, 0.001)  # 10bps = 0.1% = 0.001 in weight space
    is_no_trade, grad_fn = no_trade_region(alpha_daily, cov, cost_per_unit, gamma)
    
    grad_at_current = grad_fn(w_current)
    print(f"\n  Utility gradient at current portfolio:")
    for i, name in enumerate(assets):
        in_nt = "NO TRADE" if abs(grad_at_current[i]) <= cost_per_unit[i] else "TRADE"
        print(f"    {name}: gradient={grad_at_current[i]*10000:.2f}bps, "
              f"threshold=±{cost_per_unit[i]*10000:.0f}bps → {in_nt}")
    
    print(f"\n  All assets in no-trade region: {is_no_trade(w_current)}")
    
    print(f"""
── TC-Aware vs Naive: Rule of Thumb ──

  When to trade:
    NET alpha from rebalancing = gross alpha gain - TC - timing risk
    Only rebalance if NET alpha > 0
    
    Rebalance threshold ≈ TC × (1 / alpha_half_life)
    
  Example: Momentum strategy
    Alpha: 10%/yr, half-life 21 days
    TC: 20bps per rebalance
    → Trade if drift from optimal > 20bps × (252/21) = 2.4% per year
    → Monthly rebalancing barely breaks even; quarterly is better

  GP key insight:
    "Aim for a portfolio in between current and target, not AT target."
    The optimal portfolio is never the Markowitz portfolio when TC > 0.

  Interview question (AQR, Citadel):
  Q: "Your Markowitz model says overweight AMZN by 10%. Do you trade?"
  A: "Depends on TC. At 30bps TC with 21-day alpha half-life, the net alpha
      from trading is 10% × (1/21yr) - 0.30% = 0.18%/yr. Barely worth it.
      TC-aware optimization would trade 2-3%, not 10%, at once."
    """)
