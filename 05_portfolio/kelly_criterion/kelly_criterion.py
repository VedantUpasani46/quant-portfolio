"""
Kelly Criterion and Optimal Leverage
======================================
The Kelly criterion (Kelly 1956) finds the bet size that maximises
the long-run geometric growth rate of wealth.

Single-bet Kelly:
  f* = p/a - q/b
  where:
    p = probability of winning
    q = 1 - p
    a = fraction lost if you lose (e.g., 1.0 = lose whole bet)
    b = fraction gained if you win (e.g., 0.5 = win 50% of bet)
  
  Equivalently: f* = edge / odds

Continuous Kelly (log-normal returns):
  f* = μ / σ²
  where μ = expected return, σ² = variance of returns.
  
  This is equivalent to maximising E[log(W)] = E[ln(W)].

Why Kelly maximises growth:
  Wealth after T periods: Wₜ = W₀ · Πₜ (1 + f·rₜ)
  ln(Wₜ/W₀) = Σₜ ln(1 + f·rₜ) ≈ T · E[ln(1 + f·r)]
  
  Maximise g(f) = E[ln(1 + f·r)]
  First-order condition: E[r / (1 + f*·r)] = 0
  For log-normal: f* = μ/σ²

Multi-asset Kelly:
  f* = Σ⁻¹ · μ   (vector of positions)
  where:
    Σ = covariance matrix of returns
    μ = vector of expected returns
  
  This is the same as the unconstrained Sharpe-maximising portfolio
  scaled by the inverse variance — but then FULLY LEVERED.

Fractional Kelly (practical):
  Full Kelly f* is theoretically optimal but causes extreme drawdowns.
  Fractional Kelly f = κ · f*  where κ ∈ (0, 1) is the "Kelly fraction".
  
  Common practice:
    κ = 1.0: Full Kelly — maximises growth, 50% drawdown in bad scenarios
    κ = 0.5: Half Kelly — ~75% of growth, much smaller drawdowns  
    κ = 0.25: Quarter Kelly — ~55% of growth, drawdowns manageable
  
  Rule of thumb: parameter uncertainty → use 0.25-0.5 Kelly.

Overbetting above Kelly:
  Betting MORE than f* REDUCES long-run growth → ruin possible.
  "Kelly is an upper bound on rational bet size."

Key relationships:
  - f* = 2 × Sharpe / (σ × √T)   [for Sharpe maximizing portfolio]
  - Max growth rate = μ - σ²/2    [at f* = 1 for normalised returns]
  - Kelly ↔ Markowitz: Kelly selects the same portfolio but answers
    "how much leverage?" not just "which weights?"

References:
  - Kelly, J.L. (1956). A New Interpretation of Information Rate.
    Bell System Technical Journal 35(4), 917–926.
  - Thorp, E.O. (1997). The Kelly Criterion in Blackjack, Sports Betting,
    and the Stock Market. Handbook of Asset and Liability Management.
  - MacLean, L.C. et al. (2011). The Kelly Capital Growth Investment Criterion.
    World Scientific.
  - Haghani, V. & Dewey, R. (2016). Rational Decision-Making Under Uncertainty.
    SSRN 2856963. [The famous experiment where quants overbet]
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Single-asset Kelly
# ---------------------------------------------------------------------------

def kelly_discrete(p: float, b: float, a: float = 1.0) -> float:
    """
    Discrete Kelly fraction for binary bet.
    f* = p/a - q/b  (fraction of bankroll to bet)
    p = win prob, b = profit per $ bet, a = loss per $ bet (=1 typically)
    """
    q = 1 - p
    return max(0.0, p / a - q / b)


def kelly_continuous(mu: float, sigma: float) -> float:
    """
    Continuous Kelly for log-normal returns.
    f* = μ / σ²
    μ = expected daily return, σ = daily vol.
    """
    return mu / (sigma ** 2 + 1e-10)


def growth_rate(f: float, mu: float, sigma: float) -> float:
    """
    Expected log-wealth growth rate per period (Kelly objective).
    g(f) ≈ f·μ - f²·σ²/2   (second-order Taylor expansion)
    """
    return f * mu - 0.5 * (f ** 2) * sigma ** 2


def fractional_kelly_tradeoff(
    mu: float,
    sigma: float,
    fractions: np.ndarray = None,
) -> pd.DataFrame:
    """
    Compute growth rate and drawdown statistics for different Kelly fractions.
    Shows the growth/drawdown tradeoff.
    """
    if fractions is None:
        fractions = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    
    f_star = kelly_continuous(mu, sigma)
    rows = []
    
    for kappa in fractions:
        f = kappa * f_star
        g = growth_rate(f, mu, sigma)
        ann_growth = g * 252
        
        # Approximate maximum drawdown under log-normal:
        # P(drawdown > d) ≈ exp(-2·g·d / σ²) for diffusion model
        # Expected max drawdown ≈ -ln(1-ε) where ε = 0.5 typically
        # Practical: from ruin probability P(W < W₀/2) with drift g
        # Simpler: MDD ≈ 2·σ / (f*μ) for Brownian motion
        if g > 0:
            mdd_expected = sigma**2 / (2 * g) if g > 0 else np.inf
            prob_halving = norm.cdf(-np.sqrt(2 * g / sigma**2))
        else:
            mdd_expected = np.inf
            prob_halving = 0.5
        
        rows.append({
            "kelly_fraction":  kappa,
            "f_actual":        f,
            "annual_growth":   ann_growth,
            "ann_growth_pct":  f"{ann_growth:.2%}",
            "growth_vs_full":  g / growth_rate(f_star, mu, sigma) if f_star > 0 else 0,
            "approx_mdd":      min(mdd_expected, 1.0),
            "is_optimal":      abs(kappa - 1.0) < 0.01,
        })
    
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulated wealth paths
# ---------------------------------------------------------------------------

def simulate_kelly_paths(
    mu: float,
    sigma: float,
    kelly_fractions: list,
    T: int = 1260,
    n_paths: int = 200,
    seed: int = 42,
) -> dict:
    """
    Simulate wealth paths for different Kelly fractions.
    Returns final wealth distribution and max drawdown for each.
    """
    rng = np.random.default_rng(seed)
    results = {}
    
    f_star = kelly_continuous(mu, sigma)
    daily_returns = rng.normal(mu, sigma, (T, n_paths))
    
    for kappa in kelly_fractions:
        f = kappa * f_star
        
        # Wealth paths
        levered_returns = f * daily_returns
        log_wealth = np.cumsum(np.log(1 + levered_returns), axis=0)
        wealth = np.exp(log_wealth)
        
        # Max drawdown per path
        running_max = np.maximum.accumulate(wealth, axis=0)
        drawdowns = (wealth - running_max) / running_max
        max_drawdowns = drawdowns.min(axis=0)
        
        results[kappa] = {
            "final_wealth": wealth[-1],
            "median_final": np.median(wealth[-1]),
            "mean_final":   np.mean(wealth[-1]),
            "pct_ruin":     (wealth[-1] < 0.5).mean(),  # lost > 50%
            "max_drawdown": np.median(max_drawdowns),
            "mdd_95pct":    np.percentile(max_drawdowns, 5),  # worst 5%
            "wealth_paths": wealth[:, :5],  # first 5 for plotting
        }
    
    return results, f_star


# ---------------------------------------------------------------------------
# Multi-asset Kelly
# ---------------------------------------------------------------------------

def kelly_multi_asset(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float = 0.0,
) -> np.ndarray:
    """
    Multi-asset Kelly portfolio.
    f* = Σ⁻¹ · (μ - rf)
    
    This is the FULLY LEVERED tangency portfolio.
    Total leverage = sum(|f*|) can be very large.
    
    In practice: use fractional Kelly (scale down) or constrain leverage.
    """
    excess_mu = mu - risk_free
    cov_inv = np.linalg.inv(cov)
    f_star = cov_inv @ excess_mu
    return f_star


def kelly_with_leverage_constraint(
    mu: np.ndarray,
    cov: np.ndarray,
    max_leverage: float = 2.0,
    risk_free: float = 0.0,
) -> np.ndarray:
    """
    Kelly portfolio with leverage constraint ||f||₁ ≤ max_leverage.
    Scales f* to satisfy the constraint.
    """
    f_star = kelly_multi_asset(mu, cov, risk_free)
    total_leverage = np.abs(f_star).sum()
    
    if total_leverage > max_leverage:
        f_star = f_star * max_leverage / total_leverage
    
    return f_star


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  Kelly Criterion and Optimal Leverage")
    print("  Single-asset, fractional Kelly, multi-asset")
    print("═" * 66)
    
    # ── 1. Discrete Kelly (casino example) ───────────────────────
    print(f"\n── 1. Discrete Kelly: Biased Coin Flip ──")
    print(f"\n  A coin with p=55% heads. Win 1× bet, lose 1× bet.")
    print(f"  f* = p - q = 0.55 - 0.45 = 0.10 (bet 10% of bankroll)")
    
    for p in [0.50, 0.51, 0.55, 0.60, 0.70]:
        f = kelly_discrete(p, b=1.0, a=1.0)
        print(f"    p={p:.2f}: f*={f:.4f} ({f:.1%} of bankroll)")
    
    # ── 2. Continuous Kelly for a stock strategy ──────────────────
    print(f"\n── 2. Continuous Kelly for a Stock Strategy ──")
    
    mu_daily    = 0.0008    # ~20% annual
    sigma_daily = 0.015     # ~24% annual
    f_star      = kelly_continuous(mu_daily, sigma_daily)
    
    print(f"\n  Strategy: μ={mu_daily*252:.0%}/yr, σ={sigma_daily*np.sqrt(252):.0%}/yr")
    print(f"  Sharpe ratio: {mu_daily/sigma_daily*np.sqrt(252):.2f}")
    print(f"  f* = μ/σ² = {mu_daily:.4f}/{sigma_daily**2:.6f} = {f_star:.2f}×")
    print(f"  Optimal leverage: {f_star:.2f}× (use {f_star:.1f}× leverage)")
    
    # ── 3. Fractional Kelly tradeoff ─────────────────────────────
    print(f"\n── 3. Fractional Kelly: Growth vs Drawdown Tradeoff ──")
    
    fractions = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00])
    df_tradeoff = fractional_kelly_tradeoff(mu_daily, sigma_daily, fractions)
    
    print(f"\n  {'κ (fraction)':>14} {'Leverage':>10} {'Ann Growth':>12} {'% of Max':>10} {'E[MDD]':>10}")
    print("  " + "─" * 60)
    for _, row in df_tradeoff.iterrows():
        marker = " ← OPTIMAL" if row["is_optimal"] else ""
        growth_of_max = row["growth_vs_full"] * 100
        print(f"  {row['kelly_fraction']:>14.2f}  {row['f_actual']:>10.2f}×  "
              f"{row['annual_growth']:>12.2%}  {growth_of_max:>10.1f}%  "
              f"{row['approx_mdd']:>10.2%}{marker}")
    
    print(f"\n  Half Kelly (κ=0.50): ~{df_tradeoff.loc[df_tradeoff.kelly_fraction==0.5, 'growth_vs_full'].values[0]*100:.0f}% of max growth")
    print(f"  Overbetting (κ=2.0): growth LESS than half Kelly!")
    
    # ── 4. Simulated wealth paths ─────────────────────────────────
    print(f"\n── 4. Simulated Wealth Paths (5 years, 200 paths) ──")
    
    sim_results, f_star_sim = simulate_kelly_paths(
        mu_daily, sigma_daily,
        kelly_fractions=[0.25, 0.5, 1.0, 1.5, 2.0],
        T=1260, n_paths=200,
    )
    
    print(f"\n  f* = {f_star_sim:.2f}×")
    print(f"\n  {'κ':>6} {'Leverage':>10} {'Median W':>12} {'Mean W':>12} {'P(ruin)':>10} {'Med MDD':>10}")
    print("  " + "─" * 62)
    for kappa, res in sim_results.items():
        lev = kappa * f_star_sim
        print(f"  {kappa:>6.2f}  {lev:>10.2f}×  "
              f"{res['median_final']:>12.3f}  {res['mean_final']:>12.3f}  "
              f"{res['pct_ruin']:>10.2%}  {res['max_drawdown']:>10.2%}")
    
    print(f"\n  Note: mean_W > median_W because distribution is right-skewed.")
    print(f"  Overbetting (κ=2.0): ruin rate spikes, median wealth collapses.")
    
    # ── 5. Multi-asset Kelly ──────────────────────────────────────
    print(f"\n── 5. Multi-Asset Kelly Portfolio ──")
    
    # 4-asset portfolio
    mu = np.array([0.0008, 0.0006, 0.0010, 0.0004])  # daily expected returns
    vol = np.array([0.015, 0.012, 0.020, 0.008])
    corr = np.array([
        [1.00, 0.60, 0.70, 0.10],
        [0.60, 1.00, 0.55, 0.05],
        [0.70, 0.55, 1.00, 0.15],
        [0.10, 0.05, 0.15, 1.00],
    ])
    cov = np.outer(vol, vol) * corr
    assets = ["US_Eq", "EU_Eq", "EM_Eq", "Bonds"]
    
    f_full = kelly_multi_asset(mu, cov)
    f_constrained = kelly_with_leverage_constraint(mu, cov, max_leverage=2.0)
    
    print(f"\n  Assets: {assets}")
    print(f"  Ann. returns: {mu*252}")
    
    print(f"\n  {'Asset':>8} {'Full Kelly':>14} {'2× constrained':>16}")
    print("  " + "─" * 42)
    for i, name in enumerate(assets):
        print(f"  {name:>8} {f_full[i]:>14.2f}× {f_constrained[i]:>16.2f}×")
    
    total_full = np.abs(f_full).sum()
    total_constr = np.abs(f_constrained).sum()
    print(f"  {'Total lev':>8} {total_full:>14.2f}× {total_constr:>16.2f}×")
    
    # Portfolio Sharpe and growth
    port_mu_full = f_full @ mu
    port_var_full = f_full @ cov @ f_full
    sharpe_full = port_mu_full / np.sqrt(port_var_full) * np.sqrt(252)
    
    port_mu_c = f_constrained @ mu
    port_var_c = f_constrained @ cov @ f_constrained
    sharpe_c = port_mu_c / np.sqrt(port_var_c) * np.sqrt(252)
    
    print(f"\n  Full Kelly:      Ann growth ≈ {(port_mu_full - 0.5*port_var_full)*252:.1%},  Sharpe ≈ {sharpe_full:.2f}")
    print(f"  2× constrained:  Ann growth ≈ {(port_mu_c - 0.5*port_var_c)*252:.1%},  Sharpe ≈ {sharpe_c:.2f}")
    
    print(f"""
── Key Insights ──

  Kelly criterion = optimal leverage for any investment:
    Under Kelly: wealth → ∞ almost surely (in infinite time)
    Under overbetting: wealth → 0 almost surely

  Why practitioners use fractional Kelly:
    1. Parameter uncertainty: μ estimated with error → f* uncertain
       True f* could be half the estimated one → full Kelly overbets
    2. Drawdowns: full Kelly can lose 50%+ before recovering
       Most investors cannot tolerate this psychologically or contractually
    3. Rule of thumb: use 0.25-0.50 Kelly when parameters uncertain

  Kelly vs Markowitz:
    Markowitz: WHAT to hold (portfolio weights)
    Kelly: HOW MUCH to leverage (scalar on top of Markowitz weights)
    Multi-asset Kelly = fully-levered tangency portfolio

  Interview question (Citadel, Jane Street):
  Q: "You have a coin with p=55% heads. What fraction do you bet?"
  A: "f* = p - q = 10% of bankroll. Betting more than 10% REDUCES
      long-run growth — Kelly is a maximum, not a target."
    """)
