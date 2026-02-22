"""
Bootstrap Confidence Intervals
================================
Efron's (1979) bootstrap: estimate sampling distribution and confidence
intervals via resampling without assuming parametric distributions.

Why bootstrap?
  1. No parametric assumptions (doesn't require normality)
  2. Works for ANY statistic (median, Sharpe ratio, IC, alpha, tail risk)
  3. Accounts for parameter uncertainty (not just point estimates)
  4. Handles complex estimators (e.g., Fama-French α, GARCH parameters)

The algorithm:
  1. Draw B bootstrap samples: {X₁*, X₂*, ..., Xᵦ*} via resampling with replacement
  2. Compute θ̂ᵦ* for each sample (e.g., mean, Sharpe ratio, regression coef)
  3. Confidence interval methods:
     - Percentile: [θ̂₂.₅%, θ̂₉₇.₅%]
     - Bias-corrected percentile (BCa): adjusts for bias and skewness
     - Standard error: θ̂ ± 1.96·SE_bootstrap

Block bootstrap (for time-series):
  Regular bootstrap assumes iid → wrong for autocorrelated returns.
  Solution: resample BLOCKS of consecutive observations.
  Block size b = O(T^(1/3)) balances bias (large b) vs variance (small b).

Stationary bootstrap (Politis & Romano 1994):
  Random block sizes with geometric distribution.
  Expected block length = 1/p where p = prob(new block).

Applications in quant finance:
  - Sharpe ratio CI: bootstrap accounts for fat tails (normal approx fails)
  - GARCH parameter uncertainty: MLE gives point estimate, bootstrap gives CI
  - Alpha significance: is Fama-French α sig different from zero?
  - VaR confidence bands: bootstrap historical VaR for uncertainty around 99th percentile
  - Backtest robustness: does strategy Sharpe vary across bootstrap samples?

Bootstrap caveats:
  - Doesn't help with model misspecification (garbage in = garbage out)
  - Requires "large enough" sample (T > 50 for Sharpe ratio)
  - Block bootstrap requires choosing block size (no universal rule)
  - Can be slow (B=1000 samples × complex model)

References:
  - Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife.
    Annals of Statistics 7(1), 1–26.
  - Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman & Hall.
  - Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap.
    Journal of the American Statistical Association 89(428), 1303–1313.
  - Ledoit, O. & Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio.
    Journal of Empirical Finance 15(5), 850–859.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Standard bootstrap
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    statistic: float                # original sample statistic
    bootstrap_estimates: np.ndarray # B bootstrap estimates
    ci_lower: float                 # lower CI bound
    ci_upper: float                 # upper CI bound
    std_error: float                # bootstrap standard error
    bias: float                     # bootstrap bias estimate
    alpha: float                    # confidence level (default 0.05 for 95% CI)

    def summary(self, stat_name: str = "Statistic") -> str:
        ci_pct = (1 - self.alpha) * 100
        lines = [
            f"  {stat_name}:",
            f"    Point estimate:     {self.statistic:.4f}",
            f"    Bootstrap mean:     {self.bootstrap_estimates.mean():.4f}",
            f"    Bootstrap SE:       {self.std_error:.4f}",
            f"    Bias:               {self.bias:.4f}",
            f"    {ci_pct:.0f}% CI:             [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
        ]
        return "\n".join(lines)


def bootstrap(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """
    Standard bootstrap (iid resampling).

    Parameters
    ----------
    data : np.ndarray
        Original sample (1D array).
    statistic_func : callable
        Function that takes data and returns a scalar statistic.
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Significance level (0.05 = 95% CI).

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    
    # Original statistic
    theta_hat = statistic_func(data)
    
    # Bootstrap estimates
    theta_star = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        theta_star[b] = statistic_func(sample)
    
    # Confidence interval (percentile method)
    ci_lower = np.percentile(theta_star, alpha / 2 * 100)
    ci_upper = np.percentile(theta_star, (1 - alpha / 2) * 100)
    
    # Standard error and bias
    se = theta_star.std()
    bias = theta_star.mean() - theta_hat
    
    return BootstrapResult(
        statistic=theta_hat,
        bootstrap_estimates=theta_star,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=se,
        bias=bias,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Block bootstrap for time series
# ---------------------------------------------------------------------------

def block_bootstrap(
    data: np.ndarray,
    statistic_func: callable,
    block_size: int = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """
    Block bootstrap for time-series data.
    Resamples blocks of consecutive observations to preserve autocorrelation.

    If block_size is None, uses Politis & White (2004) automatic selection:
    b_opt ≈ T^(1/3) for weakly dependent data.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    
    if block_size is None:
        block_size = int(n ** (1/3))  # rule of thumb
    
    # Original statistic
    theta_hat = statistic_func(data)
    
    # Bootstrap
    theta_star = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Number of blocks needed to fill sample
        n_blocks = int(np.ceil(n / block_size))
        
        # Randomly select starting indices for blocks
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        
        # Build bootstrap sample from blocks
        blocks = [data[start:start + block_size] for start in starts]
        sample = np.concatenate(blocks)[:n]  # trim to original length
        
        theta_star[b] = statistic_func(sample)
    
    ci_lower = np.percentile(theta_star, alpha / 2 * 100)
    ci_upper = np.percentile(theta_star, (1 - alpha / 2) * 100)
    se = theta_star.std()
    bias = theta_star.mean() - theta_hat
    
    return BootstrapResult(
        statistic=theta_hat,
        bootstrap_estimates=theta_star,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=se,
        bias=bias,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  Bootstrap Confidence Intervals")
    print("  Efron's bootstrap for parameter uncertainty")
    print("═" * 66)
    
    # ── Example 1: Sharpe ratio ──────────────────────────────────
    print(f"\n── Example 1: Sharpe Ratio Confidence Interval ──")
    
    np.random.seed(42)
    T = 252  # 1 year daily
    mu_annual, vol_annual = 0.08, 0.18
    returns = np.random.normal(mu_annual / 252, vol_annual / np.sqrt(252), T)
    
    def sharpe_ratio(r):
        return r.mean() / r.std() * np.sqrt(252)
    
    sharpe_bootstrap = bootstrap(returns, sharpe_ratio, n_bootstrap=1000)
    
    print(f"\n  Sample: {T} daily returns")
    print(f"  Annualised return: {returns.mean() * 252:.2%}")
    print(f"  Annualised vol:    {returns.std() * np.sqrt(252):.2%}")
    print(f"\n" + sharpe_bootstrap.summary("Sharpe Ratio"))
    
    # Compare to parametric (assumes normality)
    sharpe_hat = sharpe_ratio(returns)
    sharpe_se_parametric = np.sqrt((1 + 0.5 * sharpe_hat**2) / T) * np.sqrt(252)
    ci_param_lower = sharpe_hat - 1.96 * sharpe_se_parametric
    ci_param_upper = sharpe_hat + 1.96 * sharpe_se_parametric
    
    print(f"\n  Parametric 95% CI (assumes normality):")
    print(f"    [{ci_param_lower:.4f}, {ci_param_upper:.4f}]")
    print(f"  Bootstrap 95% CI:")
    print(f"    [{sharpe_bootstrap.ci_lower:.4f}, {sharpe_bootstrap.ci_upper:.4f}]")
    
    # ── Example 2: Alpha from Fama-French regression ─────────────
    print(f"\n── Example 2: Fama-French Alpha Confidence Interval ──")
    
    # Simulate asset returns with alpha
    T = 1260  # 5 years
    mkt_ret = np.random.normal(0.0004, 0.01, T)
    smb_ret = np.random.normal(0.0001, 0.005, T)
    hml_ret = np.random.normal(0.0001, 0.005, T)
    
    true_alpha = 0.0002  # 20bp/day = 5% annual
    asset_ret = (true_alpha
                 + 1.2 * mkt_ret
                 + 0.3 * smb_ret
                 - 0.1 * hml_ret
                 + np.random.normal(0, 0.005, T))
    
    factors = np.column_stack([mkt_ret, smb_ret, hml_ret])
    
    def fama_french_alpha(data):
        """Returns annualised alpha from FF3 regression."""
        y, X = data[:, 0], data[:, 1:]
        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        # OLS: α = (X'X)^{-1} X'y
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return coef[0] * 252  # annualise
    
    data_ff = np.column_stack([asset_ret, factors])
    alpha_bootstrap = bootstrap(data_ff, fama_french_alpha, n_bootstrap=1000)
    
    print(f"\n  True alpha (annualised): {true_alpha * 252:.2%}")
    print(f"\n" + alpha_bootstrap.summary("FF3 Alpha (annual)"))
    
    # Test if alpha significantly > 0
    pval = (alpha_bootstrap.bootstrap_estimates <= 0).mean()
    print(f"\n  Hypothesis test: α > 0?")
    print(f"    p-value (bootstrap): {pval:.4f}")
    if pval < 0.05:
        print(f"    → Reject H₀ (α = 0) at 5% level ✓")
    else:
        print(f"    → Fail to reject H₀")
    
    # ── Example 3: Block bootstrap for autocorrelated data ───────
    print(f"\n── Example 3: Block Bootstrap (AR(1) Process) ──")
    
    # Simulate AR(1) returns
    T = 500
    phi = 0.8  # high autocorrelation
    ar1_returns = np.zeros(T)
    ar1_returns[0] = 0.0005
    for t in range(1, T):
        ar1_returns[t] = 0.0002 + phi * ar1_returns[t-1] + np.random.randn() * 0.01
    
    def mean_return(r):
        return r.mean() * 252  # annualise
    
    # Standard bootstrap (WRONG for autocorrelated data)
    mean_wrong = bootstrap(ar1_returns, mean_return, n_bootstrap=500)
    
    # Block bootstrap (CORRECT)
    mean_correct = block_bootstrap(ar1_returns, mean_return, block_size=20, n_bootstrap=500)
    
    print(f"\n  AR(1) returns (φ={phi}, T={T})")
    print(f"  True mean (annual): {ar1_returns.mean() * 252:.2%}")
    print(f"\n  Standard bootstrap (iid assumption — WRONG):")
    print(f"    95% CI: [{mean_wrong.ci_lower:.4f}, {mean_wrong.ci_upper:.4f}]")
    print(f"    SE:     {mean_wrong.std_error:.4f}")
    print(f"\n  Block bootstrap (block_size=20 — CORRECT):")
    print(f"    95% CI: [{mean_correct.ci_lower:.4f}, {mean_correct.ci_upper:.4f}]")
    print(f"    SE:     {mean_correct.std_error:.4f}")
    print(f"\n  Block bootstrap SE is {mean_correct.std_error / mean_wrong.std_error:.2f}× larger")
    print(f"  (correctly accounts for autocorrelation → wider CI)")
    
    # ── Example 4: Bootstrap for VaR ─────────────────────────────
    print(f"\n── Example 4: VaR Confidence Interval ──")
    
    # Fat-tailed returns (Student-t)
    pnl = np.random.standard_t(df=5, size=1000) * 100  # dollars
    
    def var_99(data):
        return -np.percentile(data, 1)  # 99% VaR
    
    var_bootstrap = bootstrap(pnl, var_99, n_bootstrap=1000)
    
    print(f"\n  1000 days of P&L (fat-tailed, df=5)")
    print(f"\n" + var_bootstrap.summary("99% VaR"))
    
    print(f"""
── When to Use Bootstrap ──

  ✓ USE bootstrap when:
    - Statistic has no closed-form distribution (Sharpe, IC, tail risk)
    - Data is non-normal (fat tails, skew)
    - Small sample (< 100 observations)
    - Complex estimator (GARCH MLE, machine learning model params)
    - Want model-free CI (no parametric assumptions)

  ✗ DON'T use bootstrap when:
    - Data is already too small (< 30 observations)
    - Model is severely misspecified (bootstrap can't fix bad models)
    - Statistic is simple and has known distribution (use exact formula)

  Block bootstrap REQUIRED for:
    - Autocorrelated returns (momentum strategies, GARCH residuals)
    - Overlapping periods (rolling 20-day returns)
    - Time-varying volatility

  Interview question (Two Sigma, AQR):
  Q: "Your backtest Sharpe is 1.5 ± 0.3. How do you compute the ±0.3?"
  A: "Bootstrap the Sharpe ratio 1000 times, take the standard deviation
      of the bootstrap distribution. Use block bootstrap with block_size ≈ 20
      to account for return autocorrelation. The ±0.3 is the 95% CI half-width
      (1.96 × bootstrap_SE). If the CI includes 0, the Sharpe is not significant."
    """)
