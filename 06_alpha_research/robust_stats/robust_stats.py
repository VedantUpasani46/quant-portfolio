"""
Robust Statistics for Quantitative Finance
============================================
Standard OLS, sample mean, and sample covariance are easily distorted
by outliers — and financial data has FAT TAILS and outliers everywhere
(earnings surprises, flash crashes, data errors, liquidity gaps).

Robust statistics provide estimators that are resistant to outliers:

1. Robust Location Estimators:
   - Sample mean:  not robust (one bad observation = large distortion)
   - Median:       robust but inefficient (ignores 50% of data)
   - Huber M-estimator: compromise — efficient near centre, robust in tails

2. M-estimators (Huber 1964):
   Instead of minimising Σ(yᵢ - μ)² (OLS = squared loss),
   minimise Σ ρ(yᵢ - μ)  where ρ is a robust loss function.
   
   The influence function tells us how much one observation affects θ̂:
     IF(x; T, F) = lim_{ε→0} [T(F + ε·δₓ) - T(F)] / ε
   
   Robust estimators have BOUNDED influence functions.

3. Huber loss:
   ρ(u) = { ½u²           if |u| ≤ c   [quadratic in centre]
           { c|u| - ½c²   if |u| > c   [linear in tails]
   
   The tuning constant c = 1.345·σ makes Huber 95% efficient at the normal.
   Key property: square residuals are "winsorised" at c.

4. Robust covariance estimation:
   - Sample covariance: masively distorted by outliers
   - Minimum Covariance Determinant (MCD): Rousseeuw (1984)
     Find the subset of h = ⌊(n+p+1)/2⌋ observations with smallest det(cov)
     Use that subset to estimate the covariance
   - Ledoit-Wolf: shrinkage (already in portfolio), handles structured outliers

5. Robust regression (Theil-Sen):
   Slope = median of all pairwise slopes β̂ᵢⱼ = (yⱼ - yᵢ)/(xⱼ - xᵢ)
   Breakdown point: 29.3% (can tolerate almost 1/3 contaminated data)
   vs OLS: breakdown point = 0% (one outlier can move slope arbitrarily)

6. Applications in quant finance:
   - Factor model estimation: outlier returns distort betas
   - Covariance for portfolio optimization: tail events inflate correlations
   - Alpha signal construction: winsorization IS a robust technique
   - Regime detection: outliers confuse HMM state estimation
   - Risk model calibration: fat tails need robust vol/corr estimation

References:
  - Huber, P.J. (1964). Robust Estimation of a Location Parameter.
    Annals of Mathematical Statistics 35(1), 73–101.
  - Rousseeuw, P.J. & Leroy, A.M. (1987). Robust Regression and Outlier Detection.
    Wiley.
  - Maronna, R. et al. (2019). Robust Statistics: Theory and Methods, 2nd ed.
  - Cont, R. (2001). Empirical Properties of Asset Returns. Quantitative Finance 1(2).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Huber M-estimator for location
# ---------------------------------------------------------------------------

def huber_loss(u: np.ndarray, c: float) -> np.ndarray:
    """Huber loss function applied elementwise."""
    return np.where(np.abs(u) <= c,
                    0.5 * u**2,
                    c * np.abs(u) - 0.5 * c**2)


def huber_psi(u: np.ndarray, c: float) -> np.ndarray:
    """
    Huber ψ-function: derivative of Huber loss.
    ψ(u) = u if |u| ≤ c,  sign(u)·c if |u| > c
    Used in iteratively reweighted least squares (IRLS).
    """
    return np.clip(u, -c, c)


def huber_weights(residuals: np.ndarray, c: float) -> np.ndarray:
    """
    Huber weights for IRLS.
    w(u) = ψ(u)/u = 1 if |u| ≤ c, c/|u| if |u| > c
    Downweights large residuals.
    """
    abs_r = np.abs(residuals) + 1e-10
    return np.minimum(1.0, c / abs_r)


def huber_location(
    data: np.ndarray,
    c: float = 1.345,  # 95% efficiency at normal
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """
    Huber M-estimator of location via IRLS (Iteratively Reweighted LS).
    
    Algorithm:
      1. Initialize μ₀ = median(data)
      2. Compute residuals rᵢ = (xᵢ - μ) / σ̂   (standardise)
      3. Compute Huber weights wᵢ = min(1, c/|rᵢ|)
      4. Update μ = Σwᵢxᵢ / Σwᵢ   (weighted mean)
      5. Repeat until convergence.
    """
    mu = np.median(data)
    sigma = np.median(np.abs(data - mu)) / 0.6745  # MAD estimate of sigma
    
    for _ in range(max_iter):
        residuals = (data - mu) / (sigma + 1e-10)
        weights = huber_weights(residuals, c)
        mu_new = np.sum(weights * data) / np.sum(weights)
        if abs(mu_new - mu) < tol:
            break
        mu = mu_new
    
    return mu


# ---------------------------------------------------------------------------
# Huber robust regression
# ---------------------------------------------------------------------------

def huber_regression(
    X: np.ndarray,
    y: np.ndarray,
    c: float = 1.345,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict:
    """
    Huber M-estimator for linear regression via IRLS.
    
    Minimise Σ ρ((yᵢ - Xᵢβ) / σ̂) where ρ is the Huber loss.
    Equivalent to: WLS with weights wᵢ = min(1, c/|rᵢ|)
    
    Returns beta coefficients, standard errors, t-stats.
    """
    n, p = X.shape
    
    # Initialize with OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    for _ in range(max_iter):
        residuals = y - X @ beta
        sigma = np.median(np.abs(residuals)) / 0.6745  # robust scale
        std_resid = residuals / (sigma + 1e-10)
        weights = huber_weights(std_resid, c)
        
        # WLS: (X'WX)⁻¹ X'Wy
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        
        try:
            beta_new = np.linalg.solve(XtWX + 1e-8 * np.eye(p), XtWy)
        except np.linalg.LinAlgError:
            break
        
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    
    # Sandwich standard errors
    residuals = y - X @ beta
    sigma = np.median(np.abs(residuals)) / 0.6745
    std_resid = residuals / (sigma + 1e-10)
    weights = huber_weights(std_resid, c)
    
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    
    # Robust variance estimator
    score = (weights * residuals)[:, None] * X
    meat = score.T @ score
    try:
        XtWX_inv = np.linalg.inv(XtWX + 1e-8 * np.eye(p))
        var_beta = XtWX_inv @ meat @ XtWX_inv
        se = np.sqrt(np.diag(var_beta))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    
    t_stats = beta / (se + 1e-10)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))
    
    return {
        'coefficients': beta,
        'std_errors': se,
        't_statistics': t_stats,
        'p_values': p_values,
        'scale': sigma,
        'weights': weights,
        'n_downweighted': (weights < 0.99).sum(),
    }


# ---------------------------------------------------------------------------
# Theil-Sen robust slope estimator
# ---------------------------------------------------------------------------

def theil_sen(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Theil-Sen estimator: slope = median of all pairwise slopes.
    Breakdown point: 29.3% (vs 0% for OLS).
    
    For N points: O(N²) pairwise slopes → take median.
    """
    n = len(x)
    slopes = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if abs(dx) > 1e-10:
                slopes.append((y[j] - y[i]) / dx)
    
    slope = np.median(slopes)
    intercept = np.median(y - slope * x)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'n_pairs': len(slopes),
    }


# ---------------------------------------------------------------------------
# Robust covariance: MCD-inspired
# ---------------------------------------------------------------------------

def robust_covariance_mcd_simple(
    X: np.ndarray,
    alpha: float = 0.5,  # fraction of data to use
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    """
    Simplified MCD (Minimum Covariance Determinant) estimator.
    
    Find the subset of h = ⌊α·n⌋ observations with smallest determinant
    of their covariance matrix.
    
    Full MCD uses FAST-MCD algorithm (Rousseeuw 1999). This is a
    Monte Carlo approximation — sufficient for portfolios < 50 assets.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    h = int(alpha * n)
    
    best_det = np.inf
    best_indices = None
    
    # Random starts
    for _ in range(n_trials):
        # Pick random h observations
        idx = rng.choice(n, h, replace=False)
        cov_h = np.cov(X[idx].T)
        
        if np.linalg.matrix_rank(cov_h) < p:
            continue
        
        det = np.linalg.det(cov_h)
        if 0 < det < best_det:
            best_det = det
            best_indices = idx
    
    if best_indices is None:
        return {'location': X.mean(axis=0), 'covariance': np.cov(X.T)}
    
    X_mcd = X[best_indices]
    location = X_mcd.mean(axis=0)
    covariance = np.cov(X_mcd.T)
    
    # Compute Mahalanobis distances for ALL observations using MCD estimates
    cov_inv = np.linalg.inv(covariance + 1e-8 * np.eye(p))
    diff = X - location
    mahal = np.array([d @ cov_inv @ d for d in diff])
    
    # Flag outliers: Mahalanobis distance > χ²(p, 0.975)
    threshold = stats.chi2.ppf(0.975, df=p)
    outlier_mask = mahal > threshold
    
    return {
        'location': location,
        'covariance': covariance,
        'mahalanobis': mahal,
        'outlier_mask': outlier_mask,
        'n_outliers': outlier_mask.sum(),
        'selected_indices': best_indices,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 68)
    print("  Robust Statistics for Quantitative Finance")
    print("  Huber estimators, Theil-Sen, robust covariance (MCD)")
    print("═" * 68)
    
    np.random.seed(42)
    n = 252  # 1 year
    
    # ── 1. Location estimators under outliers ─────────────────────
    print(f"\n── 1. Location Estimators: Robustness to Outliers ──")
    
    # Clean data: daily returns with a fat-tailed distribution
    clean_returns = np.random.standard_t(df=5, size=n) * 0.015
    
    # Contaminated: 5% outliers (data errors, flash crashes)
    contaminated = clean_returns.copy()
    n_outliers = int(0.05 * n)
    outlier_idx = np.random.choice(n, n_outliers, replace=False)
    contaminated[outlier_idx] = np.random.choice([-0.20, 0.20], n_outliers)
    
    true_mean = 0.0
    
    print(f"\n  True mean: {true_mean:.4f}")
    print(f"  Contamination: {n_outliers} outliers ({n_outliers/n:.0%} of data) at ±20%")
    
    estimators = {
        'Sample mean':        clean_returns.mean(),
        'Median':             np.median(clean_returns),
        'Huber (clean)':      huber_location(clean_returns),
        'Sample mean (cont.)':contaminated.mean(),
        'Median (cont.)':     np.median(contaminated),
        'Huber (cont.)':      huber_location(contaminated),
    }
    
    print(f"\n  {'Estimator':>24} {'Value':>10} {'|Error|':>10}")
    print("  " + "─" * 48)
    for name, val in estimators.items():
        print(f"  {name:>24} {val:>10.6f} {abs(val - true_mean):>10.6f}")
    
    print(f"\n  Huber is barely affected by the 5% contamination.")
    print(f"  Sample mean shifts by {abs(estimators['Sample mean (cont.)'] - estimators['Sample mean']):.4f} — large!")
    
    # ── 2. Factor model with robust regression ────────────────────
    print(f"\n── 2. Robust Factor Regression: OLS vs Huber ──")
    
    # Simulate Fama-French style data
    T = 500
    mkt  = np.random.normal(0.0005, 0.010, T)
    smb  = np.random.normal(0.0001, 0.005, T)
    hml  = np.random.normal(0.0001, 0.005, T)
    
    # True betas
    true_alpha, true_beta_mkt, true_beta_smb = 0.0002, 1.2, 0.4
    
    returns = (true_alpha
               + true_beta_mkt * mkt
               + true_beta_smb * smb
               + np.random.normal(0, 0.008, T))
    
    # Add 2% outliers
    n_bad = int(0.02 * T)
    bad_idx = np.random.choice(T, n_bad, replace=False)
    returns[bad_idx] += np.random.choice([-0.15, 0.15], n_bad)
    
    X = np.column_stack([np.ones(T), mkt, smb, hml])
    feature_names = ['alpha', 'β_mkt', 'β_smb', 'β_hml']
    
    # OLS
    ols_beta = np.linalg.lstsq(X, returns, rcond=None)[0]
    ols_resid = returns - X @ ols_beta
    ols_se = np.sqrt(np.diag(np.var(ols_resid) * np.linalg.inv(X.T @ X + 1e-10*np.eye(4))))
    ols_t = ols_beta / (ols_se + 1e-10)
    
    # Huber regression
    hub = huber_regression(X, returns)
    
    true_vals = [true_alpha, true_beta_mkt, true_beta_smb, 0.0]
    
    print(f"\n  {n_bad} outliers injected ({n_bad/T:.0%} of data)")
    print(f"\n  {'Parameter':>10} {'True':>8} {'OLS':>10} {'OLS t':>8} {'Huber':>10} {'Hub t':>8} {'Winner':>8}")
    print("  " + "─" * 64)
    for i, name in enumerate(feature_names):
        ols_better = abs(ols_beta[i] - true_vals[i]) < abs(hub['coefficients'][i] - true_vals[i])
        winner = "OLS" if ols_better else "Huber"
        print(f"  {name:>10} {true_vals[i]:>8.4f} {ols_beta[i]:>10.4f} {ols_t[i]:>8.2f} "
              f"{hub['coefficients'][i]:>10.4f} {hub['t_statistics'][i]:>8.2f} {winner:>8}")
    
    print(f"\n  Huber downweighted {hub['n_downweighted']} observations ({hub['n_downweighted']/T:.1%})")
    print(f"  These are the likely outliers/data errors")
    
    # ── 3. Theil-Sen vs OLS for beta estimation ───────────────────
    print(f"\n── 3. Theil-Sen Robust Regression ──")
    
    # Single-factor: estimate beta with contaminated data
    n_small = 50
    mkt_s = np.random.normal(0.0005, 0.010, n_small)
    true_beta = 1.5
    stock_r = true_beta * mkt_s + np.random.normal(0, 0.008, n_small)
    
    # Inject 2 extreme outliers
    stock_r[0] = 0.25   # flash crash
    stock_r[1] = -0.20
    
    # OLS
    cov = np.cov(mkt_s, stock_r)
    ols_slope = cov[0, 1] / cov[0, 0]
    
    # Theil-Sen
    ts = theil_sen(mkt_s, stock_r)
    
    print(f"\n  True beta: {true_beta}")
    print(f"  2 extreme outliers injected ({2/n_small:.0%} of data)")
    print(f"  OLS beta:      {ols_slope:.4f}   (error: {abs(ols_slope - true_beta):.4f})")
    print(f"  Theil-Sen β:   {ts['slope']:.4f}   (error: {abs(ts['slope'] - true_beta):.4f})")
    
    # ── 4. Robust covariance for portfolio optimization ───────────
    print(f"\n── 4. Robust Covariance (MCD) for Portfolio Optimization ──")
    
    N, p = 200, 4  # 200 days, 4 assets
    cov_true = np.array([
        [0.0004, 0.0002, 0.0002, 0.0001],
        [0.0002, 0.0003, 0.0001, 0.0001],
        [0.0002, 0.0001, 0.0005, 0.0001],
        [0.0001, 0.0001, 0.0001, 0.0002],
    ])
    
    returns_mat = np.random.multivariate_normal(np.zeros(p), cov_true, N)
    
    # Inject 5% multivariate outliers (correlated extreme moves)
    n_mv_outliers = int(0.05 * N)
    outlier_rows = np.random.choice(N, n_mv_outliers, replace=False)
    returns_mat[outlier_rows] = np.random.multivariate_normal(
        np.zeros(p), 100 * cov_true, n_mv_outliers)
    
    # Sample covariance
    cov_sample = np.cov(returns_mat.T)
    
    # Robust MCD covariance
    mcd = robust_covariance_mcd_simple(returns_mat, alpha=0.75, n_trials=50)
    cov_mcd = mcd['covariance']
    
    print(f"\n  {n_mv_outliers} multivariate outliers injected ({n_mv_outliers/N:.0%})")
    print(f"  {mcd['n_outliers']} flagged by Mahalanobis distance (threshold χ²₄,0.975)")
    
    print(f"\n  Diagonal (variances) comparison:")
    print(f"  {'Asset':>8} {'True var':>12} {'Sample var':>12} {'MCD var':>10} {'Best':>8}")
    print("  " + "─" * 50)
    for i in range(p):
        true_v  = cov_true[i, i]
        samp_v  = cov_sample[i, i]
        mcd_v   = cov_mcd[i, i]
        best = "MCD" if abs(mcd_v - true_v) < abs(samp_v - true_v) else "Sample"
        print(f"  {'Asset ' + str(i+1):>8} {true_v:>12.6f} {samp_v:>12.6f} "
              f"{mcd_v:>10.6f} {best:>8}")
    
    print(f"\n  Sample covariance inflated by outliers — MCD more accurate ✓")
    
    print(f"""
── When to Use Robust Methods in Finance ──

  USE robust estimators when:
    - Return data has fat tails (Student-t, not normal)
    - Data may contain errors (corporate actions, stale prices)
    - Flash crashes / extreme events in sample
    - Factor model estimation with many stocks (some will have outliers)
    - Pairs trading: cointegration test sensitive to outliers

  DON'T over-trim when:
    - You're modelling tail risk (EVT, VaR) — keep the tails
    - Outliers are informative (earnings surprises are real signals)

  Practical hierarchy (most commonly used):
    1. Winsorization (clipping at 1st/99th percentile) — simple, universal
    2. Huber regression for factor models — 5-10% contamination handled
    3. Theil-Sen for pair-wise betas — very robust, O(N²)
    4. MCD covariance — for portfolio optimization under contamination

  Interview question (Two Sigma, DE Shaw):
  Q: "Your factor model alpha is 20bps, but it disappears when you
      remove the 3 best days of the last year. What happened?"
  A: "3 outlier days drove the estimated alpha. The mean return is not
      robust to these extreme observations. Fix: use Huber M-estimator
      for alpha, or bootstrap and check alpha distribution is not driven
      by a handful of observations."
    """)
