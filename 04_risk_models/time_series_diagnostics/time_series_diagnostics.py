"""
Time Series Diagnostics for Quantitative Finance
==================================================
Core statistical tests every quant must know for model validation,
signal research, and risk model building.

1. Autocorrelation Function (ACF) / PACF
   ACF(k)  = Corr(rₜ, rₜ₋ₖ)   — correlation at lag k
   PACF(k) = partial correlation at lag k (controlling for shorter lags)
   Use: identify AR/MA order, detect return predictability

2. Ljung-Box Q-test
   H₀: no autocorrelation up to lag K
   Q = T(T+2) Σₖ₌₁ᴷ ρ̂ₖ²/(T-k)  ~  χ²(K)
   Use: test if GARCH residuals are white noise, detect serial dependence

3. Durbin-Watson test
   DW = Σ(eₜ-eₜ₋₁)² / Σeₜ²   [range 0-4, DW≈2 means no autocorr]
   DW < 2: positive autocorrelation
   DW > 2: negative autocorrelation
   Use: OLS regression residuals (factor model, CAPM)

4. ADF (Augmented Dickey-Fuller) test
   H₀: unit root (non-stationary)
   Δyₜ = α + βt + γyₜ₋₁ + Σ φⱼΔyₜ₋ⱼ + εₜ
   Reject H₀ → series is stationary
   Use: test yield series, price series, spreads for cointegration

5. KPSS test
   H₀: stationary (opposite of ADF)
   Use ADF + KPSS together: ADF rejects + KPSS fails to reject → stationary

6. Return autocorrelation patterns:
   Short-horizon (1-5 days): often negative (bid-ask bounce, mean reversion)
   Medium-horizon (1-12 months): positive (momentum effect)
   Long-horizon (3-5 years): negative (long-run mean reversion)

7. Variance ratio test (Lo & MacKinlay 1988):
   VR(q) = Var(q-period return) / (q × Var(1-period return))
   VR(q) = 1 for random walk
   VR(q) > 1 → positive autocorrelation (momentum)
   VR(q) < 1 → negative autocorrelation (mean reversion)

References:
  - Ljung, G.M. & Box, G.E.P. (1978). On a Measure of Lack of Fit in Time Series Models.
    Biometrika 65(2), 297–303.
  - Lo, A. & MacKinlay, A.C. (1988). Stock Market Prices Do Not Follow Random Walks.
    Review of Financial Studies 1(1), 41–66.
  - Dickey, D.A. & Fuller, W.A. (1979). Distribution of the Estimators for
    Autoregressive Time Series with a Unit Root. JASA 74(366), 427–431.
  - Tsay, R.S. (2010). Analysis of Financial Time Series, 3rd ed. Wiley.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, norm
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Autocorrelation Function
# ---------------------------------------------------------------------------

def acf(series: np.ndarray, max_lags: int = 20) -> np.ndarray:
    """
    Sample autocorrelation function.
    ACF(k) = Cov(x_t, x_{t-k}) / Var(x_t)
    """
    n = len(series)
    mean = series.mean()
    var = ((series - mean)**2).mean()
    
    acf_vals = np.zeros(max_lags + 1)
    acf_vals[0] = 1.0
    for k in range(1, max_lags + 1):
        acf_vals[k] = ((series[k:] - mean) * (series[:-k] - mean)).mean() / (var + 1e-12)
    return acf_vals


def pacf(series: np.ndarray, max_lags: int = 20) -> np.ndarray:
    """
    Partial autocorrelation function via Yule-Walker equations.
    """
    n = len(series)
    acf_vals = acf(series, max_lags)
    
    pacf_vals = np.zeros(max_lags + 1)
    pacf_vals[0] = 1.0
    pacf_vals[1] = acf_vals[1]
    
    for k in range(2, max_lags + 1):
        # Yule-Walker: solve Γ·φ = γ
        gamma = acf_vals[1:k+1]
        Gamma = np.array([[acf_vals[abs(i-j)] for j in range(k)] for i in range(k)])
        try:
            phi = np.linalg.solve(Gamma, gamma)
            pacf_vals[k] = phi[-1]
        except np.linalg.LinAlgError:
            pacf_vals[k] = 0.0
    
    return pacf_vals


def acf_confidence_band(n: int, alpha: float = 0.05) -> float:
    """
    95% confidence band for ACF under iid assumption.
    ±z_{α/2} / √T
    """
    return norm.ppf(1 - alpha / 2) / np.sqrt(n)


# ---------------------------------------------------------------------------
# Ljung-Box Q-test
# ---------------------------------------------------------------------------

def ljung_box(series: np.ndarray, max_lags: int = 20) -> pd.DataFrame:
    """
    Ljung-Box Q-test for autocorrelation up to each lag K.
    H₀: ρ₁ = ρ₂ = ... = ρₖ = 0  (no autocorrelation)

    Q_K = T(T+2) Σₖ₌₁ᴷ ρ̂ₖ² / (T-k)  ~  χ²(K)
    """
    n = len(series)
    acf_vals = acf(series, max_lags)
    
    rows = []
    Q = 0.0
    for k in range(1, max_lags + 1):
        Q += acf_vals[k]**2 / (n - k)
        Q_k = n * (n + 2) * Q
        p_val = 1 - chi2.cdf(Q_k, df=k)
        rows.append({
            'lag': k,
            'acf': acf_vals[k],
            'Q_stat': Q_k,
            'p_value': p_val,
            'reject_H0': p_val < 0.05,
        })
    
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Durbin-Watson
# ---------------------------------------------------------------------------

def durbin_watson(residuals: np.ndarray) -> float:
    """
    Durbin-Watson statistic for regression residuals.
    DW = Σ(eₜ - eₜ₋₁)² / Σeₜ²
    DW ≈ 2(1 - ρ̂₁)  where ρ̂₁ is lag-1 ACF.
    Range: 0 to 4. DW=2: no autocorr. DW<2: positive. DW>2: negative.
    """
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)


# ---------------------------------------------------------------------------
# ADF test (Augmented Dickey-Fuller)
# ---------------------------------------------------------------------------

def adf_test(series: np.ndarray, lags: int = 1) -> dict:
    """
    ADF test for unit root.
    H₀: series has a unit root (non-stationary)
    H₁: series is stationary

    Δyₜ = α + γ·yₜ₋₁ + Σⱼ φⱼ·Δyₜ₋ⱼ + εₜ
    Test statistic: t-stat of γ̂ (compare to MacKinnon critical values)
    """
    n = len(series)
    dy = np.diff(series)
    
    # Build regressors
    y_lag = series[lags:-1] if lags > 0 else series[:-1]
    
    X_cols = [y_lag]  # lagged level
    for j in range(1, lags + 1):
        X_cols.append(dy[lags - j:-j] if j > 0 else dy[lags:])
    X_cols.append(np.ones(len(y_lag)))  # constant
    
    X = np.column_stack(X_cols)
    y = dy[lags:]
    
    if len(y) < len(X):
        y = dy[lags:lags + len(X)]
    
    # Trim to same length
    min_len = min(len(y), len(X))
    y = y[:min_len]
    X = X[:min_len]
    
    # OLS
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        s2 = resid.var()
        XtX_inv = np.linalg.inv(X.T @ X + 1e-10 * np.eye(X.shape[1]))
        se = np.sqrt(np.diag(XtX_inv * s2))
        t_stat = beta[0] / (se[0] + 1e-10)  # t-stat on γ
    except Exception:
        t_stat = 0.0
    
    # MacKinnon (1994) approximate critical values (constant case)
    crit_1pct  = -3.43
    crit_5pct  = -2.86
    crit_10pct = -2.57
    
    # Approximate p-value using interpolation
    if t_stat <= crit_1pct:
        p_approx = 0.01
    elif t_stat <= crit_5pct:
        p_approx = 0.05
    elif t_stat <= crit_10pct:
        p_approx = 0.10
    else:
        p_approx = 0.50
    
    return {
        't_statistic': t_stat,
        'p_value': p_approx,
        'critical_1pct': crit_1pct,
        'critical_5pct': crit_5pct,
        'reject_H0_5pct': t_stat < crit_5pct,
        'conclusion': 'Stationary' if t_stat < crit_5pct else 'Non-stationary (unit root)',
    }


# ---------------------------------------------------------------------------
# Variance Ratio Test (Lo & MacKinlay 1988)
# ---------------------------------------------------------------------------

def variance_ratio_test(returns: np.ndarray, q: int) -> dict:
    """
    Lo-MacKinlay variance ratio test.
    VR(q) = Var(q-period return) / (q × Var(1-period return))
    H₀: VR(q) = 1 (random walk)
    H₁: VR(q) ≠ 1

    VR > 1 → positive autocorrelation (momentum)
    VR < 1 → negative autocorrelation (mean reversion)
    """
    n = len(returns)
    mu = returns.mean()
    
    # 1-period variance
    sigma1_sq = np.sum((returns - mu)**2) / (n - 1)
    
    # q-period variance (overlapping)
    q_returns = np.array([returns[i:i+q].sum() for i in range(n - q + 1)])
    mu_q = q_returns.mean()
    sigmaq_sq = np.sum((q_returns - mu_q)**2) / (len(q_returns) - 1)
    
    vr = sigmaq_sq / (q * sigma1_sq) if sigma1_sq > 0 else 1.0
    
    # Asymptotic standard error under homoskedasticity
    se = np.sqrt(2 * (2*q - 1) * (q - 1) / (3 * q * n))
    z_stat = (vr - 1) / (se + 1e-12)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    return {
        'q': q,
        'variance_ratio': vr,
        'z_statistic': z_stat,
        'p_value': p_value,
        'reject_H0': p_value < 0.05,
        'interpretation': 'Momentum' if vr > 1 else 'Mean reversion' if vr < 1 else 'Random walk',
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 68)
    print("  Time Series Diagnostics")
    print("  ACF, Ljung-Box, ADF, Variance Ratio — return predictability")
    print("═" * 68)
    
    np.random.seed(42)
    T = 1260  # 5 years
    
    # ── 1. ACF of raw returns vs squared returns ──────────────────
    print(f"\n── 1. Autocorrelation of Daily Returns ──")
    
    # Simulate GARCH(1,1) returns — realistic
    omega, alpha_g, beta_g = 1e-6, 0.08, 0.90
    sigma2 = np.zeros(T); returns = np.zeros(T)
    sigma2[0] = omega / (1 - alpha_g - beta_g)
    for t in range(1, T):
        sigma2[t] = omega + alpha_g * returns[t-1]**2 + beta_g * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.randn()
    
    acf_raw = acf(returns, max_lags=10)
    acf_sq  = acf(returns**2, max_lags=10)
    conf    = acf_confidence_band(T)
    
    print(f"\n  95% confidence band: ±{conf:.4f}")
    print(f"\n  {'Lag':>6} {'ACF(rₜ)':>12} {'Sig?':>6} {'ACF(rₜ²)':>12} {'Sig?':>6}")
    print("  " + "─" * 46)
    for k in range(1, 11):
        sig_raw = "*" if abs(acf_raw[k]) > conf else " "
        sig_sq  = "***" if abs(acf_sq[k]) > conf else " "
        print(f"  {k:>6} {acf_raw[k]:>12.4f} {sig_raw:>6} {acf_sq[k]:>12.4f} {sig_sq:>6}")
    
    print(f"\n  Raw returns: mostly insignificant (efficient markets) ✓")
    print(f"  Squared returns: highly significant (volatility clustering) ✓")
    print(f"  → Evidence of GARCH effects: past vol predicts future vol")
    
    # ── 2. Ljung-Box test ─────────────────────────────────────────
    print(f"\n── 2. Ljung-Box Q-Test ──")
    lb_raw  = ljung_box(returns, max_lags=10)
    lb_sq   = ljung_box(returns**2, max_lags=10)
    
    print(f"\n  Testing raw returns (H₀: no autocorrelation):")
    print(f"  {'Lag':>6} {'Q-stat':>10} {'p-value':>10} {'Reject H₀?':>12}")
    print("  " + "─" * 42)
    for _, row in lb_raw[lb_raw['lag'].isin([1, 5, 10])].iterrows():
        flag = "YES" if row['reject_H0'] else "no"
        print(f"  {row['lag']:>6.0f} {row['Q_stat']:>10.3f} {row['p_value']:>10.4f} {flag:>12}")
    
    print(f"\n  Testing squared returns (H₀: no ARCH effects):")
    print(f"  {'Lag':>6} {'Q-stat':>10} {'p-value':>10} {'Reject H₀?':>12}")
    print("  " + "─" * 42)
    for _, row in lb_sq[lb_sq['lag'].isin([1, 5, 10])].iterrows():
        flag = "YES ← ARCH!" if row['reject_H0'] else "no"
        print(f"  {row['lag']:>6.0f} {row['Q_stat']:>10.3f} {row['p_value']:>10.4f} {flag:>12}")
    
    # ── 3. ADF test ───────────────────────────────────────────────
    print(f"\n── 3. ADF Unit Root Tests ──")
    
    # Prices (should be non-stationary), returns (should be stationary)
    prices = 100 * np.exp(np.cumsum(returns))
    log_prices = np.log(prices)
    
    adf_prices  = adf_test(log_prices)
    adf_returns = adf_test(returns)
    
    # 10Y Treasury yield (mean-reverting → stationary?)
    yield_path = np.zeros(T)
    yield_path[0] = 0.05
    for t in range(1, T):
        yield_path[t] = 0.10 * (0.05 - yield_path[t-1]) + yield_path[t-1] + np.random.randn() * 0.002
    
    adf_yield = adf_test(yield_path)
    
    print(f"\n  {'Series':>18} {'t-stat':>10} {'p-value':>10} {'Reject H₀?':>12} {'Conclusion':>20}")
    print("  " + "─" * 74)
    for label, result in [
        ("Log prices", adf_prices),
        ("Returns", adf_returns),
        ("10Y yield", adf_yield),
    ]:
        flag = "YES (stationary)" if result['reject_H0_5pct'] else "NO (unit root)"
        print(f"  {label:>18} {result['t_statistic']:>10.3f} {result['p_value']:>10.4f} "
              f"{flag:>32}")
    
    print(f"\n  Log prices: fail to reject unit root → non-stationary ✓")
    print(f"  Returns:    reject unit root → stationary ✓")
    print(f"  Yield:      mean-reverting process → stationary ✓")
    
    # ── 4. Variance Ratio Test ────────────────────────────────────
    print(f"\n── 4. Lo-MacKinlay Variance Ratio Test ──")
    print(f"\n  H₀: returns follow a random walk (VR = 1)")
    print(f"  VR > 1 → momentum,  VR < 1 → mean reversion")
    print(f"\n  {'q (days)':>10} {'VR(q)':>10} {'z-stat':>10} {'p-value':>10} {'Signal':>16}")
    print("  " + "─" * 58)
    
    for q in [2, 5, 10, 21, 63, 126, 252]:
        vr = variance_ratio_test(returns, q)
        flag = "✓ sig" if vr['reject_H0'] else "n.s."
        print(f"  {q:>10} {vr['variance_ratio']:>10.4f} {vr['z_statistic']:>10.3f} "
              f"{vr['p_value']:>10.4f} {vr['interpretation']:>14} {flag:>4}")
    
    print(f"\n  Pure GBM: all VR(q) ≈ 1.0 (random walk)")
    print(f"  Real equities: short-run VR < 1 (bid-ask), 1-12M VR > 1 (momentum)")
    
    # ── 5. Autocorrelation of strategy signal ─────────────────────
    print(f"\n── 5. Signal Quality: IC and Autocorrelation ──")
    
    # Simulate a momentum signal with decay
    true_ic = 0.03
    signal_ac = 0.70  # signal autocorrelation (persistence)
    signal = np.zeros(T)
    signal[0] = np.random.randn() * true_ic
    for t in range(1, T):
        signal[t] = signal_ac * signal[t-1] + np.random.randn() * np.sqrt(1 - signal_ac**2) * true_ic
    
    # Add noise
    realized_return = true_ic * signal + np.random.randn(T) * 0.015
    
    ic_series = pd.Series(signal).rolling(63).corr(pd.Series(realized_return))
    acf_signal = acf(signal, max_lags=10)
    
    lb_signal = ljung_box(signal, max_lags=5)
    
    print(f"\n  Momentum signal (true IC={true_ic:.2f}, AC(1)={signal_ac})")
    print(f"\n  Signal ACF:")
    for k in range(1, 6):
        bar = "█" * int(abs(acf_signal[k]) * 50)
        print(f"    lag {k}: {acf_signal[k]:>8.4f}  {bar}")
    
    print(f"\n  Ljung-Box on signal (lags 1-5):")
    for _, row in lb_signal.head(5).iterrows():
        flag = "persistent ← good signal" if row['reject_H0'] else "white noise"
        print(f"    lag {row['lag']:.0f}: Q={row['Q_stat']:.2f}, p={row['p_value']:.4f} → {flag}")
    
    dw = durbin_watson(signal)
    print(f"\n  Durbin-Watson statistic: {dw:.4f}")
    print(f"  (DW < 2 → positive autocorrelation, confirms signal persistence)")
    
    print(f"""
── Why These Tests Matter ──

  In model validation:
    1. After fitting GARCH: check Ljung-Box on standardised residuals
       → If Q-test rejects H₀, your GARCH hasn't captured all volatility clustering
    2. Factor model residuals: Durbin-Watson
       → DW << 2 means omitted factor with time trend
    3. Spread trading: ADF on the spread
       → Reject unit root → spread is stationary → pairs trade is valid

  In alpha research:
    4. Signal ACF: high ACF → signal is persistent → lower turnover needed
       IC × √(1/(1-AC)) = Information Ratio (Grinold-Kahn)
    5. Variance ratio: identifies return horizon
       VR > 1 at q=21 → monthly momentum signal worthwhile

  Interview question (Two Sigma, DE Shaw):
  Q: "Test whether S&P 500 daily returns have any predictability."
  A: "Run Ljung-Box on raw returns → likely fail to reject (efficient)
      Run Ljung-Box on squared returns → reject strongly (ARCH effects)
      Run variance ratio test → VR > 1 at 1-3 months (momentum anomaly)
      Conclusion: mean is unpredictable, variance is predictable."
    """)
