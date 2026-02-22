"""
DCC-GARCH: Dynamic Conditional Correlation
============================================
Extends GARCH to model time-varying correlations between assets.
Critical for multi-asset portfolios where correlations spike during crises.

The DCC-GARCH model (Engle 2002):
  
  Step 1: Univariate GARCH(1,1) for each asset i:
    rᵢₜ = μᵢ + εᵢₜ
    εᵢₜ = σᵢₜ·zᵢₜ,  zᵢₜ ~ N(0,1)
    σᵢₜ² = ωᵢ + αᵢ·εᵢₜ₋₁² + βᵢ·σᵢₜ₋₁²
  
  Step 2: Standardised residuals:
    ηᵢₜ = εᵢₜ / σᵢₜ
  
  Step 3: Dynamic correlation:
    Qₜ = (1 − α − β)·Q̄ + α·ηₜ₋₁·ηₜ₋₁ᵀ + β·Qₜ₋₁
    Rₜ = diag(Qₜ)⁻¹/² · Qₜ · diag(Qₜ)⁻¹/²
  
  where:
    Q̄ = unconditional correlation of η
    Rₜ = time-varying correlation matrix at time t
    α, β = DCC parameters (α + β < 1 for stationarity)

Why DCC-GARCH matters:
  - Correlations are NOT constant: they spike to 0.8+ during crises,
    drop to 0.3 in calm markets
  - Static correlation underestimates portfolio risk during stress
  - Risk parity, minimum variance portfolios need dynamic correlations
  - Every systematic fund (Two Sigma, AQR, Man AHL) uses DCC or similar

Applications:
  - Dynamic portfolio optimization (MVO with time-varying Σₜ)
  - Risk budgeting (correlations → diversification benefit)
  - Contagion analysis (correlation spikes = crisis transmission)
  - Pairs trading (correlation breakdown detection)

References:
  - Engle, R.F. (2002). Dynamic Conditional Correlation: A Simple Class
    of Multivariate GARCH Models. Journal of Business & Economic Statistics 20(3).
  - Cappiello, L. et al. (2006). Asymmetric Dynamics in the Correlations of
    Global Equity and Bond Returns. Journal of Financial Econometrics 4(4).
  - Bauwens, L. et al. (2006). Multivariate GARCH Models: A Survey. Journal of
    Applied Econometrics 21(1).
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Step 1: Univariate GARCH(1,1) fitting
# ---------------------------------------------------------------------------

@dataclass
class GARCHParams:
    omega: float
    alpha: float
    beta: float
    mu: float = 0.0

    def is_stationary(self) -> bool:
        return self.alpha + self.beta < 1.0


def garch11_volatility(
    returns: np.ndarray,
    params: GARCHParams,
) -> np.ndarray:
    """
    Compute GARCH(1,1) conditional volatility path.
    σₜ² = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
    """
    T = len(returns)
    var = np.zeros(T)
    var[0] = returns.var()  # initial variance = unconditional
    
    eps = returns - params.mu
    
    for t in range(1, T):
        var[t] = params.omega + params.alpha * eps[t-1]**2 + params.beta * var[t-1]
    
    return np.sqrt(var)


def fit_garch11_mle(returns: np.ndarray) -> GARCHParams:
    """
    MLE estimation of GARCH(1,1) parameters.
    Log-likelihood: Σₜ [-0.5·ln(2π) - 0.5·ln(σₜ²) - 0.5·εₜ²/σₜ²]
    """
    mu_hat = returns.mean()
    eps = returns - mu_hat
    
    # Unconditional variance for initialization
    sigma2_uncond = eps.var()
    
    def neg_log_likelihood(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        T = len(eps)
        var = np.zeros(T)
        var[0] = sigma2_uncond
        
        for t in range(1, T):
            var[t] = omega + alpha * eps[t-1]**2 + beta * var[t-1]
            if var[t] <= 0:
                return 1e10
        
        # Log-likelihood (up to constant)
        ll = -0.5 * np.sum(np.log(var) + eps**2 / var)
        return -ll
    
    # Initial guess: typical GARCH values
    x0 = [sigma2_uncond * 0.01, 0.08, 0.90]
    result = minimize(neg_log_likelihood, x0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-8})
    
    omega, alpha, beta = result.x
    return GARCHParams(omega=omega, alpha=alpha, beta=beta, mu=mu_hat)


# ---------------------------------------------------------------------------
# Step 2 & 3: DCC estimation
# ---------------------------------------------------------------------------

@dataclass
class DCCParams:
    alpha_dcc: float    # DCC α parameter
    beta_dcc: float     # DCC β parameter
    Q_bar: np.ndarray   # Unconditional correlation matrix

    def is_stationary(self) -> bool:
        return self.alpha_dcc + self.beta_dcc < 1.0


class DCCGARCHModel:
    """
    Dynamic Conditional Correlation GARCH model.
    
    Workflow:
      1. fit() — estimate univariate GARCH + DCC parameters
      2. dynamic_correlation() — extract Rₜ time series
      3. conditional_covariance() — Σₜ = Dₜ·Rₜ·Dₜ where Dₜ = diag(σₜ)
    """
    
    def __init__(self):
        self.garch_params: list[GARCHParams] = []
        self.dcc_params: Optional[DCCParams] = None
        self.returns: Optional[np.ndarray] = None
        self.volatilities: Optional[np.ndarray] = None
        self.std_residuals: Optional[np.ndarray] = None
    
    def fit(self, returns: np.ndarray) -> None:
        """
        Estimate DCC-GARCH on T×N return matrix.
        
        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
            Daily returns for N assets over T time periods.
        """
        T, N = returns.shape
        self.returns = returns
        
        # Step 1: Fit univariate GARCH(1,1) for each asset
        print(f"  Fitting {N} univariate GARCH(1,1) models...")
        self.garch_params = []
        self.volatilities = np.zeros((T, N))
        
        for i in range(N):
            params = fit_garch11_mle(returns[:, i])
            self.garch_params.append(params)
            self.volatilities[:, i] = garch11_volatility(returns[:, i], params)
        
        # Step 2: Standardised residuals
        self.std_residuals = np.zeros((T, N))
        for i in range(N):
            eps = returns[:, i] - self.garch_params[i].mu
            self.std_residuals[:, i] = eps / self.volatilities[:, i]
        
        # Step 3: Estimate DCC parameters
        print(f"  Estimating DCC parameters...")
        Q_bar = np.corrcoef(self.std_residuals.T)  # unconditional correlation
        
        self.dcc_params = self._fit_dcc(self.std_residuals, Q_bar)
        
        print(f"  DCC α={self.dcc_params.alpha_dcc:.4f}, β={self.dcc_params.beta_dcc:.4f}")
        print(f"  Persistence (α+β) = {self.dcc_params.alpha_dcc + self.dcc_params.beta_dcc:.4f}")
    
    def _fit_dcc(self, eta: np.ndarray, Q_bar: np.ndarray) -> DCCParams:
        """
        Estimate DCC(1,1) parameters via quasi-MLE.
        Qₜ = (1−α−β)·Q̄ + α·ηₜ₋₁·ηₜ₋₁ᵀ + β·Qₜ₋₁
        """
        T, N = eta.shape
        
        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            Q = Q_bar.copy()
            ll = 0.0
            
            for t in range(1, T):
                # Q update
                Q = (1 - alpha - beta) * Q_bar + alpha * np.outer(eta[t-1], eta[t-1]) + beta * Q
                
                # Correlation matrix Rₜ
                Q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Q)))
                R = Q_diag_inv_sqrt @ Q @ Q_diag_inv_sqrt
                
                # Ensure positive definite
                eigvals = np.linalg.eigvalsh(R)
                if np.any(eigvals <= 0):
                    return 1e10
                
                # Log-likelihood contribution (up to constant)
                try:
                    sign, logdet = np.linalg.slogdet(R)
                    if sign <= 0:
                        return 1e10
                    ll += -0.5 * (logdet + eta[t] @ np.linalg.inv(R) @ eta[t])
                except np.linalg.LinAlgError:
                    return 1e10
            
            return -ll
        
        x0 = [0.05, 0.90]
        result = minimize(neg_log_likelihood, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6})
        
        alpha_dcc, beta_dcc = result.x
        return DCCParams(alpha_dcc=alpha_dcc, beta_dcc=beta_dcc, Q_bar=Q_bar)
    
    def dynamic_correlation(self) -> np.ndarray:
        """
        Compute Rₜ for all t.
        Returns array of shape (T, N, N).
        """
        if self.dcc_params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        T, N = self.std_residuals.shape
        R_series = np.zeros((T, N, N))
        Q = self.dcc_params.Q_bar.copy()
        
        for t in range(T):
            if t > 0:
                eta_prev = self.std_residuals[t-1]
                Q = ((1 - self.dcc_params.alpha_dcc - self.dcc_params.beta_dcc) * self.dcc_params.Q_bar
                     + self.dcc_params.alpha_dcc * np.outer(eta_prev, eta_prev)
                     + self.dcc_params.beta_dcc * Q)
            
            Q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Q)))
            R = Q_diag_inv_sqrt @ Q @ Q_diag_inv_sqrt
            R_series[t] = R
        
        return R_series
    
    def conditional_covariance(self) -> np.ndarray:
        """
        Compute time-varying covariance Σₜ = Dₜ·Rₜ·Dₜ.
        Returns array of shape (T, N, N).
        """
        R_series = self.dynamic_correlation()
        T, N = self.volatilities.shape
        Sigma_series = np.zeros((T, N, N))
        
        for t in range(T):
            D = np.diag(self.volatilities[t])
            Sigma_series[t] = D @ R_series[t] @ D
        
        return Sigma_series


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  DCC-GARCH: Dynamic Conditional Correlation")
    print("  Time-varying correlations for multi-asset portfolios")
    print("═" * 66)
    
    # Simulate 3 assets with time-varying correlations
    np.random.seed(42)
    T = 1260  # 5 years daily
    N = 3
    
    # Simulate returns with correlation regime change
    returns = np.zeros((T, N))
    vol_base = [0.015, 0.020, 0.025]  # 1.5%, 2.0%, 2.5% daily vol
    
    # First half: low correlation (normal market)
    # Second half: high correlation (crisis)
    for t in range(T):
        if t < T // 2:
            corr = 0.3  # normal market
        else:
            corr = 0.7  # crisis: correlations spike
        
        # Build correlation matrix
        R = np.eye(N)
        for i in range(N):
            for j in range(i+1, N):
                R[i, j] = R[j, i] = corr
        
        # Covariance = D·R·D
        D = np.diag(vol_base)
        Sigma = D @ R @ D
        
        # Draw multivariate normal
        z = np.random.multivariate_normal(np.zeros(N), Sigma)
        returns[t] = z
    
    print(f"\n  Simulated {T} days of {N}-asset returns")
    print(f"  Correlation regime: 0.3 (days 1-630) → 0.7 (days 631-1260)")
    
    # Fit DCC-GARCH
    print(f"\n── Fitting DCC-GARCH Model ──")
    model = DCCGARCHModel()
    model.fit(returns)
    
    # Extract univariate GARCH results
    print(f"\n── Univariate GARCH(1,1) Results ──")
    print(f"\n  {'Asset':>8} {'ω':>10} {'α':>8} {'β':>8} {'α+β':>8} {'Ann Vol':>10}")
    print("  " + "─" * 60)
    for i, params in enumerate(model.garch_params):
        ann_vol = np.sqrt(252) * returns[:, i].std()
        print(f"  Asset {i+1:>2} {params.omega:>10.6f} {params.alpha:>8.4f} "
              f"{params.beta:>8.4f} {params.alpha + params.beta:>8.4f} {ann_vol:>10.2%}")
    
    # Dynamic correlations
    R_series = model.dynamic_correlation()
    
    print(f"\n── Dynamic Correlation: Asset 1 vs Asset 2 ──")
    print(f"\n  {'Period':>20} {'Mean ρ₁₂':>12} {'Min':>8} {'Max':>8}")
    print("  " + "─" * 52)
    
    # First half vs second half
    rho12_first = R_series[:T//2, 0, 1]
    rho12_second = R_series[T//2:, 0, 1]
    
    print(f"  {'Days 1-630 (normal)':>20} {rho12_first.mean():>12.4f} "
          f"{rho12_first.min():>8.4f} {rho12_first.max():>8.4f}")
    print(f"  {'Days 631-1260 (crisis)':>20} {rho12_second.mean():>12.4f} "
          f"{rho12_second.min():>8.4f} {rho12_second.max():>8.4f}")
    
    # Rolling correlation (120-day window for comparison)
    rolling_corr = pd.Series(returns[:, 0]).rolling(120).corr(pd.Series(returns[:, 1]))
    
    # Compare last 100 days
    last_100 = slice(-100, None)
    print(f"\n  {'Last 100 days':>20} {'DCC ρ₁₂':>12} {'Rolling 120d ρ':>18}")
    print("  " + "─" * 54)
    print(f"  {'Mean':>20} {R_series[last_100, 0, 1].mean():>12.4f} "
          f"{rolling_corr.iloc[last_100].mean():>18.4f}")
    print(f"  {'Std':>20} {R_series[last_100, 0, 1].std():>12.4f} "
          f"{rolling_corr.iloc[last_100].std():>18.4f}")
    
    # Conditional covariance
    Sigma_series = model.conditional_covariance()
    
    print(f"\n── Portfolio Volatility (Equal-Weighted) ──")
    w = np.ones(N) / N
    port_var_static = w @ np.cov(returns.T) @ w
    port_vol_static = np.sqrt(port_var_static) * np.sqrt(252)
    
    port_var_dcc = np.array([w @ Sigma_series[t] @ w for t in range(T)])
    port_vol_dcc = np.sqrt(port_var_dcc) * np.sqrt(252)
    
    print(f"\n  Static covariance (sample cov): {port_vol_static:.2%} annualised")
    print(f"  DCC time-varying (mean):        {port_vol_dcc.mean():.2%}")
    print(f"  DCC (5th percentile):           {np.percentile(port_vol_dcc, 5):.2%}")
    print(f"  DCC (95th percentile):          {np.percentile(port_vol_dcc, 95):.2%}")
    
    print(f"\n  Static vol understates risk in high-correlation regime by:")
    print(f"    {np.percentile(port_vol_dcc, 95) - port_vol_static:.2%} "
          f"({(np.percentile(port_vol_dcc, 95) / port_vol_static - 1) * 100:.1f}%)")
    
    print(f"""
── Why DCC-GARCH Matters for Systematic Funds ──

  1. Diversification breakdown during crises:
     Normal market: ρ ≈ 0.3  → diversification benefit
     Crisis:        ρ → 0.7  → "all assets fall together"
     Static models miss this → underestimate tail risk

  2. Dynamic portfolio optimization:
     MVO with Σₜ instead of static Σ̄
     Rebalance when correlations spike (reduce equity exposure)
     Risk parity: scale positions by 1/σₜ dynamically

  3. Pairs trading:
     Correlation breakdown = signal deterioration
     DCC detects this in real-time → exit before blowup

  4. Contagion analysis:
     Spike in ρ(SPY, EEM) = EM crisis transmission
     Two Sigma, AQR use DCC for global macro strategies

  Interview question (DE Shaw, Two Sigma):
  Q: "Your MVO portfolio blows up in 2008. Why?"
  A: "Static correlation from 2005-2007 (ρ≈0.4) underestimated
      crisis correlation (ρ≈0.85). DCC-GARCH would have detected
      the correlation regime shift and reduced exposure."
    """)
