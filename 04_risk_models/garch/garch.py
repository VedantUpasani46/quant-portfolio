"""
GARCH(1,1) Volatility Model
=============================
Estimates and forecasts volatility using the Generalised Autoregressive
Conditional Heteroskedasticity (GARCH) model of Bollerslev (1986).

Model specification:
  Return equation:   r_t = μ + ε_t,     ε_t = σ_t · z_t,   z_t ~ N(0,1)
  Variance equation: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Parameters:
  ω (omega) > 0  — long-run variance intercept
  α (alpha) ≥ 0  — ARCH effect: sensitivity of σ² to recent shocks
  β (beta)  ≥ 0  — GARCH persistence: how quickly shocks decay
  α + β < 1      — stationarity condition (mean-reversion to long-run variance)

Long-run (unconditional) variance: σ²_∞ = ω / (1 - α - β)
Half-life of volatility shock: h = ln(0.5) / ln(α + β)

Estimation:
  Maximum likelihood estimation (MLE) assuming Gaussian innovations.
  Log-likelihood: ℓ = -T/2 · ln(2π) - 1/2 · Σ [ln(σ²_t) + ε²_t/σ²_t]

  Uses scipy.optimize.minimize with L-BFGS-B and multiple starting points
  to avoid local optima.

Why GARCH matters:
  Asset returns exhibit volatility clustering — calm periods are followed
  by calm, turbulent by turbulent (Mandelbrot, 1963). GARCH captures this
  stylised fact and is the foundation of options market-making, VaR, and
  risk-weighted asset calculation under Basel.

References:
  - Bollerslev, T. (1986). Generalised Autoregressive Conditional
    Heteroskedasticity. Journal of Econometrics, 31, 307–327.
  - Engle, R.F. (1982). ARCH. Econometrica, 50(4), 987–1007.
    [Nobel Prize 2003]
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 23.
  - Lopez de Prado, M. (2018). Advances in Financial ML, Ch. 2.
"""

import math
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize, OptimizeResult


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GARCHParams:
    """Estimated GARCH(1,1) parameters."""
    mu: float       # return mean
    omega: float    # variance intercept
    alpha: float    # ARCH coefficient
    beta: float     # GARCH coefficient
    log_likelihood: float = 0.0

    @property
    def persistence(self) -> float:
        """α + β — closer to 1 means longer memory."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """σ²_∞ = ω / (1 - α - β)"""
        denom = 1 - self.alpha - self.beta
        if denom <= 0:
            return float("inf")
        return self.omega / denom

    @property
    def long_run_vol(self) -> float:
        """Annualised long-run volatility: √(σ²_∞ · 252)"""
        return math.sqrt(max(0, self.long_run_variance * 252))

    @property
    def half_life(self) -> float:
        """Days for a vol shock to decay to half its initial magnitude."""
        if self.persistence >= 1:
            return float("inf")
        return math.log(0.5) / math.log(self.persistence)

    def summary(self) -> str:
        return (
            f"  GARCH(1,1) Estimates\n"
            f"  {'─'*38}\n"
            f"  μ  (mean return)        : {self.mu:>10.6f}\n"
            f"  ω  (omega, intercept)   : {self.omega:>10.8f}\n"
            f"  α  (alpha, ARCH)        : {self.alpha:>10.6f}\n"
            f"  β  (beta, GARCH)        : {self.beta:>10.6f}\n"
            f"  α + β (persistence)     : {self.persistence:>10.6f}\n"
            f"  Long-run vol (annual)   : {self.long_run_vol:>10.4%}\n"
            f"  Half-life (days)        : {self.half_life:>10.2f}\n"
            f"  Log-likelihood          : {self.log_likelihood:>10.4f}"
        )


@dataclass
class GARCHForecast:
    """Multi-step-ahead variance and volatility forecasts."""
    horizon: int
    conditional_variances: np.ndarray   # daily variances, shape (horizon,)
    sigma2_0: float                     # current (t=0) conditional variance

    @property
    def conditional_vols_annual(self) -> np.ndarray:
        return np.sqrt(self.conditional_variances * 252)

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Horizon (days)": range(1, self.horizon + 1),
            "Cond. Var (daily)": self.conditional_variances,
            "Cond. Vol (annual)": self.conditional_vols_annual,
        })


# ---------------------------------------------------------------------------
# Core GARCH model
# ---------------------------------------------------------------------------

class GARCH11:
    """
    GARCH(1,1) model fitted via maximum likelihood.

    Parameters
    ----------
    returns : array-like
        Daily log returns (as decimals, not percentages).

    Usage
    -----
    >>> import numpy as np
    >>> returns = np.random.normal(0, 0.01, 1000)
    >>> model = GARCH11(returns)
    >>> params = model.fit()
    >>> print(params.summary())
    >>> forecast = model.forecast(horizon=10)
    """

    def __init__(self, returns: np.ndarray | pd.Series):
        if isinstance(returns, pd.Series):
            self.returns = returns.dropna().values
            self.dates = returns.dropna().index
        else:
            self.returns = np.asarray(returns, dtype=float)
            self.dates = None
        self.T = len(self.returns)
        self._params: GARCHParams | None = None
        self._conditional_variances: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def _filter_variances(self, mu: float, omega: float, alpha: float, beta: float) -> np.ndarray:
        """
        Compute the sequence of conditional variances σ²_t given parameters.
        Initialise σ²_1 with the sample variance (robust to starting values).
        """
        eps = self.returns - mu
        T = self.T
        sigma2 = np.empty(T)
        sigma2[0] = np.var(eps)   # initialisation

        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

        return sigma2

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Gaussian log-likelihood (negated for minimisation).

        ℓ(θ) = -T/2 · ln(2π) - 1/2 · Σ_t [ln σ²_t + ε²_t / σ²_t]
        """
        mu, omega, alpha, beta = params

        # Parameter constraints (handled via penalty rather than boundary failure)
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
            return 1e10

        sigma2 = self._filter_variances(mu, omega, alpha, beta)

        if np.any(sigma2 <= 0):
            return 1e10

        eps = self.returns - mu
        ll = -0.5 * np.sum(np.log(sigma2) + eps ** 2 / sigma2)
        return -ll   # negated for minimisation

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fit(self, n_starts: int = 5) -> GARCHParams:
        """
        Fit GARCH(1,1) by MLE using L-BFGS-B with multiple starting points.

        Multiple initialisations guard against local optima — common in
        GARCH estimation with near-unit-root persistence.

        Parameters
        ----------
        n_starts : int
            Number of random starting points (in addition to a sensible default).

        Returns
        -------
        GARCHParams
            Estimated parameters at the MLE.
        """
        sample_var = np.var(self.returns)
        mu_init = np.mean(self.returns)

        # Starting configurations
        start_configs = [
            [mu_init, sample_var * 0.05, 0.10, 0.85],   # typical persistence
            [mu_init, sample_var * 0.10, 0.15, 0.80],
            [mu_init, sample_var * 0.20, 0.20, 0.70],
            [0.0,     sample_var * 0.05, 0.08, 0.90],   # high persistence
            [0.0,     sample_var * 0.10, 0.25, 0.60],   # lower persistence
        ]

        # Add random starts
        rng = np.random.default_rng(42)
        for _ in range(n_starts):
            alpha_r = rng.uniform(0.02, 0.25)
            beta_r = rng.uniform(0.60, 0.95 - alpha_r)
            omega_r = sample_var * rng.uniform(0.01, 0.15)
            start_configs.append([mu_init, omega_r, alpha_r, beta_r])

        bounds = [
            (-0.1, 0.1),        # mu
            (1e-8, 0.01),       # omega
            (1e-6, 0.5),        # alpha
            (1e-6, 0.9999),     # beta
        ]

        best_result: OptimizeResult | None = None
        best_ll = float("inf")

        for x0 in start_configs:
            try:
                result = minimize(
                    self._log_likelihood,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 5000, "ftol": 1e-12},
                )
                if result.success and result.fun < best_ll:
                    best_ll = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None:
            raise RuntimeError("GARCH optimisation failed across all starting points.")

        mu, omega, alpha, beta = best_result.x
        self._params = GARCHParams(
            mu=mu, omega=omega, alpha=alpha, beta=beta,
            log_likelihood=-best_ll
        )
        self._conditional_variances = self._filter_variances(mu, omega, alpha, beta)
        return self._params

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, horizon: int = 10) -> GARCHForecast:
        """
        Multi-step-ahead variance forecasts.

        For GARCH(1,1), the h-step-ahead forecast has the closed form:
            E[σ²_{T+h} | F_T] = σ²_∞ + (α+β)^{h-1} · (σ²_{T+1} - σ²_∞)

        where σ²_∞ = ω / (1-α-β) is the long-run variance.
        Forecasts mean-revert to the long-run variance at rate (α+β) per period.
        """
        if self._params is None:
            raise RuntimeError("Call fit() before forecast().")

        p = self._params
        lr_var = p.long_run_variance

        eps_T = self.returns[-1] - p.mu
        sigma2_T = self._conditional_variances[-1]
        sigma2_T1 = p.omega + p.alpha * eps_T ** 2 + p.beta * sigma2_T

        forecasts = np.empty(horizon)
        for h in range(1, horizon + 1):
            if h == 1:
                forecasts[0] = sigma2_T1
            else:
                forecasts[h - 1] = lr_var + (p.alpha + p.beta) ** (h - 1) * (sigma2_T1 - lr_var)

        return GARCHForecast(
            horizon=horizon,
            conditional_variances=forecasts,
            sigma2_0=sigma2_T1,
        )

    # ------------------------------------------------------------------
    # Volatility series
    # ------------------------------------------------------------------

    def conditional_vol_series(self) -> pd.Series:
        """Return the in-sample conditional volatility series (annualised)."""
        if self._conditional_variances is None:
            raise RuntimeError("Call fit() first.")
        vols = np.sqrt(self._conditional_variances * 252)
        if self.dates is not None:
            return pd.Series(vols, index=self.dates, name="GARCH_Vol_Annual")
        return pd.Series(vols, name="GARCH_Vol_Annual")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def standardised_residuals(self) -> np.ndarray:
        """
        Standardised residuals: z_t = ε_t / σ_t
        Should be approximately N(0,1) if model is correctly specified.
        Check: kurtosis ≈ 3, Ljung-Box on z²_t not significant.
        """
        if self._conditional_variances is None:
            raise RuntimeError("Call fit() first.")
        eps = self.returns - self._params.mu
        return eps / np.sqrt(self._conditional_variances)

    def ljung_box_test(self, lags: int = 10) -> dict:
        """
        Ljung-Box test on squared standardised residuals.
        A large p-value (> 0.05) indicates no remaining ARCH effects.
        """
        from scipy.stats import chi2
        z2 = self.standardised_residuals() ** 2
        T = len(z2)
        autocorrs = [np.corrcoef(z2[:-k], z2[k:])[0, 1] for k in range(1, lags + 1)]
        lb_stat = T * (T + 2) * sum(rk ** 2 / (T - k) for k, rk in enumerate(autocorrs, 1))
        p_value = 1 - chi2.cdf(lb_stat, df=lags)
        return {
            "LB statistic": round(lb_stat, 4),
            "degrees of freedom": lags,
            "p-value": round(p_value, 6),
            "no remaining ARCH effects": p_value > 0.05,
        }

    def print_diagnostics(self) -> None:
        """Print model diagnostics."""
        z = self.standardised_residuals()
        print("\n── Model Diagnostics ──")
        print(f"  Standardised residuals — mean   : {z.mean():.4f}  (expect ≈ 0)")
        print(f"  Standardised residuals — std    : {z.std():.4f}   (expect ≈ 1)")

        from scipy.stats import kurtosis, skew, shapiro
        print(f"  Excess kurtosis                 : {kurtosis(z):.4f}  (0 = normal)")
        print(f"  Skewness                        : {skew(z):.4f}   (0 = normal)")
        lb = self.ljung_box_test()
        print(f"  Ljung-Box Q(10) stat            : {lb['LB statistic']:.4f}")
        print(f"  Ljung-Box p-value               : {lb['p-value']:.4f}")
        print(f"  No remaining ARCH effects       : {lb['no remaining ARCH effects']}")


# ---------------------------------------------------------------------------
# Data loader — synthetic data with GARCH structure
# ---------------------------------------------------------------------------

def simulate_garch_returns(
    T: int = 2000, mu: float = 0.0003, omega: float = 2e-6,
    alpha: float = 0.08, beta: float = 0.88, seed: int = 42
) -> np.ndarray:
    """
    Simulate returns from a GARCH(1,1) process — useful for testing.
    Real usage: load from Yahoo Finance via yfinance.
    """
    rng = np.random.default_rng(seed)
    eps = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = math.sqrt(sigma2[t]) * rng.standard_normal()

    return mu + eps


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 60)
    print("  GARCH(1,1) Volatility Model")
    print("  Simulated equity returns (GARCH structure)")
    print("═" * 60)

    # Simulate returns with known parameters for validation
    true_omega, true_alpha, true_beta = 2e-6, 0.08, 0.88
    returns = simulate_garch_returns(T=2500, omega=true_omega, alpha=true_alpha, beta=true_beta)

    print(f"\n  Sample size: {len(returns):,} observations")
    print(f"  Return stats: μ={returns.mean():.5f}, σ={returns.std():.5f}")
    print(f"\n  True params: ω={true_omega:.2e}, α={true_alpha}, β={true_beta}")
    print(f"  True α+β = {true_alpha + true_beta:.3f}")
    print(f"  True long-run vol (annual): {math.sqrt(true_omega/(1-true_alpha-true_beta)*252):.4%}")

    model = GARCH11(returns)

    print("\n  Fitting GARCH(1,1) via MLE...")
    params = model.fit()

    print("\n" + params.summary())

    print("\n── Forecast: 1–10 Day Ahead Volatility ──")
    forecast = model.forecast(horizon=10)
    print(forecast.summary_table().to_string(index=False))

    model.print_diagnostics()

    print("\n── Parameter Recovery (true vs estimated) ──")
    print(f"  {'Param':<8} {'True':>10} {'Estimated':>12} {'Error':>10}")
    print(f"  {'─'*42}")
    for name, true_val, est_val in [
        ("omega", true_omega, params.omega),
        ("alpha", true_alpha, params.alpha),
        ("beta",  true_beta,  params.beta),
    ]:
        err = abs(est_val - true_val) / true_val
        print(f"  {name:<8} {true_val:>10.6f} {est_val:>12.6f} {err:>10.4%}")

    print("\n  Note: To use with real data, replace simulate_garch_returns() with:")
    print("  >>> import yfinance as yf")
    print("  >>> prices = yf.download('SPY', start='2015-01-01')['Adj Close']")
    print("  >>> returns = np.log(prices / prices.shift(1)).dropna().values")
    print("  >>> model = GARCH11(returns)")
