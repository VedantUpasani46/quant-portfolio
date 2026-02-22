"""
Value at Risk (VaR) Calculator
================================
Computes portfolio Value at Risk and Expected Shortfall (CVaR) using:
  1. Historical Simulation  — non-parametric; uses empirical return distribution.
  2. Parametric (Variance-Covariance) — assumes returns are normally distributed.
  3. Monte Carlo VaR — simulates future portfolio values via correlated GBM.

What is VaR?
  VaR(α, T) answers: "What is the minimum loss we expect to exceed only (1-α)%
  of the time over horizon T?"
  E.g., 1-day 99% VaR of $1M means there is a 1% chance of losing more than
  $1M in a single day.

Regulatory context:
  - Basel III requires banks to compute 10-day 99% VaR (FRTB uses ES instead).
  - Expected Shortfall (ES / CVaR) is the average loss beyond the VaR threshold,
    now preferred by regulators as it is sub-additive (coherent risk measure).

References:
  - Hull, J.C. (2022). Risk Management and Financial Institutions, Ch. 9–12.
  - Jorion, P. (2006). Value at Risk, 3rd ed. McGraw-Hill.
  - Basel Committee on Banking Supervision (2019). FRTB Standards.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    """
    Output of a VaR calculation.

    Attributes
    ----------
    var : float
        Value at Risk as a positive dollar loss figure.
    es : float
        Expected Shortfall (CVaR): average loss given VaR is breached.
    confidence_level : float
        e.g. 0.99 for 99%.
    horizon_days : int
        Holding period in calendar days.
    method : str
        Calculation method used.
    portfolio_value : float
        Current portfolio value.
    """
    var: float
    es: float
    confidence_level: float
    horizon_days: int
    method: str
    portfolio_value: float

    @property
    def var_pct(self) -> float:
        return self.var / self.portfolio_value

    @property
    def es_pct(self) -> float:
        return self.es / self.portfolio_value

    def summary(self) -> str:
        return (
            f"  Method          : {self.method}\n"
            f"  Confidence      : {self.confidence_level:.0%}\n"
            f"  Horizon         : {self.horizon_days} day(s)\n"
            f"  Portfolio Value : ${self.portfolio_value:,.2f}\n"
            f"  VaR             : ${self.var:,.2f}  ({self.var_pct:.3%})\n"
            f"  Expected Shortfall : ${self.es:,.2f}  ({self.es_pct:.3%})"
        )


# ---------------------------------------------------------------------------
# Return preprocessing utilities
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from a DataFrame of prices.

    Log returns: r_t = ln(P_t / P_{t-1})
    Preferred over simple returns because they are time-additive and
    more likely to be normally distributed.
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_portfolio_returns(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """
    Portfolio return series given asset returns and weights.
    weights should sum to 1.
    """
    if not math.isclose(weights.sum(), 1.0, abs_tol=1e-6):
        warnings.warn(f"Weights sum to {weights.sum():.6f}, expected 1.0.")
    return (returns * weights).sum(axis=1)


# ---------------------------------------------------------------------------
# VaR Calculator
# ---------------------------------------------------------------------------

class VaRCalculator:
    """
    Computes Value at Risk and Expected Shortfall for a portfolio.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price series, one column per asset, indexed by date.
    weights : array-like
        Portfolio weights (must sum to 1); same order as columns in prices.
    portfolio_value : float
        Total current portfolio value in dollars.
    confidence : float
        VaR confidence level (default 0.99 for 99%).
    horizon : int
        VaR holding period in days (default 1).

    Usage
    -----
    >>> import pandas as pd, numpy as np
    >>> prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    >>> calc = VaRCalculator(prices, weights=np.array([0.5, 0.5]), portfolio_value=1_000_000)
    >>> result = calc.historical_simulation()
    >>> print(result.summary())
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float = 1_000_000,
        confidence: float = 0.99,
        horizon: int = 1,
    ):
        self.prices = prices
        self.weights = np.asarray(weights, dtype=float)
        self.portfolio_value = portfolio_value
        self.confidence = confidence
        self.horizon = horizon

        if len(self.weights) != prices.shape[1]:
            raise ValueError(
                f"weights length {len(self.weights)} != number of assets {prices.shape[1]}"
            )

        self._returns = compute_log_returns(prices)
        self._port_returns = compute_portfolio_returns(self._returns, self.weights)

    # ------------------------------------------------------------------
    # Method 1: Historical Simulation
    # ------------------------------------------------------------------

    def historical_simulation(self) -> VaRResult:
        """
        Non-parametric VaR from the empirical return distribution.

        Steps:
          1. Compute historical portfolio returns.
          2. Scale to the desired horizon: r_h = r_1 * √h (independent returns assumption).
          3. Sort returns; the VaR is the (1-α) quantile.
          4. ES is the mean of returns worse than VaR.

        Advantages:
          - Captures fat tails and skewness from actual data.
          - No distributional assumption.
        Disadvantages:
          - Limited by the historical window; misses tail events not in the sample.
        """
        r = self._port_returns.values
        scaled = r * math.sqrt(self.horizon)   # scale to h-day horizon

        loss_quantile = np.percentile(scaled, (1 - self.confidence) * 100)
        var_dollar = -loss_quantile * self.portfolio_value

        tail_losses = scaled[scaled <= loss_quantile]
        es_dollar = -tail_losses.mean() * self.portfolio_value if len(tail_losses) > 0 else var_dollar

        return VaRResult(
            var=var_dollar,
            es=es_dollar,
            confidence_level=self.confidence,
            horizon_days=self.horizon,
            method="Historical Simulation",
            portfolio_value=self.portfolio_value,
        )

    # ------------------------------------------------------------------
    # Method 2: Parametric (Variance-Covariance / Delta-Normal)
    # ------------------------------------------------------------------

    def parametric(self) -> VaRResult:
        """
        Parametric VaR assuming normally distributed portfolio returns.

        VaR = μ_p·h + z_α · σ_p · √h

        where:
          μ_p = portfolio mean daily return
          σ_p = portfolio daily volatility = √(w'Σw)
          z_α = standard normal quantile at (1-α)
          h   = horizon in days

        Advantages: fast, analytically transparent.
        Disadvantages: normality assumption underestimates tail risk (fat tails).

        Component VaR (per asset contribution) is also computed.
        """
        w = self.weights
        R = self._returns.values                  # (T, n_assets)
        mu = R.mean(axis=0)                       # (n_assets,)
        cov = np.cov(R.T)                         # (n_assets, n_assets) — covariance matrix

        # Portfolio statistics
        mu_p = w @ mu                             # scalar
        sigma_p = math.sqrt(w @ cov @ w)          # portfolio daily vol

        # Scale to horizon
        mu_h = mu_p * self.horizon
        sigma_h = sigma_p * math.sqrt(self.horizon)

        # VaR (loss convention: positive = loss)
        z = norm.ppf(1 - self.confidence)         # e.g. -2.326 for 99%
        var_return = -(mu_h + z * sigma_h)        # flip sign for loss
        var_dollar = var_return * self.portfolio_value

        # ES for normal: ES = μ - σ · φ(z) / (1-α)
        phi_z = norm.pdf(norm.ppf(1 - self.confidence))
        es_return = -(mu_h - sigma_h * phi_z / (1 - self.confidence))
        es_dollar = es_return * self.portfolio_value

        return VaRResult(
            var=var_dollar,
            es=es_dollar,
            confidence_level=self.confidence,
            horizon_days=self.horizon,
            method="Parametric (Delta-Normal)",
            portfolio_value=self.portfolio_value,
        )

    # ------------------------------------------------------------------
    # Method 3: Monte Carlo VaR
    # ------------------------------------------------------------------

    def monte_carlo(self, n_sims: int = 100_000, seed: int = 42) -> VaRResult:
        """
        Monte Carlo VaR via correlated GBM simulation.

        Steps:
          1. Estimate mean vector μ and covariance matrix Σ from history.
          2. Cholesky-decompose Σ to obtain correlated normals.
          3. Simulate n_sims portfolio P&L scenarios over the horizon.
          4. VaR = quantile of the simulated P&L distribution.

        Advantages:
          - Handles non-linear instruments (with delta-gamma approximation).
          - Easily incorporates stress scenarios.
        Disadvantages:
          - Computationally expensive; depends on model choice for Σ.
        """
        rng = np.random.default_rng(seed)
        n_assets = len(self.weights)
        R = self._returns.values

        mu = R.mean(axis=0)
        cov = np.cov(R.T)

        # Cholesky decomposition for correlated draws
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Covariance matrix not positive definite — add small regularisation
            cov += np.eye(n_assets) * 1e-8
            L = np.linalg.cholesky(cov)

        dt = self.horizon  # h-day horizon

        # Simulate: each draw is an (n_assets,) daily return over h days
        # For simplicity, we simulate compound daily returns over horizon
        portfolio_pnl = np.zeros(n_sims)
        for _ in range(self.horizon):
            Z = rng.standard_normal((n_assets, n_sims))   # (n_assets, n_sims)
            correlated = L @ Z                              # (n_assets, n_sims)
            daily_returns = mu[:, None] + correlated        # (n_assets, n_sims)
            port_daily = self.weights @ daily_returns       # (n_sims,)
            portfolio_pnl += port_daily                     # accumulate over horizon

        dollar_pnl = portfolio_pnl * self.portfolio_value
        loss_quantile = np.percentile(dollar_pnl, (1 - self.confidence) * 100)
        var_dollar = -loss_quantile

        tail_losses = dollar_pnl[dollar_pnl <= loss_quantile]
        es_dollar = -tail_losses.mean() if len(tail_losses) > 0 else var_dollar

        return VaRResult(
            var=var_dollar,
            es=es_dollar,
            confidence_level=self.confidence,
            horizon_days=self.horizon,
            method=f"Monte Carlo ({n_sims:,} sims)",
            portfolio_value=self.portfolio_value,
        )

    # ------------------------------------------------------------------
    # Backtesting: Basel Traffic Light Test
    # ------------------------------------------------------------------

    def backtest(self, n_days: int = 250) -> dict:
        """
        Basel Traffic Light Test — count VaR breaches over the last n_days.

        At 99% confidence we expect ~2.5 breaches per year.
        Basel zones:
          Green  : 0–4 exceptions  (model likely correct)
          Yellow : 5–9 exceptions  (investigate)
          Red    : 10+ exceptions  (model likely flawed)

        Returns
        -------
        dict with breach count, rate, and Basel zone.
        """
        returns = self._port_returns.values[-n_days:]
        var_result = self.parametric()
        daily_var_return = var_result.var / self.portfolio_value / math.sqrt(self.horizon)

        # A breach occurs when the actual daily loss exceeds VaR
        losses = -returns   # positive = loss
        breaches = int((losses > daily_var_return).sum())
        breach_rate = breaches / n_days

        if breaches <= 4:
            zone = "🟢 Green — model likely adequate"
        elif breaches <= 9:
            zone = "🟡 Yellow — investigate model assumptions"
        else:
            zone = "🔴 Red — model likely flawed"

        return {
            "n_days": n_days,
            "breaches": breaches,
            "expected_breaches": round((1 - self.confidence) * n_days, 1),
            "breach_rate": round(breach_rate, 4),
            "zone": zone,
        }

    # ------------------------------------------------------------------
    # Component VaR
    # ------------------------------------------------------------------

    def component_var(self) -> pd.DataFrame:
        """
        Decompose total portfolio VaR into per-asset contributions.

        Component VaR_i = w_i · (∂VaR/∂w_i) = w_i · ρ_{i,p} · σ_p · z_α · √h

        where ρ_{i,p} is the correlation of asset i with the portfolio.
        Component VaRs sum exactly to total VaR (Euler decomposition).
        """
        w = self.weights
        R = self._returns.values
        cov = np.cov(R.T)

        sigma_p = math.sqrt(w @ cov @ w)
        z = -norm.ppf(1 - self.confidence)  # positive multiplier
        scalar = z * math.sqrt(self.horizon)

        # Marginal VaR: d(VaR)/d(w_i)
        marginal_var = (cov @ w) / sigma_p * scalar

        component_var_dollar = w * marginal_var * self.portfolio_value
        pct_contribution = component_var_dollar / component_var_dollar.sum()

        return pd.DataFrame({
            "Asset": self.prices.columns,
            "Weight": w,
            "Component VaR ($)": component_var_dollar.round(2),
            "% of Total VaR": (pct_contribution * 100).round(2),
        })

    # ------------------------------------------------------------------
    # Convenience: compare all methods
    # ------------------------------------------------------------------

    def compare_methods(self) -> None:
        """Print comparison of all three VaR methods."""
        methods = [self.historical_simulation(), self.parametric(), self.monte_carlo()]
        print("\n" + "=" * 75)
        print(f"  {'Method':<32} {'VaR ($)':>12} {'VaR (%)':>10} {'ES ($)':>12} {'ES (%)':>10}")
        print("=" * 75)
        for r in methods:
            print(
                f"  {r.method:<32} {r.var:>12,.2f} {r.var_pct:>10.4%} "
                f"{r.es:>12,.2f} {r.es_pct:>10.4%}"
            )
        print("=" * 75)


# ---------------------------------------------------------------------------
# Demo with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    n_days = 504  # ~2 years of daily data

    # Synthetic price series for two correlated assets
    cov_true = np.array([[0.0004, 0.0002], [0.0002, 0.0009]])  # daily covariance
    L = np.linalg.cholesky(cov_true)
    mu_true = np.array([0.0003, 0.0005])

    Z = np.random.standard_normal((n_days, 2))
    returns_sim = mu_true + (L @ Z.T).T
    prices_sim = 100 * np.exp(np.cumsum(returns_sim, axis=0))

    dates = pd.bdate_range("2022-01-01", periods=n_days)
    prices_df = pd.DataFrame(prices_sim, index=dates, columns=["AAPL", "MSFT"])
    weights = np.array([0.6, 0.4])

    calc = VaRCalculator(
        prices=prices_df, weights=weights,
        portfolio_value=5_000_000, confidence=0.99, horizon=1
    )

    print("═" * 75)
    print("  VaR Calculator — $5M Portfolio (60% AAPL / 40% MSFT)")
    print("  99% Confidence, 1-Day Horizon")
    print("═" * 75)
    calc.compare_methods()

    print("\n── Component VaR Decomposition ──")
    print(calc.component_var().to_string(index=False))

    print("\n── Basel Traffic Light Backtest ──")
    bt = calc.backtest(n_days=250)
    for k, v in bt.items():
        print(f"  {k}: {v}")

    print("\n── 10-Day 99% VaR (Basel Square Root of Time Scaling) ──")
    calc_10d = VaRCalculator(
        prices=prices_df, weights=weights,
        portfolio_value=5_000_000, confidence=0.99, horizon=10
    )
    r10 = calc_10d.parametric()
    print(r10.summary())
