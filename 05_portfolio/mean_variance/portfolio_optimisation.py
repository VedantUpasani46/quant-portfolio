"""
Portfolio Optimisation: Markowitz Mean-Variance & Black-Litterman
==================================================================
Implements the two fundamental portfolio construction frameworks:

1. Markowitz (1952) Mean-Variance Optimisation
   - Efficient frontier computation
   - Minimum variance portfolio
   - Maximum Sharpe ratio portfolio (tangency portfolio)
   - Constrained optimisation (long-only, weight bounds, sector limits)

2. Black-Litterman (1990) Model
   - Combines market equilibrium returns (reverse-optimised from market weights)
     with investor views via Bayesian updating
   - Solves the problem of extreme corner solutions from classical MVO
   - Standard at sovereign wealth funds, pension funds, and large asset managers

3. Risk Decomposition
   - Marginal risk contributions
   - Risk parity portfolio (equal risk contribution)

References:
  - Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77–91.
  - Black, F. & Litterman, R. (1992). Global Portfolio Optimization. FAJ.
  - He, G. & Litterman, R. (1999). The Intuition Behind Black-Litterman. Goldman Sachs.
  - Meucci, A. (2010). The Black-Litterman Approach. Risk Magazine.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PortfolioStats:
    """Statistics for a single portfolio on the efficient frontier."""
    weights: np.ndarray
    expected_return: float      # annualised
    volatility: float           # annualised
    sharpe_ratio: float
    labels: list[str] = None

    def summary(self) -> str:
        lines = [
            f"  Expected Return  : {self.expected_return:.4%}",
            f"  Volatility       : {self.volatility:.4%}",
            f"  Sharpe Ratio     : {self.sharpe_ratio:.4f}",
            "  Weights:",
        ]
        labels = self.labels or [f"Asset {i+1}" for i in range(len(self.weights))]
        for label, w in zip(labels, self.weights):
            lines.append(f"    {label:<16}: {w:.4%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markowitz Optimiser
# ---------------------------------------------------------------------------

class MarkowitzOptimiser:
    """
    Mean-Variance portfolio optimiser.

    Parameters
    ----------
    expected_returns : np.ndarray   Annualised expected returns, shape (n,).
    cov_matrix : np.ndarray         Annualised covariance matrix, shape (n, n).
    asset_names : list[str]         Asset labels (optional).
    risk_free_rate : float          Annualised risk-free rate for Sharpe calculation.

    Usage
    -----
    >>> mu = np.array([0.08, 0.12, 0.06, 0.10])
    >>> cov = np.diag([0.04, 0.09, 0.01, 0.06])
    >>> opt = MarkowitzOptimiser(mu, cov, ['Bonds', 'Equities', 'Cash', 'RE'])
    >>> mvp = opt.minimum_variance()
    >>> tangency = opt.maximum_sharpe()
    >>> frontier = opt.efficient_frontier(n_points=50)
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: Optional[list[str]] = None,
        risk_free_rate: float = 0.04,
    ):
        self.mu = np.asarray(expected_returns, dtype=float)
        self.cov = np.asarray(cov_matrix, dtype=float)
        self.n = len(self.mu)
        self.rf = risk_free_rate
        self.labels = asset_names or [f"Asset {i+1}" for i in range(self.n)]

        if self.cov.shape != (self.n, self.n):
            raise ValueError(f"Covariance matrix must be ({self.n},{self.n}), got {self.cov.shape}")
        if not np.allclose(self.cov, self.cov.T):
            raise ValueError("Covariance matrix must be symmetric.")

    # ------------------------------------------------------------------
    # Portfolio statistics
    # ------------------------------------------------------------------

    def portfolio_return(self, w: np.ndarray) -> float:
        return float(self.mu @ w)

    def portfolio_variance(self, w: np.ndarray) -> float:
        return float(w @ self.cov @ w)

    def portfolio_vol(self, w: np.ndarray) -> float:
        return math.sqrt(self.portfolio_variance(w))

    def sharpe(self, w: np.ndarray) -> float:
        ret = self.portfolio_return(w)
        vol = self.portfolio_vol(w)
        return (ret - self.rf) / vol if vol > 1e-10 else 0.0

    def _make_stats(self, w: np.ndarray) -> PortfolioStats:
        return PortfolioStats(
            weights=w,
            expected_return=self.portfolio_return(w),
            volatility=self.portfolio_vol(w),
            sharpe_ratio=self.sharpe(w),
            labels=self.labels,
        )

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _base_constraints(self) -> list[dict]:
        """Weights sum to 1."""
        return [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    def _long_only_bounds(self) -> list[tuple]:
        """Each weight in [0, 1]."""
        return [(0.0, 1.0)] * self.n

    def minimum_variance(self, long_only: bool = True, weight_bounds: Optional[list] = None) -> PortfolioStats:
        """
        Minimum Variance Portfolio (MVP): lowest risk regardless of return.

        Closed-form (unconstrained): w* = Σ^{-1} 1 / (1' Σ^{-1} 1)
        With constraints: numerical optimisation.
        """
        bounds = weight_bounds or (self._long_only_bounds() if long_only else [(None, None)] * self.n)
        w0 = np.ones(self.n) / self.n

        result = minimize(
            lambda w: self.portfolio_variance(w),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=self._base_constraints(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            warnings.warn(f"MVP optimisation: {result.message}")
        return self._make_stats(result.x)

    def maximum_sharpe(self, long_only: bool = True, weight_bounds: Optional[list] = None) -> PortfolioStats:
        """
        Maximum Sharpe (Tangency) Portfolio: optimal risky portfolio on the CML.

        The tangency portfolio is the unique portfolio where the Capital Market
        Line is tangent to the efficient frontier.

        With a risk-free asset available, all investors should hold a combination
        of the risk-free asset and the tangency portfolio (Tobin's Two-Fund Theorem).
        """
        bounds = weight_bounds or (self._long_only_bounds() if long_only else [(None, None)] * self.n)
        w0 = np.ones(self.n) / self.n

        result = minimize(
            lambda w: -self.sharpe(w),   # maximise Sharpe = minimise -Sharpe
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=self._base_constraints(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            warnings.warn(f"Max Sharpe optimisation: {result.message}")
        return self._make_stats(result.x)

    def target_return(self, target: float, long_only: bool = True) -> PortfolioStats:
        """
        Minimum variance portfolio achieving a specified target return.
        Traces out the efficient frontier when called across a range of targets.
        """
        bounds = self._long_only_bounds() if long_only else [(None, None)] * self.n
        constraints = self._base_constraints() + [
            {"type": "eq", "fun": lambda w: self.portfolio_return(w) - target}
        ]
        w0 = np.ones(self.n) / self.n

        result = minimize(
            lambda w: self.portfolio_variance(w),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise ValueError(f"No feasible portfolio for target return {target:.4%}: {result.message}")
        return self._make_stats(result.x)

    def efficient_frontier(self, n_points: int = 50, long_only: bool = True) -> pd.DataFrame:
        """
        Trace the efficient frontier between the MVP and the maximum return portfolio.
        Returns a DataFrame of (volatility, return, sharpe) points.
        """
        mvp = self.minimum_variance(long_only=long_only)
        max_ret = max(self.mu)

        target_returns = np.linspace(mvp.expected_return, max_ret * 0.99, n_points)
        frontier = []

        for target in target_returns:
            try:
                port = self.target_return(target, long_only=long_only)
                frontier.append({
                    "Expected Return": port.expected_return,
                    "Volatility": port.volatility,
                    "Sharpe Ratio": port.sharpe_ratio,
                })
            except ValueError:
                pass

        return pd.DataFrame(frontier)

    # ------------------------------------------------------------------
    # Risk decomposition
    # ------------------------------------------------------------------

    def marginal_risk_contribution(self, w: np.ndarray) -> np.ndarray:
        """
        Marginal Risk Contribution (MRC): ∂σ_p/∂w_i = (Σw)_i / σ_p

        MRC_i shows how much portfolio volatility increases per unit increase in w_i.
        """
        sigma_p = self.portfolio_vol(w)
        return (self.cov @ w) / sigma_p

    def risk_contribution(self, w: np.ndarray) -> np.ndarray:
        """
        Absolute Risk Contribution (ARC): w_i · MRC_i

        ARC_i sums to portfolio volatility: Σ ARC_i = σ_p
        """
        return w * self.marginal_risk_contribution(w)

    def risk_parity(self) -> PortfolioStats:
        """
        Risk Parity (Equal Risk Contribution) portfolio.

        Each asset contributes equally to portfolio volatility:
        w_i · MRC_i = σ_p / n  for all i

        Used by Bridgewater (All Weather), AQR, and many pension funds.
        Tends to concentrate in low-volatility assets (bonds) relative to MVO.
        """
        n = self.n
        w0 = np.ones(n) / n
        target_contribution = 1.0 / n  # each asset = 1/n of total risk

        def risk_parity_objective(w: np.ndarray) -> float:
            rc = self.risk_contribution(w)
            total_risk = rc.sum()
            # Minimise sum of squared deviations from equal contributions
            return np.sum((rc / total_risk - target_contribution) ** 2)

        bounds = [(0.001, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        result = minimize(
            risk_parity_objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 2000},
        )
        return self._make_stats(result.x)

    def risk_decomposition_table(self, w: np.ndarray) -> pd.DataFrame:
        """Full risk decomposition for a given portfolio."""
        mrc = self.marginal_risk_contribution(w)
        arc = self.risk_contribution(w)
        sigma_p = self.portfolio_vol(w)
        return pd.DataFrame({
            "Asset": self.labels,
            "Weight": [f"{wi:.4%}" for wi in w],
            "MRC": [f"{m:.6f}" for m in mrc],
            "ARC": [f"{a:.6f}" for a in arc],
            "% of Risk": [f"{a/sigma_p:.4%}" for a in arc],
        })


# ---------------------------------------------------------------------------
# Black-Litterman Model
# ---------------------------------------------------------------------------

class BlackLitterman:
    """
    Black-Litterman model for combining equilibrium returns with investor views.

    The BL approach solves two classical MVO problems:
    1. Sensitivity: tiny changes in expected returns → large weight swings
    2. Non-intuitive: MVO often generates extreme or concentrated portfolios

    BL starts from equilibrium (CAPM) returns and Bayesian-updates them
    with investor views, producing stable, diversified portfolios.

    Parameters
    ----------
    market_weights : np.ndarray   Market capitalisation weights (sum to 1).
    cov_matrix : np.ndarray       Annualised covariance matrix.
    asset_names : list[str]       Asset labels.
    risk_aversion : float         Market risk aversion parameter λ (default 2.5).
    tau : float                   Uncertainty in prior (default 0.05).
    risk_free_rate : float        Risk-free rate for equilibrium calculation.
    """

    def __init__(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: Optional[list[str]] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.04,
    ):
        self.w_mkt = np.asarray(market_weights, dtype=float)
        self.cov = np.asarray(cov_matrix, dtype=float)
        self.n = len(self.w_mkt)
        self.labels = asset_names or [f"Asset {i+1}" for i in range(self.n)]
        self.lam = risk_aversion
        self.tau = tau
        self.rf = risk_free_rate

        # Equilibrium (implied) returns: Π = λ · Σ · w_mkt
        # This is the CAPM-implied expected excess return for each asset
        self.pi = risk_aversion * cov_matrix @ self.w_mkt

    def posterior_returns(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Black-Litterman posterior expected returns and covariance.

        Bayesian update:
          Prior: μ ~ N(Π, τΣ)
          Views: P·μ = Q + ε,   ε ~ N(0, Ω)
          Posterior: μ_BL ~ N(μ*, M)

          M = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
          μ* = M · [(τΣ)^{-1}Π + P'Ω^{-1}Q]

        Parameters
        ----------
        P : np.ndarray, shape (k, n)   View matrix: each row is a view.
        Q : np.ndarray, shape (k,)     View returns: P·μ = Q.
        Omega : np.ndarray, optional   View uncertainty, shape (k, k).
                                       Default: proportional to τ·P·Σ·P'.

        Returns
        -------
        mu_bl : np.ndarray   Posterior expected returns.
        cov_bl : np.ndarray  Posterior covariance matrix.
        """
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)
        k = P.shape[0]

        if Omega is None:
            # He & Litterman (1999): Ω = diag(τ · P·Σ·P')
            Omega = np.diag(np.diag(self.tau * P @ self.cov @ P.T))

        tau_cov = self.tau * self.cov
        tau_cov_inv = np.linalg.inv(tau_cov)
        omega_inv = np.linalg.inv(Omega)

        # Posterior precision matrix
        M_inv = tau_cov_inv + P.T @ omega_inv @ P
        M = np.linalg.inv(M_inv)

        # Posterior mean
        mu_bl = M @ (tau_cov_inv @ self.pi + P.T @ omega_inv @ Q)

        # Full posterior covariance (includes parameter uncertainty)
        cov_bl = self.cov + M

        return mu_bl, cov_bl

    def bl_optimal_portfolio(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None,
    ) -> PortfolioStats:
        """
        Compute the BL posterior returns and optimise for maximum Sharpe.

        Returns the portfolio that best incorporates both market equilibrium
        and investor views.
        """
        mu_bl, cov_bl = self.posterior_returns(P, Q, Omega)
        opt = MarkowitzOptimiser(
            mu_bl + self.rf, cov_bl,   # add rf to convert excess to total returns
            asset_names=self.labels,
            risk_free_rate=self.rf,
        )
        return opt.maximum_sharpe()

    def print_equilibrium(self) -> None:
        print("  Black-Litterman Equilibrium Returns (Π = λΣw_mkt):")
        for label, pi in zip(self.labels, self.pi):
            print(f"    {label:<16}: {pi + self.rf:.4%}  (excess: {pi:.4%})")


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Asset universe: Global 60/40-style allocation
    assets = ["US Equities", "Intl Equities", "EM Equities", "US Bonds", "Intl Bonds", "Commodities"]
    n = len(assets)

    # Approximate annualised parameters
    mu = np.array([0.09, 0.10, 0.12, 0.04, 0.035, 0.06])
    vols = np.array([0.16, 0.18, 0.24, 0.05, 0.07, 0.20])

    # Correlation matrix
    corr = np.array([
        [1.00, 0.85, 0.75, -0.20, -0.15,  0.15],
        [0.85, 1.00, 0.80, -0.18, -0.12,  0.18],
        [0.75, 0.80, 1.00, -0.10, -0.08,  0.22],
        [-0.20,-0.18,-0.10, 1.00,  0.65, -0.05],
        [-0.15,-0.12,-0.08, 0.65,  1.00, -0.02],
        [0.15, 0.18, 0.22, -0.05, -0.02,  1.00],
    ])
    cov = np.outer(vols, vols) * corr

    print("═" * 65)
    print("  Portfolio Optimisation Suite")
    print("  6-Asset Global Multi-Asset Portfolio")
    print("═" * 65)

    opt = MarkowitzOptimiser(mu, cov, asset_names=assets, risk_free_rate=0.04)

    print("\n── Minimum Variance Portfolio ──")
    mvp = opt.minimum_variance()
    print(mvp.summary())

    print("\n── Maximum Sharpe (Tangency) Portfolio ──")
    tangency = opt.maximum_sharpe()
    print(tangency.summary())

    print("\n── Risk Parity Portfolio ──")
    rp = opt.risk_parity()
    print(rp.summary())

    print("\n── Risk Decomposition (Tangency Portfolio) ──")
    print(opt.risk_decomposition_table(tangency.weights).to_string(index=False))

    print("\n── Efficient Frontier (10 points) ──")
    frontier = opt.efficient_frontier(n_points=10)
    print(frontier.round(6).to_string(index=False))

    # --- Black-Litterman ---
    print("\n" + "═" * 65)
    print("  Black-Litterman Model")
    print("═" * 65)

    # Market cap weights (approximate global allocation)
    w_mkt = np.array([0.35, 0.20, 0.10, 0.25, 0.08, 0.02])
    bl = BlackLitterman(w_mkt, cov, asset_names=assets, risk_aversion=2.5)

    bl.print_equilibrium()

    # Investor views:
    # View 1: EM Equities will outperform Intl Equities by 2%
    # View 2: US Bonds will return 5%
    P = np.array([
        [0, -1, 1, 0, 0, 0],    # EM - Intl
        [0,  0, 0, 1, 0, 0],    # US Bonds absolute
    ])
    Q = np.array([0.02, 0.05])

    print("\n  Views:")
    print("    View 1: EM Equities outperform Intl Equities by 2%")
    print("    View 2: US Bonds return 5%")

    mu_bl, cov_bl = bl.posterior_returns(P, Q)
    print("\n  Posterior Expected Returns (BL vs Prior):")
    print(f"  {'Asset':<18} {'Prior':>10} {'BL Posterior':>14} {'Change':>10}")
    print(f"  {'─'*52}")
    for i, asset in enumerate(assets):
        prior = mu[i]
        posterior = mu_bl[i] + 0.04   # add rf
        print(f"  {asset:<18} {prior:>10.4%} {posterior:>14.4%} {posterior-prior:>+10.4%}")

    print("\n── BL Optimal Portfolio ──")
    bl_port = bl.bl_optimal_portfolio(P, Q)
    print(bl_port.summary())
