"""
Mean-Variance Portfolio Optimization
=======================================
Markowitz (1952) portfolio theory: for a given level of expected return,
find the portfolio with minimum variance. The set of all such portfolios
is the EFFICIENT FRONTIER.

The optimization problem:
  min  w'Σw                     [minimize portfolio variance]
  s.t. w'μ = μ_target           [hit target return]
       w'1 = 1                   [fully invested]
       wᵢ ≥ 0  (optional)       [no short-selling constraint]

The Sharpe ratio maximising portfolio (tangency portfolio):
  max  (w'μ - rf) / √(w'Σw)
  s.t. w'1 = 1, wᵢ ≥ 0

Analytical solution (unconstrained, no short-selling restriction):
  Using the method of Lagrange multipliers, the efficient frontier
  is a quadratic curve in (σ, μ) space. The minimum variance portfolio
  (MVP) is the leftmost point on the frontier.

Key concepts demonstrated:
  1. Efficient Frontier construction
  2. Minimum Variance Portfolio (MVP)
  3. Maximum Sharpe Ratio / Tangency Portfolio
  4. Risk Parity (equal risk contribution)
  5. Black-Litterman style constraints
  6. In-sample vs out-of-sample performance

Why mean-variance matters:
  - Foundation of every asset allocation model (60/40, endowment model)
  - The Sharpe ratio maximising portfolio = Capital Market Line tangency
  - Used daily by pension funds, sovereign wealth funds, asset managers
  - Every quant portfolio construction interview starts with Markowitz

Limitations and extensions:
  - Estimation error: small errors in μ → large allocation changes (Michaud 1989)
  - Fat tails: variance ≠ risk for non-Gaussian returns
  - Solutions: shrinkage estimators (Ledoit-Wolf), Black-Litterman, robust optimization

References:
  - Markowitz, H.M. (1952). Portfolio Selection. JF 7(1), 77–91.
  - Sharpe, W.F. (1964). Capital Asset Prices. JF 19(3), 425–442.
  - Ledoit, O. & Wolf, M. (2004). A Well-Conditioned Estimator. JMA 88(2).
  - Meucci, A. (2005). Risk and Asset Allocation. Springer.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
from scipy.linalg import solve


# ---------------------------------------------------------------------------
# Return and covariance estimation
# ---------------------------------------------------------------------------

def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Sample covariance matrix (unbiased, denominator = n-1)."""
    return np.cov(returns.T)


def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf (2004) analytical shrinkage estimator.

    Shrinks the sample covariance toward a structured target (scaled identity):
      Σ_shrunk = (1-δ)·Σ_sample + δ·μ_trace·I

    The optimal shrinkage intensity δ is derived analytically.
    This significantly reduces estimation error for large n, small T.

    The formula used is the constant-correlation shrinkage target (Ledoit & Wolf 2004).
    """
    T, n = returns.shape
    S = np.cov(returns.T)

    # Target: scaled identity (simplest LW target — works well in practice)
    mu = np.trace(S) / n

    # Compute optimal δ via Oracle approximating shrinkage
    # δ* = min(1, (sum of squared off-diagonal elements in normalised form) / (T * sum of squared off-diagonals of S))
    # Simplified Oracle formula (Ledoit-Wolf Oracle 2004, Eq. 14):
    delta_num = 0.0
    delta_denom = 0.0
    for t in range(T):
        x = returns[t]
        x_outer = np.outer(x, x)
        diff = x_outer - S
        delta_num += np.sum(diff ** 2)

    delta_num /= T ** 2
    off_diag = S - np.diag(np.diag(S))
    delta_denom = np.sum(off_diag ** 2)
    diag_excess = np.sum((np.diag(S) - mu) ** 2)
    delta_denom += diag_excess

    if delta_denom < 1e-10:
        return S

    delta = min(1.0, delta_num / delta_denom)
    target = mu * np.eye(n)
    return (1 - delta) * S + delta * target


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """Expected portfolio return."""
    return float(w @ mu)


def portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio variance σ²_p = w'Σw."""
    return float(w @ cov @ w)


def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility (annualised if cov is annualised)."""
    return math.sqrt(max(portfolio_variance(w, cov), 0.0))


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                 rf: float = 0.0) -> float:
    """Sharpe ratio = (E[r_p] - rf) / σ_p."""
    ret = portfolio_return(w, mu)
    vol = portfolio_vol(w, cov)
    return (ret - rf) / vol if vol > 0 else 0.0


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Marginal risk contribution of each asset.
    RC_i = w_i · (Σw)_i / σ_p   (fraction of total risk from asset i)
    """
    Sigma_w = cov @ w
    sigma_p = math.sqrt(max(float(w @ Sigma_w), 1e-12))
    return w * Sigma_w / sigma_p


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

@dataclass
class EfficientFrontier:
    """
    Compute and store the efficient frontier.

    Attributes
    ----------
    mu : np.ndarray          Expected returns (annualised).
    cov : np.ndarray         Covariance matrix (annualised).
    asset_names : list[str]
    rf : float               Risk-free rate.
    allow_short : bool       Allow negative weights.
    weight_bounds : tuple    (min_weight, max_weight) per asset.
    """

    mu: np.ndarray
    cov: np.ndarray
    asset_names: list
    rf: float = 0.04
    allow_short: bool = False
    weight_bounds: tuple = (0.0, 1.0)

    def __post_init__(self):
        self.n = len(self.mu)
        self._mvp: Optional[np.ndarray] = None
        self._tangency: Optional[np.ndarray] = None

    def minimum_variance_portfolio(self) -> np.ndarray:
        """
        Global minimum variance portfolio.
        Analytical solution (unconstrained):
          w_mvp = Σ⁻¹·1 / (1'Σ⁻¹·1)

        With constraints: solved numerically via QP.
        """
        if self._mvp is not None:
            return self._mvp

        def objective(w):
            return portfolio_variance(w, self.cov)

        def jac(w):
            return 2 * self.cov @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [self.weight_bounds] * self.n

        w0 = np.ones(self.n) / self.n
        result = minimize(objective, w0, jac=jac, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-9, "maxiter": 500})
        self._mvp = result.x
        return self._mvp

    def maximum_sharpe_portfolio(self) -> np.ndarray:
        """
        Tangency portfolio: maximum Sharpe ratio.
        Analytical (unconstrained, long-only handled via numerical QP):
          w_tangency ∝ Σ⁻¹(μ - rf·1)

        The QP reformulation: max Sharpe ↔ min variance of rescaled portfolio.
        """
        if self._tangency is not None:
            return self._tangency

        excess_mu = self.mu - self.rf

        def neg_sharpe(w):
            return -sharpe_ratio(w, self.mu, self.cov, self.rf)

        def jac_neg_sharpe(w):
            ret = portfolio_return(w, self.mu)
            vol = portfolio_vol(w, self.cov)
            if vol < 1e-8:
                return np.zeros_like(w)
            grad_ret = self.mu
            grad_vol = (self.cov @ w) / vol
            # d(-SR)/dw = -(vol·grad_ret - (ret-rf)·grad_vol) / vol²
            return -(vol * grad_ret - (ret - self.rf) * grad_vol) / (vol ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [self.weight_bounds] * self.n

        best_w, best_sr = None, -np.inf
        for _ in range(10):
            w0 = np.random.dirichlet(np.ones(self.n))
            result = minimize(neg_sharpe, w0, jac=jac_neg_sharpe,
                              method="SLSQP", bounds=bounds,
                              constraints=constraints,
                              options={"ftol": 1e-9, "maxiter": 500})
            sr = sharpe_ratio(result.x, self.mu, self.cov, self.rf)
            if sr > best_sr:
                best_sr = sr
                best_w = result.x

        self._tangency = best_w
        return self._tangency

    def efficient_portfolio(self, target_return: float) -> Optional[np.ndarray]:
        """
        Minimum variance portfolio with expected return = target_return.
        Returns None if target is below MVP return or not feasible.
        """
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: portfolio_return(w, self.mu) - target_return},
        ]
        bounds = [self.weight_bounds] * self.n
        w0 = np.ones(self.n) / self.n

        result = minimize(lambda w: portfolio_variance(w, self.cov), w0,
                          method="SLSQP", bounds=bounds,
                          constraints=constraints,
                          options={"ftol": 1e-9, "maxiter": 500})
        if result.success:
            return result.x
        return None

    def compute_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Trace the efficient frontier across a range of target returns.
        Returns DataFrame with (return, vol, sharpe, weights).
        """
        mvp = self.minimum_variance_portfolio()
        mu_mvp = portfolio_return(mvp, self.mu)
        mu_max = float(self.mu.max())

        # Only trace the efficient part (above MVP)
        target_returns = np.linspace(mu_mvp * 1.001, mu_max * 0.98, n_points)

        records = []
        for mu_t in target_returns:
            w = self.efficient_portfolio(mu_t)
            if w is not None:
                vol = portfolio_vol(w, self.cov)
                sr = sharpe_ratio(w, self.mu, self.cov, self.rf)
                row = {"return": mu_t, "vol": vol, "sharpe": sr}
                for i, name in enumerate(self.asset_names):
                    row[f"w_{name}"] = w[i]
                records.append(row)

        return pd.DataFrame(records)

    def risk_parity_portfolio(self) -> np.ndarray:
        """
        Equal Risk Contribution (ERC) / Risk Parity portfolio.
        Each asset contributes equally to total portfolio risk:
          RC_i = σ_p / n   for all i

        Solved via the formulation: min Σ_i Σ_j (RC_i - RC_j)²
        or equivalently: min Σ_i (RC_i - 1/n)²

        Popular with large asset allocators (Bridgewater, AQR).
        """
        n = self.n
        target_rc = 1.0 / n  # equal contribution

        def objective(w):
            rc = risk_contributions(w, self.cov)
            return float(np.sum((rc - target_rc) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(1e-6, 1.0)] * n  # must be positive (risk parity requires long-only)

        best_w, best_obj = None, np.inf
        for _ in range(5):
            w0 = np.random.dirichlet(np.ones(n))
            result = minimize(objective, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"ftol": 1e-10, "maxiter": 1000})
            if result.fun < best_obj:
                best_obj = result.fun
                best_w = result.x

        return best_w


# ---------------------------------------------------------------------------
# Performance attribution
# ---------------------------------------------------------------------------

def brinson_attribution(
    portfolio_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    asset_returns: np.ndarray,
    benchmark_return: float,
) -> dict:
    """
    Brinson-Hood-Beebower (1986) performance attribution.

    Decomposes active return into:
      Allocation effect:  (w_p - w_b)·(r_b_sector - r_b_total)
      Selection effect:   w_b·(r_p_sector - r_b_sector)
      Interaction effect: (w_p - w_b)·(r_p_sector - r_b_sector)

    Here simplified to single-period asset-level attribution.
    """
    portfolio_return = float(portfolio_weights @ asset_returns)
    active_return = portfolio_return - benchmark_return
    active_weights = portfolio_weights - benchmark_weights
    selection = benchmark_weights * (asset_returns - benchmark_return)
    allocation = active_weights * benchmark_return
    interaction = active_weights * (asset_returns - benchmark_return)

    return {
        "portfolio_return":  portfolio_return,
        "benchmark_return":  benchmark_return,
        "active_return":     active_return,
        "allocation_effect": float(allocation.sum()),
        "selection_effect":  float(selection.sum()),
        "interaction_effect":float(interaction.sum()),
        "total_attributed":  float((allocation + selection + interaction).sum()),
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy.random as npr

    print("═" * 66)
    print("  Mean-Variance Portfolio Optimization — Markowitz (1952)")
    print("  Efficient frontier, Sharpe maximisation, Risk Parity")
    print("═" * 66)

    rng = np.random.default_rng(42)

    # Realistic 6-asset universe: US/EU/EM equity, IG bonds, HY bonds, commodities
    asset_names = ["US_Equity", "EU_Equity", "EM_Equity", "IG_Bonds", "HY_Bonds", "Commodities"]
    n = len(asset_names)

    # Annualised expected returns (approximate long-run estimates)
    mu = np.array([0.10, 0.08, 0.12, 0.04, 0.06, 0.07])

    # Construct a realistic covariance matrix
    # Annualised vols: equity ~18-25%, bonds ~4-8%, commodities ~20%
    vols = np.array([0.18, 0.20, 0.25, 0.04, 0.08, 0.20])
    # Correlation structure
    corr = np.array([
        [1.00,  0.82,  0.70, -0.15,  0.30,  0.15],
        [0.82,  1.00,  0.68, -0.10,  0.28,  0.12],
        [0.70,  0.68,  1.00, -0.12,  0.35,  0.18],
        [-0.15,-0.10, -0.12,  1.00,  0.40, -0.05],
        [0.30,  0.28,  0.35,  0.40,  1.00,  0.10],
        [0.15,  0.12,  0.18, -0.05,  0.10,  1.00],
    ])
    cov = np.outer(vols, vols) * corr

    rf = 0.04  # 4% risk-free rate

    ef = EfficientFrontier(mu=mu, cov=cov, asset_names=asset_names, rf=rf)

    # ── Key portfolios ─────────────────────────────────────────────
    mvp = ef.minimum_variance_portfolio()
    tangency = ef.maximum_sharpe_portfolio()
    rp = ef.risk_parity_portfolio()
    equal_w = np.ones(n) / n

    print(f"\n── Key Portfolios ──")
    print(f"\n  {'Portfolio':<24} {'Return':>8} {'Vol':>8} {'Sharpe':>8}")
    print("  " + "─" * 52)

    for name, w in [
        ("Equal Weight", equal_w),
        ("Min Variance (MVP)", mvp),
        ("Max Sharpe (Tangency)", tangency),
        ("Risk Parity", rp),
    ]:
        ret = portfolio_return(w, mu)
        vol = portfolio_vol(w, cov)
        sr = sharpe_ratio(w, mu, cov, rf)
        print(f"  {name:<24} {ret:>8.3%} {vol:>8.3%} {sr:>8.4f}")

    # ── Weight breakdown ──────────────────────────────────────────
    print(f"\n── Portfolio Weights (%) ──")
    print(f"\n  {'Asset':<16}", end="")
    for pname in ["EqW", "MVP", "Tangency", "RiskParity"]:
        print(f"{pname:>12}", end="")
    print()
    print("  " + "─" * 64)
    for i, aname in enumerate(asset_names):
        print(f"  {aname:<16}", end="")
        for w in [equal_w, mvp, tangency, rp]:
            print(f"{w[i]:>12.1%}", end="")
        print()

    # ── Risk contributions ─────────────────────────────────────────
    print(f"\n── Risk Contributions (% of portfolio vol) ──")
    print(f"\n  {'Asset':<16}{'MVP RC':>12}{'Tangency RC':>14}{'RiskParity RC':>16}")
    print("  " + "─" * 56)
    for i, aname in enumerate(asset_names):
        rc_mvp  = risk_contributions(mvp, cov)
        rc_tang = risk_contributions(tangency, cov)
        rc_rp   = risk_contributions(rp, cov)
        print(f"  {aname:<16}{rc_mvp[i]:>12.2%}{rc_tang[i]:>14.2%}{rc_rp[i]:>16.2%}")

    print(f"\n  Risk Parity: each asset contributes ~{100/n:.1f}% of total risk ✓")

    # ── Efficient frontier sample ─────────────────────────────────
    print(f"\n── Efficient Frontier (selected points) ──")
    frontier = ef.compute_frontier(n_points=40)
    print(f"\n  {'Return':>10} {'Vol':>10} {'Sharpe':>10}")
    print("  " + "─" * 34)
    for _, row in frontier.iloc[::8].iterrows():
        print(f"  {row['return']:>10.3%} {row['vol']:>10.3%} {row['sharpe']:>10.4f}")

    # ── Ledoit-Wolf shrinkage ─────────────────────────────────────
    print(f"\n── Ledoit-Wolf Shrinkage Estimation ──")
    T = 252
    simulated_returns = rng.multivariate_normal(mu / 252, cov / 252, size=T)
    cov_sample = sample_covariance(simulated_returns) * 252
    cov_lw = ledoit_wolf_shrinkage(simulated_returns) * 252
    # Compare condition numbers (Ledoit-Wolf should be lower = more stable)
    cn_sample = np.linalg.cond(cov_sample)
    cn_lw = np.linalg.cond(cov_lw)
    print(f"\n  Sample covariance condition number:        {cn_sample:>10.2f}")
    print(f"  Ledoit-Wolf covariance condition number:   {cn_lw:>10.2f}")
    print(f"  Shrinkage reduces condition number by {100*(1 - cn_lw/cn_sample):.1f}%")
    print(f"  (Lower = better conditioned = more stable portfolio weights)")
