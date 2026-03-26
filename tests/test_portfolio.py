"""
Tests for portfolio construction models.

Covers: Black-Litterman, Markowitz mean-variance, Almgren-Chriss optimal
execution, and Kelly criterion.

All tests verify mathematical properties using synthetic data with fixed seeds.
"""

import numpy as np
import pytest
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════════════
# Black-Litterman helpers
# ═══════════════════════════════════════════════════════════════════════════

def market_implied_returns(cov, weights, risk_aversion=2.5, risk_free=0.03):
    """Reverse-optimise equilibrium excess returns: π = δ·Σ·w_mkt."""
    return risk_aversion * cov @ weights


def black_litterman_posterior(cov, pi, P, Q, omega=None, tau=0.05):
    """
    Black-Litterman posterior mean and covariance.

    Parameters
    ----------
    cov    : N×N covariance matrix
    pi     : N-vector of equilibrium returns
    P      : K×N pick matrix (views)
    Q      : K-vector of view returns
    omega  : K×K uncertainty of views (default: τ·P·Σ·P')
    tau    : scalar, uncertainty of equilibrium

    Returns
    -------
    mu_bl  : posterior expected returns (N,)
    cov_bl : posterior covariance (N×N)
    """
    tau_cov = tau * cov
    if omega is None:
        omega = np.diag(np.diag(P @ tau_cov @ P.T))

    tau_cov_inv = np.linalg.inv(tau_cov)
    omega_inv = np.linalg.inv(omega)

    cov_bl = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
    mu_bl = cov_bl @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

    return mu_bl, cov_bl


# ═══════════════════════════════════════════════════════════════════════════
# Markowitz helpers
# ═══════════════════════════════════════════════════════════════════════════

def min_variance_portfolio(cov, allow_short=False):
    """Find the minimum variance portfolio weights."""
    n = cov.shape[0]
    x0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if allow_short:
        bounds = [(-1, 1)] * n
    else:
        bounds = [(0, 1)] * n

    res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                   bounds=bounds, constraints=constraints)
    return res.x, res.fun


def efficient_frontier(mu, cov, n_points=30, allow_short=False):
    """
    Compute efficient frontier: for each target return, minimise variance.
    Returns (target_returns, portfolio_vols, weights_list).
    """
    n = len(mu)
    w_min, var_min = min_variance_portfolio(cov, allow_short)
    ret_min = mu @ w_min

    target_returns = np.linspace(ret_min, max(mu), n_points)
    vols = []
    weights_list = []

    for target in target_returns:
        x0 = np.ones(n) / n
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: mu @ w - t},
        ]
        bounds = [(-1, 1) if allow_short else (0, 1)] * n

        res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints)
        vols.append(np.sqrt(res.fun))
        weights_list.append(res.x)

    return target_returns, np.array(vols), weights_list


def max_sharpe_portfolio(mu, cov, rf=0.03, allow_short=False):
    """Find the tangency (max Sharpe) portfolio."""
    n = len(mu)
    x0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(-1, 1) if allow_short else (0, 1)] * n

    def neg_sharpe(w):
        ret = mu @ w
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / vol if vol > 1e-10 else 0

    res = minimize(neg_sharpe, x0, method="SLSQP",
                   bounds=bounds, constraints=constraints)
    return res.x, -res.fun  # weights, sharpe


# ═══════════════════════════════════════════════════════════════════════════
# Almgren-Chriss optimal execution helpers
# ═══════════════════════════════════════════════════════════════════════════

def almgren_chriss_trajectory(X, T, N, sigma, eta, gamma, lam):
    """
    Almgren-Chriss (2001) optimal execution trajectory.

    Parameters
    ----------
    X     : total shares to liquidate
    T     : total time horizon
    N     : number of time steps
    sigma : volatility ($/share/√time)
    eta   : temporary impact coefficient
    gamma : permanent impact coefficient
    lam   : risk aversion parameter

    Returns
    -------
    x : array of remaining holdings at each time step (N+1,)
    n : array of trade sizes at each step (N,)
    """
    tau = T / N
    kappa_sq = lam * sigma**2 / eta
    kappa = np.sqrt(kappa_sq) if kappa_sq > 0 else 1e-10

    # Sinh formula
    times = np.arange(N + 1) * tau
    x = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

    # Trade sizes
    n = -np.diff(x)  # positive = selling

    return x, n


# ═══════════════════════════════════════════════════════════════════════════
# Kelly criterion helpers
# ═══════════════════════════════════════════════════════════════════════════

def kelly_fraction(p, b, a=1.0):
    """
    Kelly criterion for a simple bet.

    Parameters
    ----------
    p : probability of winning
    b : net odds received on a win (e.g., 1.0 for even money)
    a : net odds lost on a loss (default 1.0 = lose entire stake)

    Returns
    -------
    f : optimal fraction of bankroll to bet
    """
    q = 1 - p
    f = (p * b - q * a) / (a * b)
    return f


def kelly_continuous(mu, sigma, r=0.0):
    """
    Continuous Kelly for a lognormal asset.
    f* = (μ - r) / σ²
    """
    return (mu - r) / sigma**2


# ═══════════════════════════════════════════════════════════════════════════
# BLACK-LITTERMAN TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBlackLittermanPosterior:
    """Posterior mean lies between prior and views."""

    def test_posterior_between_prior_and_views(self, portfolio_params):
        """
        BL posterior mean should be a weighted average (shrinkage)
        between equilibrium (prior) and investor views.
        """
        pp = portfolio_params
        cov = pp["cov_matrix"]
        n = pp["n_assets"]
        w_mkt = np.ones(n) / n  # equal weight market portfolio

        pi = market_implied_returns(cov, w_mkt)

        # View: asset 0 outperforms by 2%
        P = np.zeros((1, n))
        P[0, 0] = 1.0
        Q = np.array([pi[0] + 0.02])

        mu_bl, _ = black_litterman_posterior(cov, pi, P, Q)

        # Posterior for asset 0 should be between prior and view
        assert mu_bl[0] > pi[0]      # pulled toward view
        assert mu_bl[0] < Q[0]       # but not all the way
        # Other assets should remain close to prior
        for i in range(1, n):
            assert abs(mu_bl[i] - pi[i]) < abs(mu_bl[0] - pi[0])

    def test_posterior_covariance_psd(self, portfolio_params):
        """Posterior covariance must be positive semi-definite."""
        pp = portfolio_params
        cov = pp["cov_matrix"]
        n = pp["n_assets"]
        w_mkt = np.ones(n) / n
        pi = market_implied_returns(cov, w_mkt)

        P = np.eye(n)[:2]  # views on first 2 assets
        Q = pi[:2] + 0.01

        _, cov_bl = black_litterman_posterior(cov, pi, P, Q)

        eigvals = np.linalg.eigvalsh(cov_bl)
        assert np.all(eigvals > -1e-10)

    def test_no_views_returns_prior(self, portfolio_params):
        """
        With very uncertain views (large Ω), posterior ≈ prior.
        """
        pp = portfolio_params
        cov = pp["cov_matrix"]
        n = pp["n_assets"]
        w_mkt = np.ones(n) / n
        pi = market_implied_returns(cov, w_mkt)

        P = np.eye(n)[:1]
        Q = np.array([0.50])  # extreme view, but very uncertain
        omega = np.array([[100.0]])  # huge uncertainty

        mu_bl, _ = black_litterman_posterior(cov, pi, P, Q, omega=omega)

        np.testing.assert_allclose(mu_bl, pi, atol=0.01)

    def test_certain_views_match_views(self, portfolio_params):
        """
        With very confident views (tiny Ω), posterior ≈ views.
        """
        pp = portfolio_params
        cov = pp["cov_matrix"]
        n = pp["n_assets"]
        w_mkt = np.ones(n) / n
        pi = market_implied_returns(cov, w_mkt)

        P = np.zeros((1, n))
        P[0, 0] = 1.0
        Q = np.array([0.15])
        omega = np.array([[1e-10]])  # extreme confidence

        mu_bl, _ = black_litterman_posterior(cov, pi, P, Q, omega=omega)

        assert abs(mu_bl[0] - Q[0]) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# MARKOWITZ TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestEfficientFrontierMonotonicity:
    """Efficient frontier should be monotonically increasing in risk-return."""

    def test_frontier_return_risk_monotonic(self, portfolio_params):
        """Higher target return ⟹ higher portfolio volatility on the frontier."""
        pp = portfolio_params
        rets, vols, _ = efficient_frontier(
            pp["expected_returns"], pp["cov_matrix"], n_points=20
        )
        # Volatilities should be non-decreasing along the frontier
        for i in range(1, len(vols)):
            assert vols[i] >= vols[i-1] - 1e-6

    def test_frontier_above_individual_assets(self, portfolio_params):
        """
        Efficient frontier should dominate individual assets
        (same return at lower risk, or higher return at same risk).
        """
        pp = portfolio_params
        mu = pp["expected_returns"]
        cov = pp["cov_matrix"]

        _, min_var = min_variance_portfolio(cov)
        min_vol = np.sqrt(min_var)

        # Individual asset volatilities
        asset_vols = np.sqrt(np.diag(cov))

        # Minimum variance portfolio should have lower vol than any single asset
        assert min_vol <= min(asset_vols) + 1e-6


class TestMinimumVariancePortfolio:
    """Properties of the global minimum variance portfolio."""

    def test_weights_sum_to_one(self, portfolio_params):
        """Weights must sum to 1."""
        pp = portfolio_params
        w, _ = min_variance_portfolio(pp["cov_matrix"])
        assert abs(np.sum(w) - 1.0) < 1e-6

    def test_min_var_is_minimum(self, portfolio_params):
        """
        Min variance portfolio should have lower variance than equal-weight.
        """
        pp = portfolio_params
        cov = pp["cov_matrix"]
        w_min, var_min = min_variance_portfolio(cov)

        w_eq = np.ones(pp["n_assets"]) / pp["n_assets"]
        var_eq = w_eq @ cov @ w_eq

        assert var_min <= var_eq + 1e-10

    def test_long_only_weights_bounded(self, portfolio_params):
        """Long-only weights ∈ [0, 1]."""
        pp = portfolio_params
        w, _ = min_variance_portfolio(pp["cov_matrix"], allow_short=False)
        assert np.all(w >= -1e-8)
        assert np.all(w <= 1.0 + 1e-8)

    def test_max_sharpe_beats_min_var_sharpe(self, portfolio_params):
        """Tangency portfolio should have higher Sharpe than min var."""
        pp = portfolio_params
        mu = pp["expected_returns"]
        cov = pp["cov_matrix"]
        rf = pp["risk_free_rate"]

        w_min, _ = min_variance_portfolio(cov)
        ret_min = mu @ w_min
        vol_min = np.sqrt(w_min @ cov @ w_min)
        sharpe_min = (ret_min - rf) / vol_min

        w_tan, sharpe_tan = max_sharpe_portfolio(mu, cov, rf)

        assert sharpe_tan >= sharpe_min - 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# ALMGREN-CHRISS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAlmgrenChrissTrajectory:
    """Optimal execution trajectory properties."""

    def test_trajectory_sums_to_total(self):
        """Total shares traded must equal initial position."""
        X = 100_000
        x, n = almgren_chriss_trajectory(
            X=X, T=1.0, N=50, sigma=0.3, eta=0.01, gamma=0.001, lam=1e-6
        )
        assert abs(np.sum(n) - X) < 1.0  # within 1 share

    def test_trajectory_starts_at_X(self):
        """Holdings at t=0 must equal total position X."""
        X = 50_000
        x, _ = almgren_chriss_trajectory(
            X=X, T=1.0, N=20, sigma=0.25, eta=0.01, gamma=0.001, lam=1e-6
        )
        assert abs(x[0] - X) < 1e-6

    def test_trajectory_ends_at_zero(self):
        """Holdings at t=T must be ≈ 0 (fully liquidated)."""
        X = 50_000
        x, _ = almgren_chriss_trajectory(
            X=X, T=1.0, N=50, sigma=0.25, eta=0.01, gamma=0.001, lam=1e-6
        )
        assert abs(x[-1]) < 1.0

    def test_monotone_decreasing_holdings(self):
        """Holdings should monotonically decrease (for a sell programme)."""
        X = 100_000
        x, _ = almgren_chriss_trajectory(
            X=X, T=1.0, N=30, sigma=0.3, eta=0.01, gamma=0.001, lam=1e-6
        )
        for i in range(1, len(x)):
            assert x[i] <= x[i-1] + 1e-6

    def test_all_trades_positive(self):
        """All trade sizes should be positive (selling)."""
        X = 100_000
        _, n = almgren_chriss_trajectory(
            X=X, T=1.0, N=30, sigma=0.3, eta=0.01, gamma=0.001, lam=1e-6
        )
        assert np.all(n > -1e-6)

    def test_higher_risk_aversion_front_loads(self):
        """
        Higher λ (risk aversion) → more front-loaded execution
        (sell more early to reduce risk).
        """
        X = 100_000
        params = dict(X=X, T=1.0, N=50, sigma=0.3, eta=0.01, gamma=0.001)

        _, n_low = almgren_chriss_trajectory(**params, lam=1e-8)
        _, n_high = almgren_chriss_trajectory(**params, lam=1e-4)

        # High risk aversion: first trade should be larger
        assert n_high[0] > n_low[0]

    def test_sinh_boundary_lambda_zero(self):
        """
        As λ→0 (risk-neutral), trajectory approaches TWAP
        (uniform execution).
        """
        X = 100_000
        N = 50
        x, n = almgren_chriss_trajectory(
            X=X, T=1.0, N=N, sigma=0.3, eta=0.01, gamma=0.001, lam=1e-12
        )
        twap_trade = X / N
        # All trades should be approximately equal
        assert np.std(n) / twap_trade < 0.05


# ═══════════════════════════════════════════════════════════════════════════
# KELLY CRITERION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestKellyFraction:
    """Kelly fraction properties for discrete bets."""

    def test_positive_edge_positive_fraction(self):
        """Positive expected value → positive Kelly fraction."""
        # Even money bet with 60% win probability
        f = kelly_fraction(p=0.60, b=1.0)
        assert f > 0

    def test_negative_edge_no_bet(self):
        """Negative expected value → Kelly says don't bet (f ≤ 0)."""
        f = kelly_fraction(p=0.40, b=1.0)
        assert f <= 0

    def test_fair_bet_no_bet(self):
        """Fair bet (p·b = q·a) → Kelly fraction = 0."""
        f = kelly_fraction(p=0.50, b=1.0)
        assert abs(f) < 1e-10

    def test_kelly_bounded_zero_one(self):
        """For reasonable edges and even money, 0 < f < 1."""
        for p in [0.51, 0.55, 0.60, 0.70, 0.80]:
            f = kelly_fraction(p=p, b=1.0)
            assert 0 < f < 1.0, f"p={p}, f={f}"

    def test_certain_win_full_bet(self):
        """With p=1, Kelly says bet everything: f = 1."""
        f = kelly_fraction(p=1.0, b=1.0)
        assert abs(f - 1.0) < 1e-10

    def test_kelly_increases_with_edge(self):
        """Higher probability of winning → larger Kelly fraction."""
        fractions = [kelly_fraction(p=p, b=1.0) for p in [0.55, 0.60, 0.70, 0.80]]
        for i in range(1, len(fractions)):
            assert fractions[i] > fractions[i-1]

    @pytest.mark.parametrize("p,b,expected_sign", [
        (0.60, 1.0, 1),    # positive edge
        (0.40, 1.0, -1),   # negative edge
        (0.30, 3.0, 1),    # positive EV despite low p (0.3*3 - 0.7 = 0.2 > 0)
        (0.20, 2.0, -1),   # negative EV (0.2*2 - 0.8 = -0.4 < 0)
    ])
    def test_kelly_sign(self, p, b, expected_sign):
        """Kelly fraction sign matches expected value sign."""
        f = kelly_fraction(p=p, b=b)
        if expected_sign > 0:
            assert f > 0
        else:
            assert f <= 0


class TestKellyContinuous:
    """Continuous Kelly for lognormal assets."""

    def test_positive_excess_return(self):
        """Positive excess return → positive Kelly allocation."""
        f = kelly_continuous(mu=0.10, sigma=0.20, r=0.05)
        assert f > 0

    def test_negative_excess_return(self):
        """Negative excess return → negative (short) Kelly allocation."""
        f = kelly_continuous(mu=0.02, sigma=0.20, r=0.05)
        assert f < 0

    def test_kelly_formula_exact(self):
        """Verify f* = (μ-r)/σ²."""
        mu, sigma, r = 0.12, 0.25, 0.03
        f = kelly_continuous(mu, sigma, r)
        expected = (mu - r) / sigma**2
        assert abs(f - expected) < 1e-12

    def test_higher_vol_lower_fraction(self):
        """Higher volatility → lower Kelly fraction (same excess return)."""
        f_low_vol = kelly_continuous(mu=0.10, sigma=0.15, r=0.03)
        f_high_vol = kelly_continuous(mu=0.10, sigma=0.30, r=0.03)
        assert f_low_vol > f_high_vol

    def test_kelly_optimal_growth_rate(self):
        """
        Kelly maximises log growth rate: g(f) = r + f(μ-r) - ½f²σ².
        Verify that f* is indeed the maximiser.
        """
        mu, sigma, r = 0.10, 0.20, 0.03
        f_star = kelly_continuous(mu, sigma, r)

        def growth_rate(f):
            return r + f * (mu - r) - 0.5 * f**2 * sigma**2

        g_star = growth_rate(f_star)

        # Check that nearby points have lower growth rate
        for delta in [-0.5, -0.1, 0.1, 0.5]:
            g_other = growth_rate(f_star + delta)
            assert g_star >= g_other - 1e-12
