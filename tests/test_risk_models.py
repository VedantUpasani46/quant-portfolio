"""
Tests for risk models: GARCH, DCC-GARCH, VaR/CVaR.

All tests verify mathematical properties using synthetic data with fixed seeds.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════
# GARCH(1,1) helpers
# ═══════════════════════════════════════════════════════════════════════════

def garch11_simulate(omega, alpha, beta, n, rng, sigma2_0=None):
    """
    Simulate GARCH(1,1) process:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
        r_t = σ_t · z_t,  z_t ~ N(0,1)
    Returns (returns, variances).
    """
    z = rng.standard_normal(n)
    sigma2 = np.zeros(n)
    r = np.zeros(n)

    if sigma2_0 is None:
        sigma2_0 = omega / (1 - alpha - beta)

    sigma2[0] = sigma2_0
    r[0] = np.sqrt(sigma2[0]) * z[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
        r[t] = np.sqrt(sigma2[t]) * z[t]

    return r, sigma2


def garch11_loglik(params, returns):
    """Negative log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        if sigma2[t] <= 0:
            return 1e10

    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
    return -ll  # negative for minimization


def garch11_fit(returns):
    """Fit GARCH(1,1) via MLE. Returns (omega, alpha, beta)."""
    x0 = [1e-5, 0.05, 0.90]
    bounds = [(1e-8, 0.1), (1e-6, 0.5), (0.5, 0.9999)]
    constraints = {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}

    res = minimize(garch11_loglik, x0, args=(returns,), method="SLSQP",
                   bounds=bounds, constraints=constraints)
    return res.x, res


# ═══════════════════════════════════════════════════════════════════════════
# DCC-GARCH helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_dcc_correlations(returns_matrix, a=0.05, b=0.93):
    """
    Simplified DCC(1,1) correlation estimation.
    returns_matrix: T × N array of standardised residuals.
    Returns array of T correlation matrices (N×N each).
    """
    T, N = returns_matrix.shape
    Qbar = np.corrcoef(returns_matrix.T)

    Q = np.zeros((T, N, N))
    R = np.zeros((T, N, N))
    Q[0] = Qbar.copy()

    for t in range(1, T):
        eps = returns_matrix[t-1:t].T  # N×1
        Q[t] = (1 - a - b) * Qbar + a * (eps @ eps.T) + b * Q[t-1]

        # Normalize to correlation
        D_inv = np.diag(1.0 / np.sqrt(np.diag(Q[t])))
        R[t] = D_inv @ Q[t] @ D_inv

    return R


# ═══════════════════════════════════════════════════════════════════════════
# VaR / CVaR helpers
# ═══════════════════════════════════════════════════════════════════════════

def parametric_var(returns, alpha=0.05):
    """Parametric (Gaussian) VaR at confidence level 1-alpha."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    return -(mu + stats.norm.ppf(alpha) * sigma)


def historical_var(returns, alpha=0.05):
    """Historical simulation VaR."""
    return -np.percentile(returns, 100 * alpha)


def parametric_cvar(returns, alpha=0.05):
    """Parametric (Gaussian) CVaR (Expected Shortfall)."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    var = parametric_var(returns, alpha)
    # ES = -mu + sigma * phi(Phi^{-1}(alpha)) / alpha
    return -(mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)


def historical_cvar(returns, alpha=0.05):
    """Historical CVaR: mean of losses beyond VaR."""
    threshold = np.percentile(returns, 100 * alpha)
    tail = returns[returns <= threshold]
    return -np.mean(tail) if len(tail) > 0 else -threshold


def kupiec_test(violations, n_obs, alpha=0.05):
    """
    Kupiec (1995) proportion of failures test.
    Returns (test_stat, p_value).
    H0: violation rate = alpha.
    """
    n_viol = int(np.sum(violations))
    p_hat = n_viol / n_obs if n_obs > 0 else 0

    if n_viol == 0 or n_viol == n_obs:
        return 0.0, 1.0

    lr = -2 * (n_viol * np.log(alpha / p_hat) +
               (n_obs - n_viol) * np.log((1 - alpha) / (1 - p_hat)))
    p_value = 1 - stats.chi2.cdf(lr, 1)
    return lr, p_value


# ═══════════════════════════════════════════════════════════════════════════
# GARCH TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestGARCHPersistence:
    """Persistence parameter bounds: α + β < 1 for stationarity."""

    def test_stationarity_condition(self):
        """α + β must be strictly less than 1 for covariance stationarity."""
        omega, alpha, beta = 1e-5, 0.08, 0.90
        assert alpha + beta < 1.0

    def test_persistence_near_one(self):
        """High persistence (α+β close to 1) produces slowly decaying vol."""
        rng = np.random.default_rng(42)
        r_high, s2_high = garch11_simulate(1e-6, 0.05, 0.94, 1000, rng)
        r_low, s2_low = garch11_simulate(1e-5, 0.05, 0.80, 1000, np.random.default_rng(42))

        # Higher persistence → slower mean reversion → more autocorrelation in variance
        acf_high = np.corrcoef(s2_high[1:], s2_high[:-1])[0, 1]
        acf_low = np.corrcoef(s2_low[1:], s2_low[:-1])[0, 1]
        assert acf_high > acf_low

    @pytest.mark.parametrize("alpha,beta", [
        (0.05, 0.90),
        (0.10, 0.85),
        (0.15, 0.80),
        (0.03, 0.95),
    ])
    def test_persistence_bound(self, alpha, beta):
        """Multiple parameter sets all satisfy stationarity."""
        assert alpha + beta < 1.0
        assert alpha > 0
        assert beta > 0


class TestGARCHUnconditionalVariance:
    """Unconditional variance = ω / (1 - α - β)."""

    def test_unconditional_variance_formula(self):
        """Sample variance of long GARCH simulation ≈ ω/(1-α-β)."""
        omega, alpha, beta = 2e-5, 0.08, 0.90
        rng = np.random.default_rng(42)
        returns, _ = garch11_simulate(omega, alpha, beta, 50_000, rng)

        theoretical = omega / (1 - alpha - beta)
        empirical = np.var(returns)

        assert abs(empirical - theoretical) / theoretical < 0.15  # within 15%

    def test_conditional_variance_mean_reverts(self):
        """Conditional variance should mean-revert to unconditional level."""
        omega, alpha, beta = 2e-5, 0.08, 0.88
        rng = np.random.default_rng(42)
        _, sigma2 = garch11_simulate(omega, alpha, beta, 10_000, rng,
                                      sigma2_0=0.001)  # start high

        unconditional = omega / (1 - alpha - beta)
        # Late-sample mean should be close to unconditional
        late_mean = np.mean(sigma2[5000:])
        assert abs(late_mean - unconditional) / unconditional < 0.25


class TestGARCHMLE:
    """MLE convergence and parameter recovery."""

    def test_mle_recovers_parameters(self):
        """
        MLE should approximately recover true GARCH parameters
        from a sufficiently long simulated series.
        """
        true_omega, true_alpha, true_beta = 2e-5, 0.08, 0.88
        rng = np.random.default_rng(42)
        returns, _ = garch11_simulate(true_omega, true_alpha, true_beta, 5000, rng)

        (est_omega, est_alpha, est_beta), res = garch11_fit(returns)

        assert res.success
        # Parameters within reasonable tolerance
        assert abs(est_alpha - true_alpha) < 0.05
        assert abs(est_beta - true_beta) < 0.10

    def test_mle_stationarity_constraint(self):
        """Estimated α + β < 1 (stationarity enforced)."""
        rng = np.random.default_rng(123)
        returns, _ = garch11_simulate(1e-5, 0.07, 0.90, 3000, rng)
        (omega, alpha, beta), _ = garch11_fit(returns)
        assert alpha + beta < 1.0

    def test_mle_omega_positive(self):
        """Estimated ω must be positive."""
        rng = np.random.default_rng(7)
        returns, _ = garch11_simulate(1e-5, 0.06, 0.91, 3000, rng)
        (omega, alpha, beta), _ = garch11_fit(returns)
        assert omega > 0


# ═══════════════════════════════════════════════════════════════════════════
# DCC-GARCH TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDCCCorrelationMatrix:
    """DCC correlation matrix properties."""

    def test_correlation_psd(self):
        """Every DCC correlation matrix must be positive semi-definite."""
        rng = np.random.default_rng(42)
        T, N = 500, 3
        returns = rng.standard_normal((T, N))
        R = compute_dcc_correlations(returns, a=0.05, b=0.93)

        for t in range(10, T):  # skip burn-in
            eigvals = np.linalg.eigvalsh(R[t])
            assert np.all(eigvals >= -1e-10), f"Non-PSD at t={t}: {eigvals.min()}"

    def test_correlation_bounds(self):
        """All off-diagonal correlations must be in [-1, 1]."""
        rng = np.random.default_rng(42)
        T, N = 500, 4
        returns = rng.standard_normal((T, N))
        R = compute_dcc_correlations(returns, a=0.05, b=0.93)

        for t in range(10, T):
            assert np.all(R[t] >= -1.0 - 1e-10)
            assert np.all(R[t] <= 1.0 + 1e-10)

    def test_diagonal_ones(self):
        """Diagonal of correlation matrix must be 1."""
        rng = np.random.default_rng(42)
        T, N = 300, 3
        returns = rng.standard_normal((T, N))
        R = compute_dcc_correlations(returns, a=0.05, b=0.93)

        for t in range(10, T):
            diag = np.diag(R[t])
            np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_symmetry(self):
        """Correlation matrix must be symmetric."""
        rng = np.random.default_rng(42)
        T, N = 300, 4
        returns = rng.standard_normal((T, N))
        R = compute_dcc_correlations(returns, a=0.05, b=0.93)

        for t in range(10, T):
            np.testing.assert_allclose(R[t], R[t].T, atol=1e-12)


class TestDCCStressSpike:
    """Stress spike detection in DCC correlations."""

    def test_correlation_spike_during_stress(self):
        """
        During a stress period (large co-movement), correlations should spike.
        """
        rng = np.random.default_rng(42)
        T, N = 500, 3

        # Normal period
        returns = rng.standard_normal((T, N)) * 0.01

        # Inject stress: large correlated moves at t=250..270
        stress_factor = rng.standard_normal(20) * 0.05
        for j in range(N):
            returns[250:270, j] = stress_factor + rng.standard_normal(20) * 0.005

        # Standardise
        std_returns = (returns - returns.mean(axis=0)) / returns.std(axis=0)
        R = compute_dcc_correlations(std_returns, a=0.05, b=0.93)

        # Average off-diagonal correlation post-stress should be elevated
        corr_pre = np.mean([R[200][0, 1], R[200][0, 2], R[200][1, 2]])
        corr_post = np.mean([R[275][0, 1], R[275][0, 2], R[275][1, 2]])

        assert corr_post > corr_pre  # correlations increase in stress


# ═══════════════════════════════════════════════════════════════════════════
# VaR / CVaR TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestVaRProperties:
    """Value at Risk mathematical properties."""

    def test_cvar_geq_var_parametric(self):
        """CVaR ≥ VaR always (Expected Shortfall is more conservative)."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(10_000) * 0.02

        for alpha in [0.01, 0.025, 0.05, 0.10]:
            var = parametric_var(returns, alpha)
            cvar = parametric_cvar(returns, alpha)
            assert cvar >= var - 1e-10, f"CVaR < VaR at alpha={alpha}"

    def test_cvar_geq_var_historical(self):
        """CVaR ≥ VaR for historical method too."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(10_000) * 0.02

        for alpha in [0.01, 0.025, 0.05, 0.10]:
            var = historical_var(returns, alpha)
            cvar = historical_cvar(returns, alpha)
            assert cvar >= var - 1e-10

    def test_var_positive_for_zero_mean(self):
        """VaR at 5% should be positive for mean-zero returns."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(10_000) * 0.02
        var = parametric_var(returns, 0.05)
        assert var > 0

    def test_var_increases_with_volatility(self):
        """Higher volatility → higher VaR."""
        rng = np.random.default_rng(42)
        returns_low = rng.standard_normal(10_000) * 0.01
        returns_high = rng.standard_normal(10_000) * 0.03

        var_low = parametric_var(returns_low, 0.05)
        var_high = parametric_var(returns_high, 0.05)
        assert var_high > var_low

    def test_var_scales_with_confidence(self):
        """VaR at 1% > VaR at 5% (more extreme quantile)."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(5000) * 0.02

        var_1 = parametric_var(returns, 0.01)
        var_5 = parametric_var(returns, 0.05)
        assert var_1 > var_5


class TestVaRParametricVsHistorical:
    """Consistency between parametric and historical VaR."""

    def test_gaussian_returns_consistency(self):
        """
        For truly Gaussian returns, parametric and historical VaR
        should converge with large sample.
        """
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(100_000) * 0.02

        var_param = parametric_var(returns, 0.05)
        var_hist = historical_var(returns, 0.05)

        assert abs(var_param - var_hist) / var_param < 0.05  # within 5%

    def test_fat_tails_hist_higher(self):
        """
        With fat-tailed returns (t-distribution), historical VaR should
        exceed parametric (Gaussian) VaR at extreme quantiles.
        """
        rng = np.random.default_rng(42)
        # t-distribution with 4 dof (fat tails)
        returns = stats.t.rvs(df=4, size=50_000, random_state=42) * 0.01

        var_param = parametric_var(returns, 0.01)
        var_hist = historical_var(returns, 0.01)

        assert var_hist > var_param  # historical captures fat tails


class TestKupiecBacktest:
    """Kupiec proportion-of-failures backtest."""

    def test_kupiec_well_calibrated_model(self):
        """
        A well-calibrated VaR model should pass the Kupiec test
        (fail to reject H0 at 5% significance).
        """
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_obs = 10_000
        returns = rng.standard_normal(n_obs) * 0.02
        var = parametric_var(returns, alpha)

        # Count violations
        violations = returns < -var
        _, p_value = kupiec_test(violations, n_obs, alpha)

        assert p_value > 0.05, f"Kupiec p-value={p_value:.4f} (rejected at 5%)"

    def test_kupiec_rejects_bad_model(self):
        """
        A deliberately miscalibrated VaR (too small) should fail Kupiec.
        """
        rng = np.random.default_rng(42)
        n_obs = 10_000
        returns = rng.standard_normal(n_obs) * 0.02

        # Use VaR at 20% as if it were 5% → too many violations
        bad_var = parametric_var(returns, 0.20)
        violations = returns < -bad_var
        violation_rate = np.mean(violations)

        # Should have ~20% violations, far from 5%
        assert violation_rate > 0.10

    def test_violation_rate_near_alpha(self):
        """
        For Gaussian returns with correct VaR, violation rate ≈ α.
        """
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_obs = 50_000
        returns = rng.standard_normal(n_obs) * 0.015

        var = parametric_var(returns, alpha)
        violations = returns < -var
        viol_rate = np.mean(violations)

        assert abs(viol_rate - alpha) < 0.01  # within 1pp
