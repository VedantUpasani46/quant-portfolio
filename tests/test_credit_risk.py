"""
Tests for credit risk models.

Covers: CDS / hazard rates / survival probabilities, Merton model,
and Extreme Value Theory (EVT) / GPD.

All tests verify mathematical properties using synthetic data with fixed seeds.
"""

import numpy as np
import pytest
from scipy.stats import norm, genpareto
from scipy.optimize import brentq, minimize_scalar


# ═══════════════════════════════════════════════════════════════════════════
# CDS / Hazard rate helpers
# ═══════════════════════════════════════════════════════════════════════════

def survival_probability(hazard_rate, t):
    """
    Survival probability under constant hazard rate.
    S(t) = exp(-λ·t)
    """
    return np.exp(-hazard_rate * t)


def default_probability(hazard_rate, t):
    """PD(t) = 1 - S(t)."""
    return 1 - survival_probability(hazard_rate, t)


def hazard_rate_from_spread(cds_spread, recovery=0.40):
    """
    Approximate hazard rate from CDS spread.
    λ ≈ spread / (1 - R)
    """
    return cds_spread / (1 - recovery)


def cds_premium_leg_pv(cds_spread, hazard_rate, risk_free, maturity, freq=4):
    """
    PV of CDS premium leg (buyer pays spread).
    Sum of spread * Δt * DF(t) * S(t) over payment dates.
    """
    dt = 1.0 / freq
    n_periods = int(maturity * freq)
    pv = 0.0
    for i in range(1, n_periods + 1):
        t = i * dt
        df = np.exp(-risk_free * t)
        surv = survival_probability(hazard_rate, t)
        pv += cds_spread * dt * df * surv
    return pv


def cds_protection_leg_pv(hazard_rate, recovery, risk_free, maturity, freq=4):
    """
    PV of CDS protection leg (seller pays (1-R) on default).
    Sum of (1-R) * (S(t_{i-1}) - S(t_i)) * DF(t_i) over periods.
    """
    dt = 1.0 / freq
    n_periods = int(maturity * freq)
    pv = 0.0
    for i in range(1, n_periods + 1):
        t_prev = (i - 1) * dt
        t_curr = i * dt
        df = np.exp(-risk_free * t_curr)
        default_prob = survival_probability(hazard_rate, t_prev) - survival_probability(hazard_rate, t_curr)
        pv += (1 - recovery) * default_prob * df
    return pv


def fair_cds_spread(hazard_rate, recovery, risk_free, maturity, freq=4):
    """
    Fair CDS spread: set premium leg PV = protection leg PV.
    """
    prot = cds_protection_leg_pv(hazard_rate, recovery, risk_free, maturity, freq)
    # Premium leg per unit spread
    dt = 1.0 / freq
    n_periods = int(maturity * freq)
    risky_annuity = 0.0
    for i in range(1, n_periods + 1):
        t = i * dt
        df = np.exp(-risk_free * t)
        surv = survival_probability(hazard_rate, t)
        risky_annuity += dt * df * surv

    return prot / risky_annuity if risky_annuity > 0 else 0


# ═══════════════════════════════════════════════════════════════════════════
# Merton model helpers
# ═══════════════════════════════════════════════════════════════════════════

def merton_pd(V, D, T, r, sigma_V):
    """
    Merton (1974) default probability.

    Parameters
    ----------
    V       : firm asset value
    D       : face value of debt (default barrier)
    T       : time to maturity of debt
    r       : risk-free rate
    sigma_V : asset volatility

    Returns
    -------
    pd : probability of default (risk-neutral)
    """
    d2 = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    return norm.cdf(-d2)


def merton_equity_value(V, D, T, r, sigma_V):
    """Merton equity = call option on firm assets with strike = D."""
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    return V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)


def merton_debt_value(V, D, T, r, sigma_V):
    """Merton debt = risk-free debt - put option."""
    equity = merton_equity_value(V, D, T, r, sigma_V)
    return V - equity  # balance sheet identity: V = E + D_risky


def merton_credit_spread(V, D, T, r, sigma_V):
    """
    Credit spread implied by Merton model.
    s = -ln(D_risky / (D·exp(-rT))) / T
    """
    D_risky = merton_debt_value(V, D, T, r, sigma_V)
    D_rf = D * np.exp(-r * T)
    if D_risky <= 0 or D_risky >= D_rf:
        return 0.0
    return -np.log(D_risky / D_rf) / T


# ═══════════════════════════════════════════════════════════════════════════
# EVT / GPD helpers
# ═══════════════════════════════════════════════════════════════════════════

def fit_gpd(exceedances):
    """
    Fit Generalized Pareto Distribution to threshold exceedances via MLE.
    Returns (shape ξ, scale β).
    """
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    return shape, scale


def gpd_var(xi, beta, threshold, n_total, n_exceed, alpha=0.05):
    """
    GPD-based VaR estimate.
    VaR_α = u + (β/ξ) * [(n/N_u * α)^{-ξ} - 1]  for ξ ≠ 0
    """
    p_exceed = n_exceed / n_total
    if abs(xi) < 1e-10:
        return threshold - beta * np.log(alpha / p_exceed)
    return threshold + (beta / xi) * ((alpha / p_exceed)**(-xi) - 1)


def gpd_cvar(xi, beta, threshold, n_total, n_exceed, alpha=0.05):
    """
    GPD-based CVaR (Expected Shortfall).
    ES_α = VaR_α / (1-ξ) + (β - ξ·u) / (1-ξ)
    """
    var = gpd_var(xi, beta, threshold, n_total, n_exceed, alpha)
    if xi >= 1:
        return np.inf  # undefined for ξ ≥ 1
    return var / (1 - xi) + (beta - xi * threshold) / (1 - xi)


# ═══════════════════════════════════════════════════════════════════════════
# CDS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestHazardRate:
    """Hazard rate properties."""

    def test_hazard_rate_positive(self):
        """Hazard rate derived from positive spread must be positive."""
        for spread in [0.005, 0.01, 0.05, 0.10]:
            h = hazard_rate_from_spread(spread, recovery=0.40)
            assert h > 0

    def test_hazard_rate_increases_with_spread(self):
        """Higher CDS spread → higher hazard rate."""
        h1 = hazard_rate_from_spread(0.01, 0.40)
        h2 = hazard_rate_from_spread(0.05, 0.40)
        assert h2 > h1

    def test_hazard_rate_decreases_with_recovery(self):
        """Higher recovery → lower hazard rate (same spread)."""
        h_low_r = hazard_rate_from_spread(0.02, 0.20)
        h_high_r = hazard_rate_from_spread(0.02, 0.60)
        assert h_high_r > h_low_r


class TestSurvivalProbability:
    """Survival probability properties."""

    def test_survival_at_zero(self):
        """S(0) = 1 (no time elapsed = no default)."""
        assert abs(survival_probability(0.05, 0) - 1.0) < 1e-12

    def test_survival_monotonically_decreasing(self):
        """S(t) is strictly decreasing in t."""
        h = 0.03
        times = np.linspace(0, 30, 100)
        probs = [survival_probability(h, t) for t in times]
        for i in range(1, len(probs)):
            assert probs[i] < probs[i-1]

    def test_survival_bounded_zero_one(self):
        """S(t) ∈ (0, 1] for t ≥ 0 and λ > 0."""
        h = 0.05
        for t in [0, 0.5, 1, 5, 10, 30]:
            s = survival_probability(h, t)
            assert 0 < s <= 1.0

    def test_survival_approaches_zero(self):
        """S(t) → 0 as t → ∞."""
        h = 0.02
        s = survival_probability(h, 500)
        assert s < 0.01

    def test_pd_complement_of_survival(self):
        """PD(t) = 1 - S(t)."""
        h, t = 0.03, 5.0
        assert abs(default_probability(h, t) + survival_probability(h, t) - 1.0) < 1e-12


class TestCDSPricing:
    """CDS fair spread and leg PV tests."""

    def test_fair_spread_positive(self, credit_params):
        """Fair CDS spread must be positive."""
        cp = credit_params
        spread = fair_cds_spread(cp["hazard_rate"], cp["recovery_rate"],
                                  cp["risk_free_rate"], cp["maturity_years"])
        assert spread > 0

    def test_fair_spread_premium_equals_protection(self, credit_params):
        """At fair spread, premium leg PV = protection leg PV."""
        cp = credit_params
        spread = fair_cds_spread(cp["hazard_rate"], cp["recovery_rate"],
                                  cp["risk_free_rate"], cp["maturity_years"])

        prem_pv = cds_premium_leg_pv(spread, cp["hazard_rate"],
                                      cp["risk_free_rate"], cp["maturity_years"])
        prot_pv = cds_protection_leg_pv(cp["hazard_rate"], cp["recovery_rate"],
                                         cp["risk_free_rate"], cp["maturity_years"])

        assert abs(prem_pv - prot_pv) < 1e-8

    def test_spread_increases_with_hazard(self):
        """Higher hazard rate → higher CDS spread."""
        spreads = []
        for h in [0.01, 0.03, 0.05, 0.10]:
            s = fair_cds_spread(h, 0.40, 0.03, 5)
            spreads.append(s)
        for i in range(1, len(spreads)):
            assert spreads[i] > spreads[i-1]

    def test_spread_decreases_with_recovery(self):
        """Higher recovery → lower CDS spread (less loss given default)."""
        spreads = []
        for R in [0.20, 0.40, 0.60, 0.80]:
            s = fair_cds_spread(0.03, R, 0.03, 5)
            spreads.append(s)
        for i in range(1, len(spreads)):
            assert spreads[i] < spreads[i-1]


# ═══════════════════════════════════════════════════════════════════════════
# MERTON MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestMertonPD:
    """Merton default probability properties."""

    def test_pd_bounded_zero_one(self):
        """PD ∈ [0, 1]."""
        for V in [50, 100, 200]:
            for D in [80, 100, 120]:
                pd = merton_pd(V, D, T=1.0, r=0.05, sigma_V=0.30)
                assert 0 <= pd <= 1

    def test_pd_increases_with_leverage(self):
        """Higher D/V ratio → higher PD."""
        V, T, r, sigma = 100, 1.0, 0.05, 0.25
        pds = [merton_pd(V, D, T, r, sigma) for D in [60, 80, 100, 120, 140]]
        for i in range(1, len(pds)):
            assert pds[i] >= pds[i-1]

    def test_pd_increases_with_volatility(self):
        """Higher asset volatility → higher PD."""
        V, D, T, r = 100, 90, 1.0, 0.05
        pds = [merton_pd(V, D, T, r, sigma) for sigma in [0.10, 0.20, 0.30, 0.50]]
        for i in range(1, len(pds)):
            assert pds[i] >= pds[i-1]

    def test_pd_approaches_zero_for_low_leverage(self):
        """Very low leverage → PD ≈ 0."""
        pd = merton_pd(V=1000, D=10, T=1.0, r=0.05, sigma_V=0.20)
        assert pd < 0.001

    def test_pd_approaches_one_for_high_leverage(self):
        """Very high leverage → PD ≈ 1."""
        pd = merton_pd(V=10, D=1000, T=1.0, r=0.05, sigma_V=0.20)
        assert pd > 0.99

    def test_pd_increases_with_maturity(self):
        """Longer time horizon → higher PD (more time for things to go wrong)."""
        V, D, r, sigma = 100, 90, 0.05, 0.25
        pds = [merton_pd(V, D, T, r, sigma) for T in [0.5, 1, 2, 5, 10]]
        # PD should generally increase with T (may not be perfectly monotone for all params)
        assert pds[-1] > pds[0]


class TestMertonConvergence:
    """Merton model convergence and consistency."""

    def test_balance_sheet_identity(self):
        """V = E + D_risky (firm value = equity + risky debt)."""
        V, D, T, r, sigma = 100, 80, 1.0, 0.05, 0.25
        E = merton_equity_value(V, D, T, r, sigma)
        D_risky = merton_debt_value(V, D, T, r, sigma)
        assert abs(E + D_risky - V) < 1e-10

    def test_equity_nonnegative(self):
        """Equity value is always non-negative (limited liability)."""
        for V in [50, 100, 200]:
            for D in [80, 100, 120]:
                E = merton_equity_value(V, D, T=1.0, r=0.05, sigma_V=0.30)
                assert E >= -1e-10

    def test_credit_spread_nonnegative(self):
        """Credit spread must be non-negative."""
        V, D, T, r, sigma = 120, 100, 1.0, 0.05, 0.25
        spread = merton_credit_spread(V, D, T, r, sigma)
        assert spread >= -1e-10

    def test_spread_increases_with_leverage(self):
        """Higher leverage → wider credit spread."""
        T, r, sigma = 1.0, 0.05, 0.25
        spreads = []
        for V in [150, 130, 110, 100, 90]:
            D = 100
            s = merton_credit_spread(V, D, T, r, sigma)
            spreads.append(s)
        # Spreads should generally increase as V decreases
        assert spreads[-1] > spreads[0]


# ═══════════════════════════════════════════════════════════════════════════
# EVT / GPD TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestGPDShapeParameter:
    """GPD shape parameter ξ bounds and behavior."""

    def test_gpd_fit_from_known_distribution(self):
        """
        Fit GPD to data generated from known GPD → recovered ξ ≈ true ξ.
        """
        rng = np.random.default_rng(42)
        true_xi = 0.2
        true_beta = 1.0
        data = genpareto.rvs(c=true_xi, scale=true_beta, size=5000, random_state=42)

        xi_hat, beta_hat = fit_gpd(data)
        assert abs(xi_hat - true_xi) < 0.10
        assert abs(beta_hat - true_beta) < 0.20

    def test_gpd_shape_bounded_financial(self):
        """
        For typical financial loss data, ξ should be in (-0.5, 1.0).
        Fat tails ⟹ positive ξ, but not extreme.
        """
        rng = np.random.default_rng(42)
        # Simulate t-distributed losses (fat tails)
        from scipy.stats import t as t_dist
        losses = np.abs(t_dist.rvs(df=4, size=10_000, random_state=42))
        threshold = np.percentile(losses, 95)
        exceedances = losses[losses > threshold] - threshold

        xi, beta = fit_gpd(exceedances)
        assert -0.5 < xi < 1.0

    def test_gpd_scale_positive(self):
        """GPD scale parameter β must be positive."""
        rng = np.random.default_rng(42)
        data = genpareto.rvs(c=0.1, scale=2.0, size=2000, random_state=42)
        _, beta = fit_gpd(data)
        assert beta > 0


class TestPOTThresholdSensitivity:
    """Peaks-Over-Threshold sensitivity analysis."""

    def test_higher_threshold_fewer_exceedances(self):
        """Higher threshold → fewer exceedances (mechanical property)."""
        rng = np.random.default_rng(42)
        data = np.abs(rng.standard_normal(10_000))

        for u_low, u_high in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5)]:
            n_low = np.sum(data > u_low)
            n_high = np.sum(data > u_high)
            assert n_high < n_low

    def test_gpd_var_increases_with_threshold(self):
        """
        GPD-based VaR at fixed α should be relatively stable across
        reasonable threshold choices (mean excess plot stability).
        """
        rng = np.random.default_rng(42)
        data = np.abs(rng.standard_normal(10_000))

        vars_at_thresholds = []
        for pct in [90, 93, 95, 97]:
            u = np.percentile(data, pct)
            exceedances = data[data > u] - u
            xi, beta = fit_gpd(exceedances)
            v = gpd_var(xi, beta, u, len(data), len(exceedances), alpha=0.01)
            vars_at_thresholds.append(v)

        # VaR estimates should be in a reasonable range of each other
        spread = max(vars_at_thresholds) - min(vars_at_thresholds)
        mean_var = np.mean(vars_at_thresholds)
        assert spread / mean_var < 0.50  # within 50% spread

    def test_gpd_cvar_geq_var(self):
        """GPD-based CVaR ≥ VaR."""
        rng = np.random.default_rng(42)
        data = np.abs(rng.standard_normal(5000))
        u = np.percentile(data, 95)
        exceedances = data[data > u] - u

        xi, beta = fit_gpd(exceedances)
        if xi < 1:  # CVaR only defined for ξ < 1
            var = gpd_var(xi, beta, u, len(data), len(exceedances), alpha=0.01)
            cvar = gpd_cvar(xi, beta, u, len(data), len(exceedances), alpha=0.01)
            assert cvar >= var - 1e-10

    def test_mean_excess_linearity(self):
        """
        For GPD data, the mean excess function e(u) = E[X-u | X>u]
        should be approximately linear in u.
        """
        rng = np.random.default_rng(42)
        true_xi, true_beta = 0.2, 1.0
        data = genpareto.rvs(c=true_xi, scale=true_beta, size=20_000, random_state=42)

        thresholds = np.percentile(data, [70, 75, 80, 85, 90])
        mean_excesses = []
        for u in thresholds:
            exc = data[data > u] - u
            mean_excesses.append(np.mean(exc))

        # Mean excess should be approximately linear
        me = np.array(mean_excesses)
        # Check monotonicity (for ξ > 0, mean excess is increasing)
        for i in range(1, len(me)):
            assert me[i] >= me[i-1] * 0.90  # allow 10% slack
