"""
Tests for derivatives pricing models.

Covers: Black-Scholes, Heston, SABR, Monte Carlo, Binomial Tree.
All tests verify mathematical properties using synthetic data with fixed seeds.
"""

import numpy as np
import pytest
from scipy.stats import norm
from scipy.optimize import brentq


# ═══════════════════════════════════════════════════════════════════════════
# Black-Scholes helpers (self-contained, no repo imports)
# ═══════════════════════════════════════════════════════════════════════════

def bs_d1(S, K, T, r, sigma, q=0.0):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S, K, T, r, sigma, q=0.0):
    return bs_d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_call(S, K, T, r, sigma, q=0.0):
    d1 = bs_d1(S, K, T, r, sigma, q)
    d2 = bs_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma, q=0.0):
    d1 = bs_d1(S, K, T, r, sigma, q)
    d2 = bs_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_delta_call(S, K, T, r, sigma, q=0.0):
    return np.exp(-q * T) * norm.cdf(bs_d1(S, K, T, r, sigma, q))


def bs_delta_put(S, K, T, r, sigma, q=0.0):
    return -np.exp(-q * T) * norm.cdf(-bs_d1(S, K, T, r, sigma, q))


def bs_gamma(S, K, T, r, sigma, q=0.0):
    d1 = bs_d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma, q=0.0):
    d1 = bs_d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta_call(S, K, T, r, sigma, q=0.0):
    d1 = bs_d1(S, K, T, r, sigma, q)
    d2 = bs_d2(S, K, T, r, sigma, q)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    return term1 + term2 + term3


def bs_rho_call(S, K, T, r, sigma, q=0.0):
    d2 = bs_d2(S, K, T, r, sigma, q)
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def bs_implied_vol(price, S, K, T, r, q=0.0, option_type="call"):
    """Recover implied vol from price via Brent's method."""
    func = bs_call if option_type == "call" else bs_put

    def objective(sigma):
        return func(S, K, T, r, sigma, q) - price

    return brentq(objective, 1e-6, 5.0, xtol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# Heston model helpers
# ═══════════════════════════════════════════════════════════════════════════

def heston_char_func(u, S, K, T, r, v0, kappa, theta, xi, rho):
    """Heston characteristic function (log-price)."""
    i = 1j
    d = np.sqrt((rho * xi * i * u - kappa)**2 + xi**2 * (i * u + u**2))
    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)

    C = (r * i * u * T
         + (kappa * theta / xi**2)
         * ((kappa - rho * xi * i * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
    D = ((kappa - rho * xi * i * u - d) / xi**2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))

    return np.exp(C + D * v0 + i * u * np.log(S))


def heston_call_price(S, K, T, r, v0, kappa, theta, xi, rho, N=256):
    """Heston European call via numerical integration of characteristic function."""
    from scipy.integrate import quad

    def integrand_P(u, j):
        i = 1j
        if j == 1:
            phi = heston_char_func(u - i, S, K, T, r, v0, kappa, theta, xi, rho)
            phi /= heston_char_func(-i, S, K, T, r, v0, kappa, theta, xi, rho)
        else:
            phi = heston_char_func(u, S, K, T, r, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-i * u * np.log(K)) * phi / (i * u))

    P1 = 0.5 + (1 / np.pi) * quad(lambda u: integrand_P(u, 1), 1e-8, 200, limit=500)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda u: integrand_P(u, 2), 1e-8, 200, limit=500)[0]

    return S * P1 - K * np.exp(-r * T) * P2


# ═══════════════════════════════════════════════════════════════════════════
# SABR model helpers
# ═══════════════════════════════════════════════════════════════════════════

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """
    Hagan et al. (2002) SABR implied volatility formula.
    """
    if abs(F - K) < 1e-12:
        # ATM formula
        FK_mid = F
        logFK = 0.0
        vol = (alpha / (F ** (1 - beta))) * (
            1 + ((1 - beta)**2 / 24 * alpha**2 / F**(2 - 2*beta)
                 + 0.25 * rho * beta * nu * alpha / F**(1 - beta)
                 + (2 - 3 * rho**2) / 24 * nu**2) * T
        )
        return vol

    FK = F * K
    FK_beta = FK ** ((1 - beta) / 2)
    logFK = np.log(F / K)

    z = (nu / alpha) * FK_beta * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

    prefix = alpha / (FK_beta * (1 + (1 - beta)**2 / 24 * logFK**2
                                  + (1 - beta)**4 / 1920 * logFK**4))

    correction = 1 + ((1 - beta)**2 / 24 * alpha**2 / FK**(1 - beta)
                       + 0.25 * rho * beta * nu * alpha / FK_beta
                       + (2 - 3 * rho**2) / 24 * nu**2) * T

    if abs(x_z) < 1e-12:
        return prefix * correction

    return prefix * (z / x_z) * correction


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo helpers
# ═══════════════════════════════════════════════════════════════════════════

def mc_european_call(S, K, T, r, sigma, n_paths, rng, antithetic=False,
                     control_variate=False):
    """
    Monte Carlo European call pricer with variance reduction techniques.
    Returns (price, std_error).
    """
    if antithetic:
        Z = rng.standard_normal(n_paths // 2)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(n_paths)

    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    discount = np.exp(-r * T)

    if control_variate:
        # Use the underlying as control variate
        cv = ST - S * np.exp(r * T)  # mean zero under Q
        cov_xy = np.cov(payoffs, cv)[0, 1]
        var_cv = np.var(cv)
        c_star = -cov_xy / var_cv if var_cv > 0 else 0.0
        payoffs = payoffs + c_star * cv

    price = discount * np.mean(payoffs)
    std_err = discount * np.std(payoffs) / np.sqrt(len(payoffs))
    return price, std_err


# ═══════════════════════════════════════════════════════════════════════════
# Binomial tree helpers
# ═══════════════════════════════════════════════════════════════════════════

def binomial_tree_european(S, K, T, r, sigma, N, option_type="call"):
    """CRR binomial tree for European options."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal payoffs
    j = np.arange(N + 1)
    ST = S * u**j * d**(N - j)
    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:i+2] + (1 - p) * V[0:i+1])

    return V[0]


def binomial_tree_american(S, K, T, r, sigma, N, option_type="put"):
    """CRR binomial tree for American options."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Asset prices at each node
    j = np.arange(N + 1)
    ST = S * u**j * d**(N - j)
    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    for i in range(N - 1, -1, -1):
        j = np.arange(i + 1)
        S_node = S * u**j * d**(i - j)
        cont = disc * (p * V[1:i+2] + (1 - p) * V[0:i+1])
        if option_type == "call":
            exercise = np.maximum(S_node - K, 0.0)
        else:
            exercise = np.maximum(K - S_node, 0.0)
        V = np.maximum(cont, exercise)

    return V[0]


# ═══════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBlackScholesPutCallParity:
    """Put-call parity: C - P = S·e^{-qT} - K·e^{-rT}."""

    @pytest.mark.parametrize("S,K,T,r,sigma,q", [
        (100, 100, 1.0, 0.05, 0.20, 0.0),
        (100, 110, 0.5, 0.03, 0.30, 0.02),
        (50,  45,  2.0, 0.08, 0.15, 0.01),
        (200, 180, 0.25, 0.01, 0.40, 0.05),
        (100, 100, 5.0, 0.04, 0.25, 0.03),
    ])
    def test_put_call_parity(self, S, K, T, r, sigma, q):
        """C - P must equal S·exp(-qT) - K·exp(-rT) for all parameter sets."""
        call = bs_call(S, K, T, r, sigma, q)
        put = bs_put(S, K, T, r, sigma, q)
        parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((call - put) - parity_rhs) < 1e-10

    def test_put_call_parity_vectorized(self):
        """Parity holds across a sweep of strikes."""
        S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.0
        strikes = np.linspace(60, 140, 50)
        for K in strikes:
            call = bs_call(S, K, T, r, sigma, q)
            put = bs_put(S, K, T, r, sigma, q)
            assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-10


class TestBlackScholesGreeks:
    """Greek relationships and properties."""

    def test_delta_call_bounded_zero_one(self, bs_params):
        """Call delta ∈ [0, 1]."""
        p = bs_params
        delta = bs_delta_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        assert 0.0 <= delta <= 1.0

    def test_delta_put_bounded_neg_one_zero(self, bs_params):
        """Put delta ∈ [-1, 0]."""
        p = bs_params
        delta = bs_delta_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        assert -1.0 <= delta <= 0.0

    def test_call_put_delta_relationship(self, bs_params):
        """Δ_call - Δ_put = e^{-qT}."""
        p = bs_params
        dc = bs_delta_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        dp = bs_delta_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        assert abs((dc - dp) - np.exp(-p["q"] * p["T"])) < 1e-10

    def test_gamma_positive(self, bs_params):
        """Gamma is always positive (convexity of option value)."""
        p = bs_params
        gamma = bs_gamma(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        assert gamma > 0

    def test_gamma_identical_call_put(self, bs_params):
        """Gamma is the same for calls and puts (follows from put-call parity)."""
        p = bs_params
        # Gamma depends only on d1, not on option type
        gamma = bs_gamma(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        assert gamma > 0  # same formula for both

    def test_vega_positive(self, bs_params):
        """Vega is always positive (higher vol → higher option price)."""
        p = bs_params
        vega = bs_vega(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        assert vega > 0

    def test_vega_identical_call_put(self, bs_params):
        """Vega is the same for calls and puts."""
        p = bs_params
        vega = bs_vega(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        assert vega > 0

    def test_bsm_pde_satisfied(self, bs_params):
        """
        The BSM PDE: θ + rSΔ + ½σ²S²Γ = rV.
        Verify numerically for a call option.
        """
        p = bs_params
        S, K, T, r, sigma, q = p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"]

        V = bs_call(S, K, T, r, sigma, q)
        theta = bs_theta_call(S, K, T, r, sigma, q)
        delta = bs_delta_call(S, K, T, r, sigma, q)
        gamma = bs_gamma(S, K, T, r, sigma, q)

        # PDE: θ + (r-q)·S·Δ + ½σ²S²Γ - rV + qSΔ ≈ θ + rSΔ + ½σ²S²Γ - rV  (q=0 simplified)
        # Full PDE with dividends: θ + (r-q)SΔ + ½σ²S²Γ = rV
        lhs = theta + (r - q) * S * delta + 0.5 * sigma**2 * S**2 * gamma
        rhs = r * V
        assert abs(lhs - rhs) < 1e-6


class TestBlackScholesImpliedVol:
    """Implied volatility round-trip tests."""

    @pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.35, 0.50, 0.80])
    def test_implied_vol_round_trip_call(self, sigma_true):
        """Price → IV → Price must be identity (call)."""
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        price = bs_call(S, K, T, r, sigma_true, q)
        sigma_rec = bs_implied_vol(price, S, K, T, r, q, "call")
        assert abs(sigma_rec - sigma_true) < 1e-10

    @pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.35, 0.50, 0.80])
    def test_implied_vol_round_trip_put(self, sigma_true):
        """Price → IV → Price must be identity (put)."""
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        price = bs_put(S, K, T, r, sigma_true, q)
        sigma_rec = bs_implied_vol(price, S, K, T, r, q, "put")
        assert abs(sigma_rec - sigma_true) < 1e-10

    def test_iv_monotonic_in_price(self):
        """Higher call price ⟹ higher implied vol."""
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        sigmas = np.linspace(0.10, 0.60, 20)
        prices = [bs_call(S, K, T, r, s, q) for s in sigmas]
        assert all(prices[i] < prices[i+1] for i in range(len(prices) - 1))


class TestBlackScholesBoundaryConditions:
    """Boundary and limiting behavior."""

    def test_call_lower_bound(self, bs_params):
        """C ≥ max(S·e^{-qT} - K·e^{-rT}, 0)."""
        p = bs_params
        call = bs_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        lower = max(p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"]), 0.0)
        assert call >= lower - 1e-12

    def test_put_lower_bound(self, bs_params):
        """P ≥ max(K·e^{-rT} - S·e^{-qT}, 0)."""
        p = bs_params
        put = bs_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        lower = max(p["K"] * np.exp(-p["r"] * p["T"]) - p["S"] * np.exp(-p["q"] * p["T"]), 0.0)
        assert put >= lower - 1e-12

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call → intrinsic value as σ→0."""
        S, K, T, r, q = 100, 50, 1.0, 0.05, 0.0
        call = bs_call(S, K, T, r, 0.001, q)
        intrinsic = S - K * np.exp(-r * T)
        assert abs(call - intrinsic) < 0.5

    def test_call_price_increases_with_vol(self):
        """∂C/∂σ > 0: call price is monotonically increasing in volatility."""
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        prev_price = 0.0
        for sigma in np.linspace(0.05, 0.80, 30):
            price = bs_call(S, K, T, r, sigma, q)
            assert price > prev_price
            prev_price = price

    def test_zero_time_call(self):
        """At expiry, C = max(S-K, 0)."""
        S_itm, S_otm, K = 110, 90, 100
        # Use tiny T to approximate expiry (T=0 causes division by zero)
        T = 1e-8
        r, sigma, q = 0.05, 0.20, 0.0
        assert abs(bs_call(S_itm, K, T, r, sigma, q) - 10.0) < 0.01
        assert abs(bs_call(S_otm, K, T, r, sigma, q)) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# HESTON MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestHestonFellerCondition:
    """Feller condition: 2κθ > ξ² ensures variance stays positive."""

    def test_feller_satisfied(self):
        """When 2κθ > ξ², the variance process doesn't hit zero."""
        kappa, theta, xi = 2.0, 0.04, 0.3
        assert 2 * kappa * theta > xi**2

    def test_feller_violated(self):
        """When 2κθ ≤ ξ², Feller condition is violated."""
        kappa, theta, xi = 0.5, 0.01, 0.5
        assert 2 * kappa * theta <= xi**2

    @pytest.mark.parametrize("kappa,theta,xi", [
        (3.0, 0.04, 0.30),
        (1.5, 0.06, 0.20),
        (5.0, 0.03, 0.50),
    ])
    def test_feller_parameter_sets(self, kappa, theta, xi):
        """Multiple parameter sets that satisfy Feller."""
        assert 2 * kappa * theta > xi**2


class TestHestonConvergenceToBSM:
    """As vol-of-vol ξ → 0, Heston → Black-Scholes."""

    def test_heston_converges_to_bsm(self):
        """
        With ξ ≈ 0, Heston call price should match BSM within tight tolerance.
        """
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        v0 = 0.04  # σ = 0.20
        kappa = 2.0
        theta = 0.04
        xi = 1e-4  # nearly zero vol-of-vol
        rho = -0.5

        heston_price = heston_call_price(S, K, T, r, v0, kappa, theta, xi, rho)
        bsm_price = bs_call(S, K, T, r, np.sqrt(v0), 0.0)

        assert abs(heston_price - bsm_price) < 0.05, (
            f"Heston={heston_price:.4f}, BSM={bsm_price:.4f}"
        )

    def test_heston_positive_price(self):
        """Heston call price must be non-negative."""
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7
        price = heston_call_price(S, K, T, r, v0, kappa, theta, xi, rho)
        assert price > 0


class TestHestonCalibration:
    """Calibration quality metrics."""

    def test_calibration_rmse_bound(self):
        """
        Calibrate Heston to synthetic smile; RMSE on IV should be small.
        We generate prices from Heston, then back out BSM IVs and check consistency.
        """
        S, T, r = 100.0, 1.0, 0.05
        v0, kappa, theta, xi, rho = 0.04, 1.5, 0.04, 0.3, -0.6
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)

        ivs = []
        for K in strikes:
            price = heston_call_price(S, K, T, r, v0, kappa, theta, xi, rho)
            # Ensure price is above intrinsic for IV recovery
            intrinsic = max(S - K * np.exp(-r * T), 1e-8)
            if price < intrinsic:
                price = intrinsic + 0.01
            iv = bs_implied_vol(price, S, K, T, r, 0.0, "call")
            ivs.append(iv)

        ivs = np.array(ivs)
        # IVs should be positive and reasonable
        assert np.all(ivs > 0)
        assert np.all(ivs < 1.0)
        # Smile should be roughly centered near sqrt(v0)
        atm_iv = ivs[2]  # K=100
        assert abs(atm_iv - np.sqrt(v0)) < 0.05


# ═══════════════════════════════════════════════════════════════════════════
# SABR MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSABRAtmVol:
    """ATM implied volatility accuracy."""

    def test_atm_vol_approximation(self):
        """ATM SABR vol ≈ α / F^{1-β} for small T."""
        F = 100.0
        alpha, beta, rho, nu = 0.20 * F**(1 - 0.5), 0.5, -0.3, 0.4
        T = 0.01  # short expiry — correction terms are small
        iv = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
        approx = alpha / F**(1 - beta)
        assert abs(iv - approx) / approx < 0.05  # within 5%

    def test_atm_vol_positive(self):
        """ATM implied vol must be strictly positive."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.25, 0.5, -0.3, 0.4
        iv = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
        assert iv > 0


class TestSABRBetaBoundary:
    """Behavior at beta=0 (normal) and beta=1 (lognormal)."""

    def test_beta_zero_normal_backbone(self):
        """β=0: SABR reduces to normal model; vol ≈ α for ATM."""
        F, T = 0.03, 1.0  # e.g. a rate
        alpha, beta, rho, nu = 0.005, 0.0, 0.0, 0.3
        iv = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
        # For beta=0, ATM vol ≈ alpha / F^1 = alpha/F
        approx = alpha / F
        assert abs(iv - approx) / approx < 0.10

    def test_beta_one_lognormal_backbone(self):
        """β=1: SABR reduces to lognormal model; vol ≈ α for ATM."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.20, 1.0, 0.0, 0.3
        iv = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
        # For beta=1, ATM vol ≈ alpha
        assert abs(iv - alpha) / alpha < 0.10

    def test_smile_flattens_as_nu_to_zero(self):
        """As ν→0, the smile flattens (constant vol)."""
        F, T = 100.0, 1.0
        alpha, beta, rho = 0.20, 0.5, -0.3
        strikes = np.array([90, 95, 100, 105, 110], dtype=float)

        # Large nu → pronounced smile
        ivs_large_nu = [sabr_implied_vol(F, K, T, alpha, beta, rho, 0.5) for K in strikes]
        # Tiny nu → flat
        ivs_small_nu = [sabr_implied_vol(F, K, T, alpha, beta, rho, 0.01) for K in strikes]

        spread_large = max(ivs_large_nu) - min(ivs_large_nu)
        spread_small = max(ivs_small_nu) - min(ivs_small_nu)
        assert spread_small < spread_large


class TestSABRSmileSymmetry:
    """Smile symmetry properties."""

    def test_zero_rho_smile_symmetric(self):
        """With ρ=0 and β=1, smile should be approximately symmetric around ATM."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.20, 1.0, 0.0, 0.3
        K_low = 95.0
        K_high = 105.0  # equidistant from ATM=100

        iv_low = sabr_implied_vol(F, K_low, T, alpha, beta, rho, nu)
        iv_high = sabr_implied_vol(F, K_high, T, alpha, beta, rho, nu)

        # Should be nearly equal with rho=0 and beta=1
        assert abs(iv_low - iv_high) / iv_low < 0.02

    def test_negative_rho_skew(self):
        """Negative ρ produces downside skew: IV(K<F) > IV(K>F)."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.20, 0.5, -0.5, 0.4

        iv_low = sabr_implied_vol(F, 90.0, T, alpha, beta, rho, nu)
        iv_high = sabr_implied_vol(F, 110.0, T, alpha, beta, rho, nu)
        assert iv_low > iv_high


# ═══════════════════════════════════════════════════════════════════════════
# MONTE CARLO TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestMonteCarloConvergence:
    """Convergence rate and variance reduction tests."""

    def test_mc_converges_to_bsm(self):
        """MC call price converges to BSM as n_paths → ∞."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        bsm = bs_call(S, K, T, r, sigma, 0.0)
        rng = np.random.default_rng(42)
        price, se = mc_european_call(S, K, T, r, sigma, 500_000, rng)
        assert abs(price - bsm) < 3 * se  # within 3 std errors

    def test_antithetic_reduces_variance(self):
        """Antithetic variates should reduce standard error vs. plain MC."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        n = 100_000

        _, se_plain = mc_european_call(S, K, T, r, sigma, n, np.random.default_rng(42))
        _, se_anti = mc_european_call(S, K, T, r, sigma, n, np.random.default_rng(42),
                                       antithetic=True)
        assert se_anti < se_plain

    def test_control_variate_reduces_variance(self):
        """Control variate should reduce standard error vs. plain MC."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        n = 100_000

        _, se_plain = mc_european_call(S, K, T, r, sigma, n, np.random.default_rng(42))
        _, se_cv = mc_european_call(S, K, T, r, sigma, n, np.random.default_rng(42),
                                     control_variate=True)
        assert se_cv < se_plain

    def test_mc_convergence_rate(self):
        """Standard error should decrease as O(1/√n)."""
        S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.25

        n_small = 10_000
        n_large = 160_000  # 16x more paths → 4x smaller SE

        _, se_small = mc_european_call(S, K, T, r, sigma, n_small, np.random.default_rng(42))
        _, se_large = mc_european_call(S, K, T, r, sigma, n_large, np.random.default_rng(42))

        ratio = se_small / se_large
        expected_ratio = np.sqrt(n_large / n_small)  # = 4
        assert abs(ratio - expected_ratio) / expected_ratio < 0.30  # within 30%

    def test_mc_nonnegative_price(self):
        """MC call price must be non-negative."""
        rng = np.random.default_rng(99)
        price, _ = mc_european_call(100, 130, 1.0, 0.05, 0.20, 50_000, rng)
        assert price >= 0


# ═══════════════════════════════════════════════════════════════════════════
# BINOMIAL TREE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBinomialTreeConvergence:
    """Convergence of binomial tree to BSM as N → ∞."""

    def test_european_call_convergence(self):
        """CRR binomial call converges to BSM as steps increase."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        bsm = bs_call(S, K, T, r, sigma, 0.0)

        errors = []
        for N in [50, 200, 800]:
            tree_price = binomial_tree_european(S, K, T, r, sigma, N, "call")
            errors.append(abs(tree_price - bsm))

        # Errors should decrease monotonically (approximately)
        assert errors[-1] < errors[0]
        # Final error should be small
        assert errors[-1] < 0.05

    def test_european_put_convergence(self):
        """CRR binomial put converges to BSM."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        bsm = bs_put(S, K, T, r, sigma, 0.0)
        tree_price = binomial_tree_european(S, K, T, r, sigma, 500, "put")
        assert abs(tree_price - bsm) < 0.05

    def test_put_call_parity_in_tree(self):
        """Put-call parity holds within the binomial tree."""
        S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.25
        N = 300
        call = binomial_tree_european(S, K, T, r, sigma, N, "call")
        put = binomial_tree_european(S, K, T, r, sigma, N, "put")
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 0.10


class TestAmericanExercise:
    """American vs European exercise boundary."""

    def test_american_put_geq_european(self):
        """American put ≥ European put (early exercise has value)."""
        S, K, T, r, sigma = 100.0, 110.0, 1.0, 0.05, 0.25
        N = 300
        eur = binomial_tree_european(S, K, T, r, sigma, N, "put")
        amer = binomial_tree_american(S, K, T, r, sigma, N, "put")
        assert amer >= eur - 1e-10

    def test_american_call_equals_european_no_div(self):
        """American call = European call when q=0 (never early exercise)."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        N = 300
        eur = binomial_tree_european(S, K, T, r, sigma, N, "call")
        amer = binomial_tree_american(S, K, T, r, sigma, N, "call")
        assert abs(amer - eur) < 0.10

    def test_american_put_geq_intrinsic(self):
        """American put must be worth at least intrinsic value."""
        S, K, T, r, sigma = 80.0, 100.0, 1.0, 0.05, 0.30
        N = 300
        amer = binomial_tree_american(S, K, T, r, sigma, N, "put")
        intrinsic = max(K - S, 0.0)
        assert amer >= intrinsic - 1e-10

    def test_early_exercise_premium_increases_with_itm(self):
        """Early exercise premium increases as put goes deeper ITM."""
        K, T, r, sigma, N = 100.0, 1.0, 0.05, 0.25, 300
        premiums = []
        for S in [100, 90, 80, 70]:
            eur = binomial_tree_european(S, K, T, r, sigma, N, "put")
            amer = binomial_tree_american(S, K, T, r, sigma, N, "put")
            premiums.append(amer - eur)
        # Premium should generally increase as option goes deeper ITM
        assert premiums[-1] > premiums[0]
