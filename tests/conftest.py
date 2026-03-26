"""
Shared fixtures for quant-portfolio test suite.

All fixtures use numpy.random with fixed seeds for reproducibility.
No external data dependencies — everything is synthetic.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# Random-state helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Reproducible NumPy random generator (seed=42)."""
    return np.random.default_rng(42)


@pytest.fixture
def rng_alt():
    """Second independent generator (seed=123) for cross-validation."""
    return np.random.default_rng(123)


# ---------------------------------------------------------------------------
# Market data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stock_prices(rng):
    """
    Synthetic daily stock prices for 5 assets over 2 years (~504 days).
    Generated via geometric Brownian motion with realistic parameters.
    """
    n_days = 504
    n_assets = 5
    dt = 1 / 252
    mu = np.array([0.08, 0.10, 0.06, 0.12, 0.09])
    sigma = np.array([0.20, 0.25, 0.15, 0.30, 0.22])
    S0 = np.array([100.0, 50.0, 200.0, 75.0, 150.0])

    Z = rng.standard_normal((n_days, n_assets))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    prices = S0 * np.exp(np.cumsum(log_returns, axis=0))
    prices = np.vstack([S0, prices])  # prepend initial prices

    dates = pd.bdate_range(start="2022-01-03", periods=n_days + 1)
    return pd.DataFrame(
        prices,
        index=dates[:n_days + 1],
        columns=["AAPL", "GOOG", "MSFT", "TSLA", "AMZN"],
    )


@pytest.fixture
def log_returns(stock_prices):
    """Daily log returns derived from stock_prices fixture."""
    return np.log(stock_prices / stock_prices.shift(1)).dropna()


@pytest.fixture
def correlation_matrix(rng):
    """
    Synthetic 5×5 positive-definite correlation matrix.
    Built via random Cholesky factor to guarantee PSD.
    """
    n = 5
    A = rng.standard_normal((n, n))
    M = A @ A.T
    D_inv = np.diag(1.0 / np.sqrt(np.diag(M)))
    corr = D_inv @ M @ D_inv
    np.fill_diagonal(corr, 1.0)
    return corr


@pytest.fixture
def covariance_matrix(correlation_matrix):
    """Covariance matrix from correlation_matrix with realistic vols."""
    vols = np.array([0.20, 0.25, 0.15, 0.30, 0.22])
    D = np.diag(vols)
    return D @ correlation_matrix @ D


# ---------------------------------------------------------------------------
# Fixed income fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def yield_curve_rates():
    """
    Synthetic par yield curve: tenors (years) and corresponding rates.
    Upward-sloping with slight concavity, typical of a normal environment.
    """
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    rates = np.array([
        0.0425, 0.0440, 0.0455, 0.0465, 0.0470,
        0.0478, 0.0482, 0.0485, 0.0488, 0.0490, 0.0492,
    ])
    return tenors, rates


@pytest.fixture
def bond_params():
    """Standard bond parameters for testing."""
    return {
        "face_value": 1000.0,
        "coupon_rate": 0.05,
        "maturity_years": 10,
        "frequency": 2,  # semi-annual
        "ytm": 0.045,
    }


# ---------------------------------------------------------------------------
# Option pricing fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bs_params():
    """Standard Black-Scholes parameters."""
    return {
        "S": 100.0,    # spot
        "K": 100.0,    # strike (ATM)
        "T": 1.0,      # time to expiry (years)
        "r": 0.05,     # risk-free rate
        "sigma": 0.20, # volatility
        "q": 0.02,     # dividend yield
    }


@pytest.fixture
def option_chain():
    """
    Synthetic option chain: strikes from 80 to 120 with ATM at 100.
    Returns dict with strikes, market implied vols (smile shape).
    """
    strikes = np.arange(80, 121, 2.5, dtype=float)
    # Quadratic smile centred at ATM
    atm_vol = 0.20
    skew = 0.0005
    ivs = atm_vol + skew * (strikes - 100.0) ** 2
    return {"strikes": strikes, "ivs": ivs, "S": 100.0, "T": 1.0, "r": 0.05}


# ---------------------------------------------------------------------------
# Credit risk fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def credit_params():
    """Standard credit-risk parameters."""
    return {
        "notional": 10_000_000.0,
        "recovery_rate": 0.40,
        "hazard_rate": 0.02,
        "maturity_years": 5,
        "risk_free_rate": 0.03,
    }


# ---------------------------------------------------------------------------
# Portfolio fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def portfolio_params(covariance_matrix):
    """Parameters for mean-variance optimisation."""
    expected_returns = np.array([0.08, 0.10, 0.06, 0.12, 0.09])
    return {
        "expected_returns": expected_returns,
        "cov_matrix": covariance_matrix,
        "risk_free_rate": 0.03,
        "n_assets": 5,
    }


# ---------------------------------------------------------------------------
# Utility helpers (not fixtures – importable functions)
# ---------------------------------------------------------------------------

def assert_psd(matrix, tol=1e-10):
    """Assert that a matrix is positive semi-definite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert np.all(eigenvalues >= -tol), (
        f"Matrix not PSD: min eigenvalue = {eigenvalues.min():.2e}"
    )


def assert_bounded(value, lo, hi, label="value"):
    """Assert lo <= value <= hi."""
    assert lo <= value <= hi, f"{label} = {value} not in [{lo}, {hi}]"
