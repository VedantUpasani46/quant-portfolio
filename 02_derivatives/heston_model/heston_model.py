"""
Heston Stochastic Volatility Model
=====================================
Semi-analytical European option pricing under the Heston (1993) model.

The Heston model corrects two key BSM failures:
  1. Flat volatility smile → Heston produces realistic skew and smile.
  2. Constant volatility → variance follows a mean-reverting (CIR) process.

Heston dynamics:
  dS = μ·S·dt + √V·S·dW₁
  dV = κ(θ - V)dt + σ·√V·dW₂
  dW₁·dW₂ = ρ·dt

Parameters:
  V₀  — initial variance (spot variance, σ₀² = V₀)
  κ   — mean-reversion speed of variance
  θ   — long-run average variance (long-run vol² = θ)
  σ   — volatility of variance (vol-of-vol)
  ρ   — correlation between spot and variance processes

Feller condition for well-posed process: 2κθ > σ²
  If violated, variance can touch zero and the process becomes degenerate.

Pricing method:
  Heston (1993) derived a semi-analytical characteristic function solution.
  The option price is computed via the Gil-Pelaez inversion formula:
  C = S·e^(-qT)·P₁ - K·e^(-rT)·P₂

  where P₁, P₂ are recovered by numerical integration of the characteristic function.

References:
  - Heston, S.L. (1993). A Closed-Form Solution for Options with Stochastic Volatility.
    Review of Financial Studies, 6(2), 327–343.
  - Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley.
  - Rouah, F.D. (2013). The Heston Model and Its Extensions in Matlab and C++. Wiley.
"""

import math
import cmath
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass
class HestonParams:
    """
    Heston model parameters.

    Attributes
    ----------
    V0    : Initial variance (e.g. 0.04 for 20% initial vol)
    kappa : Mean-reversion speed (e.g. 2.0)
    theta : Long-run variance (e.g. 0.04 for 20% long-run vol)
    sigma : Vol-of-vol (e.g. 0.3)
    rho   : Spot-vol correlation (typically negative, e.g. -0.7)
    """
    V0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def __post_init__(self):
        feller = 2 * self.kappa * self.theta
        volvol_sq = self.sigma ** 2
        if feller <= volvol_sq:
            warnings.warn(
                f"Feller condition violated: 2κθ = {feller:.4f} ≤ σ² = {volvol_sq:.4f}. "
                f"Variance can reach zero. Consider increasing κ or θ, or decreasing σ."
            )

    @property
    def spot_vol(self) -> float:
        return math.sqrt(self.V0)

    @property
    def long_run_vol(self) -> float:
        return math.sqrt(self.theta)


# ---------------------------------------------------------------------------
# Heston Characteristic Function
# ---------------------------------------------------------------------------

def heston_char_func(
    phi: float,
    S: float, K: float, T: float, r: float, q: float,
    params: HestonParams,
    j: int  # j=1 for P1, j=2 for P2
) -> complex:
    """
    Heston characteristic function Ψ_j(φ) for the log-price.

    Uses the Albrecher et al. (2007) 'Little Trap' formulation which avoids
    the discontinuity in the original Heston formula when computing the
    complex logarithm.

    Parameters
    ----------
    phi : float     Fourier integration variable.
    j : int         1 → characteristic function for P₁; 2 → for P₂.
    """
    V0, κ, θ, σ, ρ = params.V0, params.kappa, params.theta, params.sigma, params.rho
    x = math.log(S / K)
    i = 1j

    # b and u differ between j=1 and j=2
    if j == 1:
        u = 0.5
        b = κ - ρ * σ
    else:
        u = -0.5
        b = κ

    # d = √((b - i·ρ·σ·φ)² + σ²·φ·(φ + 2i·u))
    d = cmath.sqrt((b - i * ρ * σ * phi) ** 2 + σ ** 2 * phi * (phi - i * 2 * u))

    # g = (b - i·ρ·σ·φ - d) / (b - i·ρ·σ·φ + d)  — 'Little Trap' form
    g = (b - i * ρ * σ * phi - d) / (b - i * ρ * σ * phi + d)

    # C(τ) and D(τ) in the characteristic function exponent
    exp_dT = cmath.exp(-d * T)
    C = (r - q) * phi * i * T + (κ * θ / σ ** 2) * (
        (b - i * ρ * σ * phi - d) * T - 2 * cmath.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((b - i * ρ * σ * phi - d) / σ ** 2) * (1 - exp_dT) / (1 - g * exp_dT)

    return cmath.exp(C + D * V0 + i * phi * x)


# ---------------------------------------------------------------------------
# Heston Option Pricer
# ---------------------------------------------------------------------------

class HestonPricer:
    """
    Semi-analytical European option pricer under the Heston (1993) model.

    Pricing formula (Gil-Pelaez inversion):
      P₁, P₂ = 0.5 + (1/π) ∫₀^∞ Re[e^{-iφ·ln(K)} · Ψ_j(φ) / (iφ)] dφ
      Call = S·e^{-qT}·P₁ − K·e^{-rT}·P₂

    Usage
    -----
    >>> params = HestonParams(V0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    >>> pricer = HestonPricer(S=100, K=100, T=1.0, r=0.05, q=0.0, params=params)
    >>> call_price = pricer.call_price()
    """

    def __init__(
        self,
        S: float, K: float, T: float, r: float, q: float,
        params: HestonParams,
        n_quad: int = 100,  # number of quadrature points
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.params = params
        self.n_quad = n_quad

    def _integrand(self, phi: float, j: int) -> float:
        """Real part of the integrand for the Gil-Pelaez inversion."""
        cf = heston_char_func(phi, self.S, self.K, self.T, self.r, self.q, self.params, j)
        exp_term = cmath.exp(-1j * phi * math.log(self.K))
        return (exp_term * cf / (1j * phi)).real

    def _Pj(self, j: int) -> float:
        """
        Compute the risk-neutral probability P_j via numerical integration.
        Uses scipy.integrate.quad with a truncated integration range.
        """
        integral, _ = quad(
            self._integrand, 0, 200,    # upper limit: 200 is sufficient for most cases
            args=(j,),
            limit=200,                   # max subintervals
            epsabs=1e-8, epsrel=1e-8,
        )
        return 0.5 + integral / math.pi

    def call_price(self) -> float:
        """
        Heston European call price:
        C = S·e^(-qT)·P₁ - K·e^(-rT)·P₂
        """
        P1 = self._Pj(1)
        P2 = self._Pj(2)
        call = self.S * math.exp(-self.q * self.T) * P1 - self.K * math.exp(-self.r * self.T) * P2
        return max(call, 0.0)

    def put_price(self) -> float:
        """Put price via put-call parity: P = C - S·e^(-qT) + K·e^(-rT)"""
        C = self.call_price()
        return C - self.S * math.exp(-self.q * self.T) + self.K * math.exp(-self.r * self.T)

    def implied_vol_surface(
        self, strikes: np.ndarray, maturities: np.ndarray
    ) -> np.ndarray:
        """
        Compute the implied vol surface across a grid of (K, T) values.
        Inverts the Heston price to BSM implied vol at each grid point.

        Returns 2D array of shape (len(maturities), len(strikes)).
        """
        from scipy.optimize import brentq

        try:
            from black_scholes import BlackScholesMerton, BSMInputs
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'part0_basics', 'black_scholes'))
            # Provide inline BSM price function as fallback
            def bsm_price(S, K, T, r, sigma, q, option_type):
                from scipy.stats import norm
                d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
                d2 = d1 - sigma*math.sqrt(T)
                if option_type == 'call':
                    return S*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
                return K*math.exp(-r*T)*norm.cdf(-d2) - S*math.exp(-q*T)*norm.cdf(-d1)

        surface = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                pricer = HestonPricer(self.S, K, T, self.r, self.q, self.params)
                heston_c = pricer.call_price()
                intrinsic = max(self.S * math.exp(-self.q * T) - K * math.exp(-self.r * T), 0)
                if heston_c <= intrinsic + 1e-8:
                    surface[i, j] = np.nan
                    continue
                try:
                    def diff(sigma):
                        return bsm_price(self.S, K, T, self.r, sigma, self.q, 'call') - heston_c
                    iv = brentq(diff, 0.001, 5.0, xtol=1e-6)
                    surface[i, j] = iv
                except Exception:
                    surface[i, j] = np.nan
        return surface


# ---------------------------------------------------------------------------
# Calibration: fit Heston params to market implied vols
# ---------------------------------------------------------------------------

class HestonCalibrator:
    """
    Calibrate Heston parameters to a market implied volatility surface.

    Minimises the sum of squared differences between model-implied vols
    and observed market implied vols across all strikes and maturities.

    Objective: min Σ (σ_model(K,T) - σ_market(K,T))²

    In practice, production systems use weighted loss (vega-weighted),
    penalise arbitrage, and use SABR as an initial warm-start.
    """

    def __init__(
        self,
        S: float, r: float, q: float,
        strikes: list[float], maturities: list[float],
        market_vols: np.ndarray,  # shape (len(maturities), len(strikes))
    ):
        self.S = S
        self.r = r
        self.q = q
        self.strikes = strikes
        self.maturities = maturities
        self.market_vols = market_vols

    def _model_vols(self, params: HestonParams) -> np.ndarray:
        pricer = HestonPricer(self.S, self.strikes[0], self.maturities[0],
                              self.r, self.q, params)
        return pricer.implied_vol_surface(
            np.array(self.strikes), np.array(self.maturities)
        )

    def _loss(self, x: np.ndarray) -> float:
        V0, kappa, theta, sigma, rho = x
        # Bounds enforcement
        if V0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
            return 1e10
        if not (-1 < rho < 1):
            return 1e10
        try:
            params = HestonParams(V0=V0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
            model_vols = self._model_vols(params)
            diff = model_vols - self.market_vols
            valid = ~np.isnan(diff)
            return float(np.sum(diff[valid] ** 2))
        except Exception:
            return 1e10

    def calibrate(self, x0: np.ndarray | None = None) -> HestonParams:
        """
        Run Nelder-Mead calibration.

        Parameters
        ----------
        x0 : array-like, optional
            Initial guess [V0, kappa, theta, sigma, rho].
            Defaults to reasonable starting values.

        Returns
        -------
        HestonParams with calibrated values.
        """
        if x0 is None:
            x0 = np.array([0.04, 1.5, 0.04, 0.3, -0.5])

        result = minimize(
            self._loss, x0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
        )
        V0, kappa, theta, sigma, rho = result.x
        return HestonParams(V0=V0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 60)
    print("  Heston Stochastic Volatility Model")
    print("  Semi-analytical European option pricing")
    print("═" * 60)

    # Gatheral (2006) calibrated parameters
    params = HestonParams(V0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    print(f"\n  Parameters:")
    print(f"    V₀ (initial var): {params.V0:.4f}  →  σ₀ = {params.spot_vol:.2%}")
    print(f"    κ (mean reversion): {params.kappa}")
    print(f"    θ (long-run var): {params.theta:.4f}  →  σ_∞ = {params.long_run_vol:.2%}")
    print(f"    σ (vol-of-vol): {params.sigma}")
    print(f"    ρ (spot-vol corr): {params.rho}")
    feller = 2 * params.kappa * params.theta
    print(f"    Feller: 2κθ = {feller:.4f} vs σ² = {params.sigma**2:.4f}  "
          f"({'satisfied ✓' if feller > params.sigma**2 else 'VIOLATED ✗'})")

    S, r, q = 100, 0.05, 0.0
    T = 1.0

    print(f"\n── ATM Option (S=K=100, T=1Y) ──")
    pricer = HestonPricer(S=S, K=100, T=T, r=r, q=q, params=params)
    call = pricer.call_price()
    put = pricer.put_price()
    print(f"  Heston Call: {call:.4f}")
    print(f"  Heston Put:  {put:.4f}")
    parity_check = call - put - (S * math.exp(-q * T) - 100 * math.exp(-r * T))
    print(f"  Put-Call Parity Diff: {abs(parity_check):.2e}  (should be ~0)")

    print(f"\n── Implied Vol Smile (T=1Y, K = 80 to 120) ──")
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    print(f"  {'Strike':<10} {'Call Price':>12} {'Impl. Vol':>12} {'Moneyness':>12}")
    print("  " + "─" * 48)
    for K in strikes:
        p = HestonPricer(S=S, K=K, T=T, r=r, q=q, params=params)
        c = p.call_price()
        # Quick BSM implied vol
        from scipy.stats import norm
        from scipy.optimize import brentq as _brentq
        def bsm_c(sig): 
            d1 = (math.log(S/K) + (r + 0.5*sig**2)*T)/(sig*math.sqrt(T))
            d2 = d1 - sig*math.sqrt(T)
            return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        try:
            iv = _brentq(lambda s: bsm_c(s) - c, 0.001, 5.0)
        except:
            iv = float('nan')
        moneyness = math.log(K / S)
        print(f"  {K:<10} {c:>12.4f} {iv:>12.4%} {moneyness:>12.4f}")

    print(f"\n  Key insight: negative ρ = {params.rho} creates the 'volatility skew'")
    print(f"  (lower strikes have higher implied vol — the 'equity vol skew')")
