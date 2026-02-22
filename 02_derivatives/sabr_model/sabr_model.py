"""
SABR Stochastic Volatility Model
===================================
The SABR (Stochastic Alpha Beta Rho) model — Hagan et al. (2002) —
is the industry standard for volatility surfaces in interest rate
derivatives (swaptions, caps, floors) and FX options.

SABR dynamics:
  dF = σ · Fᵝ · dW₁
  dσ = ν · σ · dW₂
  dW₁ · dW₂ = ρ · dt

Parameters:
  F   — forward rate / forward price
  α   — initial volatility (σ(0)); scale of implied vol
  β   — CEV exponent ∈ [0,1]:
          β=0 → normal SABR (basis-point vol; used for rates near zero)
          β=1 → log-normal SABR (same backbone as BSM)
          β=0.5 → square-root diffusion (CIR backbone)
  ρ   — spot-vol correlation (negative → vol skew, like Heston)
  ν   — vol-of-vol (controls smile curvature)

Why SABR over Heston for rates?
  - SABR gives a closed-form approximate implied vol formula (Hagan 2002)
    — no numerical integration needed → fast calibration
  - β parameter handles the "backbone" of the vol curve
  - Industry standard: every rates desk in every major bank uses SABR
  - Directly observable from cap/floor and swaption market quotes

SABR implied vol formula (Hagan et al. 2002, Eq. 2.17b):
  σ_BSM(K,F) ≈ A(F,K) · B(F,K) · C(F,K)

  where A, B, C are functions of (F, K, T, α, β, ρ, ν)
  derived from singular perturbation of the Kolmogorov PDE.

Limitations of the Hagan formula:
  - Approximation error for long-dated options
  - Can produce negative densities for low β (SABR arbitrage)
  - Fix: Antonov-Konikov-Spector (2015) or SABR with absorbing barrier

References:
  - Hagan, P.S. et al. (2002). Managing Smile Risk. Wilmott Magazine, 84–108.
  - Obloj, J. (2008). Fine-Tune Your SABR Smile. Wilmott Magazine.
  - West, G. (2005). Calibration of the SABR Model in Illiquid Markets.
    Applied Mathematical Finance, 12(4), 371–385.
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 27.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class SABRParams:
    """
    SABR model parameters.

    Attributes
    ----------
    alpha : float   Initial vol level (> 0). ATM vol ≈ α·F^(β-1).
    beta  : float   CEV backbone exponent ∈ [0, 1].
    rho   : float   Spot-vol correlation ∈ (-1, 1). Negative → skew.
    nu    : float   Vol-of-vol (≥ 0). Larger → more pronounced smile.
    """
    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self):
        if not (0 <= self.beta <= 1):
            raise ValueError(f"beta must be in [0,1], got {self.beta}")
        if not (-1 < self.rho < 1):
            raise ValueError(f"rho must be in (-1,1), got {self.rho}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.nu < 0:
            raise ValueError(f"nu must be non-negative, got {self.nu}")


# ---------------------------------------------------------------------------
# SABR implied vol formula (Hagan et al. 2002)
# ---------------------------------------------------------------------------

def sabr_implied_vol(
    F: float, K: float, T: float, params: SABRParams,
    vol_type: str = "lognormal"
) -> float:
    """
    Hagan et al. (2002) approximate implied volatility for SABR.

    Parameters
    ----------
    F : float       Forward price / forward rate.
    K : float       Strike.
    T : float       Time to expiry in years.
    params          SABRParams instance.
    vol_type : str  'lognormal' (returns σ_BSM) or 'normal' (returns σ_N in bp).

    Returns
    -------
    float   Approximate BSM implied vol (or normal vol if vol_type='normal').

    Notes
    -----
    The ATM case (F=K) requires a limiting form — handled separately
    to avoid division by zero.
    """
    α, β, ρ, ν = params.alpha, params.beta, params.rho, params.nu

    if T <= 0:
        return max(α * F ** (β - 1), 0.0)

    # ── ATM case (F = K, or very close) ──────────────────────────────
    if abs(F - K) < 1e-10 * F:
        FK_mid = F
        A_atm = α / (FK_mid ** (1 - β))
        B_atm = 1 + (
            ((1 - β) ** 2 / 24) * α ** 2 / (FK_mid ** (2 - 2 * β)) +
            (ρ * β * ν * α) / (4 * FK_mid ** (1 - β)) +
            ((2 - 3 * ρ ** 2) / 24) * ν ** 2
        ) * T
        return A_atm * B_atm

    # ── General case (F ≠ K) ─────────────────────────────────────────
    FK = F * K                         # geometric mean
    FK_mid = FK ** ((1 - β) / 2)      # midpoint factor

    # Logarithm and z variable
    log_FK = math.log(F / K)
    z = (ν / α) * FK_mid * log_FK

    # χ(z): maps z to the "chi" variable
    # χ = ln[(√(1 - 2ρz + z²) + z - ρ) / (1 - ρ)]
    sqrt_term = math.sqrt(max(1 - 2 * ρ * z + z ** 2, 1e-12))
    chi_z = math.log((sqrt_term + z - ρ) / (1 - ρ))

    # Numerator expansion (Eq. 2.17b, numerator)
    # A = α / [FK_mid · (1 + ((1-β)²/24)·log²(F/K) + ((1-β)⁴/1920)·log⁴(F/K))]
    A = α / (FK_mid * (
        1 +
        ((1 - β) ** 2 / 24) * log_FK ** 2 +
        ((1 - β) ** 4 / 1920) * log_FK ** 4
    ))

    # z/χ(z) term: if z≈0, use L'Hôpital (→1)
    if abs(chi_z) < 1e-12:
        z_over_chi = 1.0
    else:
        z_over_chi = z / chi_z

    # Correction term B: time-value expansion
    B = 1 + (
        ((1 - β) ** 2 / 24) * α ** 2 / FK_mid ** 2 +
        (ρ * β * ν * α) / (4 * FK_mid) +
        ((2 - 3 * ρ ** 2) / 24) * ν ** 2
    ) * T

    sigma_bsm = A * z_over_chi * B
    return max(sigma_bsm, 1e-6)  # floor at near-zero


def sabr_normal_vol(F: float, K: float, T: float, params: SABRParams) -> float:
    """
    Normal (basis-point) implied vol from SABR.
    Used when rates are near or below zero (Bachelier framework).

    σ_N ≈ σ_BSM · F^β · (1 + ...) for F ≈ K
    More precisely: σ_N = σ_BSM · F (for β=1, log-normal → normal conversion)
    """
    σ_bsm = sabr_implied_vol(F, K, T, params)
    # Approximate normal vol: σ_N ≈ σ_BSM · F · N'(d) / [call_price_derivative]
    # Simple approximation: σ_N ≈ σ_BSM · F for ATM
    return σ_bsm * math.sqrt(F * K)  # geometric mean approximation


# ---------------------------------------------------------------------------
# SABR option pricing
# ---------------------------------------------------------------------------

def sabr_option_price(
    F: float, K: float, T: float, r: float,
    params: SABRParams, option_type: str = "call"
) -> float:
    """
    Price a European option using SABR-implied vol fed into BSM.

    Step 1: Compute SABR implied vol σ(K,T)
    Step 2: Price via BSM using σ(K,T)

    This is the market-standard approach — SABR is a vol model, not a direct pricer.
    """
    sigma = sabr_implied_vol(F, K, T, params)
    if sigma <= 0 or T <= 0:
        intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
        return math.exp(-r * T) * intrinsic

    # BSM with forward price (discount factor already applied in F)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    if option_type == "call":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# SABR calibration
# ---------------------------------------------------------------------------

class SABRCalibrator:
    """
    Calibrate SABR parameters to a strip of market implied vols.

    Given quotes σ_mkt(Kᵢ) for a set of strikes at a fixed maturity T,
    find (α, β, ρ, ν) that minimise Σ (σ_SABR(Kᵢ) - σ_mkt(Kᵢ))².

    Practice note:
      β is often fixed by convention for the asset class:
        Rates (swaptions): β = 0 or β = 0.5
        FX:                β = 1
        Equity:            β = 1
      This reduces the optimisation to 3 free parameters (α, ρ, ν),
      making calibration faster and more stable.

    Two-step calibration (recommended):
      Step 1: Fix β. Use ATM vol to get initial α (closed-form from ATM formula).
      Step 2: Calibrate (α, ρ, ν) to full strike strip.
    """

    def __init__(
        self,
        F: float,
        T: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,   # market BSM implied vols at each strike
        beta: float = 0.5,         # fixed β (common choice)
        weights: Optional[np.ndarray] = None,
    ):
        self.F = F
        self.T = T
        self.strikes = np.asarray(strikes)
        self.market_vols = np.asarray(market_vols)
        self.beta = beta
        self.weights = weights if weights is not None else np.ones(len(strikes))
        # Vega-weight: ATM options weighted more (they are most liquid)
        # In practice: w_i = vega_i / Σ vega_j

    def _alpha_from_atm(self, rho: float, nu: float, atm_vol: float) -> float:
        """
        Solve the cubic equation for α given ATM implied vol and (ρ, ν).

        From SABR ATM formula:
        σ_ATM = α/F^(1-β) · [1 + ((1-β)²α²/24F^(2-2β) + ρβνα/4F^(1-β) + (2-3ρ²)ν²/24) T]

        This is a cubic in α — solved via numpy polynomial roots.
        Returns the smallest positive real root.
        """
        F, T, β = self.F, self.T, self.beta
        F_mid = F ** (1 - β)

        # Rearrange to polynomial: c3·α³ + c2·α² + c1·α + c0 = 0
        c3 = ((1 - self.beta) ** 2 * T) / (24 * F_mid ** 2)
        c2 = (rho * self.beta * nu * T) / (4 * F_mid)
        c1 = 1 + ((2 - 3 * rho ** 2) / 24) * nu ** 2 * T
        c0 = -atm_vol * F_mid

        roots = np.roots([c3, c2, c1, c0])
        real_positive = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
        if not real_positive:
            return atm_vol * F_mid  # fallback
        return min(real_positive)

    def _loss(self, params_reduced: np.ndarray) -> float:
        """Loss function: weighted sum of squared vol errors."""
        alpha, rho, nu = params_reduced
        if alpha <= 0 or not (-0.999 < rho < 0.999) or nu < 0:
            return 1e10
        try:
            p = SABRParams(alpha=alpha, beta=self.beta, rho=rho, nu=nu)
            model_vols = np.array([
                sabr_implied_vol(self.F, K, self.T, p) for K in self.strikes
            ])
            diff = model_vols - self.market_vols
            return float(np.sum(self.weights * diff ** 2))
        except Exception:
            return 1e10

    def calibrate(self) -> SABRParams:
        """
        Two-step calibration:
          1. Get α₀ from ATM vol using the cubic formula.
          2. Optimise (α, ρ, ν) over the full strike strip.

        Returns calibrated SABRParams.
        """
        # Step 1: ATM initial α
        atm_idx = np.argmin(np.abs(self.strikes - self.F))
        atm_vol = self.market_vols[atm_idx]
        alpha0 = self._alpha_from_atm(rho=-0.3, nu=0.3, atm_vol=atm_vol)

        x0 = np.array([alpha0, -0.3, 0.3])

        result = minimize(
            self._loss, x0, method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-9},
        )
        alpha, rho, nu = result.x
        return SABRParams(
            alpha=max(alpha, 1e-6),
            beta=self.beta,
            rho=max(-0.999, min(0.999, rho)),
            nu=max(nu, 0.0),
        )


# ---------------------------------------------------------------------------
# Vol surface generation
# ---------------------------------------------------------------------------

def sabr_vol_surface(
    F: float,
    T_range: np.ndarray,
    moneyness_range: np.ndarray,
    base_params: SABRParams,
) -> dict:
    """
    Generate a SABR implied vol surface across (T, K) grid.

    In practice, α is re-calibrated at each maturity. Here we
    demonstrate the shape for fixed parameters.

    Parameters
    ----------
    F : float              Current forward price.
    T_range : array        Array of maturities.
    moneyness_range : array  K/F ratios (e.g. np.linspace(0.7, 1.3, 13)).
    base_params : SABRParams  Parameters (α may be scaled by √T for realism).

    Returns
    -------
    dict with keys 'strikes', 'maturities', 'vols' (2D array).
    """
    strikes = F * moneyness_range
    surface = np.zeros((len(T_range), len(strikes)))

    for i, T in enumerate(T_range):
        # Scale α with √T (a common market-consistent adjustment)
        scaled_alpha = base_params.alpha / math.sqrt(T) * math.sqrt(T_range[0])
        p = SABRParams(alpha=scaled_alpha, beta=base_params.beta,
                       rho=base_params.rho, nu=base_params.nu)
        for j, K in enumerate(strikes):
            surface[i, j] = sabr_implied_vol(F, K, T, p)

    return {"strikes": strikes, "maturities": T_range, "vols": surface}


# ---------------------------------------------------------------------------
# Density check: detect SABR arbitrage
# ---------------------------------------------------------------------------

def sabr_density_check(
    F: float, T: float, params: SABRParams,
    n_points: int = 100
) -> dict:
    """
    Verify that the SABR smile does not imply negative probability density.
    Negative density = arbitrage (butterfly spread with negative value).

    Method: Breeden-Litzenberger (1978)
      p(K) = e^(rT) · ∂²C/∂K²

    For no-arbitrage: p(K) ≥ 0 for all K.

    Returns dict with min density, arbitrage flag, and problematic strikes.
    """
    K_range = np.linspace(F * 0.5, F * 1.5, n_points)
    r = 0.0  # simplified: zero rate for density check
    dK = K_range[1] - K_range[0]

    prices = np.array([sabr_option_price(F, K, T, r, params) for K in K_range])

    # Second derivative via finite difference
    density = np.diff(np.diff(prices)) / dK ** 2

    min_density = float(density.min())
    arbitrage_strikes = K_range[1:-1][density < -1e-6]

    return {
        "min_density": round(min_density, 6),
        "has_arbitrage": len(arbitrage_strikes) > 0,
        "n_negative_density_points": len(arbitrage_strikes),
        "arbitrage_strikes": arbitrage_strikes.tolist()[:5],  # first 5
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 64)
    print("  SABR Stochastic Volatility Model — Hagan et al. (2002)")
    print("  Industry standard for swaptions, caps, FX vol surfaces")
    print("═" * 64)

    # Typical swaption market params (rates context, β=0.5)
    F = 0.04   # 4% forward rate
    T = 1.0    # 1-year expiry

    # Representative parameters: slight skew (ρ < 0), moderate vol-of-vol
    params = SABRParams(alpha=0.02, beta=0.5, rho=-0.25, nu=0.40)

    print(f"\n  Forward rate: {F:.2%}  |  T = {T}Y")
    print(f"  α={params.alpha}  β={params.beta}  ρ={params.rho}  ν={params.nu}")

    print(f"\n── SABR Implied Vol Smile (T=1Y) ──")
    print(f"  {'K':>8} {'K/F':>8} {'SABR Vol':>12} {'Normal Vol (bp)':>18}")
    print("  " + "─" * 50)
    for k_mult in [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]:
        K = F * k_mult
        sigma_bsm = sabr_implied_vol(F, K, T, params)
        sigma_n = sigma_bsm * math.sqrt(F * K) * 10000  # in bps
        print(f"  {K:>8.4f} {k_mult:>8.2f} {sigma_bsm:>12.4%} {sigma_n:>18.2f}")

    # Effect of ρ on skew
    print(f"\n── Effect of ρ on Vol Skew (α=0.02, β=0.5, ν=0.40) ──")
    print(f"  {'ρ':>6}  {'K=0.8F':>12}  {'K=F':>12}  {'K=1.2F':>12}  Skew")
    print("  " + "─" * 56)
    for rho in [-0.5, -0.25, 0.0, 0.25, 0.5]:
        p = SABRParams(alpha=0.02, beta=0.5, rho=rho, nu=0.40)
        v_otm = sabr_implied_vol(F, F * 0.8, T, p)
        v_atm = sabr_implied_vol(F, F, T, p)
        v_itm = sabr_implied_vol(F, F * 1.2, T, p)
        skew = v_otm - v_itm
        print(f"  {rho:>6.2f}  {v_otm:>12.4%}  {v_atm:>12.4%}  {v_itm:>12.4%}  {skew:>+.4%}")

    # Effect of ν on smile curvature
    print(f"\n── Effect of ν on Smile Curvature (α=0.02, β=0.5, ρ=-0.25) ──")
    print(f"  {'ν':>6}  {'K=0.8F':>12}  {'K=F':>12}  {'K=1.2F':>12}  Curvature")
    print("  " + "─" * 60)
    for nu in [0.10, 0.25, 0.40, 0.60, 0.80]:
        p = SABRParams(alpha=0.02, beta=0.5, rho=-0.25, nu=nu)
        v_otm = sabr_implied_vol(F, F * 0.8, T, p)
        v_atm = sabr_implied_vol(F, F, T, p)
        v_itm = sabr_implied_vol(F, F * 1.2, T, p)
        curv = v_otm + v_itm - 2 * v_atm  # smile curvature
        print(f"  {nu:>6.2f}  {v_otm:>12.4%}  {v_atm:>12.4%}  {v_itm:>12.4%}  {curv:>+.4%}")

    # Calibration demo
    print(f"\n── Calibration Demo ──")
    print(f"  Generate market vols with known true params, then recover them.")
    true_params = SABRParams(alpha=0.025, beta=0.5, rho=-0.30, nu=0.45)
    strikes = F * np.array([0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30])
    mkt_vols = np.array([sabr_implied_vol(F, K, T, true_params) for K in strikes])

    cal = SABRCalibrator(F=F, T=T, strikes=strikes, market_vols=mkt_vols, beta=0.5)
    fitted = cal.calibrate()

    print(f"\n  {'Param':<8} {'True':>12} {'Fitted':>12} {'Error':>10}")
    print("  " + "─" * 46)
    for name, true_v, fit_v in [
        ("alpha", true_params.alpha, fitted.alpha),
        ("rho",   true_params.rho,   fitted.rho),
        ("nu",    true_params.nu,    fitted.nu),
    ]:
        err = abs(fit_v - true_v) / abs(true_v)
        print(f"  {name:<8} {true_v:>12.6f} {fit_v:>12.6f} {err:>10.4%}")

    # No-arbitrage check
    print(f"\n── No-Arbitrage Density Check ──")
    density_result = sabr_density_check(F, T, true_params)
    for k, v in density_result.items():
        print(f"  {k}: {v}")
