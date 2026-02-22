"""
Dupire Local Volatility Model
================================
Extracts the local volatility surface σ_loc(S, t) from a market
implied volatility surface, using Dupire's (1994) formula.

Key insight:
  BSM assumes a SINGLE constant vol σ for all strikes and maturities.
  This contradicts the observed implied vol surface: different (K, T) pairs
  have different implied vols (the "vol smile").

  Local vol (Dupire 1994) is a generalisation: under the model
    dS = r·S·dt + σ_loc(S, t)·S·dW
  the local vol function σ_loc(S, t) is uniquely determined by the market
  implied vol surface — there is exactly ONE local vol surface consistent
  with any given set of option prices.

Dupire's formula (derived from the forward Kolmogorov PDE):

  σ_loc²(K, T) = [∂C/∂T + (r-q)K·∂C/∂K + q·C] /
                  [½·K²·∂²C/∂K²]

where C(K, T) is the market call price surface.

In practice:
  1. Fit a smooth parametric surface to market implied vols (e.g. using SABR per maturity)
  2. Numerically differentiate to get ∂C/∂T, ∂C/∂K, ∂²C/∂K²
  3. Apply Dupire formula

Why local vol matters:
  - Uniquely consistent with any market smile (unlike BSM or Heston)
  - Used extensively for exotic pricing (barrier, autocall, CLN)
  - Required by regulators for model validation benchmarks
  - Foundation for local-stochastic vol models (Heston-SLV, SABR-LV)

References:
  - Dupire, B. (1994). Pricing with a Smile. Risk Magazine, 7(1), 18–20.
  - Gatheral, J. (2006). The Volatility Surface. Wiley. Ch. 1–2.
  - Derman, E. & Kani, I. (1994). Riding on a Smile. Risk Magazine, 7(2), 32–39.
  - Andreasen, J. & Huge, B. (2011). Volatility Interpolation. Risk Magazine.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm


# ---------------------------------------------------------------------------
# BSM helpers
# ---------------------------------------------------------------------------

def bsm_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Standard BSM call price."""
    if sigma <= 0 or T <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bsm_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """BSM vega: ∂C/∂σ."""
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


# ---------------------------------------------------------------------------
# Implied vol surface
# ---------------------------------------------------------------------------

@dataclass
class ImpliedVolSurface:
    """
    A smooth implied volatility surface interpolated over a (K, T) grid.

    Internally uses a 2D bicubic spline on the implied vol matrix.
    This is then differentiated to extract the call price surface
    and its derivatives for the Dupire formula.
    """
    strikes: np.ndarray           # shape (n_K,)
    maturities: np.ndarray        # shape (n_T,)
    implied_vols: np.ndarray      # shape (n_T, n_K)
    S0: float                     # current spot
    r: float                      # risk-free rate
    q: float = 0.0                # dividend yield

    def __post_init__(self):
        """Fit bicubic spline to the implied vol surface."""
        # Validate surface has no negatives or NaNs
        valid_mask = (self.implied_vols > 0) & ~np.isnan(self.implied_vols)
        if not valid_mask.all():
            # Fill NaN/zero with nearest neighbour
            from scipy.ndimage import label
            self.implied_vols = np.where(
                valid_mask, self.implied_vols,
                np.nanmedian(self.implied_vols)
            )
        # Bicubic spline: smooth interpolation over (T, K) grid
        # kx=ky=3 → cubic; smoothing=0 → interpolating spline
        self._spline = RectBivariateSpline(
            self.maturities, self.strikes,
            self.implied_vols, kx=3, ky=3, s=0
        )

    def vol(self, K: float, T: float) -> float:
        """Interpolated implied vol at (K, T)."""
        return max(float(self._spline(np.array([T]), np.array([K]))[0, 0]), 1e-4)

    def call_price(self, K: float, T: float) -> float:
        """BSM call price using interpolated implied vol."""
        sigma = self.vol(K, T)
        return bsm_call(self.S0, K, T, self.r, self.q, sigma)

    def call_grid(self, strikes: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        """Compute call prices on a grid (n_T × n_K)."""
        grid = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                grid[i, j] = self.call_price(K, T)
        return grid


# ---------------------------------------------------------------------------
# Dupire Local Volatility Extractor
# ---------------------------------------------------------------------------

class DupireLocalVol:
    """
    Extract the Dupire (1994) local volatility surface from market implied vols.

    Algorithm:
    ──────────────────────────────────────────────────────────────────────
    For each interior point (K, T) of the implied vol surface:

    1. Compute the BSM call price: C(K, T) = BSM(σ_impl(K,T))
    2. Differentiate numerically:
         ∂C/∂T   via forward finite difference
         ∂C/∂K   via central finite difference
         ∂²C/∂K² via central second difference
    3. Apply Dupire formula:
         σ_loc²(K,T) = [∂C/∂T + (r-q)·K·∂C/∂K + q·C] / [½·K²·∂²C/∂K²]
    4. Floor at ε > 0 (can be negative in poorly extrapolated regions)

    Key implementation details:
    - Step sizes: dT ≈ T/100, dK ≈ K/200 (balance accuracy vs noise)
    - The denominator ½K²·∂²C/∂K² is the (undiscounted) risk-neutral density
      — if negative, the implied vol surface has calendar spread arbitrage
    - Dupire local vol is the UNIQUE diffusion consistent with the market surface
    ──────────────────────────────────────────────────────────────────────

    Usage
    -----
    >>> surface = ImpliedVolSurface(strikes, maturities, vols, S0=100, r=0.05)
    >>> dupire = DupireLocalVol(surface)
    >>> local_vol = dupire.local_vol(K=100, T=1.0)
    """

    def __init__(self, surface: ImpliedVolSurface):
        self.surface = surface

    def local_vol(self, K: float, T: float,
                  dK_frac: float = 0.005,
                  dT_frac: float = 0.01) -> float:
        """
        Dupire local volatility at a single point (K, T).

        Parameters
        ----------
        K : float    Strike.
        T : float    Maturity.
        dK_frac : float  Fractional step for K derivatives (as fraction of K).
        dT_frac : float  Fractional step for T derivative (as fraction of T).

        Returns
        -------
        float  Local volatility σ_loc(K, T) ≥ 0.
        """
        surf = self.surface
        r, q = surf.r, surf.q

        dK = K * dK_frac
        dT = max(T * dT_frac, 1e-4)

        # ── Call price and its derivatives ──────────────────────────
        C_0  = surf.call_price(K, T)
        C_Kp = surf.call_price(K + dK, T)
        C_Km = surf.call_price(K - dK, T)
        C_Tp = surf.call_price(K, T + dT)

        # ∂C/∂T — forward difference (future info only needed for next maturity)
        dCdT = (C_Tp - C_0) / dT

        # ∂C/∂K — central difference
        dCdK = (C_Kp - C_Km) / (2 * dK)

        # ∂²C/∂K² — central second difference
        d2CdK2 = (C_Kp - 2 * C_0 + C_Km) / (dK ** 2)

        # ── Dupire numerator and denominator ────────────────────────
        numerator = dCdT + (r - q) * K * dCdK + q * C_0
        denominator = 0.5 * K ** 2 * d2CdK2

        if abs(denominator) < 1e-10:
            # Near-zero denominator: return ATM local vol as fallback
            return surf.vol(K, T)

        local_var = numerator / denominator

        if local_var <= 0:
            # Implied vol surface has arbitrage at this point
            # Return implied vol as a conservative fallback
            return surf.vol(K, T)

        return math.sqrt(local_var)

    def local_vol_surface(
        self, strikes: np.ndarray, maturities: np.ndarray
    ) -> np.ndarray:
        """
        Compute the full local vol surface on a (T, K) grid.

        Returns array of shape (len(maturities), len(strikes)).
        """
        surface = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                surface[i, j] = self.local_vol(K, T)
        return surface

    def compare_to_implied(
        self, strikes: np.ndarray, maturities: np.ndarray
    ) -> dict:
        """
        Compare local vol to implied vol across the surface.

        Key relationship (Gatheral 2006):
          σ_loc²(K, T) ≈ E_Q[σ_inst² | S_T = K]
          — local vol is the conditional expectation of instantaneous variance

        Relationship to implied vol (at-the-money, Dupire):
          σ_loc(F_T, T) ≈ σ_impl(F_T, T) + T·∂σ_impl/∂T  (approximately)
        """
        impl_vols = np.zeros((len(maturities), len(strikes)))
        local_vols = np.zeros((len(maturities), len(strikes)))

        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                impl_vols[i, j] = self.surface.vol(K, T)
                local_vols[i, j] = self.local_vol(K, T)

        return {
            "implied_vols": impl_vols,
            "local_vols": local_vols,
            "ratio": local_vols / np.where(impl_vols > 0, impl_vols, np.nan),
        }


# ---------------------------------------------------------------------------
# Generate a synthetic market-consistent implied vol surface
# ---------------------------------------------------------------------------

def generate_market_surface(
    S0: float = 100.0, r: float = 0.05, q: float = 0.0,
    base_vol: float = 0.20,
) -> ImpliedVolSurface:
    """
    Create a realistic market-like implied vol surface:
    - Downward skew (negative ρ effect): OTM puts trade richer
    - Term structure: short-dated vols higher (inverted), long-end flatter
    - Smile: curvature from vol-of-vol effect

    This approximates an equity index surface (e.g. S&P 500).
    """
    maturities = np.array([0.083, 0.25, 0.5, 1.0, 1.5, 2.0])  # 1M to 2Y
    # Moneyness range: 75% to 130% of spot
    strikes = S0 * np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                               1.05, 1.10, 1.15, 1.20, 1.25, 1.30])

    n_T, n_K = len(maturities), len(strikes)
    vols = np.zeros((n_T, n_K))

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            moneyness = math.log(K / S0)

            # Skew: negative slope in moneyness (OTM puts richer)
            skew_coef = -0.10 * math.sqrt(T)       # skew increases with sqrt(T)

            # Smile: quadratic in moneyness (convex surface)
            smile_coef = 0.05 / T                    # more pronounced at short end

            # Term structure: inverted short end (higher near-term vol)
            term_adj = 0.02 * math.exp(-2 * T)

            vol = base_vol + term_adj + skew_coef * moneyness + smile_coef * moneyness ** 2
            vols[i, j] = max(vol, 0.03)  # floor at 3%

    return ImpliedVolSurface(
        strikes=strikes, maturities=maturities,
        implied_vols=vols, S0=S0, r=r, q=q
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 64)
    print("  Dupire Local Volatility Model")
    print("  Extracts σ_loc(K,T) from the market implied vol surface")
    print("═" * 64)

    S0, r, q = 100.0, 0.05, 0.0
    surface = generate_market_surface(S0=S0, r=r, q=q, base_vol=0.20)
    dupire = DupireLocalVol(surface)

    print(f"\n── Implied Vol Surface (market input) ──")
    print(f"  {'Maturity':<10}", end="")
    moneyness_labels = ["75%", "80%", "90%", "100%", "110%", "120%"]
    key_K_mult = [0.75, 0.80, 0.90, 1.00, 1.10, 1.20]
    for lbl in moneyness_labels:
        print(f"{lbl:>9}", end="")
    print()
    print("  " + "─" * 64)
    for T in [0.083, 0.25, 0.5, 1.0, 2.0]:
        T_lbl = f"{T:.3f}Y"
        print(f"  {T_lbl:<10}", end="")
        for km in key_K_mult:
            K = S0 * km
            iv = surface.vol(K, T)
            print(f"{iv:>8.2%}", end=" ")
        print()

    print(f"\n── Local Vol Surface (Dupire extraction) ──")
    print(f"  {'Maturity':<10}", end="")
    for lbl in moneyness_labels:
        print(f"{lbl:>9}", end="")
    print()
    print("  " + "─" * 64)
    for T in [0.083, 0.25, 0.5, 1.0, 2.0]:
        T_lbl = f"{T:.3f}Y"
        print(f"  {T_lbl:<10}", end="")
        for km in key_K_mult:
            K = S0 * km
            lv = dupire.local_vol(K, T)
            print(f"{lv:>8.2%}", end=" ")
        print()

    print(f"\n── Local Vol vs Implied Vol Ratio ──")
    print(f"  (Dupire: σ_loc ≈ sqrt(2·∂σ²_impl/∂T / σ_impl) at ATM — local vol")
    print(f"   flattens the smile: more skew implies σ_loc > σ_impl in tails)")
    print(f"\n  {'Maturity':<10}", end="")
    for lbl in moneyness_labels:
        print(f"{lbl:>9}", end="")
    print()
    print("  " + "─" * 64)
    for T in [0.25, 0.5, 1.0, 2.0]:
        T_lbl = f"{T:.2f}Y"
        print(f"  {T_lbl:<10}", end="")
        for km in key_K_mult:
            K = S0 * km
            iv = surface.vol(K, T)
            lv = dupire.local_vol(K, T)
            ratio = lv / iv if iv > 0 else 0.0
            print(f"{ratio:>8.3f}", end=" ")
        print()

    print(f"\n── Key Properties to Know for Interviews ──")
    print(f"""
  1. Dupire local vol is the UNIQUE diffusion σ(S,t) consistent with the
     full market implied vol surface — no other diffusion can reproduce
     all option prices simultaneously.

  2. ATM local vol vs implied vol:
       σ_loc(F_T, T) ≈ σ_impl(F_T, T) + T·∂σ_impl/∂T
     Local vol equals implied vol only for a flat surface.

  3. OTM local vol and implied vol:
       σ_loc > σ_impl in the tails (skew flattening effect)
       The local vol skew is roughly TWICE the implied vol skew.

  4. Limitations of local vol:
     - Forward smile is flat (smile reverts as T increases)
     - Poor dynamics for exotics sensitive to future vol (e.g. cliquets)
     - Fix: Local-Stochastic Vol (LSV) = Heston with Dupire local vol floor

  5. Dupire can only be implemented where the surface is free of arbitrage:
     - Butterfly arbitrage: ∂²C/∂K² < 0 (negative density)
     - Calendar arbitrage: total implied variance not increasing in T
""")
