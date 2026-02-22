"""
Short-Rate Interest Rate Models: Vasicek & Cox-Ingersoll-Ross (CIR)
=====================================================================
Implements two foundational equilibrium short-rate models:

1. Vasicek (1977) Model
   dr = κ(θ - r)dt + σ dW_t
   - Ornstein-Uhlenbeck process: mean-reverting with Gaussian increments
   - Allows negative rates (a weakness in practice; relevant for ECB era)
   - Affine term structure → analytical bond and option prices

2. Cox-Ingersoll-Ross (CIR, 1985) Model
   dr = κ(θ - r)dt + σ√r dW_t
   - Mean-reverting; volatility proportional to √r
   - Rates stay non-negative if 2κθ ≥ σ² (Feller condition)
   - Widely used for real rates, mortgage prepayment, and credit models

Both models belong to the affine term structure class:
  P(t,T) = A(t,T) · e^{-B(t,T)·r_t}

where A(t,T) and B(t,T) have analytical forms depending on parameters.

Applications at top institutions:
  - Fed, ECB, BIS: short-rate models for monetary policy analysis
  - Fixed income desks: pricing interest rate derivatives (caps, floors, swaptions)
  - Central bank stress testing: scenario generation for rate paths
  - Risk: duration in a stochastic rate environment

References:
  - Vasicek, O. (1977). An Equilibrium Characterization of the Term Structure.
    JFE, 5(2), 177–188.
  - Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985). A Theory of the Term Structure.
    Econometrica, 53(2), 385–408.
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 31–32.
  - Brigo, D. & Mercurio, F. (2006). Interest Rate Models. Springer.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, ncx2


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ShortRateModel:
    """Abstract base for affine short-rate models."""

    def zero_rate(self, r0: float, T: float) -> float:
        """Continuously compounded zero rate for maturity T."""
        raise NotImplementedError

    def bond_price(self, r0: float, T: float) -> float:
        """Zero-coupon bond price P(0,T) = e^{-y(T)·T}"""
        return math.exp(-self.zero_rate(r0, T) * T)

    def simulate_paths(self, r0: float, T: float, n_steps: int, n_paths: int, seed: int = 42) -> np.ndarray:
        """Simulate rate paths via Euler-Maruyama. Override for exact simulation."""
        raise NotImplementedError

    def term_structure(self, r0: float, maturities: list[float]) -> pd.DataFrame:
        """Generate the model-implied yield curve."""
        rows = []
        for T in maturities:
            y = self.zero_rate(r0, T)
            rows.append({
                "Maturity (Y)": T,
                "Zero Rate": f"{y:.4%}",
                "Bond Price": f"{self.bond_price(r0, T):.8f}",
                "Fwd Rate (3M)": f"{(y*T - self.zero_rate(r0,max(T-0.25,0.01))*(T-0.25))/0.25:.4%}" if T > 0.25 else "—"
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Vasicek Model
# ---------------------------------------------------------------------------

class VasicekModel(ShortRateModel):
    """
    Vasicek (1977) short-rate model.

    dr = κ(θ - r)dt + σ dW_t

    Parameters
    ----------
    kappa : float   Speed of mean reversion (κ > 0).
    theta : float   Long-run mean rate (θ).
    sigma : float   Volatility of the short rate (σ).

    Analytical bond price P(t,T):
        P(t,T) = A(t,T) · e^{-B(t,T)·r_t}

    B(t,T) = (1 - e^{-κτ}) / κ        where τ = T - t

    A(t,T) = exp[(B-τ)(κ²θ - σ²/2)/κ² - σ²B²/(4κ)]
    """

    def __init__(self, kappa: float, theta: float, sigma: float):
        if kappa <= 0:
            raise ValueError(f"κ must be positive, got {kappa}")
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    @property
    def long_run_rate(self) -> float:
        """Long-run mean: E[r_∞] = θ"""
        return self.theta

    @property
    def long_run_vol(self) -> float:
        """Stationary standard deviation: σ / √(2κ)"""
        return self.sigma / math.sqrt(2 * self.kappa)

    @property
    def half_life(self) -> float:
        """Half-life of mean reversion: ln(2)/κ"""
        return math.log(2) / self.kappa

    def _B(self, tau: float) -> float:
        """B(τ) = (1 - e^{-κτ}) / κ"""
        return (1 - math.exp(-self.kappa * tau)) / self.kappa

    def _A(self, tau: float) -> float:
        """A(τ) = exp[(B-τ)(κ²θ - σ²/2)/κ² - σ²B²/(4κ)]"""
        B = self._B(tau)
        exponent = (
            (B - tau) * (self.kappa ** 2 * self.theta - 0.5 * self.sigma ** 2) / self.kappa ** 2
            - self.sigma ** 2 * B ** 2 / (4 * self.kappa)
        )
        return math.exp(exponent)

    def bond_price(self, r0: float, T: float) -> float:
        """P(0,T) = A(T) · e^{-B(T)·r₀}"""
        if T <= 0:
            return 1.0
        return self._A(T) * math.exp(-self._B(T) * r0)

    def zero_rate(self, r0: float, T: float) -> float:
        """Continuously compounded zero rate: R(T) = -ln(P(0,T))/T"""
        if T <= 0:
            return r0
        return -math.log(self.bond_price(r0, T)) / T

    def simulate_paths(
        self, r0: float, T: float, n_steps: int = 252, n_paths: int = 1000, seed: int = 42
    ) -> np.ndarray:
        """
        Exact simulation of Vasicek paths using the conditional distribution:
        r_{t+dt} | r_t ~ N(r_t·e^{-κdt} + θ(1-e^{-κdt}),  σ²(1-e^{-2κdt})/(2κ))
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0

        exp_kdt = math.exp(-self.kappa * dt)
        cond_mean_coeff = self.theta * (1 - exp_kdt)
        cond_var = self.sigma ** 2 * (1 - exp_kdt ** 2) / (2 * self.kappa)
        cond_std = math.sqrt(cond_var)

        Z = rng.standard_normal((n_paths, n_steps))
        for t in range(n_steps):
            paths[:, t + 1] = (
                paths[:, t] * exp_kdt + cond_mean_coeff + cond_std * Z[:, t]
            )
        return paths

    def bond_option_price(self, r0: float, T_option: float, T_bond: float,
                           K: float, option_type: str = "call") -> float:
        """
        Closed-form price of a European option on a zero-coupon bond.
        (Jamshidian 1989 formula for Vasicek)

        Parameters
        ----------
        T_option : float   Option expiry.
        T_bond : float     Bond maturity (> T_option).
        K : float          Strike (on the bond price).
        """
        if T_bond <= T_option:
            raise ValueError("Bond maturity must exceed option expiry.")

        B_s_T = self._B(T_bond - T_option)
        P_t_s = self.bond_price(r0, T_option)
        P_t_T = self.bond_price(r0, T_bond)

        sigma_P = (self.sigma * B_s_T *
                   math.sqrt((1 - math.exp(-2 * self.kappa * T_option)) / (2 * self.kappa)))

        if sigma_P < 1e-10:
            return max(P_t_T - K * P_t_s, 0) if option_type == "call" else max(K * P_t_s - P_t_T, 0)

        h = (1 / sigma_P) * math.log(P_t_T / (P_t_s * K)) + sigma_P / 2

        if option_type == "call":
            return P_t_T * norm.cdf(h) - K * P_t_s * norm.cdf(h - sigma_P)
        else:
            return K * P_t_s * norm.cdf(-(h - sigma_P)) - P_t_T * norm.cdf(-h)


# ---------------------------------------------------------------------------
# 2. CIR Model
# ---------------------------------------------------------------------------

class CIRModel(ShortRateModel):
    """
    Cox-Ingersoll-Ross (1985) short-rate model.

    dr = κ(θ - r)dt + σ√r dW_t

    Non-negative rates when Feller condition holds: 2κθ ≥ σ²

    Analytical bond price:
        P(t,T) = A(t,T) · e^{-B(t,T)·r_t}

    γ = √(κ² + 2σ²)

    B(τ) = 2(e^{γτ} - 1) / [(γ+κ)(e^{γτ}-1) + 2γ]

    A(τ) = [2γ·e^{(κ+γ)τ/2} / ((γ+κ)(e^{γτ}-1) + 2γ)]^{2κθ/σ²}
    """

    def __init__(self, kappa: float, theta: float, sigma: float):
        if kappa <= 0:
            raise ValueError(f"κ must be positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"θ must be positive, got {theta}")
        if sigma <= 0:
            raise ValueError(f"σ must be positive, got {sigma}")
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.gamma = math.sqrt(kappa ** 2 + 2 * sigma ** 2)

    @property
    def feller_satisfied(self) -> bool:
        """2κθ ≥ σ²: rates stay strictly positive."""
        return 2 * self.kappa * self.theta >= self.sigma ** 2

    @property
    def long_run_rate(self) -> float:
        return self.theta

    @property
    def long_run_vol(self) -> float:
        """Stationary std: σ√(θ/(2κ))"""
        return self.sigma * math.sqrt(self.theta / (2 * self.kappa))

    def _B(self, tau: float) -> float:
        g, k = self.gamma, self.kappa
        numer = 2 * (math.exp(g * tau) - 1)
        denom = (g + k) * (math.exp(g * tau) - 1) + 2 * g
        return numer / denom

    def _A(self, tau: float) -> float:
        g, k = self.gamma, self.kappa
        numer = 2 * g * math.exp((k + g) * tau / 2)
        denom = (g + k) * (math.exp(g * tau) - 1) + 2 * g
        power = 2 * k * self.theta / self.sigma ** 2
        return (numer / denom) ** power

    def bond_price(self, r0: float, T: float) -> float:
        """P(0,T) = A(T) · e^{-B(T)·r₀}"""
        if T <= 0:
            return 1.0
        return self._A(T) * math.exp(-self._B(T) * r0)

    def zero_rate(self, r0: float, T: float) -> float:
        if T <= 0:
            return r0
        return -math.log(self.bond_price(r0, T)) / T

    def simulate_paths(
        self, r0: float, T: float, n_steps: int = 252, n_paths: int = 1000, seed: int = 42
    ) -> np.ndarray:
        """
        Euler-Maruyama simulation for CIR.
        The reflection method ensures non-negative rates:
        r_{t+dt} = |r_t + κ(θ-r_t)dt + σ√r_t · √dt · Z|
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0

        Z = rng.standard_normal((n_paths, n_steps))
        for t in range(n_steps):
            r = paths[:, t]
            dr = (self.kappa * (self.theta - r) * dt
                  + self.sigma * np.sqrt(np.maximum(r, 0)) * sqrt_dt * Z[:, t])
            paths[:, t + 1] = np.abs(r + dr)  # reflection ensures non-negativity

        return paths


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def compare_term_structures(
    r0: float = 0.05, maturities: list[float] | None = None
) -> pd.DataFrame:
    """
    Compare Vasicek and CIR yield curves for same parameters.
    Illustrates the difference in rate distribution tails.
    """
    if maturities is None:
        maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

    vasicek = VasicekModel(kappa=0.5, theta=0.06, sigma=0.02)
    cir = CIRModel(kappa=0.5, theta=0.06, sigma=0.10)

    rows = []
    for T in maturities:
        rows.append({
            "Maturity": T,
            "Vasicek Zero Rate": f"{vasicek.zero_rate(r0, T):.4%}",
            "CIR Zero Rate": f"{cir.zero_rate(r0, T):.4%}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 65)
    print("  Short-Rate Interest Rate Models")
    print("═" * 65)

    # --- Vasicek ---
    print("\n── Vasicek Model: κ=0.5, θ=6%, σ=2%, r₀=5% ──")
    vasicek = VasicekModel(kappa=0.5, theta=0.06, sigma=0.02)

    print(f"  Half-life of mean reversion : {vasicek.half_life:.2f} years")
    print(f"  Long-run rate               : {vasicek.long_run_rate:.2%}")
    print(f"  Long-run vol (stationary)   : {vasicek.long_run_vol:.4%}")
    print(f"  Negative rates possible     : Yes (Gaussian)")

    print("\n  Vasicek Term Structure:")
    ts = vasicek.term_structure(r0=0.05, maturities=[0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    print(ts.to_string(index=False))

    print("\n  Bond Option (Call on 5Y bond, option expires in 1Y, strike=0.85):")
    opt = vasicek.bond_option_price(r0=0.05, T_option=1.0, T_bond=5.0, K=0.85, option_type="call")
    print(f"  Call price: {opt:.6f}")

    # --- CIR ---
    print("\n── CIR Model: κ=0.5, θ=6%, σ=10%, r₀=5% ──")
    cir = CIRModel(kappa=0.5, theta=0.06, sigma=0.10)

    print(f"  Feller condition (2κθ≥σ²): {cir.feller_satisfied} "
          f"[2·{0.5}·{0.06:.2f}={2*0.5*0.06:.4f} vs σ²={0.10**2:.4f}]")
    print(f"  Half-life: {math.log(2)/cir.kappa:.2f}Y | Long-run vol: {cir.long_run_vol:.4%}")

    print("\n  CIR Term Structure:")
    ts_cir = cir.term_structure(r0=0.05, maturities=[0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    print(ts_cir.to_string(index=False))

    # --- Comparison ---
    print("\n── Vasicek vs CIR Term Structure Comparison ──")
    print(compare_term_structures(r0=0.05).to_string(index=False))

    # --- Rate path simulation ---
    print("\n── Simulated Rate Paths (10 paths, 1Y, daily steps) ──")
    paths = vasicek.simulate_paths(r0=0.05, T=1.0, n_steps=252, n_paths=10, seed=0)
    print(f"  Min rate across paths: {paths.min():.4%}")
    print(f"  Max rate across paths: {paths.max():.4%}")
    print(f"  Mean terminal rate:    {paths[:, -1].mean():.4%}  (expect ≈ θ={vasicek.theta:.2%})")
    print(f"  Std terminal rate:     {paths[:, -1].std():.4%}")

    print("\n── CIR Paths (rates stay non-negative) ──")
    cir_paths = cir.simulate_paths(r0=0.05, T=1.0, n_steps=252, n_paths=1000, seed=0)
    print(f"  Min rate observed:  {cir_paths.min():.6%}  (should be ≥ 0)")
    print(f"  Mean terminal rate: {cir_paths[:,-1].mean():.4%}")
