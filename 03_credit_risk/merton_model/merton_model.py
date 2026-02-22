"""
Merton Structural Credit Risk Model
======================================
The Merton (1974) model treats a firm's equity as a call option on its assets.

Key insight:
  Shareholders own the firm but have limited liability.
  If asset value V_T > D at maturity → shareholders get V_T - D (call payoff)
  If asset value V_T ≤ D at maturity → firm defaults → shareholders get 0

  Equity = max(V_T - D, 0)   ←→   European call on firm assets

This allows us to:
  1. Infer the unobservable asset value V_A and asset volatility σ_A
     from observable equity price E and equity volatility σ_E
  2. Compute Distance-to-Default (DD) — a z-score to default
  3. Estimate the risk-neutral Probability of Default (PD)
  4. Price risky debt and credit spreads

Distance to Default (KMV model extension):
  DD = (ln(V_A/D) + (μ - σ_A²/2)·T) / (σ_A·√T)
  PD = N(-DD)

The KMV extension (Moody's Analytics) uses:
  - "Default point" = STD + 0.5·LTD (empirically calibrated)
  - Expected Default Frequency (EDF) from empirical default database

Limitations:
  - Assumes a simple capital structure (single zero-coupon debt)
  - Firm value follows GBM (no jumps → underestimates short-term default risk)
  - Risk-neutral vs physical PD: must adjust for market price of risk
  - Reduced-form models (Jarrow-Turnbull, Duffie-Singleton) are alternatives

References:
  - Merton, R.C. (1974). On the Pricing of Corporate Debt. JF 29(2), 449–470.
  - Vasicek, O. (1984). Credit Valuation. KMV Corporation. (basis of EDF)
  - Crosbie, P. & Bohn, J. (2003). Modeling Default Risk. Moody's KMV.
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 24.
  - Lando, D. (2004). Credit Risk Modeling. Princeton UP.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve, brentq


# ---------------------------------------------------------------------------
# Merton model equations
# ---------------------------------------------------------------------------

def merton_equity_value(V_A: float, D: float, T: float, r: float,
                         sigma_A: float) -> float:
    """
    Equity value = BSM call on firm assets at strike D.

    E = V_A·N(d₁) - D·e^(-rT)·N(d₂)
    d₁ = [ln(V_A/D) + (r + σ_A²/2)T] / (σ_A√T)
    d₂ = d₁ - σ_A√T
    """
    if V_A <= 0 or sigma_A <= 0 or T <= 0:
        return max(V_A - D * math.exp(-r * T), 0.0)
    d1 = (math.log(V_A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * math.sqrt(T))
    d2 = d1 - sigma_A * math.sqrt(T)
    return V_A * norm.cdf(d1) - D * math.exp(-r * T) * norm.cdf(d2)


def merton_equity_delta(V_A: float, D: float, T: float, r: float,
                         sigma_A: float) -> float:
    """
    ∂E/∂V_A = N(d₁)   (delta of the equity call).
    Used in the equity volatility equation: σ_E = (V_A/E)·N(d₁)·σ_A
    """
    if V_A <= 0 or sigma_A <= 0 or T <= 0:
        return 1.0 if V_A > D else 0.0
    d1 = (math.log(V_A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * math.sqrt(T))
    return norm.cdf(d1)


def merton_debt_value(V_A: float, D: float, T: float, r: float,
                       sigma_A: float) -> float:
    """
    Risky debt value = risk-free bond - put on firm assets.

    D_risky = D·e^(-rT)·N(d₂) + V_A·N(-d₁)
    = D·e^(-rT) - put(V_A, D, T, r, σ_A)
    """
    if V_A <= 0 or sigma_A <= 0 or T <= 0:
        return min(V_A, D * math.exp(-r * T))
    equity = merton_equity_value(V_A, D, T, r, sigma_A)
    return V_A - equity


def merton_credit_spread(V_A: float, D: float, T: float, r: float,
                          sigma_A: float) -> float:
    """
    Credit spread = yield on risky debt - risk-free yield.

    y_risky = -ln(D_risky/D) / T   (yield of risky zero-coupon bond)
    spread = y_risky - r
    """
    D_risky = merton_debt_value(V_A, D, T, r, sigma_A)
    if D_risky <= 0 or D <= 0:
        return float("inf")
    y_risky = -math.log(D_risky / (D * math.exp(-r * T))) / T + r
    spread = y_risky - r
    return max(spread, 0.0)


def merton_distance_to_default(V_A: float, D: float, T: float,
                                 mu: float, sigma_A: float) -> float:
    """
    Distance-to-Default (DD): number of standard deviations between
    expected future asset value and the default boundary.

    DD = [ln(V_A/D) + (μ - σ_A²/2)·T] / (σ_A·√T)

    Under the physical (real-world) measure, using drift μ (not risk-free r).
    The KMV model uses the 1-year DD as the primary credit metric.
    """
    if V_A <= 0 or D <= 0 or sigma_A <= 0 or T <= 0:
        return 0.0
    return (math.log(V_A / D) + (mu - 0.5 * sigma_A ** 2) * T) / (sigma_A * math.sqrt(T))


def merton_probability_of_default(DD: float) -> float:
    """
    Risk-neutral PD = N(-d₂) where d₂ uses risk-free rate r.
    Physical PD = N(-DD) where DD uses real-world drift μ.

    The KMV EDF uses DD with an empirical mapping to observed default rates
    rather than N(-DD) directly (Gaussian assumption is too thin-tailed).
    """
    return norm.cdf(-DD)


# ---------------------------------------------------------------------------
# Merton model calibration
# ---------------------------------------------------------------------------

class MertonModel:
    """
    Calibrate the Merton model to observable equity price and volatility.

    Given (E, σ_E, D, T, r), solve for (V_A, σ_A) via the system:
      E = V_A·N(d₁) - D·e^(-rT)·N(d₂)          [equity value equation]
      σ_E = (V_A/E)·N(d₁)·σ_A                    [equity vol equation]

    This is a 2×2 nonlinear system solved by iterative methods.

    Usage
    -----
    >>> model = MertonModel(E=80, sigma_E=0.30, D=100, T=1, r=0.05, mu=0.08)
    >>> result = model.calibrate()
    >>> print(result.summary())
    """

    def __init__(
        self,
        E: float,        # equity market cap ($)
        sigma_E: float,  # annualised equity volatility
        D: float,        # face value of debt (default point)
        T: float,        # maturity of debt (years)
        r: float,        # risk-free rate
        mu: float,       # expected asset return (physical measure)
    ):
        self.E = E
        self.sigma_E = sigma_E
        self.D = D
        self.T = T
        self.r = r
        self.mu = mu

    def _equations(self, x: np.ndarray) -> np.ndarray:
        """
        System of equations to solve for (V_A, σ_A).
        Returns residuals that should be zero at the solution.
        """
        V_A, sigma_A = x
        if V_A <= 0 or sigma_A <= 0:
            return np.array([1e10, 1e10])

        eq1 = merton_equity_value(V_A, self.D, self.T, self.r, sigma_A) - self.E
        delta = merton_equity_delta(V_A, self.D, self.T, self.r, sigma_A)
        eq2 = (V_A / self.E) * delta * sigma_A - self.sigma_E
        return np.array([eq1, eq2])

    def calibrate(self) -> "MertonResult":
        """
        Solve the Merton system via Newton's method with multiple starting points.
        Returns a MertonResult with calibrated parameters and credit metrics.
        """
        best = None
        best_residual = float("inf")

        # Multiple starting points for robustness
        V_A_inits = [self.E + self.D * math.exp(-self.r * self.T),
                     (self.E + self.D) * 1.05,
                     (self.E + self.D) * 0.95]

        for V_A0 in V_A_inits:
            x0 = np.array([V_A0, self.sigma_E * self.E / V_A0])
            try:
                sol = fsolve(self._equations, x0, full_output=True)
                x_sol, info, ier, msg = sol
                residual = float(np.sum(self._equations(x_sol) ** 2))
                if ier == 1 and residual < best_residual and x_sol[0] > 0 and x_sol[1] > 0:
                    best_residual = residual
                    best = x_sol
            except Exception:
                pass

        if best is None:
            raise ValueError("Merton calibration failed to converge.")

        V_A, sigma_A = best

        # Derived credit metrics
        d1 = (math.log(V_A / self.D) + (self.r + 0.5 * sigma_A**2) * self.T) / (sigma_A * math.sqrt(self.T))
        d2 = d1 - sigma_A * math.sqrt(self.T)
        dd = merton_distance_to_default(V_A, self.D, self.T, self.mu, sigma_A)
        pd_rn = norm.cdf(-d2)       # risk-neutral PD
        pd_physical = norm.cdf(-dd)  # physical PD
        D_risky = merton_debt_value(V_A, self.D, self.T, self.r, sigma_A)
        spread = merton_credit_spread(V_A, self.D, self.T, self.r, sigma_A)
        leverage = self.D / V_A

        return MertonResult(
            V_A=V_A,
            sigma_A=sigma_A,
            d1=d1,
            d2=d2,
            dd=dd,
            pd_risk_neutral=pd_rn,
            pd_physical=pd_physical,
            debt_value=D_risky,
            credit_spread_bps=spread * 10000,
            leverage=leverage,
            E=self.E,
            sigma_E=self.sigma_E,
            D=self.D,
            T=self.T,
            r=self.r,
            mu=self.mu,
        )

    def credit_curve(self, maturities: np.ndarray) -> pd.DataFrame:
        """
        Compute the credit spread term structure for different maturities.
        Requires a calibrated (V_A, σ_A) — runs calibrate() first.
        """
        result = self.calibrate()
        V_A, sigma_A = result.V_A, result.sigma_A

        records = []
        for T in maturities:
            model_T = MertonModel(self.E, self.sigma_E, self.D, T, self.r, self.mu)
            try:
                res_T = model_T.calibrate()
                records.append({
                    "maturity": T,
                    "credit_spread_bps": res_T.credit_spread_bps,
                    "pd_risk_neutral": res_T.pd_risk_neutral,
                    "dd": res_T.dd,
                })
            except Exception:
                pass
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MertonResult:
    """Calibrated Merton model output with full credit analytics."""
    V_A: float              # Implied asset value
    sigma_A: float          # Implied asset volatility
    d1: float               # BSM d₁
    d2: float               # BSM d₂
    dd: float               # Distance-to-Default (physical)
    pd_risk_neutral: float  # Risk-neutral PD = N(-d₂)
    pd_physical: float      # Physical PD = N(-DD)
    debt_value: float       # Implied risky debt value
    credit_spread_bps: float
    leverage: float         # D / V_A
    # Inputs
    E: float
    sigma_E: float
    D: float
    T: float
    r: float
    mu: float

    def summary(self, name: str = "Firm") -> str:
        lines = [
            "=" * 56,
            f"  Merton Credit Risk Model: {name}",
            "=" * 56,
            f"  Inputs:",
            f"    Equity value:     ${self.E:>12,.1f}",
            f"    Equity vol:       {self.sigma_E:>12.4%}",
            f"    Debt face value:  ${self.D:>12,.1f}",
            f"    Maturity:         {self.T:>12.1f} years",
            f"    Risk-free rate:   {self.r:>12.4%}",
            f"    Asset drift (μ):  {self.mu:>12.4%}",
            "─" * 56,
            f"  Calibrated:",
            f"    Asset value V_A:  ${self.V_A:>12,.1f}",
            f"    Asset vol σ_A:    {self.sigma_A:>12.4%}",
            f"    Leverage D/V_A:   {self.leverage:>12.4%}",
            "─" * 56,
            f"  Credit Metrics:",
            f"    d₁:               {self.d1:>12.4f}",
            f"    d₂:               {self.d2:>12.4f}",
            f"    Distance-to-Default: {self.dd:>9.4f} σ",
            f"    PD (risk-neutral): {self.pd_risk_neutral:>11.4%}",
            f"    PD (physical):    {self.pd_physical:>12.4%}",
            f"    Credit Spread:    {self.credit_spread_bps:>9.1f} bps",
            f"    Implied debt val: ${self.debt_value:>12,.1f}",
            "=" * 56,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-firm credit portfolio
# ---------------------------------------------------------------------------

class MertonCreditPortfolio:
    """
    Applies the Merton model to a portfolio of firms.
    Useful for credit risk monitoring or constructing a credit risk dashboard.
    """

    def __init__(self, firms: list[dict]):
        """
        Parameters
        ----------
        firms : list of dicts, each with keys:
          name, E, sigma_E, D, T, r, mu
        """
        self.firms = firms

    def analyse_all(self) -> pd.DataFrame:
        """Run Merton calibration on all firms and return a summary DataFrame."""
        rows = []
        for firm in self.firms:
            try:
                model = MertonModel(
                    E=firm["E"], sigma_E=firm["sigma_E"],
                    D=firm["D"], T=firm["T"],
                    r=firm["r"], mu=firm.get("mu", 0.08)
                )
                res = model.calibrate()
                rows.append({
                    "name":            firm["name"],
                    "equity":          firm["E"],
                    "debt_face":       firm["D"],
                    "asset_value":     res.V_A,
                    "asset_vol":       res.sigma_A,
                    "leverage":        res.leverage,
                    "dd":              res.dd,
                    "pd_rn_pct":       res.pd_risk_neutral * 100,
                    "pd_phys_pct":     res.pd_physical * 100,
                    "spread_bps":      res.credit_spread_bps,
                })
            except Exception as e:
                rows.append({"name": firm["name"], "error": str(e)})

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 60)
    print("  Merton Structural Credit Risk Model")
    print("  Equity-as-option framework for PD and credit spread estimation")
    print("═" * 60)

    # ── Single firm example ────────────────────────────────────────
    print("\n── Example Firm: $80B equity, $100B debt ──")
    model = MertonModel(
        E=80e9,     # $80B equity market cap
        sigma_E=0.30,   # 30% equity vol
        D=100e9,    # $100B debt
        T=1.0,
        r=0.05,
        mu=0.08,
    )
    result = model.calibrate()
    print(result.summary("Example Corp"))

    # ── Credit spread term structure ───────────────────────────────
    print("\n── Credit Spread Term Structure ──")
    mats = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    print(f"  {'Maturity':>8}  {'Spread (bps)':>14}  {'PD (RN)':>10}  {'DD':>8}")
    print("  " + "─" * 46)
    for T in mats:
        m = MertonModel(80e9, 0.30, 100e9, T, 0.05, 0.08)
        try:
            res = m.calibrate()
            print(f"  {T:>8.1f}Y  {res.credit_spread_bps:>14.1f}  "
                  f"{res.pd_risk_neutral:>10.4%}  {res.dd:>8.4f}")
        except Exception:
            print(f"  {T:>8.1f}Y  calibration failed")

    # ── Effect of leverage on DD and spread ───────────────────────
    print("\n── Effect of Leverage on Distance-to-Default ──")
    print(f"  {'D/V':>8}  {'DD':>8}  {'PD (phys)':>12}  {'Spread (bps)':>14}")
    print("  " + "─" * 48)
    V_total = 180e9
    for leverage in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        D = V_total * leverage
        E = V_total * (1 - leverage)
        m = MertonModel(E=E, sigma_E=0.35, D=D, T=1, r=0.05, mu=0.08)
        try:
            res = m.calibrate()
            print(f"  {leverage:>8.0%}  {res.dd:>8.3f}  {res.pd_physical:>12.4%}  {res.credit_spread_bps:>14.1f}")
        except Exception:
            pass

    # ── Multi-firm portfolio ───────────────────────────────────────
    print("\n── Credit Portfolio (5 stylised firms) ──")
    firms = [
        {"name": "Investment Grade A",  "E": 50e9,  "sigma_E": 0.20, "D": 30e9,  "T": 1, "r": 0.05, "mu": 0.08},
        {"name": "Investment Grade BBB","E": 30e9,  "sigma_E": 0.25, "D": 40e9,  "T": 1, "r": 0.05, "mu": 0.08},
        {"name": "Sub-Investment BB",   "E": 10e9,  "sigma_E": 0.35, "D": 30e9,  "T": 1, "r": 0.05, "mu": 0.08},
        {"name": "High Yield B",        "E": 5e9,   "sigma_E": 0.45, "D": 25e9,  "T": 1, "r": 0.05, "mu": 0.08},
        {"name": "Distressed CCC",      "E": 2e9,   "sigma_E": 0.70, "D": 20e9,  "T": 1, "r": 0.05, "mu": 0.08},
    ]

    portfolio = MertonCreditPortfolio(firms)
    df = portfolio.analyse_all()

    display_cols = ["name", "leverage", "dd", "pd_rn_pct", "spread_bps"]
    print(f"\n  {'Firm':<25}  {'D/V':>8}  {'DD':>8}  {'PD %':>8}  {'Spread':>8}")
    print("  " + "─" * 65)
    for _, row in df.iterrows():
        if "error" not in row:
            print(f"  {row['name']:<25}  {row['leverage']:>8.1%}  "
                  f"{row['dd']:>8.3f}  {row['pd_rn_pct']:>8.3f}%  {row['spread_bps']:>7.0f}bp")
