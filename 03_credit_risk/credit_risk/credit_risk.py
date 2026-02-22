"""
Credit Risk: Merton Structural Model & CDS Pricing
====================================================
Implements two foundational credit risk frameworks:

1. Merton (1974) Structural Model
   - Models equity as a call option on the firm's assets
   - Derives probability of default (PD), distance-to-default (DD),
     and credit spread from observable market data
   - The theoretical foundation of KMV / Moody's Analytics models

2. Credit Default Swap (CDS) Pricing
   - Marks-to-market a CDS from the survival probability curve
   - CDS spreads as a measure of credit risk (credit triangle)
   - Basis: CDS spread ≈ hazard rate for small spreads

3. Credit Risk Metrics (Basel / IRB Framework)
   - PD (Probability of Default)
   - LGD (Loss Given Default)
   - EAD (Exposure at Default)
   - Expected Loss (EL) = PD × LGD × EAD
   - Unexpected Loss (UL) — basis of capital requirement

Why this matters:
  Every major bank has a model validation team reviewing structural credit models.
  BIS Basel III/IV capital rules are built on PD/LGD/EAD inputs. The Fed,
  ECB, and regulators mandate stress testing of credit portfolios.

References:
  - Merton, R.C. (1974). On the Pricing of Corporate Debt. JF, 29(2), 449–470.
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 24–25.
  - Hull, J.C. (2022). Risk Management and Financial Institutions, Ch. 19–22.
  - BCBS (2017). Basel III: Finalising Post-Crisis Reforms. BIS.
"""

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MertonResult:
    """Outputs from the Merton structural model."""
    asset_value: float          # V_A: implied total asset value
    asset_volatility: float     # σ_A: implied asset volatility
    distance_to_default: float  # DD = (ln(V_A/D) + (μ-σ²/2)T) / (σ_A√T)
    prob_default_rn: float      # Risk-neutral PD = N(-d2)
    prob_default_real: float    # Real-world PD (using asset drift μ instead of r)
    credit_spread_bps: float    # Implied CDS/bond spread in basis points
    equity_check: float         # Verify: BSM(V_A, D, σ_A) ≈ E (market cap)

    def summary(self) -> str:
        return (
            f"  Asset Value (V_A)         : ${self.asset_value:>15,.2f}\n"
            f"  Asset Volatility (σ_A)    : {self.asset_volatility:>15.4%}\n"
            f"  Distance to Default (DD)  : {self.distance_to_default:>15.4f}σ\n"
            f"  PD (risk-neutral)         : {self.prob_default_rn:>15.6%}\n"
            f"  PD (real-world, μ=8%)     : {self.prob_default_real:>15.6%}\n"
            f"  Implied Credit Spread     : {self.credit_spread_bps:>15.2f} bps\n"
            f"  Equity Check (BSM vs mkt) : ${self.equity_check:>15,.2f}"
        )


@dataclass
class CDSResult:
    """CDS pricing output."""
    fair_spread_bps: float      # Fair CDS spread at which NPV = 0
    mtm_value: float            # MTM of a CDS with existing coupon
    survival_prob_maturity: float
    risky_annuity: float        # PV of 1bp stream (risky duration)
    protection_leg_pv: float
    premium_leg_pv: float

    def summary(self) -> str:
        return (
            f"  Fair CDS Spread        : {self.fair_spread_bps:.2f} bps\n"
            f"  Survival Prob (T)      : {self.survival_prob_maturity:.6%}\n"
            f"  Risky Annuity          : {self.risky_annuity:.6f}\n"
            f"  Protection Leg PV      : {self.protection_leg_pv:.6f}\n"
            f"  Premium Leg PV (1bp)   : {self.premium_leg_pv:.6f}"
        )


# ---------------------------------------------------------------------------
# 1. Merton Structural Model
# ---------------------------------------------------------------------------

class MertonModel:
    """
    Merton (1974) structural model of credit risk.

    Firm value V_A follows GBM: dV_A = μ·V_A dt + σ_A·V_A dW_t

    Equity is a call option on firm assets:
        E = V_A·N(d1) - D·e^{-rT}·N(d2)

    where:
        d1 = [ln(V_A/D) + (r + σ_A²/2)T] / (σ_A√T)
        d2 = d1 - σ_A√T
        D = face value of debt (default threshold)

    We observe E (equity = market cap) and σ_E (equity vol),
    and solve for V_A and σ_A via two equations:
        1. E = BSM_call(V_A, D, σ_A, r, T)
        2. σ_E = (V_A/E)·N(d1)·σ_A  (Ito's lemma link)

    Distance to Default (DD): number of standard deviations the firm's
    asset value is from the default point. KMV empirically maps DD → PD.

    Parameters
    ----------
    equity_value : float    Market capitalisation (E).
    equity_vol : float      Annualised equity volatility (σ_E).
    debt_face : float       Total face value of debt (D), often 1-year horizon.
    risk_free : float       Risk-free rate (r).
    T : float               Time horizon in years (typically 1Y).
    asset_drift : float     Real-world asset drift (μ) for PD calculation.
    recovery_rate : float   LGD = 1 - recovery_rate.
    """

    def __init__(
        self,
        equity_value: float,
        equity_vol: float,
        debt_face: float,
        risk_free: float = 0.05,
        T: float = 1.0,
        asset_drift: float = 0.08,
        recovery_rate: float = 0.40,
    ):
        self.E = equity_value
        self.sigma_E = equity_vol
        self.D = debt_face
        self.r = risk_free
        self.T = T
        self.mu = asset_drift
        self.recovery = recovery_rate

    def _d1(self, V_A: float, sigma_A: float) -> float:
        return (math.log(V_A / self.D) + (self.r + 0.5 * sigma_A ** 2) * self.T) / (sigma_A * math.sqrt(self.T))

    def _d2(self, V_A: float, sigma_A: float) -> float:
        return self._d1(V_A, sigma_A) - sigma_A * math.sqrt(self.T)

    def _equity_from_assets(self, V_A: float, sigma_A: float) -> float:
        """BSM call price: equity value as a call on assets."""
        d1 = self._d1(V_A, sigma_A)
        d2 = self._d2(V_A, sigma_A)
        return V_A * norm.cdf(d1) - self.D * math.exp(-self.r * self.T) * norm.cdf(d2)

    def solve(self) -> MertonResult:
        """
        Solve for implied asset value (V_A) and asset volatility (σ_A)
        from observed equity value and equity volatility.

        System of two equations (Ronn & Verma, 1986):
            E = V_A·N(d1) - D·e^{-rT}·N(d2)           ... (1) BSM equation
            σ_E·E = V_A·N(d1)·σ_A                      ... (2) Ito's lemma

        Solved via Newton-like iteration (fsolve).
        """
        def equations(x: np.ndarray) -> np.ndarray:
            V_A, sigma_A = x
            if V_A <= 0 or sigma_A <= 0:
                return [1e10, 1e10]
            d1 = self._d1(V_A, sigma_A)
            d2 = self._d2(V_A, sigma_A)
            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)

            eq1 = V_A * Nd1 - self.D * math.exp(-self.r * self.T) * Nd2 - self.E
            eq2 = V_A * Nd1 * sigma_A - self.sigma_E * self.E

            return [eq1, eq2]

        # Initial guess: V_A ≈ E + D·e^{-rT}, sigma_A ≈ sigma_E * E/(E+D)
        V_A_init = self.E + self.D * math.exp(-self.r * self.T)
        sigma_A_init = self.sigma_E * self.E / V_A_init

        solution, info, ier, msg = fsolve(
            equations, [V_A_init, sigma_A_init], full_output=True
        )
        if ier != 1:
            raise RuntimeError(f"Merton model did not converge: {msg}")

        V_A, sigma_A = solution

        # Distance to default (real-world): use drift μ instead of r
        dd_rn = self._d2(V_A, sigma_A)   # risk-neutral (under Q): uses r
        dd_real = (math.log(V_A / self.D) + (self.mu - 0.5 * sigma_A ** 2) * self.T) / (sigma_A * math.sqrt(self.T))

        pd_rn = norm.cdf(-dd_rn)
        pd_real = norm.cdf(-dd_real)

        # Implied credit spread (from zero-coupon debt price)
        # P_risky = D·e^{-rT}·N(d2) + V_A·(1-recovery)·N(-d1)  [Merton debt pricing]
        d1 = self._d1(V_A, sigma_A)
        d2 = self._d2(V_A, sigma_A)
        debt_pv = self.D * math.exp(-self.r * self.T) * norm.cdf(d2) + V_A * (1 - self.recovery) * norm.cdf(-d1)
        # Implied yield on risky debt
        if debt_pv > 0 and self.T > 0:
            risky_yield = -math.log(debt_pv / self.D) / self.T
            credit_spread = (risky_yield - self.r) * 10_000  # in bps
        else:
            credit_spread = 0.0

        return MertonResult(
            asset_value=V_A,
            asset_volatility=sigma_A,
            distance_to_default=dd_real,
            prob_default_rn=pd_rn,
            prob_default_real=pd_real,
            credit_spread_bps=max(0, credit_spread),
            equity_check=self._equity_from_assets(V_A, sigma_A),
        )


# ---------------------------------------------------------------------------
# 2. CDS Pricing
# ---------------------------------------------------------------------------

class CDSPricer:
    """
    Credit Default Swap pricer using a constant hazard rate model.

    CDS structure:
      - Protection buyer pays a quarterly premium (spread) until default or maturity
      - Protection seller pays (1 - recovery_rate) at default

    Under a constant hazard rate λ:
      Survival probability: Q(τ > t) = e^{-λt}
      Default probability: Q(τ ≤ t) = 1 - e^{-λt}

    Fair spread:
      s = (1-R) · ∫₀ᵀ e^{-(r+λ)t} λ dt / ∫₀ᵀ e^{-(r+λ)t} dt
        ≈ λ(1-R)  for small spreads ("credit triangle")

    Parameters
    ----------
    hazard_rate : float    Constant hazard rate λ (per year).
    recovery : float       Recovery rate R (e.g. 0.40).
    risk_free : float      Risk-free rate for discounting.
    maturity : float       CDS maturity in years.
    freq : int             Premium payment frequency per year (4=quarterly).
    """

    def __init__(
        self,
        hazard_rate: float,
        recovery: float = 0.40,
        risk_free: float = 0.05,
        maturity: float = 5.0,
        freq: int = 4,
    ):
        self.lam = hazard_rate
        self.R = recovery
        self.r = risk_free
        self.T = maturity
        self.freq = freq
        self._dt = 1.0 / freq
        self._payment_times = [i * self._dt for i in range(1, int(round(maturity * freq)) + 1)]

    def survival_prob(self, t: float) -> float:
        """Q(τ > t) = e^{-λt}"""
        return math.exp(-self.lam * t)

    def discount(self, t: float) -> float:
        """Risk-free discount factor: e^{-rt}"""
        return math.exp(-self.r * t)

    def protection_leg_pv(self) -> float:
        """
        PV of protection (contingent) leg:
        Protection_PV = (1-R) · ∫₀ᵀ e^{-rt} · f(t) dt
                     ≈ (1-R) · Σ e^{-r·t_i} · [Q(t_{i-1}) - Q(t_i)]  (discrete approx)

        where f(t) = λ·e^{-λt} is the default density.
        """
        protection = 0.0
        t_prev = 0.0
        for t in self._payment_times:
            q_prev = self.survival_prob(t_prev)
            q_curr = self.survival_prob(t)
            # PV of protection payment if default occurs in (t_prev, t]
            # Use midpoint discount
            t_mid = (t_prev + t) / 2
            protection += self.discount(t_mid) * (q_prev - q_curr)
            t_prev = t
        return (1 - self.R) * protection

    def premium_leg_pv(self, spread: float = 1.0) -> float:
        """
        PV of premium leg for a 1bp spread (the risky annuity / RPV01):
        Premium_PV = spread · Δt · Σ Q(t_i) · df(t_i)

        Full premium PV = spread × risky_annuity
        """
        annuity = sum(
            self._dt * self.survival_prob(t) * self.discount(t)
            for t in self._payment_times
        )
        return spread * annuity

    def fair_spread(self) -> float:
        """
        Fair spread s* such that NPV = 0:
        s* = Protection_PV / Risky_Annuity

        In the constant hazard rate model:
        s* ≈ λ · (1-R)  (credit triangle approximation)
        """
        prot = self.protection_leg_pv()
        annuity = self.premium_leg_pv(spread=1.0)
        return prot / annuity

    def mtm(self, existing_spread: float) -> float:
        """
        Mark-to-market value of an existing CDS position.
        For protection buyer: MTM = Protection_PV - existing_spread × Annuity
        """
        fair_s = self.fair_spread()
        annuity = self.premium_leg_pv(spread=1.0)
        return (fair_s - existing_spread) * annuity

    def price(self) -> CDSResult:
        fair_s = self.fair_spread()
        return CDSResult(
            fair_spread_bps=fair_s * 10_000,
            mtm_value=0.0,   # at inception, MTM = 0
            survival_prob_maturity=self.survival_prob(self.T),
            risky_annuity=self.premium_leg_pv(1.0),
            protection_leg_pv=self.protection_leg_pv(),
            premium_leg_pv=self.premium_leg_pv(fair_s),
        )


# ---------------------------------------------------------------------------
# 3. Expected Loss / IRB Capital Framework
# ---------------------------------------------------------------------------

def irb_capital_requirement(
    pd: float, lgd: float, ead: float,
    maturity: float = 2.5, correlation: Optional[float] = None
) -> dict:
    """
    Basel III IRB (Internal Ratings-Based) capital requirement.

    Under the Basel Asymptotic Single Risk Factor (ASRF) model:
        K = LGD · N[(N^{-1}(PD) + √ρ·N^{-1}(0.999)) / √(1-ρ)] - PD·LGD

    where ρ is the asset correlation (specified by Basel for each exposure class).

    Parameters
    ----------
    pd : float          Probability of Default (1-year).
    lgd : float         Loss Given Default.
    ead : float         Exposure at Default ($).
    maturity : float    Effective maturity M (years).
    correlation : float Asset correlation ρ (None → Basel formula for corporate).

    References
    ----------
    BCBS (2006). Basel II: International Convergence, §272–279.
    """
    if correlation is None:
        # Basel corporate asset correlation formula
        correlation = (
            0.12 * (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
            + 0.24 * (1 - (1 - math.exp(-50 * pd)) / (1 - math.exp(-50)))
        )

    rho = correlation
    sqrt_rho = math.sqrt(rho)
    sqrt_1_rho = math.sqrt(1 - rho)

    z_pd = norm.ppf(pd)
    z_conf = norm.ppf(0.999)

    wcdr = norm.cdf((z_pd + sqrt_rho * z_conf) / sqrt_1_rho)  # Worst Case Default Rate

    # Maturity adjustment (Basel III)
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    maturity_adj = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)

    k = (wcdr - pd) * lgd * maturity_adj   # Capital requirement per unit EAD
    rwa = k * 12.5 * ead                    # Risk-Weighted Assets
    capital = rwa * 0.08                    # Minimum capital at 8% ratio

    return {
        "PD": f"{pd:.4%}",
        "LGD": f"{lgd:.1%}",
        "EAD": f"${ead:,.0f}",
        "Asset Correlation (ρ)": f"{rho:.4f}",
        "WCDR (99.9%)": f"{wcdr:.4%}",
        "Capital Requirement (K)": f"{k:.4%}",
        "Risk-Weighted Assets": f"${rwa:,.0f}",
        "Minimum Capital (8%)": f"${capital:,.0f}",
        "Expected Loss": f"${pd * lgd * ead:,.0f}",
        "Unexpected Loss (UL≈K·EAD)": f"${k * ead:,.0f}",
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 60)
    print("  Credit Risk Analytics Suite")
    print("═" * 60)

    # --- 1. Merton Model ---
    print("\n── Merton Structural Model ──")
    print("  Firm: $500M market cap, 30% equity vol, $400M debt face, 1Y horizon")
    merton = MertonModel(
        equity_value=500e6,
        equity_vol=0.30,
        debt_face=400e6,
        risk_free=0.05,
        T=1.0,
        asset_drift=0.08,
        recovery_rate=0.40,
    )
    result = merton.solve()
    print(result.summary())

    print("\n  Sensitivity: Equity vol → Credit Spread")
    print(f"  {'σ_E':>8} {'DD':>10} {'PD(real)':>12} {'Spread(bps)':>14}")
    print(f"  {'─'*46}")
    for sigma_e in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        try:
            m = MertonModel(500e6, sigma_e, 400e6, 0.05, 1.0)
            r = m.solve()
            print(f"  {sigma_e:>8.0%} {r.distance_to_default:>10.4f} {r.prob_default_real:>12.6%} {r.credit_spread_bps:>14.2f}")
        except Exception:
            pass

    # --- 2. CDS Pricing ---
    print("\n── CDS Pricing (5Y, Quarterly Premium) ──")
    print("  Hazard rate: 1.5%/yr, Recovery: 40%, Risk-free: 5%")
    cds = CDSPricer(hazard_rate=0.015, recovery=0.40, risk_free=0.05, maturity=5.0)
    cds_result = cds.price()
    print(cds_result.summary())
    print(f"  Credit Triangle approx: {0.015 * 0.60 * 10000:.2f} bps (vs exact {cds_result.fair_spread_bps:.2f})")

    print("\n  Hazard Rate → CDS Spread Table")
    print(f"  {'λ':>8} {'Surv(5Y)':>10} {'CDS Spread':>12}")
    print(f"  {'─'*32}")
    for lam in [0.005, 0.010, 0.020, 0.050, 0.100, 0.200]:
        c = CDSPricer(hazard_rate=lam, recovery=0.40, risk_free=0.05, maturity=5.0)
        s = c.fair_spread()
        q5 = c.survival_prob(5.0)
        print(f"  {lam:>8.3f} {q5:>10.4%} {s*10000:>12.2f} bps")

    # --- 3. Basel IRB Capital ---
    print("\n── Basel III IRB Capital Requirement ──")
    print("  Corporate loan: PD=1%, LGD=45%, EAD=$10M")
    irb = irb_capital_requirement(pd=0.01, lgd=0.45, ead=10_000_000, maturity=2.5)
    for k, v in irb.items():
        print(f"  {k:<35}: {v}")
