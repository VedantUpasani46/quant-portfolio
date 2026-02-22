"""
Credit Valuation Adjustment (CVA) and XVA Framework
======================================================
CVA is the market value of counterparty credit risk on a derivatives portfolio.
It's the difference between the risk-free value and the true value accounting
for the possibility that your counterparty defaults.

CVA = risk-free PV − true PV (with default risk)
    = E[loss from counterparty default]
    = ∫₀ᵀ DF(t) · EE(t) · dPD(t)   (simplified, no wrong-way risk)

where:
  DF(t)   = OIS discount factor to time t
  EE(t)   = Expected Exposure at time t (average of max(V(t), 0))
  dPD(t)  = Marginal probability of default in [t, t+dt]
           = PD(0,t) − PD(0,t+dt)   (from CDS spreads)

Discrete approximation (industry standard):
  CVA ≈ (1 − R) · Σᵢ DF(tᵢ) · EE(tᵢ) · [PD(0,tᵢ₋₁) − PD(0,tᵢ)]
  where R = recovery rate (typically 40% for senior unsecured)

The XVA family:
  CVA  — Credit Valuation Adjustment (counterparty default risk)
  DVA  — Debit Valuation Adjustment (own credit risk: your right to default)
  FVA  — Funding Valuation Adjustment (cost of funding uncollateralised trades)
  MVA  — Margin Valuation Adjustment (cost of initial margin posting)
  KVA  — Capital Valuation Adjustment (cost of regulatory capital)

Bilateral CVA:
  BCVA = CVA − DVA
  DVA: if YOU default, your counterparty loses money → benefit to you
  Controversial: accounting recognizes DVA gain when your own credit worsens

Expected Exposure metrics:
  EE(t)    = E[max(V(t), 0)]      Average over all scenarios
  PFE(t,p) = percentile_p(max(V(t), 0))  Potential Future Exposure (95% typical)
  EEPE     = time-average of EE over the margin period of risk
  EPE      = Effective Positive Exposure (Basel regulatory metric)

CDS-implied survival probability:
  PD(0,t) = 1 − exp(−λ·t)   where λ = CDS_spread / (1 − R)
  This is the constant hazard rate model (widely used in practice).

References:
  - Gregory, J. (2015). The xVA Challenge, 3rd ed. Wiley.
  - Brigo, D. & Morini, M. (2011). Close-Out Convention Tensions.
    Risk Magazine.
  - Basel Committee (2011). Basel III Counterparty Credit Risk Rules.
  - Hull, J.C. & White, A. (2012). CVA and Wrong-Way Risk. FAJ 68(5).
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Survival probability from CDS spreads
# ---------------------------------------------------------------------------

def cds_hazard_rate(cds_spread_bp: float, recovery: float = 0.40) -> float:
    """
    Constant hazard rate implied by a flat CDS spread.
    λ = spread / (1 − R)
    CDS spread is the annual premium paid on CDS protection.
    """
    spread = cds_spread_bp / 10000
    return spread / (1 - recovery)


def survival_probability(t: float, hazard_rate: float) -> float:
    """Q(survive to t) = exp(−λ·t)"""
    return math.exp(-hazard_rate * t)


def default_probability(t1: float, t2: float, hazard_rate: float) -> float:
    """Marginal PD in [t1, t2]: Q(default in [t1,t2]) = SP(t1) − SP(t2)"""
    return survival_probability(t1, hazard_rate) - survival_probability(t2, hazard_rate)


# ---------------------------------------------------------------------------
# Expected Exposure simulation (for an IRS)
# ---------------------------------------------------------------------------

def simulate_irs_exposure(
    fixed_rate: float,
    maturity: float,
    notional: float,
    initial_rate: float,
    rate_vol: float,        # annualised vol of short rate
    n_steps: int = 20,
    n_paths: int = 5000,
    payment_freq: int = 2,
    payer: bool = True,
    seed: int = 42,
) -> dict:
    """
    Simulate exposure profile for a fixed-for-floating IRS.
    Uses simplified Hull-White rate paths to reprice the swap at each step.

    At each time step t:
      V(t) = PV of remaining swap cash flows, using simulated short rate r(t)
      Exposure(t) = max(V(t), 0)    if we receive from counterparty
      EE(t) = E[Exposure(t)]
      PFE(t, 95%) = 95th percentile of Exposure(t)

    Parameters
    ----------
    payer: True = we pay fixed, receive floating (exposure > 0 when rates rise)
    """
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    times = np.linspace(0, maturity, n_steps + 1)

    # Simulate short rates via simple GBM (Hull-White with zero mean-reversion for clarity)
    a, sigma = 0.10, rate_vol
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = initial_rate

    for step in range(n_steps):
        Z = rng.standard_normal(n_paths)
        drift = a * (initial_rate - rates[:, step]) * dt  # weak mean reversion
        diffusion = sigma * math.sqrt(dt) * Z
        rates[:, step + 1] = rates[:, step] + drift + diffusion

    # Reprice swap at each time step using simulated rate as par rate proxy
    exposures = np.zeros((n_paths, n_steps + 1))
    dt_pay = 1.0 / payment_freq

    for step in range(n_steps + 1):
        t = times[step]
        remaining = maturity - t
        if remaining < dt_pay / 2:
            exposures[:, step] = 0
            continue

        # Swap PV ≈ (simulated_rate − fixed_rate) × DV01
        # DV01 for remaining tenor (annuity at 1bp)
        n_remaining = int(remaining * payment_freq)
        if n_remaining == 0:
            continue

        # Approximate annuity using simulated rate for discounting
        r_sim = rates[:, step]
        r_sim = np.maximum(r_sim, 0.001)  # floor at 0

        # Annuity: Σᵢ τ · exp(−r_sim · i·dt_pay)
        annuity = np.zeros(n_paths)
        for i in range(1, n_remaining + 1):
            annuity += dt_pay * np.exp(-r_sim * i * dt_pay)

        sign = 1 if payer else -1
        # Payer swap value = (float_rate − fixed_rate) × annuity × notional
        # Floating ≈ r_sim (since flat yield curve), Fixed = fixed_rate
        swap_val = sign * (r_sim - fixed_rate) * annuity * notional
        exposures[:, step] = np.maximum(swap_val, 0)

    ee  = exposures.mean(axis=0)
    pfe = np.percentile(exposures, 95, axis=0)
    return {"times": times, "ee": ee, "pfe95": pfe, "exposures": exposures}


# ---------------------------------------------------------------------------
# CVA calculation
# ---------------------------------------------------------------------------

def compute_cva(
    ee: np.ndarray,               # Expected Exposure profile
    times: np.ndarray,            # time grid
    hazard_rate: float,           # counterparty hazard rate λ
    discount_rates: float,        # flat OIS rate for discounting
    recovery: float = 0.40,
) -> dict:
    """
    CVA using the industry-standard discrete integral:

    CVA = (1-R) · Σᵢ DF(tᵢ) · EE(tᵢ) · [SP(tᵢ₋₁) − SP(tᵢ)]

    where SP = survival probability, DF = OIS discount factor.
    """
    lgd = 1 - recovery    # Loss Given Default

    cva = 0.0
    cva_profile = np.zeros(len(times))
    cva_density = np.zeros(len(times))

    for i in range(1, len(times)):
        t = times[i]
        df   = math.exp(-discount_rates * t)
        sp_prev = survival_probability(times[i-1], hazard_rate)
        sp_curr = survival_probability(t, hazard_rate)
        marginal_pd = sp_prev - sp_curr

        contribution = lgd * df * ee[i] * marginal_pd
        cva += contribution
        cva_profile[i] = cva
        cva_density[i] = contribution

    return {
        "cva": cva,
        "cva_bps": cva / (ee[0] if ee[0] > 0 else 1) * 10000,
        "cva_profile": cva_profile,
        "cva_density": cva_density,
        "lgd": lgd,
        "hazard_rate": hazard_rate,
    }


def compute_dva(
    ee_negative: np.ndarray,    # Expected Negative Exposure (own default exposure)
    times: np.ndarray,
    own_hazard_rate: float,     # our own hazard rate
    discount_rate: float,
    recovery_self: float = 0.40,
) -> float:
    """
    DVA: benefit from our own right to default.
    DVA = (1-R_self) · Σᵢ DF(tᵢ) · ENE(tᵢ) · [SP_self(tᵢ₋₁) − SP_self(tᵢ)]
    ENE = E[max(-V(t), 0)] = Expected Negative Exposure
    """
    lgd_self = 1 - recovery_self
    dva = 0.0
    for i in range(1, len(times)):
        t = times[i]
        df = math.exp(-discount_rate * t)
        sp_prev = survival_probability(times[i-1], own_hazard_rate)
        sp_curr = survival_probability(t, own_hazard_rate)
        dva += lgd_self * df * ee_negative[i] * (sp_prev - sp_curr)
    return dva


# ---------------------------------------------------------------------------
# FVA
# ---------------------------------------------------------------------------

def compute_fva(
    ee: np.ndarray,               # positive exposure (funding cost for receivables)
    times: np.ndarray,
    funding_spread: float,        # our funding spread over OIS (in decimal)
    discount_rate: float,
) -> float:
    """
    FVA: cost of funding uncollateralised derivatives.

    FVA ≈ −Σᵢ funding_spread · DF(tᵢ) · EE(tᵢ) · Δt
    Interpretation: we must fund the expected positive exposure at a spread
    above the risk-free rate — this has a cost.
    """
    fva = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        df = math.exp(-discount_rate * times[i])
        fva -= funding_spread * df * ee[i] * dt
    return fva


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  CVA / XVA: Counterparty Credit Risk Valuation Adjustment")
    print("  CVA, DVA, FVA for an interest rate swap book")
    print("═" * 66)

    notional     = 10_000_000    # $10M
    fixed_rate   = 0.050         # 5% fixed
    initial_rate = 0.048         # 4.8% current rate
    maturity     = 5.0           # 5-year swap
    discount_r   = 0.045         # OIS discount rate
    rate_vol     = 0.015         # 1.5% short-rate vol
    recovery     = 0.40          # 40% recovery on default

    # ── CDS-implied hazard rates ─────────────────────────────────
    print(f"\n── Counterparty Credit Profile (CDS-implied) ──")
    print(f"\n  {'CDS Spread':>12} {'Hazard Rate λ':>16} {'1Y PD':>10} {'5Y PD':>10}")
    print("  " + "─" * 52)
    for cds_bp in [50, 100, 200, 300, 500]:
        lam = cds_hazard_rate(cds_bp, recovery)
        pd1 = 1 - survival_probability(1, lam)
        pd5 = 1 - survival_probability(5, lam)
        print(f"  {cds_bp:>12}bp {lam:>16.4f} {pd1:>10.4%} {pd5:>10.4%}")

    # Use a single counterparty: CDS = 150bp
    cds_cp = 150
    lambda_cp  = cds_hazard_rate(cds_cp, recovery)
    lambda_self = cds_hazard_rate(80, recovery)    # our own CDS spread
    funding_spread = 0.0060                         # 60bp funding spread

    print(f"\n  Counterparty: CDS = {cds_cp}bp → λ = {lambda_cp:.4f}, 5Y PD = {1-survival_probability(5,lambda_cp):.2%}")
    print(f"  Ourselves:    CDS = 80bp  → λ = {lambda_self:.4f}, 5Y PD = {1-survival_probability(5,lambda_self):.2%}")

    # ── Exposure simulation ───────────────────────────────────────
    print(f"\n── Expected Exposure Profile (5Y Payer IRS) ──")
    print(f"  5000 Monte Carlo paths, 20 time steps")

    exp_result = simulate_irs_exposure(
        fixed_rate=fixed_rate, maturity=maturity, notional=notional,
        initial_rate=initial_rate, rate_vol=rate_vol,
        n_steps=20, n_paths=5000, payer=True
    )

    times = exp_result["times"]
    ee    = exp_result["ee"]
    pfe   = exp_result["pfe95"]

    # Negative exposure for DVA (when swap is out-of-the-money to us)
    exposures_all = exp_result["exposures"]
    ene = np.maximum(-exposures_all + notional * abs(initial_rate - fixed_rate),
                      0).mean(axis=0)  # simplified ENE

    print(f"\n  {'Time':>6} {'EE ($)':>14} {'PFE 95% ($)':>14}")
    print("  " + "─" * 38)
    for i, (t, e, p) in enumerate(zip(times, ee, pfe)):
        if i % 4 == 0:
            print(f"  {t:>6.2f}Y ${e:>12,.0f} ${p:>12,.0f}")

    # ── CVA calculation ───────────────────────────────────────────
    print(f"\n── CVA Calculation ──")
    cva_result = compute_cva(ee, times, lambda_cp, discount_r, recovery)

    print(f"\n  CVA (counterparty risk charge):  ${cva_result['cva']:>10,.0f}")
    print(f"  CVA as % of notional:            {cva_result['cva']/notional*100:>10.4f}%")
    print(f"  CVA in basis points:             {cva_result['cva']/notional*10000:>10.2f}bp")

    # DVA
    dva = compute_dva(ene, times, lambda_self, discount_r)
    print(f"\n  DVA (own credit benefit):        ${dva:>10,.0f}")
    print(f"  BCVA (bilateral = CVA − DVA):    ${cva_result['cva'] - dva:>10,.0f}")

    # FVA
    fva = compute_fva(ee, times, funding_spread, discount_r)
    print(f"\n  FVA (funding cost):              ${fva:>10,.0f}")
    total_xva = cva_result['cva'] - dva + abs(fva)
    print(f"\n  Total XVA (CVA − DVA + |FVA|):  ${total_xva:>10,.0f}")
    print(f"  True PV = Risk-free PV − Total XVA")

    # ── Sensitivity to CDS spread ─────────────────────────────────
    print(f"\n── CVA Sensitivity to Counterparty CDS Spread ──")
    print(f"\n  {'CDS (bp)':>10} {'CVA ($)':>14} {'CVA (bp)':>12} {'CVA/DV01':>12}")
    print("  " + "─" * 50)
    # Approximate DV01 of the swap
    dv01_swap = maturity * notional / 10000 * 0.5  # rough

    for cds_bp in [50, 100, 150, 200, 300, 500]:
        lam = cds_hazard_rate(cds_bp, recovery)
        cva_s = compute_cva(ee, times, lam, discount_r, recovery)["cva"]
        cva_bps = cva_s / notional * 10000
        print(f"  {cds_bp:>10}bp ${cva_s:>12,.0f} {cva_bps:>12.2f}bp {cva_s/dv01_swap:>12.2f}×")

    print(f"""
── Key Concepts ──

  CVA is a P&L charge booked at trade inception:
    Dealer sells 5Y swap to BBB-rated counterparty (CDS=150bp)
    Risk-free value = $178,000  (swap is in-the-money to dealer)
    CVA charge = −${cva_result['cva']:,.0f}  (expected loss from CP default)
    True value = ${notional*(initial_rate-fixed_rate)*maturity - cva_result['cva']:,.0f}
    CVA desk hedges this with CDS protection

  DVA (controversial): When YOUR credit worsens, your DVA 'gains'
    → Some banks reported DVA gains during the 2008 crisis
    → Basel III now restricts DVA recognition for regulatory capital

  FVA (since 2012): Banks cannot borrow at OIS to fund swaps
    → Gap between OIS and bank's funding cost (typically 30-100bp)
    → FVA = funding cost of uncollateralised receivables

  FRTB (2022): Regulatory capital for CVA moved to SA-CVA / IMA-CVA
    → Banks must hold capital for CVA risk (sensitivity-based approach)
    """)
