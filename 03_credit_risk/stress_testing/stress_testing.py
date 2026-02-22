"""
Stress Testing Framework
=========================
Implements three complementary stress testing methodologies used by
regulators (Fed CCAR/DFAST, ECB SREP, BoE ILAAP, IMF FSAP) and
internal risk management:

  1. Historical scenario replay — 2008 GFC, 2020 COVID, 2022 rates shock
  2. Hypothetical factor shocks — parallel/tilt/twist rate moves, equity crashes
  3. Reverse stress testing — find the portfolio loss that causes insolvency

Regulatory context:
  Basel III Pillar 2 / ICAAP requires banks to demonstrate solvency
  under severe but plausible stress scenarios. CCAR (Fed) and DFAST
  mandate specific macroeconomic scenarios projected 9 quarters forward.

Stress testing vs VaR:
  VaR is a statistical measure (normal market conditions, e.g. 99% 1-day).
  Stress testing covers TAIL EVENTS not captured by historical VaR:
    - Regime changes (correlation breakdown, liquidity crises)
    - Low-probability high-impact scenarios (pandemic, war, sovereign crisis)
    - Non-linear risks (option gamma, credit cliff effects)

Full integration with factor model:
  P&L under stress = Σᵢ (ΔFᵢ · dP/dFᵢ + ½·(ΔFᵢ)²·d²P/dFᵢ²)
  where ΔFᵢ are the shocked risk factors and dP/dF are sensitivities.

References:
  - Basel Committee (2009). Principles for Sound Stress Testing.
  - Fed (2020). CCAR 2020 Severely Adverse Scenario Description.
  - IMF (2012). Macrofinancial Stress Testing — Principles and Practices.
  - Hull, J.C. (2022). Risk Management and Financial Institutions, Ch. 14.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Risk factor definitions
# ---------------------------------------------------------------------------

@dataclass
class RiskFactor:
    """A single market risk factor with a name and current value."""
    name: str
    value: float
    unit: str = "level"   # "level", "bps", "pct"


@dataclass
class Portfolio:
    """
    A portfolio described by its linear and second-order sensitivities
    to a set of risk factors.

    For a bond portfolio:
      DV01[i] = dollar sensitivity of bond i to a 1bp rate shift
      Convexity[i] = second-order rate sensitivity

    For an equity portfolio:
      Delta[i] = price sensitivity to equity move
      Gamma[i] = second-order sensitivity (for options)
    """
    name: str
    nav: float                               # current NAV ($)
    factor_names: list[str]                  # risk factors this portfolio is exposed to
    deltas: np.ndarray                       # dV/dF_i per unit of F_i (first-order)
    gammas: Optional[np.ndarray] = None      # d²V/dF_i² (second-order; optional)
    positions: Optional[dict] = None         # raw position details for reporting

    def pnl_from_shocks(self, shocks: np.ndarray) -> float:
        """
        P&L approximation: ΔV ≈ Σ δᵢ·ΔFᵢ + ½·Σ γᵢ·(ΔFᵢ)²

        Parameters
        ----------
        shocks : np.ndarray   Factor shocks aligned with self.factor_names.
        """
        linear = float(self.deltas @ shocks)
        if self.gammas is not None:
            quadratic = 0.5 * float(self.gammas @ (shocks ** 2))
        else:
            quadratic = 0.0
        return linear + quadratic


# ---------------------------------------------------------------------------
# Historical scenarios
# ---------------------------------------------------------------------------

# Each scenario: dict mapping factor_name → shock (absolute move)
# Rate shocks in bps, equity in decimal (e.g. -0.35 = -35%), credit spreads in bps

HISTORICAL_SCENARIOS: dict[str, dict] = {

    "2008_GFC_Lehman": {
        "description": "Sep–Nov 2008: Lehman collapse. Global equity -40%, credit spreads +500bp, rates -200bp (flight to quality), IG spreads +400bp.",
        "shocks": {
            "US_10Y_rate_bps":      -200.0,
            "US_2Y_rate_bps":       -250.0,
            "EUR_10Y_rate_bps":     -150.0,
            "SP500_equity_pct":     -0.40,
            "EUROSTOXX_equity_pct": -0.45,
            "EM_equity_pct":        -0.55,
            "IG_credit_spread_bps": +400.0,
            "HY_credit_spread_bps": +1200.0,
            "VIX_change":           +40.0,
            "USDEUR_fx_pct":        +0.20,   # USD strengthens
        }
    },

    "2011_EuroZone_Sovereign": {
        "description": "Aug–Nov 2011: Eurozone sovereign crisis. Italian/Spanish spreads +300bp, EUR weakens, bunds rally.",
        "shocks": {
            "US_10Y_rate_bps":      -50.0,
            "EUR_10Y_rate_bps":     -80.0,
            "IT_10Y_spread_bps":    +300.0,
            "ES_10Y_spread_bps":    +250.0,
            "SP500_equity_pct":     -0.20,
            "EUROSTOXX_equity_pct": -0.30,
            "IG_credit_spread_bps": +150.0,
            "HY_credit_spread_bps": +500.0,
            "USDEUR_fx_pct":        +0.10,
        }
    },

    "2020_COVID_March": {
        "description": "Feb–Mar 2020: COVID-19 shock. Fastest equity bear market in history, rates to zero, credit spreads +400bp.",
        "shocks": {
            "US_10Y_rate_bps":      -100.0,
            "US_2Y_rate_bps":       -150.0,
            "EUR_10Y_rate_bps":     -30.0,
            "SP500_equity_pct":     -0.35,
            "EUROSTOXX_equity_pct": -0.38,
            "EM_equity_pct":        -0.30,
            "IG_credit_spread_bps": +200.0,
            "HY_credit_spread_bps": +700.0,
            "VIX_change":           +55.0,
            "USDEUR_fx_pct":        +0.05,
        }
    },

    "2022_Rates_Shock": {
        "description": "Jan–Oct 2022: Fastest rate hiking cycle since 1980. US 10Y +300bp, global equity -25%, IG bonds -20%.",
        "shocks": {
            "US_10Y_rate_bps":      +300.0,
            "US_2Y_rate_bps":       +350.0,
            "EUR_10Y_rate_bps":     +200.0,
            "SP500_equity_pct":     -0.25,
            "EUROSTOXX_equity_pct": -0.22,
            "EM_equity_pct":        -0.30,
            "IG_credit_spread_bps": +100.0,
            "HY_credit_spread_bps": +300.0,
            "VIX_change":           +15.0,
            "USDEUR_fx_pct":        +0.15,
        }
    },

    "1994_Bond_Massacre": {
        "description": "Feb–Dec 1994: Fed surprised markets with 300bp of hikes. Bond market -5%, mortgage market disrupted.",
        "shocks": {
            "US_10Y_rate_bps":      +250.0,
            "US_2Y_rate_bps":       +300.0,
            "EUR_10Y_rate_bps":     +150.0,
            "SP500_equity_pct":     -0.05,
            "IG_credit_spread_bps": +80.0,
            "HY_credit_spread_bps": +200.0,
        }
    },

    "1997_Asian_Crisis": {
        "description": "Jul–Dec 1997: Thai baht devaluation triggers EM contagion. EM currencies -30-50%, EM equity -50%, flight to quality.",
        "shocks": {
            "US_10Y_rate_bps":      -30.0,
            "EM_equity_pct":        -0.50,
            "SP500_equity_pct":     -0.10,
            "EM_credit_spread_bps": +600.0,
            "IG_credit_spread_bps": +50.0,
        }
    },
}

# Hypothetical regulatory scenarios (CCAR-style)
REGULATORY_SCENARIOS: dict[str, dict] = {
    "CCAR_Severely_Adverse_2024": {
        "description": "Fed CCAR Severely Adverse: Severe global recession. US unemployment +6%, equity -55%, RE -40%.",
        "shocks": {
            "US_10Y_rate_bps":      -200.0,
            "US_2Y_rate_bps":       -300.0,
            "SP500_equity_pct":     -0.55,
            "EUROSTOXX_equity_pct": -0.50,
            "IG_credit_spread_bps": +500.0,
            "HY_credit_spread_bps": +1500.0,
            "VIX_change":           +50.0,
        }
    },

    "ECB_Adverse_2024": {
        "description": "ECB SREP Adverse: stagflation + financial instability. EUR rates +150bp then -100bp, equity -30%.",
        "shocks": {
            "EUR_10Y_rate_bps":     +150.0,
            "IT_10Y_spread_bps":    +200.0,
            "EUROSTOXX_equity_pct": -0.30,
            "SP500_equity_pct":     -0.20,
            "IG_credit_spread_bps": +200.0,
            "USDEUR_fx_pct":        +0.08,
        }
    },
}


# ---------------------------------------------------------------------------
# Hypothetical scenario builder
# ---------------------------------------------------------------------------

class HypotheticalScenario:
    """
    Build custom factor shock scenarios for specific risk concerns.

    Scenario types:
      - Parallel rate shift: all rates move by the same amount
      - Steepening/flattening: short rates and long rates move differently
      - Equity crash: equity falls by a specified percentage
      - Credit widening: credit spreads widen across the board
      - Combined macro shock: multiple correlated factor moves
    """

    @staticmethod
    def parallel_rate_shift(shift_bps: float) -> dict:
        """All tenors shift by the same amount."""
        return {
            "US_10Y_rate_bps": shift_bps,
            "US_2Y_rate_bps":  shift_bps,
            "EUR_10Y_rate_bps": shift_bps,
            "description": f"Parallel rate shift: {shift_bps:+.0f}bp all tenors"
        }

    @staticmethod
    def bear_steepener(short_shift_bps: float, long_shift_bps: float) -> dict:
        """
        Bear steepener: long rates rise more than short rates.
        Common in an inflation-fear scenario.
        """
        return {
            "US_2Y_rate_bps":  short_shift_bps,
            "US_10Y_rate_bps": long_shift_bps,
            "description": f"Bear steepener: 2Y {short_shift_bps:+.0f}bp, 10Y {long_shift_bps:+.0f}bp"
        }

    @staticmethod
    def bull_flattener(short_shift_bps: float, long_shift_bps: float) -> dict:
        """
        Bull flattener: short rates fall more than long rates.
        Common in recession / flight-to-quality scenario.
        """
        return {
            "US_2Y_rate_bps":  short_shift_bps,
            "US_10Y_rate_bps": long_shift_bps,
            "description": f"Bull flattener: 2Y {short_shift_bps:+.0f}bp, 10Y {long_shift_bps:+.0f}bp"
        }

    @staticmethod
    def equity_crash(equity_pct: float, credit_widening_bps: float = 200.0) -> dict:
        """
        Equity crash with correlated credit widening.
        Captures the typical equity-credit correlation in risk-off events.
        """
        return {
            "SP500_equity_pct":     equity_pct,
            "EUROSTOXX_equity_pct": equity_pct * 1.05,  # EU tends to fall slightly more
            "EM_equity_pct":        equity_pct * 1.20,
            "IG_credit_spread_bps": credit_widening_bps,
            "HY_credit_spread_bps": credit_widening_bps * 3.5,
            "VIX_change":           max(0, -equity_pct * 100),
            "description": f"Equity crash {equity_pct:.0%} with credit widening"
        }

    @staticmethod
    def rate_credit_combo(rate_bps: float, ig_bps: float, hy_bps: float) -> dict:
        """Combined rate and credit shock."""
        return {
            "US_10Y_rate_bps": rate_bps,
            "US_2Y_rate_bps":  rate_bps,
            "IG_credit_spread_bps": ig_bps,
            "HY_credit_spread_bps": hy_bps,
            "description": f"Rate {rate_bps:+.0f}bp + IG {ig_bps:+.0f}bp + HY {hy_bps:+.0f}bp"
        }


# ---------------------------------------------------------------------------
# Stress test engine
# ---------------------------------------------------------------------------

@dataclass
class StressResult:
    """Result of a single stress scenario applied to a portfolio."""
    scenario_name: str
    description: str
    portfolio_name: str
    pnl: float                    # estimated P&L under stress
    pnl_pct_nav: float            # P&L as % of NAV
    factor_contributions: dict    # which factors drove the P&L
    breached_limit: bool          # did loss exceed the stop-loss limit?
    stop_loss_limit: float


class StressTestEngine:
    """
    Applies stress scenarios to a portfolio and produces a full stress report.

    Usage
    -----
    >>> engine = StressTestEngine(portfolio, stop_loss_limit=-0.15)
    >>> results = engine.run_all_scenarios()
    >>> engine.print_report(results)
    """

    def __init__(self, portfolio: Portfolio, stop_loss_limit: float = -0.20):
        """
        Parameters
        ----------
        portfolio : Portfolio
        stop_loss_limit : float
            P&L fraction below which a breach is flagged (e.g. -0.15 = -15% of NAV).
        """
        self.portfolio = portfolio
        self.stop_loss = stop_loss_limit

    def apply_scenario(self, scenario_name: str, scenario: dict) -> StressResult:
        """Apply a single scenario and return the P&L attribution."""
        shocks = np.zeros(len(self.portfolio.factor_names))
        factor_contribs = {}

        for i, fname in enumerate(self.portfolio.factor_names):
            shock = scenario.get("shocks", scenario).get(fname, 0.0)
            shocks[i] = shock
            # Contribution of this factor to total P&L
            linear_contrib = self.portfolio.deltas[i] * shock
            gamma_contrib = 0.0
            if self.portfolio.gammas is not None:
                gamma_contrib = 0.5 * self.portfolio.gammas[i] * shock ** 2
            factor_contribs[fname] = linear_contrib + gamma_contrib

        total_pnl = self.portfolio.pnl_from_shocks(shocks)
        pnl_pct = total_pnl / self.portfolio.nav

        desc = scenario.get("description", scenario_name)

        return StressResult(
            scenario_name=scenario_name,
            description=desc,
            portfolio_name=self.portfolio.name,
            pnl=total_pnl,
            pnl_pct_nav=pnl_pct,
            factor_contributions=factor_contribs,
            breached_limit=pnl_pct < self.stop_loss,
            stop_loss_limit=self.stop_loss,
        )

    def run_all_scenarios(
        self,
        include_historical: bool = True,
        include_regulatory: bool = True,
        custom_scenarios: Optional[dict] = None,
    ) -> list[StressResult]:
        """Run all registered scenarios and return sorted results (worst first)."""
        all_scenarios = {}
        if include_historical:
            all_scenarios.update(HISTORICAL_SCENARIOS)
        if include_regulatory:
            all_scenarios.update(REGULATORY_SCENARIOS)
        if custom_scenarios:
            all_scenarios.update(custom_scenarios)

        results = [self.apply_scenario(name, sc) for name, sc in all_scenarios.items()]
        return sorted(results, key=lambda r: r.pnl)   # worst P&L first

    def reverse_stress_test(self, target_loss_pct: float = -0.20) -> dict:
        """
        Reverse stress test: find the minimum parallel equity crash (or rate
        shock) that would produce a loss equal to target_loss_pct of NAV.

        This answers: "What scenario breaks us?"
        Required by Basel III / Pillar 2 and by the Fed's horizontal review.
        """
        target_loss = target_loss_pct * self.portfolio.nav
        portfolio = self.portfolio

        results = {}

        # 1. Equity crash reverse: find equity_pct that gives target_loss
        eq_factors = [i for i, n in enumerate(portfolio.factor_names)
                      if "equity" in n.lower()]
        if eq_factors:
            # Binary search: find equity_pct such that pnl = target_loss
            def equity_pnl(eq_pct):
                shocks = np.zeros(len(portfolio.factor_names))
                for i in eq_factors:
                    shocks[i] = eq_pct
                return portfolio.pnl_from_shocks(shocks) - target_loss

            from scipy.optimize import brentq
            try:
                eq_pct_break = brentq(equity_pnl, -0.99, 0.0)
                results["equity_crash_breakeven"] = {
                    "equity_shock_pct": eq_pct_break,
                    "interpretation": f"Portfolio breaks (loses {abs(target_loss_pct):.0%} NAV) at equity crash of {eq_pct_break:.1%}"
                }
            except Exception:
                results["equity_crash_breakeven"] = {"error": "Could not solve"}

        # 2. Parallel rate rise reverse
        rate_factors = [i for i, n in enumerate(portfolio.factor_names) if "rate" in n.lower()]
        if rate_factors:
            def rate_pnl(rate_bps):
                shocks = np.zeros(len(portfolio.factor_names))
                for i in rate_factors:
                    shocks[i] = rate_bps
                return portfolio.pnl_from_shocks(shocks) - target_loss

            try:
                rate_break = brentq(rate_pnl, -1000, 1000)
                results["rate_shock_breakeven"] = {
                    "rate_shock_bps": rate_break,
                    "interpretation": f"Portfolio breaks at {rate_break:+.0f}bp parallel rate move"
                }
            except Exception:
                results["rate_shock_breakeven"] = {"error": "Could not solve"}

        return results

    def print_report(self, results: list[StressResult]) -> None:
        """Print a formatted stress test report."""
        port = self.portfolio
        print(f"\n{'═' * 72}")
        print(f"  Stress Test Report: {port.name}")
        print(f"  Current NAV: ${port.nav:,.0f}  |  Stop-loss limit: {self.stop_loss:.0%} of NAV")
        print(f"  Stop-loss = ${self.stop_loss * port.nav:,.0f}")
        print(f"{'═' * 72}")
        print(f"\n  {'Scenario':<35} {'P&L ($M)':>10} {'P&L %':>8} {'Breach?':>8}")
        print("  " + "─" * 65)
        for r in results:
            breach_str = "⚠ BREACH" if r.breached_limit else ""
            pnl_m = r.pnl / 1e6
            print(f"  {r.scenario_name:<35} {pnl_m:>10.2f}  {r.pnl_pct_nav:>7.2%}  {breach_str}")

        # Top risk drivers across all scenarios
        breaches = [r for r in results if r.breached_limit]
        print(f"\n  Scenarios breaching stop-loss: {len(breaches)}/{len(results)}")

        # Worst scenario deep-dive
        worst = results[0]
        print(f"\n  Worst Scenario: {worst.scenario_name}")
        print(f"  {worst.description}")
        print(f"  P&L: ${worst.pnl/1e6:.2f}M  ({worst.pnl_pct_nav:.2%} of NAV)")
        print(f"\n  Top factor contributions:")
        sorted_contribs = sorted(worst.factor_contributions.items(),
                                 key=lambda x: x[1])
        for fname, contrib in sorted_contribs[:5]:
            if abs(contrib) > 1000:
                print(f"    {fname:<35} ${contrib/1e6:>8.3f}M")


# ---------------------------------------------------------------------------
# Helper: build a realistic mixed portfolio for demo
# ---------------------------------------------------------------------------

def build_demo_portfolio() -> Portfolio:
    """
    A $500M mixed portfolio with rate, credit, and equity exposures.
    Typical of a bank's Treasury or insurance company's investment portfolio.

    Risk factor sensitivities:
      Rate factors: DV01 in $/bp
      Equity factors: $ per 1% move = NAV * equity_weight / 100
      Credit spread factors: DV01 in $/bp
    """
    factor_names = [
        "US_10Y_rate_bps",       # Rate risk (bond portfolio)
        "US_2Y_rate_bps",        # Short-rate risk
        "EUR_10Y_rate_bps",      # EUR rate risk
        "SP500_equity_pct",      # Equity (as fraction: -0.35 = -35%)
        "EUROSTOXX_equity_pct",  # EU equity
        "EM_equity_pct",         # EM equity
        "IG_credit_spread_bps",  # IG credit spread DV01
        "HY_credit_spread_bps",  # HY credit spread DV01
        "IT_10Y_spread_bps",     # Italian sovereign spread
        "ES_10Y_spread_bps",     # Spanish sovereign spread
        "VIX_change",            # Vol exposure (short vol position)
        "USDEUR_fx_pct",         # USD/EUR FX
    ]

    nav = 500_000_000  # $500M

    # Deltas: $ P&L per unit shock
    # Rate DV01: $500K/bp (typical for $500M bond portfolio, 10Y duration)
    deltas = np.array([
        -500_000,    # US 10Y DV01: lose $500K per bp rise
        -150_000,    # US 2Y DV01 (shorter duration)
        -200_000,    # EUR 10Y DV01 (smaller EUR allocation)
        nav * 0.25,  # Equity delta: 25% of NAV in equities ($ per 1% move = 1.25M per 1%)
        nav * 0.10,  # EU equity 10%
        nav * 0.05,  # EM equity 5%
        -100_000,    # IG credit DV01 (long credit, lose on widening)
        -50_000,     # HY credit DV01
        -80_000,     # Italian BTP exposure
        -40_000,     # Spanish bonos
        -20_000,     # Short vol (lose $20K per VIX point rise)
        nav * 0.05,  # 5% unhedged FX: gain when USD strengthens
    ])

    # Gamma (convexity): second-order effects
    gammas = np.array([
        200,         # Rate convexity: gain from convexity on large rate moves
        50,
        80,
        0,           # Linear equity exposure
        0,
        0,
        30,          # Credit convexity
        20,
        25,
        15,
        -5_000,      # Short gamma on vol (short vol = negative gamma)
        0,
    ])

    return Portfolio(
        name="Treasury / Investment Portfolio ($500M)",
        nav=nav,
        factor_names=factor_names,
        deltas=deltas,
        gammas=gammas,
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 72)
    print("  Stress Testing Framework")
    print("  Historical scenarios + Hypothetical shocks + Reverse stress test")
    print("═" * 72)

    portfolio = build_demo_portfolio()
    engine = StressTestEngine(portfolio, stop_loss_limit=-0.15)  # -15% of NAV

    # Run all scenarios
    results = engine.run_all_scenarios()
    engine.print_report(results)

    # Custom hypothetical scenarios
    print(f"\n── Custom Hypothetical Scenarios ──")
    custom = {
        "Bear_Steepener_100bp": HypotheticalScenario.bear_steepener(-50, +150),
        "Equity_Crash_30pct":   HypotheticalScenario.equity_crash(-0.30, credit_widening_bps=200),
        "Rate_Rise_200bp":      HypotheticalScenario.parallel_rate_shift(+200),
    }
    for name, sc in custom.items():
        res = engine.apply_scenario(name, sc)
        breach = " ⚠ BREACH" if res.breached_limit else ""
        print(f"  {name:<35} ${res.pnl/1e6:>8.2f}M  ({res.pnl_pct_nav:>7.2%}){breach}")

    # Reverse stress test
    print(f"\n── Reverse Stress Test (find break point at -15% NAV = -$75M) ──")
    rss = engine.reverse_stress_test(target_loss_pct=-0.15)
    for key, val in rss.items():
        if "error" not in val:
            print(f"\n  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
