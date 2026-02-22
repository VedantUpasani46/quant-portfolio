"""
P&L Attribution Engine
========================
Decomposes daily portfolio P&L into its sources:
  1. Factor P&L   — how much came from beta to known risk factors?
  2. Alpha P&L    — what's left after removing factor exposures?
  3. Greeks P&L   — for derivatives books: delta, gamma, vega, theta
  4. Brinson attribution — allocation vs selection vs interaction effects

Used daily by every portfolio manager, risk manager, and trader to
understand *why* the portfolio made or lost money.

Without attribution, you cannot:
  - Know if a good day was skill or just market beta
  - Identify which factor is hurting you in a drawdown
  - Verify that your hedge is working
  - Communicate P&L drivers to investors and senior management

The P&L identity:
  Total P&L = Σᵢ wᵢ · rᵢ
            = Σₖ (exposure_k · factor_return_k) + alpha P&L
            = [Delta P&L + Gamma P&L + Vega P&L + Theta P&L] + residual

Factor P&L decomposition:
  P&L_factor = Σₖ βₖ · Fₖ
  where βₖ = portfolio beta to factor k, Fₖ = factor return

Alpha P&L:
  P&L_alpha = Total P&L − P&L_factor
  This should be uncorrelated with known factors if alpha is genuine.

Greeks P&L (Taylor expansion):
  P&L ≈ Δ·ΔS + ½Γ·(ΔS)² + V·Δσ + Θ·Δt + ρ·Δr
  Unexplained residual = actual P&L minus Greeks estimate

References:
  - Brinson, G.P., Hood, L.R. & Beebower, G.L. (1986). Determinants of
    Portfolio Performance. FAJ 42(4), 39–44.
  - Grinold, R.C. & Kahn, R.N. (2000). Active Portfolio Management. McGraw-Hill.
  - Meucci, A. (2007). Risk and Asset Allocation. Springer.
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 19.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Factor P&L attribution
# ---------------------------------------------------------------------------

@dataclass
class FactorAttribution:
    """
    Factor-based P&L decomposition for an equity portfolio.

    Given:
      - Portfolio factor exposures β = (β_market, β_size, β_value, ...)
      - Realised factor returns F = (F_market, F_size, F_value, ...)

    P&L_factor_k = β_k × F_k
    P&L_total_factor = Σₖ β_k × F_k
    P&L_alpha = P&L_total − P&L_total_factor

    Attributes
    ----------
    factor_names : list[str]
    factor_exposures : np.ndarray  Portfolio beta to each factor
    """
    factor_names: list
    factor_exposures: np.ndarray    # portfolio beta to each factor
    portfolio_nav: float

    def attribute(
        self,
        factor_returns: np.ndarray,   # realised factor returns today
        total_pnl: float,             # actual portfolio P&L today
    ) -> dict:
        """
        Decompose total_pnl into factor contributions + alpha.

        Returns dict with per-factor $ P&L and residual alpha.
        """
        factor_pnls = self.factor_exposures * factor_returns * self.portfolio_nav
        total_factor_pnl = float(factor_pnls.sum())
        alpha_pnl = total_pnl - total_factor_pnl

        result = {
            "total_pnl":        total_pnl,
            "total_factor_pnl": total_factor_pnl,
            "alpha_pnl":        alpha_pnl,
            "alpha_pct":        alpha_pnl / self.portfolio_nav * 100,
            "factor_breakdown": {
                name: float(pnl)
                for name, pnl in zip(self.factor_names, factor_pnls)
            },
        }
        return result

    def rolling_attribution(
        self,
        factor_returns_matrix: np.ndarray,  # (T, K)
        daily_pnl_series: np.ndarray,       # (T,)
    ) -> pd.DataFrame:
        """
        Apply attribution across T days.
        Returns DataFrame with daily factor P&Ls and alpha.
        """
        T, K = factor_returns_matrix.shape
        records = []
        for t in range(T):
            result = self.attribute(factor_returns_matrix[t], daily_pnl_series[t])
            row = {"total_pnl": result["total_pnl"],
                   "factor_pnl": result["total_factor_pnl"],
                   "alpha_pnl": result["alpha_pnl"]}
            for name, pnl in result["factor_breakdown"].items():
                row[f"pnl_{name}"] = pnl
            records.append(row)
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Greeks P&L attribution (derivatives)
# ---------------------------------------------------------------------------

@dataclass
class GreeksPnL:
    """
    Decomposes a derivatives book's daily P&L using Taylor expansion.

    P&L ≈ Δ·ΔS + ½Γ·(ΔS)² + V·Δσ + Θ·Δt + ρ·Δr

    The "unexplained" residual should be small if Greeks are accurate.
    Large residual → need higher-order terms, or Greeks are stale.
    """
    delta: float    # position delta (shares equivalent)
    gamma: float    # position gamma
    vega: float     # $ per 1% vol
    theta: float    # $ per calendar day
    rho: float      # $ per 1% rate
    vanna: float = 0.0   # ∂²V/∂S∂σ (optional higher-order)
    volga: float = 0.0   # ∂²V/∂σ²

    def explain_pnl(
        self,
        delta_S: float,        # spot price move ($)
        delta_sigma: float,    # vol move (e.g. 0.01 = +1 vol point)
        delta_t: float,        # time elapsed (calendar days)
        delta_r: float,        # rate move (e.g. 0.0001 = +1bp)
        actual_pnl: float,     # actual observed P&L
        S: float = 100.0,      # spot price for pct calculations
    ) -> dict:
        """
        Decompose actual_pnl into Greeks contributions.
        """
        # First-order
        delta_pnl  = self.delta  * delta_S
        vega_pnl   = self.vega   * delta_sigma * 100   # vega is per 1% vol
        theta_pnl  = self.theta  * delta_t
        rho_pnl    = self.rho    * delta_r * 100       # rho is per 1%

        # Second-order
        gamma_pnl  = 0.5 * self.gamma * delta_S ** 2
        vanna_pnl  = self.vanna * delta_S * delta_sigma * 100
        volga_pnl  = 0.5 * self.volga * (delta_sigma * 100) ** 2

        greeks_total = (delta_pnl + gamma_pnl + vega_pnl + theta_pnl +
                        rho_pnl + vanna_pnl + volga_pnl)
        unexplained  = actual_pnl - greeks_total

        return {
            "actual_pnl":      actual_pnl,
            "greeks_total":    greeks_total,
            "explained_pct":   greeks_total / actual_pnl * 100 if actual_pnl != 0 else 0,
            "unexplained":     unexplained,
            "breakdown": {
                "delta_pnl":   delta_pnl,
                "gamma_pnl":   gamma_pnl,
                "vega_pnl":    vega_pnl,
                "theta_pnl":   theta_pnl,
                "rho_pnl":     rho_pnl,
                "vanna_pnl":   vanna_pnl,
                "volga_pnl":   volga_pnl,
            }
        }


# ---------------------------------------------------------------------------
# Brinson-Hood-Beebower attribution
# ---------------------------------------------------------------------------

@dataclass
class BrinsonAttribution:
    """
    Brinson-Hood-Beebower (1986) performance attribution.

    Decomposes active return into three effects:

    Allocation:  (w_p - w_b)·(r_b - R_b)
      Did over/underweighting sectors add value relative to benchmark?

    Selection:   w_b·(r_p - r_b)
      Did stock selection within each sector add value?

    Interaction: (w_p - w_b)·(r_p - r_b)
      Combined effect of allocation × selection.

    Total active return = Allocation + Selection + Interaction.

    Parameters
    ----------
    sectors : list[str]         Sector labels
    port_weights : np.ndarray   Portfolio sector weights
    bench_weights : np.ndarray  Benchmark sector weights
    port_returns : np.ndarray   Portfolio sector returns
    bench_returns : np.ndarray  Benchmark sector returns
    """
    sectors: list
    port_weights: np.ndarray
    bench_weights: np.ndarray
    port_returns: np.ndarray
    bench_returns: np.ndarray

    def compute(self) -> dict:
        """
        Run Brinson attribution. Returns full decomposition.
        """
        w_p = self.port_weights
        w_b = self.bench_weights
        r_p = self.port_returns
        r_b = self.bench_returns

        R_p = float(w_p @ r_p)    # total portfolio return
        R_b = float(w_b @ r_b)    # total benchmark return
        R_b_total = R_b            # benchmark total for allocation

        # Brinson components
        allocation   = (w_p - w_b) * (r_b - R_b_total)
        selection    = w_b * (r_p - r_b)
        interaction  = (w_p - w_b) * (r_p - r_b)

        total_active = float(allocation.sum() + selection.sum() + interaction.sum())

        sector_detail = {
            sector: {
                "port_weight":    float(w_p[i]),
                "bench_weight":   float(w_b[i]),
                "active_weight":  float(w_p[i] - w_b[i]),
                "port_return":    float(r_p[i]),
                "bench_return":   float(r_b[i]),
                "active_return":  float(r_p[i] - r_b[i]),
                "allocation":     float(allocation[i]),
                "selection":      float(selection[i]),
                "interaction":    float(interaction[i]),
                "total_contrib":  float(allocation[i] + selection[i] + interaction[i]),
            }
            for i, sector in enumerate(self.sectors)
        }

        return {
            "portfolio_return":    R_p,
            "benchmark_return":    R_b,
            "active_return":       R_p - R_b,
            "allocation_total":    float(allocation.sum()),
            "selection_total":     float(selection.sum()),
            "interaction_total":   float(interaction.sum()),
            "attributed_total":    total_active,
            "attribution_error":   (R_p - R_b) - total_active,  # should be ~0
            "sectors":             sector_detail,
        }

    def print_report(self) -> None:
        result = self.compute()
        print(f"\n  {'Sector':<20} {'w_p':>6} {'w_b':>6} {'r_p':>8} {'r_b':>8} "
              f"{'Alloc':>8} {'Select':>8} {'Interact':>8} {'Total':>8}")
        print("  " + "─" * 90)
        for sector, d in result["sectors"].items():
            print(f"  {sector:<20} {d['port_weight']:>6.1%} {d['bench_weight']:>6.1%} "
                  f"{d['port_return']:>8.3%} {d['bench_return']:>8.3%} "
                  f"{d['allocation']:>8.4%} {d['selection']:>8.4%} "
                  f"{d['interaction']:>8.4%} {d['total_contrib']:>8.4%}")
        print("  " + "─" * 90)
        print(f"  {'TOTAL':<20} {'':>6} {'':>6} "
              f"{result['portfolio_return']:>8.3%} {result['benchmark_return']:>8.3%} "
              f"{result['allocation_total']:>8.4%} {result['selection_total']:>8.4%} "
              f"{result['interaction_total']:>8.4%} {result['attributed_total']:>8.4%}")
        print(f"\n  Active return: {result['active_return']:.4%}  |  "
              f"Attribution error: {result['attribution_error']:.6%}")


# ---------------------------------------------------------------------------
# Integrated daily P&L report
# ---------------------------------------------------------------------------

class DailyPnLReport:
    """
    Generates a complete daily P&L attribution report combining:
      - Factor exposure attribution
      - Greeks P&L explanation
      - Brinson sector attribution
      - P&L summary table
    """

    def __init__(
        self,
        portfolio_name: str,
        nav: float,
    ):
        self.portfolio_name = portfolio_name
        self.nav = nav
        self._factor_attr: Optional[FactorAttribution] = None
        self._greeks_pnl: Optional[GreeksPnL] = None
        self._brinson: Optional[BrinsonAttribution] = None

    def set_factor_attribution(self, fa: FactorAttribution) -> None:
        self._factor_attr = fa

    def set_greeks(self, gp: GreeksPnL) -> None:
        self._greeks_pnl = gp

    def set_brinson(self, br: BrinsonAttribution) -> None:
        self._brinson = br

    def generate(
        self,
        actual_pnl: float,
        factor_returns: Optional[np.ndarray] = None,
        spot_move: Optional[float] = None,
        vol_move: Optional[float] = None,
        time_elapsed: float = 1.0,
        rate_move: float = 0.0,
        S: float = 100.0,
    ) -> str:
        lines = [
            f"\n{'═'*68}",
            f"  Daily P&L Attribution Report: {self.portfolio_name}",
            f"{'═'*68}",
            f"  Date NAV: ${self.nav:,.0f}",
            f"  Actual P&L: ${actual_pnl:,.0f}  ({actual_pnl/self.nav*100:.3f}% of NAV)",
        ]

        # Factor attribution
        if self._factor_attr and factor_returns is not None:
            lines.append(f"\n── Factor Attribution ──")
            fa_result = self._factor_attr.attribute(factor_returns, actual_pnl)
            lines.append(f"  {'Factor':<22} {'Exposure':>10} {'Factor Ret':>12} {'P&L ($)':>12}")
            lines.append("  " + "─" * 58)
            for name, pnl in fa_result["factor_breakdown"].items():
                exp = self._factor_attr.factor_exposures[
                    self._factor_attr.factor_names.index(name)]
                idx = self._factor_attr.factor_names.index(name)
                fret = factor_returns[idx]
                lines.append(f"  {name:<22} {exp:>10.3f} {fret:>12.4%} ${pnl:>10,.0f}")
            lines.append(f"  {'':─<58}")
            lines.append(f"  {'Total Factor P&L':<22} {'':>10} {'':>12} "
                         f"${fa_result['total_factor_pnl']:>10,.0f}")
            lines.append(f"  {'Alpha P&L':<22} {'':>10} {'':>12} "
                         f"${fa_result['alpha_pnl']:>10,.0f}")

        # Greeks attribution
        if self._greeks_pnl and spot_move is not None and vol_move is not None:
            lines.append(f"\n── Greeks P&L Explanation ──")
            lines.append(f"  Spot move: {spot_move/S*100:+.2f}%  "
                         f"Vol move: {vol_move*100:+.1f}bp  "
                         f"Time: {time_elapsed:.1f} day(s)")
            gresult = self._greeks_pnl.explain_pnl(
                spot_move, vol_move, time_elapsed, rate_move, actual_pnl, S)
            lines.append(f"\n  {'Greek':<18} {'P&L ($)':>14}")
            lines.append("  " + "─" * 34)
            labels = {
                "delta_pnl":  "Delta",
                "gamma_pnl":  "Gamma (½Γ·ΔS²)",
                "vega_pnl":   "Vega",
                "theta_pnl":  "Theta",
                "rho_pnl":    "Rho",
                "vanna_pnl":  "Vanna",
                "volga_pnl":  "Volga",
            }
            for key, label in labels.items():
                pnl = gresult["breakdown"][key]
                if abs(pnl) > 0.01:
                    lines.append(f"  {label:<18} ${pnl:>12,.0f}")
            lines.append(f"  {'':─<32}")
            lines.append(f"  {'Greeks total':<18} ${gresult['greeks_total']:>12,.0f}")
            lines.append(f"  {'Unexplained':<18} ${gresult['unexplained']:>12,.0f}")
            lines.append(f"  Explained: {gresult['explained_pct']:.1f}% of actual P&L")

        lines.append(f"\n{'═'*68}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 68)
    print("  P&L Attribution Engine")
    print("  Factor + Greeks + Brinson decomposition of daily portfolio P&L")
    print("═" * 68)

    rng = np.random.default_rng(42)

    # ── Factor attribution ─────────────────────────────────────────
    print(f"\n── Factor P&L Attribution (Equity Portfolio) ──")
    print(f"  Factors: Market, Size, Value, Momentum, Low-Volatility")

    factor_names = ["Market (MKT)", "Size (SMB)", "Value (HML)",
                    "Momentum (MOM)", "Low-Vol (BAB)"]
    # Portfolio betas
    betas = np.array([0.95, 0.15, -0.10, 0.30, 0.20])
    nav = 100_000_000  # $100M

    fa = FactorAttribution(
        factor_names=factor_names,
        factor_exposures=betas,
        portfolio_nav=nav,
    )

    # Simulate 20 trading days
    T = 20
    true_factor_returns = rng.multivariate_normal(
        mean=[0.0004, 0.0001, 0.0001, 0.0002, 0.0001],
        cov=np.diag([0.0001, 0.00003, 0.00003, 0.00005, 0.00002]),
        size=T
    )
    true_alpha = rng.normal(0.00003, 0.0001, T)  # small positive daily alpha
    total_pnl_series = (
        (true_factor_returns * betas[None, :]).sum(axis=1) * nav
        + true_alpha * nav
        + rng.normal(0, 5000, T)  # noise
    )

    attr_df = fa.rolling_attribution(true_factor_returns, total_pnl_series)
    print(f"\n  20-day rolling attribution (cumulative $):")
    print(f"  Total P&L:     ${attr_df['total_pnl'].sum():>14,.0f}")
    print(f"  Factor P&L:    ${attr_df['factor_pnl'].sum():>14,.0f}")
    print(f"  Alpha P&L:     ${attr_df['alpha_pnl'].sum():>14,.0f}")
    print(f"\n  Factor breakdown (20-day cumulative):")
    for fname in factor_names:
        col = f"pnl_{fname}"
        print(f"    {fname:<20} ${attr_df[col].sum():>12,.0f}")

    # ── Greeks P&L ─────────────────────────────────────────────────
    print(f"\n── Greeks P&L Explanation (Options Book) ──")
    # Stylised derivatives book
    gk = GreeksPnL(
        delta=1200.0,     # long 1200 share-equivalents
        gamma=45.0,       # positive gamma (long options)
        vega=850.0,       # long vega (in $ per 1% vol)
        theta=-620.0,     # short theta (cost of long options)
        rho=320.0,        # small rho
        vanna=150.0,      # positive vanna
        volga=2100.0,     # positive volga (long vol convexity)
    )

    S = 450.0
    # A down day: S falls 1.5%, vol rises 2 points
    delta_S = -6.75     # S falls $6.75 (-1.5%)
    delta_vol = 0.02    # vol rises 2 pp
    actual_pnl = -3_850.0   # actual loss on the book

    result = gk.explain_pnl(delta_S, delta_vol, 1.0, 0.0, actual_pnl, S)
    print(f"\n  Market moves: S {delta_S:+.2f} ({delta_S/S*100:.2f}%), "
          f"vol {delta_vol*100:+.1f}bp, 1 day")
    print(f"  Actual P&L: ${actual_pnl:,.0f}")
    print(f"\n  {'Greek':<20} {'P&L ($)':>14}")
    print("  " + "─" * 36)
    labels_order = [
        ("delta_pnl", "Delta (Δ·ΔS)"),
        ("gamma_pnl", "Gamma (½Γ·ΔS²)"),
        ("vega_pnl",  "Vega (V·Δσ)"),
        ("theta_pnl", "Theta (Θ·Δt)"),
        ("vanna_pnl", "Vanna (cross)"),
        ("volga_pnl", "Volga (½Vg·Δσ²)"),
    ]
    for key, label in labels_order:
        pnl = result["breakdown"][key]
        print(f"  {label:<20} ${pnl:>12,.0f}")
    print("  " + "─" * 36)
    print(f"  {'Greeks total':<20} ${result['greeks_total']:>12,.0f}")
    print(f"  {'Unexplained':<20} ${result['unexplained']:>12,.0f}")
    print(f"  Explanation ratio: {result['explained_pct']:.1f}%")

    # ── Brinson attribution ────────────────────────────────────────
    print(f"\n── Brinson Sector Attribution ──")
    sectors = ["Technology", "Healthcare", "Financials",
               "Energy", "Consumer", "Industrials"]
    w_port  = np.array([0.30, 0.15, 0.18, 0.05, 0.18, 0.14])
    w_bench = np.array([0.25, 0.13, 0.15, 0.08, 0.20, 0.19])
    r_port  = np.array([0.025, 0.018, 0.012, -0.005, 0.010, 0.008])
    r_bench = np.array([0.022, 0.014, 0.015, -0.003, 0.012, 0.006])

    brinson = BrinsonAttribution(sectors, w_port, w_bench, r_port, r_bench)
    brinson.print_report()

    # ── Integrated report ─────────────────────────────────────────
    print(f"\n── Integrated Daily P&L Report ──")
    report = DailyPnLReport("Equity Long/Short + Options Overlay", nav)
    report.set_factor_attribution(fa)
    report.set_greeks(gk)

    day_pnl = float(total_pnl_series[0])
    print(report.generate(
        actual_pnl=day_pnl,
        factor_returns=true_factor_returns[0],
        spot_move=delta_S,
        vol_move=delta_vol,
        time_elapsed=1.0,
        S=S,
    ))
