"""
Options Portfolio Risk Engine
================================
Aggregates Greeks and runs scenario analysis across a multi-leg options book.
Used by options market-makers, derivatives desks, and risk functions.

Functionality:
  1. Position-level Greeks: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga
  2. Portfolio-level aggregation (by underlying, by expiry, total)
  3. P&L scenarios: price × vol grid (the "risk slide" or "stress matrix")
  4. Dollar Greeks: translate unit Greeks into $ risk per market move
  5. Gamma P&L: estimate daily P&L from realized vs implied vol
  6. Hedging targets: delta-neutral and vega-neutral hedge quantities

Higher-order Greeks:
  Vanna (∂Δ/∂σ = ∂V/∂σ):    How delta changes with vol. Important for
                              skew positions and barrier options.
  Volga (∂²V/∂σ²):          Convexity in vol. Long options = long volga.
  Charm (∂Δ/∂t):            How delta changes with time. Important for
                              overnight hedges (delta at open vs close).
  Speed (∂Γ/∂S):            Rate of change of gamma with price.

The "greeks ladder":
  P&L ≈ Δ·ΔS + ½Γ·(ΔS)² + V·Δσ + Θ·Δt + ρ·Δr
        + Vanna·ΔS·Δσ + ½Volga·(Δσ)²  [second order]

References:
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 19-20.
  - Natenberg, S. (1994). Option Volatility & Pricing. McGraw-Hill.
  - Haug, E.G. (2007). The Complete Guide to Option Pricing Formulas. McGraw-Hill.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# BSM Greeks (single option)
# ---------------------------------------------------------------------------

def bsm_greeks(S: float, K: float, T: float, r: float, sigma: float,
               option_type: Literal["call", "put"] = "call",
               q: float = 0.0) -> dict:
    """
    Full BSM Greeks for a single European option.

    Returns dict with: price, delta, gamma, vega, theta, rho, vanna, volga,
    charm, speed, color, ultima.
    """
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return {"price": intrinsic, "delta": 0.0, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "rho": 0.0, "vanna": 0.0, "volga": 0.0,
                "charm": 0.0, "speed": 0.0}

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    Kd = K * math.exp(-r * T)
    Sq = S * math.exp(-q * T)

    if option_type == "call":
        price = Sq * Nd1 - Kd * Nd2
        delta = math.exp(-q * T) * Nd1
        rho = K * T * math.exp(-r * T) * Nd2 / 100   # per 1% change
        theta = (-(Sq * nd1 * sigma) / (2 * math.sqrt(T))
                 - r * Kd * Nd2 + q * Sq * Nd1) / 365
    else:
        price = Kd * norm.cdf(-d2) - Sq * norm.cdf(-d1)
        delta = -math.exp(-q * T) * norm.cdf(-d1)
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (-(Sq * nd1 * sigma) / (2 * math.sqrt(T))
                 + r * Kd * norm.cdf(-d2) - q * Sq * norm.cdf(-d1)) / 365

    gamma = math.exp(-q * T) * nd1 / (S * sigma * math.sqrt(T))
    vega = Sq * nd1 * math.sqrt(T) / 100  # per 1% vol change

    # Higher-order Greeks
    vanna = -math.exp(-q * T) * nd1 * d2 / sigma         # ∂²V/∂S∂σ
    volga = vega * d1 * d2 / sigma                        # ∂²V/∂σ²  (per 1%² )
    charm = (-math.exp(-q * T) * nd1 *
             (2 * (r - q) * T - d2 * sigma * math.sqrt(T)) /
             (2 * T * sigma * math.sqrt(T))) / 365
    speed = -gamma / S * (d1 / (sigma * math.sqrt(T)) + 1)  # ∂³V/∂S³

    return {
        "price": price, "delta": delta, "gamma": gamma,
        "vega": vega, "theta": theta, "rho": rho,
        "vanna": vanna, "volga": volga, "charm": charm, "speed": speed,
        "d1": d1, "d2": d2,
    }


# ---------------------------------------------------------------------------
# Option position
# ---------------------------------------------------------------------------

@dataclass
class OptionPosition:
    """
    A single option position in the book.

    Attributes
    ----------
    underlying : str   Ticker of the underlying.
    option_type : str  'call' or 'put'.
    K : float          Strike price.
    T : float          Time to expiry in years.
    quantity : int     Number of contracts (negative = short).
                       Convention: 1 contract = 100 shares (US standard).
    multiplier : int   Shares per contract (default 100).
    S : float          Current underlying price.
    sigma : float      Implied volatility.
    r : float          Risk-free rate.
    q : float          Dividend yield.
    """
    underlying: str
    option_type: Literal["call", "put"]
    K: float
    T: float          # years to expiry
    quantity: int     # contracts (neg = short)
    S: float
    sigma: float
    r: float = 0.04
    q: float = 0.0
    multiplier: int = 100

    def greeks(self) -> dict:
        g = bsm_greeks(self.S, self.K, self.T, self.r, self.sigma,
                       self.option_type, self.q)
        sign = 1 if self.quantity > 0 else -1
        contracts = abs(self.quantity)
        mult = contracts * self.multiplier
        return {k: v * mult * sign for k, v in g.items()}

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.multiplier * self.S

    @property
    def moneyness(self) -> str:
        ratio = self.S / self.K
        if abs(ratio - 1) < 0.02:
            return "ATM"
        return "OTM" if (
            (self.option_type == "call" and self.S < self.K) or
            (self.option_type == "put" and self.S > self.K)
        ) else "ITM"

    def label(self) -> str:
        T_days = int(self.T * 365)
        sign = "+" if self.quantity > 0 else ""
        return (f"{sign}{self.quantity}x {self.underlying} "
                f"{T_days}d {self.K:.0f}{self.option_type[0].upper()} "
                f"[{self.moneyness}] σ={self.sigma:.0%}")


# ---------------------------------------------------------------------------
# Portfolio Greeks aggregation
# ---------------------------------------------------------------------------

class OptionsBook:
    """
    Aggregates Greeks across a multi-leg options portfolio.

    Usage
    -----
    >>> book = OptionsBook()
    >>> book.add(OptionPosition("SPX", "call", K=4500, T=0.25, ...))
    >>> book.add(OptionPosition("SPX", "put",  K=4300, T=0.25, ...))
    >>> report = book.risk_report()
    """

    def __init__(self):
        self.positions: list[OptionPosition] = []

    def add(self, pos: OptionPosition) -> None:
        self.positions.append(pos)

    def aggregate_greeks(self) -> dict:
        """Sum Greeks across all positions."""
        totals = {k: 0.0 for k in
                  ["price", "delta", "gamma", "vega", "theta", "rho",
                   "vanna", "volga", "charm", "speed"]}
        for pos in self.positions:
            g = pos.greeks()
            for k in totals:
                totals[k] += g.get(k, 0.0)
        return totals

    def greeks_by_underlying(self) -> pd.DataFrame:
        """Break down Greeks by underlying."""
        data = {}
        for pos in self.positions:
            u = pos.underlying
            if u not in data:
                data[u] = {k: 0.0 for k in
                           ["delta", "gamma", "vega", "theta", "rho", "vanna"]}
            g = pos.greeks()
            for k in data[u]:
                data[u][k] += g.get(k, 0.0)
        return pd.DataFrame(data).T

    def dollar_greeks(self) -> dict:
        """
        Dollar Greeks: translate unit Greeks into $ P&L per unit move.

        Dollar Delta: $/1% move in S     = delta × S × 0.01
        Dollar Gamma: $/1% move² in S    = 0.5 × gamma × (S × 0.01)²
        Dollar Vega:  $/1 vol-point move  = vega (already per 1% vol)
        Dollar Theta: $/calendar day      = theta
        """
        g = self.aggregate_greeks()
        S_avg = np.mean([pos.S for pos in self.positions]) if self.positions else 100
        return {
            "dollar_delta_per1pct":  g["delta"] * S_avg * 0.01,
            "dollar_gamma_per1pct":  0.5 * g["gamma"] * (S_avg * 0.01) ** 2,
            "dollar_vega_per1vol":   g["vega"],
            "dollar_theta_perday":   g["theta"],
            "dollar_rho_per1bps":    g["rho"] / 100,
        }

    def scenario_pnl(
        self,
        spot_shocks: np.ndarray,
        vol_shocks: np.ndarray,
    ) -> pd.DataFrame:
        """
        P&L grid across (spot shock, vol shock) scenarios.
        This is the standard "risk slide" / "ladder" used at every options desk.

        Parameters
        ----------
        spot_shocks : array of % spot changes, e.g. np.linspace(-0.20, 0.20, 9)
        vol_shocks  : array of absolute vol changes, e.g. np.arange(-0.08, 0.09, 0.02)

        Returns DataFrame indexed by spot_shock, columns = vol_shock.
        """
        rows = []
        for ds in spot_shocks:
            row = {"spot_shock": ds}
            for dv in vol_shocks:
                total_pnl = 0.0
                for pos in self.positions:
                    # Shocked Greeks (revalue at S*(1+ds), sigma+dv)
                    g_shock = bsm_greeks(
                        pos.S * (1 + ds), pos.K, pos.T, pos.r,
                        max(pos.sigma + dv, 0.01), pos.option_type, pos.q
                    )
                    g_base = bsm_greeks(
                        pos.S, pos.K, pos.T, pos.r,
                        pos.sigma, pos.option_type, pos.q
                    )
                    pnl = (g_shock["price"] - g_base["price"]) * abs(pos.quantity) * pos.multiplier
                    if pos.quantity < 0:
                        pnl = -pnl
                    total_pnl += pnl
                row[f"vol{dv:+.0%}"] = total_pnl
            rows.append(row)

        df = pd.DataFrame(rows).set_index("spot_shock")
        return df

    def risk_report(self) -> str:
        """Format a full risk report for the book."""
        g = self.aggregate_greeks()
        dg = self.dollar_greeks()
        lines = [
            "═" * 64,
            "  Options Book Risk Report",
            "═" * 64,
            f"\n  Positions: {len(self.positions)}",
        ]
        for pos in self.positions:
            lines.append(f"    {pos.label()}")

        lines.append(f"\n── Aggregate Greeks ──")
        lines.append(f"  {'Delta':>18} {g['delta']:>12.2f}  shares")
        lines.append(f"  {'Gamma':>18} {g['gamma']:>12.4f}  shares / $ move")
        lines.append(f"  {'Vega':>18} {g['vega']:>12.2f}  $ / 1% vol")
        lines.append(f"  {'Theta':>18} {g['theta']:>12.2f}  $ / calendar day")
        lines.append(f"  {'Rho':>18} {g['rho']:>12.4f}  $ / 1% rate")
        lines.append(f"  {'Vanna':>18} {g['vanna']:>12.4f}  shares / 1% vol")
        lines.append(f"  {'Volga':>18} {g['volga']:>12.4f}  $ / 1%² vol")

        lines.append(f"\n── Dollar Greeks ($ P&L per market move) ──")
        lines.append(f"  {'DollarDelta/1%S':>25} ${dg['dollar_delta_per1pct']:>10,.0f}")
        lines.append(f"  {'DollarGamma/1%²S':>25} ${dg['dollar_gamma_per1pct']:>10,.0f}")
        lines.append(f"  {'DollarVega/1vol':>25} ${dg['dollar_vega_per1vol']:>10,.0f}")
        lines.append(f"  {'DollarTheta/day':>25} ${dg['dollar_theta_perday']:>10,.0f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 64)
    print("  Options Portfolio Risk Engine")
    print("  Greeks, scenarios, dollar risk across multi-leg book")
    print("═" * 64)

    S = 450.0  # SPY @ $450
    r, q = 0.04, 0.015
    sigma_atm = 0.18  # 18% ATM vol

    book = OptionsBook()

    # A realistic options book: delta-hedged with some vol positions
    positions_config = [
        # Core long delta position (covered call strategy)
        ("SPY", "call", 460, 0.083, +50, sigma_atm * 1.02),   # long 50 ATM+ calls, 1M
        # Protective puts
        ("SPY", "put",  430, 0.083, +30, sigma_atm * 1.12),   # long 30 OTM puts (skew)
        # Short vol via straddle
        ("SPY", "call", 450, 0.25,  -20, sigma_atm * 1.00),   # short ATM call, 3M
        ("SPY", "put",  450, 0.25,  -20, sigma_atm * 1.00),   # short ATM put, 3M
        # Calendar spread
        ("SPY", "call", 455, 0.50,  +10, sigma_atm * 0.97),   # long 6M call
        ("SPY", "call", 455, 0.083, -10, sigma_atm * 1.01),   # short 1M call
    ]

    for ul, otype, K, T_yr, qty, sig in positions_config:
        book.add(OptionPosition(ul, otype, K, T_yr, qty, S, sig, r, q))

    # ── Risk report ───────────────────────────────────────────────
    print(book.risk_report())

    # ── Greeks by expiry ──────────────────────────────────────────
    print(f"\n── Greeks by Expiry ──")
    expiry_groups = {}
    for pos in book.positions:
        T_label = f"{int(pos.T*365)}d"
        if T_label not in expiry_groups:
            expiry_groups[T_label] = []
        expiry_groups[T_label].append(pos)

    print(f"\n  {'Expiry':>8} {'Delta':>10} {'Gamma':>10} {'Vega':>10} {'Theta':>10}")
    print("  " + "─" * 54)
    for exp, positions in sorted(expiry_groups.items(), key=lambda x: int(x[0][:-1])):
        sub = OptionsBook()
        for p in positions:
            sub.add(p)
        g = sub.aggregate_greeks()
        print(f"  {exp:>8} {g['delta']:>10.2f} {g['gamma']:>10.4f} "
              f"{g['vega']:>10.2f} {g['theta']:>10.2f}")

    # ── Scenario P&L grid ─────────────────────────────────────────
    print(f"\n── Scenario P&L Grid ($) — Spot × Vol ──")
    spot_shocks = np.array([-0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10])
    vol_shocks  = np.array([-0.05, -0.02, 0.00, 0.02, 0.05])
    grid = book.scenario_pnl(spot_shocks, vol_shocks)

    # Format header
    print(f"\n  {'S\\V':>8}", end="")
    for col in grid.columns:
        print(f"  {col:>10}", end="")
    print()
    print("  " + "─" * (12 + 12 * len(grid.columns)))
    for idx, row in grid.iterrows():
        s_label = f"S{idx:+.0%}"
        print(f"  {s_label:>8}", end="")
        for pnl in row:
            marker = " ←" if abs(pnl) < 100 else ""
            print(f"  {pnl:>+10,.0f}", end="")
        print()

    print(f"\n  ← Cells near zero indicate hedge effectiveness at that scenario.")
    print(f"\n── Gamma P&L Analysis ──")
    g = book.aggregate_greeks()
    dg = book.dollar_greeks()
    print(f"""
  With Gamma = {g['gamma']:.4f} and Theta = ${g['theta']:,.0f}/day:

  Daily Gamma P&L = ½Γ·(ΔS)² × {len(book.positions)} positions

  Break-even daily move: √(−2·Theta/Gamma) per share
  = √(−2 × {g['theta']:.2f} / {g['gamma']:.6f}) = {math.sqrt(abs(-2*g['theta']/max(abs(g['gamma']),1e-8))):.2f}

  Gamma P&L for various realized moves (daily):
  {'Move':>8}  {'Gamma P&L':>14}  {'Net (+ Theta)':>14}
  """)
    for move_pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
        move = S * move_pct / 100
        gamma_pnl = 0.5 * g["gamma"] * move ** 2
        net = gamma_pnl + g["theta"]
        print(f"  {move_pct:.1f}%     ${gamma_pnl:>12,.0f}   ${net:>12,.0f}")
