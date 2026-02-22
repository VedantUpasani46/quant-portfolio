# P&L Attribution Engine

Decomposes daily portfolio P&L into factor exposures, Greeks contributions, and Brinson sector effects. Used by portfolio managers, risk managers, and traders to explain *why* the book made or lost money.

## Three decomposition methods

**1. Factor Attribution** (equity portfolios)
```
P&L_factor = Σₖ βₖ · Fₖ · NAV
P&L_alpha  = P&L_total − P&L_factor
```

**2. Greeks P&L** (derivatives books)
```
P&L ≈ Δ·ΔS + ½Γ·(ΔS)² + V·Δσ + Θ·Δt + ρ·Δr + Vanna·ΔS·Δσ + ½Volga·(Δσ)²
```

**3. Brinson Sector Attribution** (benchmark-relative)
```
Allocation effect:   (w_p − w_b)·(r_b − R_b)
Selection effect:    w_b·(r_p − r_b)  
Interaction effect:  (w_p − w_b)·(r_p − r_b)
```

## Results

Greeks attribution (down 1.5%, vol +2bp):
```
Delta (Δ·ΔS)        −$8,100
Gamma (½Γ·ΔS²)      +$1,025   ← positive gamma offsets delta
Vega (V·Δσ)         +$1,700   ← long vol gains on spike
Theta (Θ·Δt)          −$620
Vanna (cross)        −$2,025
Volga (½Vg·Δσ²)     +$4,200   ← long volga gains on vol jump

Greeks total:        −$3,820
Unexplained:            −$30
Explanation ratio:    99.2%   ← near-perfect attribution
```

Brinson (attribution error = 0.000000% — mathematically exact):
```
Technology:  +13.6bp  (allocation +4.6bp, selection +7.5bp)
Financials:   −4.8bp  (underperformed benchmark within sector)
Total active: +21.6bp explained
```

## Usage

```python
from pnl_attribution import FactorAttribution, GreeksPnL, BrinsonAttribution, DailyPnLReport

# Factor attribution
fa = FactorAttribution(
    factor_names=["Market", "Size", "Value", "Momentum"],
    factor_exposures=np.array([0.95, 0.15, -0.10, 0.30]),
    portfolio_nav=100_000_000
)
result = fa.attribute(factor_returns_today, actual_pnl_today)

# Greeks P&L
gk = GreeksPnL(delta=1200, gamma=45, vega=850, theta=-620, rho=320)
explained = gk.explain_pnl(delta_S=-6.75, delta_sigma=0.02, delta_t=1.0,
                            delta_r=0.0, actual_pnl=-3850, S=450)
print(f"Explanation ratio: {explained['explained_pct']:.1f}%")

# Integrated report
report = DailyPnLReport("My Portfolio", nav=100e6)
report.set_factor_attribution(fa)
report.set_greeks(gk)
print(report.generate(actual_pnl=-28866, factor_returns=..., spot_move=-6.75, vol_move=0.02))
```

## References
- Brinson, G.P., Hood, L.R. & Beebower, G.L. (1986). Determinants of Portfolio Performance. *FAJ* 42(4).
- Hull, J.C. (2022). *Options, Futures and Other Derivatives*, Ch. 19.
- Grinold, R.C. & Kahn, R.N. (2000). *Active Portfolio Management*. McGraw-Hill.
