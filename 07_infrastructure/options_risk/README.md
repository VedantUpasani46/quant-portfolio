# Options Portfolio Risk Engine

Multi-leg options book aggregator: full BSM Greeks (including Vanna, Volga, Charm, Speed), dollar risk metrics, scenario P&L grid, and gamma P&L analysis.

## Greeks computed

| Greek | Formula | Interpretation |
|-------|---------|---------------|
| Delta | ∂V/∂S | $/$ spot move |
| Gamma | ∂²V/∂S² | Rate of delta change |
| Vega | ∂V/∂σ | $ per 1% vol move |
| Theta | ∂V/∂t | $ per calendar day |
| Rho | ∂V/∂r | $ per 1% rate move |
| Vanna | ∂²V/∂S∂σ | How delta changes with vol |
| Volga | ∂²V/∂σ² | Vol convexity |

## Results (6-leg SPY book)

```
Aggregate Greeks:
  Delta:   +1121.6 shares          DollarDelta/1%S:  +$5,047
  Gamma:   +62.05  shares/$        DollarGamma/1%²S: +$628
  Vega:    +703.3  $/1% vol        DollarVega/1vol:  +$703
  Theta:   -688.4  $/day           DollarTheta/day:  -$688

Break-even daily move: 4.71 (need ~1% SPY move to cover theta cost)
```

Scenario P&L grid (the "risk slide" used by every options desk):
```
       S-5%    S+0%    S+5%   S+10%
vol-5% -10.8K    -2.7K  +38.0K +101.3K
vol+5%  -8.2K   +4.1K  +42.7K +100.8K
```

## Usage

```python
from options_risk_engine import OptionsBook, OptionPosition
import numpy as np

book = OptionsBook()
book.add(OptionPosition("SPY", "call", K=450, T=0.25, quantity=+10, S=450, sigma=0.18, r=0.04))
book.add(OptionPosition("SPY", "put",  K=430, T=0.25, quantity=+10, S=450, sigma=0.20, r=0.04))

print(book.risk_report())          # full Greek aggregation
dg = book.dollar_greeks()          # dollar risk metrics

# Scenario grid: 7 spot × 5 vol scenarios
grid = book.scenario_pnl(
    spot_shocks=np.linspace(-0.10, 0.10, 7),
    vol_shocks=np.array([-0.05, -0.02, 0, 0.02, 0.05])
)
```

## References
- Hull, J.C. (2022). *Options, Futures and Other Derivatives*, Ch. 19–20.
- Natenberg, S. (1994). *Option Volatility & Pricing*. McGraw-Hill.
