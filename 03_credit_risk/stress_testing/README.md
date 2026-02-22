# Stress Testing Framework

Historical scenario replay, hypothetical factor shocks, and reverse stress testing for a multi-factor portfolio. Implements the methodology underlying CCAR (Fed), DFAST, SREP (ECB), ICAAP (PRA), and Basel III Pillar 2 requirements.

## Why stress testing exists

VaR is a statistical measure — it describes *normal* market conditions. Regulators require banks to demonstrate solvency under **extreme but plausible** scenarios that VaR cannot capture:
- Correlation breakdown (assets that normally hedge each other crash together)
- Regime shifts (2008 GFC, 2020 COVID)
- Policy discontinuities (2022 rate shock, 1994 bond massacre)

**Basel III Pillar 2 / ICAAP mandates:** "Banks must have robust processes to identify, measure and manage all material risks, including stress testing."

## Scenarios included

**6 historical scenarios:**
| Scenario | Key shocks |
|----------|-----------|
| 2008 GFC (Lehman) | Equity -40%, credit +400bp, rates -200bp |
| 2011 Eurozone crisis | IT/ES spreads +300bp, EUR weakens |
| 2020 COVID (March) | Equity -35%, credit +200bp, rates to zero |
| **2022 Rates shock** | Rates +300bp, equity -25% — **worst for bond portfolios** |
| 1994 Bond Massacre | Rates +250bp across the curve |
| 1997 Asian Crisis | EM equity -50%, EM spreads +600bp |

**2 regulatory scenarios:** CCAR Severely Adverse 2024, ECB Adverse 2024

## Results (on $500M Treasury portfolio)

```
Scenario                              P&L ($M)    P&L %   Breach?
─────────────────────────────────────────────────────────────────
2022_Rates_Shock                       -299.65  -59.93%   ⚠ BREACH
1994_Bond_Massacre                     -214.35  -42.87%   ⚠ BREACH
ECB_Adverse_2024                       -102.00  -20.40%   ⚠ BREACH
2008_GFC_Lehman                          +4.71   +0.94%   (flight to quality!)
```

The 2008 GFC actually shows a small *gain* for a bond-heavy portfolio: rate rally on flight to quality offset credit/equity losses. The 2022 scenario is catastrophic for any duration-long portfolio — exactly what happened to SVB.

## P&L attribution (Worst scenario — 2022 Rates)

```
US 10Y rate (+300bp):   −$141M   (DV01 × shock)
US 2Y rate (+350bp):    − $49M
EUR 10Y rate (+200bp):  − $38M
S&P 500 (−25%):         − $31M
HY credit (+300bp):     − $14M
```

## Usage

```python
from stress_testing import build_demo_portfolio, StressTestEngine, HypotheticalScenario

portfolio = build_demo_portfolio()
engine = StressTestEngine(portfolio, stop_loss_limit=-0.15)  # -15% NAV

# Run all historical + regulatory scenarios
results = engine.run_all_scenarios()
engine.print_report(results)

# Custom hypothetical scenario
custom = {
    "Bear_Steepener": HypotheticalScenario.bear_steepener(-50, +150),
    "Equity_Crash":   HypotheticalScenario.equity_crash(-0.30, credit_widening_bps=200),
}
for name, sc in custom.items():
    res = engine.apply_scenario(name, sc)
    print(f"{name}: ${res.pnl/1e6:.1f}M  ({res.pnl_pct_nav:.1%})")

# Reverse stress test: what breaks us?
rss = engine.reverse_stress_test(target_loss_pct=-0.15)
# → equity_crash_breakeven: -37.5%
# → rate_shock_breakeven: +90bp
```

## P&L formula

```
ΔV ≈ Σᵢ δᵢ·ΔFᵢ + ½·Σᵢ γᵢ·(ΔFᵢ)²
```
where δᵢ = first-order sensitivity (DV01, delta), γᵢ = second-order (convexity, gamma).

## References

- Basel Committee (2009). *Principles for Sound Stress Testing Practices*.
- Fed (2024). *CCAR Severely Adverse Scenario Description*.
- Hull, J.C. (2022). *Risk Management and Financial Institutions*, 6th ed., Ch. 14.
