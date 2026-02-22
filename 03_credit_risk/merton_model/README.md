# Merton Structural Credit Risk Model

Implementation of the Merton (1974) structural model: treats firm equity as a call option on firm assets, extracting implied asset value, Distance-to-Default, and credit spread from observable equity prices and volatility.

## The equity-as-option insight

Merton (1974) observed that equity holders have **limited liability**:
- If firm assets V_T > D at debt maturity: shareholders receive V_T − D
- If firm assets V_T ≤ D: firm defaults, shareholders receive 0

This is identical to a European **call option** on V with strike D:
```
Equity = max(V_T − D, 0)
```

Since equity is observable (market cap, equity vol), we can **invert the BSM formula** to recover the unobservable asset value V_A and asset volatility σ_A.

## The calibration system

Two equations, two unknowns (V_A, σ_A):

```
E = V_A·N(d₁) - D·e^(-rT)·N(d₂)      [equity = call on assets]
σ_E = (V_A/E)·N(d₁)·σ_A               [equity vol = asset vol × leverage × delta]
```

Solved via Newton-Raphson with multiple starting points for robustness.

## Key output: Distance-to-Default

```
DD = [ln(V_A/D) + (μ - σ_A²/2)·T] / (σ_A·√T)
```

This is the number of standard deviations between the **expected future asset value** and the default boundary. The physical PD = N(−DD).

The Moody's KMV model (the commercial implementation) uses empirical mappings from DD to observed default frequencies, rather than N(−DD) directly, because the Gaussian assumption underestimates short-horizon PDs.

## Results

```
Example: $80B equity, $100B debt, σ_E = 30%, T = 1Y

Calibrated:
  Asset value V_A:   $175B
  Asset volatility:  13.7%
  Leverage D/V_A:    57%

Credit metrics:
  d₂ = 4.38
  Distance-to-Default: 4.6σ
  PD (risk-neutral):   0.0006%
  Credit spread:       0bp  (very low leverage)
```

```
Credit term structure (same firm):
  Maturity   Spread (bps)   PD (RN)
  0.5Y              0        0.000%
  2.0Y              0.3      0.103%
  5.0Y              7.4      3.08%
  10.0Y            27.4     12.05%
```

```
Credit portfolio (5 stylised firms):
  Firm                  D/V     DD     PD%    Spread
  Investment Grade A    38%    8.1     0.000%      0bp
  Investment Grade BBB  59%    5.5     0.000%      0bp
  Sub-Investment BB     78%    3.6     0.056%      0bp
  High Yield B          87%    2.8     0.861%      2bp
  Distressed CCC        95%    1.7     9.856%     33bp
```

## Usage

```python
from merton_model import MertonModel, MertonCreditPortfolio

# Single firm
model = MertonModel(
    E=10e9,        # $10B market cap
    sigma_E=0.40,  # 40% equity vol
    D=15e9,        # $15B debt
    T=1.0,
    r=0.05,
    mu=0.08,
)
result = model.calibrate()
print(result.summary("Levered Corp"))

# Credit spread term structure
import numpy as np
term_structure = model.credit_curve(np.array([1, 2, 3, 5, 7, 10]))
print(term_structure)

# Portfolio of firms
portfolio = MertonCreditPortfolio(firms_list)
df = portfolio.analyse_all()
print(df[["name", "dd", "pd_rn_pct", "spread_bps"]])
```

## Why this matters

The Merton model is the foundation of:
- **KMV / Moody's Analytics EDF** — the global standard for PD estimation on public firms
- **CreditMetrics (J.P. Morgan 1997)** — portfolio credit risk framework used industrywide
- **Basel II/III IRB** — internal ratings-based approach to credit capital
- **CVA desks** — counterparty credit risk, hazard rate calibration

Any credit risk or model validation role will test Merton in interviews.

## References

- Merton, R.C. (1974). On the Pricing of Corporate Debt. *JF* 29(2), 449–470.
- Crosbie, P. & Bohn, J. (2003). *Modeling Default Risk*. Moody's KMV.
- Lando, D. (2004). *Credit Risk Modeling*. Princeton UP.
- Hull, J.C. (2022). *Options, Futures and Other Derivatives*, Ch. 24.
