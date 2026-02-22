# Bootstrap Confidence Intervals

Efron's bootstrap for parameter uncertainty. Model-free CIs for Sharpe, alpha, VaR.

## Key results
```
Sharpe ratio (252 days):
  Point estimate: 0.40
  Bootstrap 95% CI: [-1.64, 2.26]
  Bootstrap SE: 1.01

Block bootstrap (AR(1) with φ=0.8):
  Standard (iid): SE = 0.16
  Block (size=20): SE = 0.37  ← 2.39× larger (correct)

VaR 99%:
  Point: $369
  95% CI: [$284, $444]
```

## References
- Efron (1979). Bootstrap Methods. *Ann. Statistics* 7(1).
- Politis & Romano (1994). Stationary Bootstrap. *JASA* 89(428).
