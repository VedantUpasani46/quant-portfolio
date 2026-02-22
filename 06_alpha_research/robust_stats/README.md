# Robust Statistics for Quantitative Finance

Huber M-estimators, Theil-Sen regression, and MCD robust covariance — all resistant to the fat tails and outliers ubiquitous in financial data.

## Key results
```
Location under 5% contamination at ±20%:
  Sample mean: shifts by 0.0002 (unstable)
  Huber:       barely affected ✓

Factor regression (2% outliers):
  Huber wins on α, β_mkt, β_smb — closer to true params
  Huber downweights 17.2% of obs as suspected outliers

Robust covariance (5% multivariate outliers):
  Sample variance: 8-16× inflated above truth
  MCD: 3-6× closer to truth ✓
  9/10 outliers correctly flagged by Mahalanobis distance
```

## References
- Huber (1964). Robust Estimation. *Ann. Math. Stats* 35(1).
- Rousseeuw & Leroy (1987). Robust Regression and Outlier Detection. Wiley.
