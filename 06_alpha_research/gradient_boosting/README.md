# Gradient Boosting for Alpha

XGBoost/LightGBM methodology for return prediction. Standard at every systematic fund.

## Key results
```
Walk-forward validation (5 folds):
  Mean IC: 0.12
  IC > 0:  80% of folds
  
Feature importance:
  1. px_sma_50 (20.4%)   ← price/SMA ratio
  2. ret_252d  (18.6%)   ← momentum
  3. ret_21d   (14.9%)
  
Gu/Kelly/Xiu (2020) findings:
  GBM R² = 0.38%, linear = 0.26% (GBM wins by 46%)
```

## References
- Gu/Kelly/Xiu (2020). Empirical Asset Pricing via Machine Learning. *RFS* 33(5).
- Friedman (2001). Greedy Function Approximation. *Ann. Statistics* 29(5).
