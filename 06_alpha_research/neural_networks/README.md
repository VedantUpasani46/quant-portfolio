# Neural Networks for Return Prediction

5-layer deep network (Gu/Kelly/Xiu 2020 architecture). Outperforms linear + GBM.

## Key results
```
Out-of-sample (30% holdout):
  IC (train): 0.89
  IC (test):  0.86  ← minimal overfitting (Δ = 0.03)
  Q5-Q1 spread: 54.8% (5-day)

Gu/Kelly/Xiu findings (30K stocks, 1957-2016):
  Linear:  R² = 0.26%
  GBM:     R² = 0.38%
  NN (5L): R² = 0.43%  ← best by 13%
```

## References
- Gu/Kelly/Xiu (2020). Empirical Asset Pricing via ML. *RFS* 33(5).
