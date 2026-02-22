# Machine Learning for Return Prediction

Walk-forward cross-sectional return prediction using Ridge, Lasso, and ElasticNet regularisation. Implements the core quantitative equity research pipeline: feature engineering, walk-forward validation, and Information Coefficient measurement.

## The prediction framework

At each rebalance date t:
1. Compute cross-sectional features X_t (momentum, reversal, vol, beta, etc.)
2. Fit regularised model on rolling training window
3. Predict next-period returns ŷ_t = X_t · β̂
4. Go long top decile, short bottom decile

**Walk-forward validation** (not k-fold): each fold trains on past data only, prevents lookahead bias.

## Key metrics

| Metric | Formula | "Good" threshold |
|--------|---------|-----------------|
| IC | Spearman ρ(ŷ, y) | > 0.05 |
| IR | IC_mean / IC_std | > 0.5 |
| Hit rate | % periods IC > 0 | > 55% |

IC = 0.05 might seem low — but if consistent (high IR), it generates significant alpha. Institutional quant funds target IC ≈ 0.03–0.08.

## Features implemented

| Feature | Signal | Source |
|---------|--------|--------|
| Momentum | Past 12-1M return | Jegadeesh & Titman (1993) |
| Short-term reversal | Past 1M return (negative) | Jegadeesh (1990) |
| Volatility | 60-day realised vol (negative) | Ang et al. (2006) |
| Beta | CAPM beta (low-beta anomaly) | Frazzini & Pedersen (2014) |
| Idiosyncratic vol | Residual vol (negative) | Ang et al. (2009) |
| Skewness | Return skewness | Harvey & Siddique (2000) |

## Regularisation comparison

**Ridge (L2):** shrinks all coefficients toward zero — no sparsity, all features retained.  
**Lasso (L1):** forces many coefficients to exactly zero — automatic feature selection.  
**ElasticNet:** L1 + L2 hybrid — sparse but handles correlated features better than Lasso.

In practice: Lasso is preferred for factor selection (identifies the 2-3 most predictive signals). Ridge is preferred when all features are expected to contribute.

## Usage

```python
from ml_return_prediction import MLReturnPredictor

predictor = MLReturnPredictor(
    returns,           # (T, n_stocks) return matrix
    model_type='lasso',
    alpha=1e-3,        # regularisation strength
    train_window=252,  # 1-year rolling window
    prediction_horizon=21,  # 1-month ahead
)
result = predictor.run()
print(f"IC: {result.ic_mean:.4f}  IR: {result.ir:.4f}  Hit: {result.hit_rate:.2%}")
print(f"Top-minus-bottom alpha: {result.alpha_bps_ann:.0f} bps/year")
print("Feature importances:", result.feature_importances)
```

## References

- Tibshirani, R. (1996). Regression Shrinkage via the Lasso. *JRSS-B*.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley, Ch. 7, 12.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via ML. *RFS* 33(5).
