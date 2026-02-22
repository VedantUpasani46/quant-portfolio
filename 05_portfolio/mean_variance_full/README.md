# Mean-Variance Portfolio Optimization

Markowitz (1952) portfolio construction: efficient frontier, maximum Sharpe ratio (tangency portfolio), minimum variance portfolio, and risk parity — all implemented from scratch using quadratic programming.

## The optimization problem

For a given target return μ*, find weights minimising portfolio variance:
```
min  w'Σw           [minimize variance]
s.t. w'μ = μ*       [hit target return]
     w'1 = 1        [fully invested]
     w ≥ 0          [no shorting]
```

The set of all such solutions traces the **efficient frontier** in (σ, μ) space.

## Key results (6-asset universe)

```
Portfolio           Return     Vol    Sharpe
─────────────────────────────────────────────
Equal Weight        7.83%   11.08%    0.346
Min Variance        4.50%    3.68%    0.136
Max Sharpe          8.03%   10.36%    0.389  ← Tangency portfolio
Risk Parity        10.02%   19.13%    0.315
```

MVP: 90% IG bonds (lowest vol, negative equity correlation).  
Tangency: 47% HY bonds + 29% US equity + 13% EM equity → SR 0.389 vs 0.346 equal weight.

## The four portfolios

| Portfolio | Construction | Use case |
|-----------|-------------|---------|
| **Equal Weight** | 1/N | Naive benchmark |
| **Min Variance** | min w'Σw | Capital preservation |
| **Max Sharpe** | max (w'μ−rf)/σ | Return/risk efficiency |
| **Risk Parity** | each asset = 1/K total risk | Diversification |

## Ledoit-Wolf shrinkage

Sample covariance Σ̂ has huge estimation error for small T/N. Ledoit-Wolf (2004) analytical shrinkage reduces condition number by **38.8%** — critical for stable portfolio weights.

```python
from mean_variance_optimization import EfficientFrontier, ledoit_wolf_shrinkage

# Build the frontier
ef = EfficientFrontier(mu=expected_returns, cov=cov_matrix,
                       asset_names=names, rf=0.04)

mvp       = ef.minimum_variance_portfolio()
tangency  = ef.maximum_sharpe_portfolio()
risk_par  = ef.risk_parity_portfolio()
frontier  = ef.compute_frontier(n_points=50)

# Risk decomposition
from mean_variance_optimization import risk_contributions
rc = risk_contributions(tangency, cov_matrix)

# Shrinkage estimation
cov_lw = ledoit_wolf_shrinkage(returns)  # more stable than sample cov
```

## References

- Markowitz, H.M. (1952). Portfolio Selection. *JF* 7(1), 77–91.
- Ledoit, O. & Wolf, M. (2004). A Well-Conditioned Estimator. *JMA* 88(2).
- Meucci, A. (2005). *Risk and Asset Allocation*. Springer.
