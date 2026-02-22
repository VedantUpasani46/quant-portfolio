# PCA Factor Analysis: Yield Curves & Equity Returns

Principal Component Analysis for financial factor extraction — from scratch implementation. Demonstrates the Litterman-Scheinkman (1991) level/slope/curvature yield curve decomposition and a statistical equity factor risk model.

## Yield curve PCA: the three factors

PCA on a yield curve with 11 tenors (3M to 30Y):

```
PC1 (Level):     98.6% of variance — parallel curve shift
PC2 (Slope):      1.3% of variance — steepening/flattening
PC3 (Curvature):  0.02% — butterfly (5Y vs 2Y+10Y)
```

Factor loadings confirm the economic interpretation:
- **PC1**: all tenors load +0.29 to +0.32 (flat → parallel shift)
- **PC2**: 3M loads −0.27, 30Y loads +0.71 (negative short, positive long → steepening)
- **PC3**: 5Y−10Y load negative, wings positive (butterfly)

**Practical use for rates desks:** A DV01-neutral trade is still exposed to slope/curvature. Proper hedging requires matching PC1, PC2, PC3 sensitivities — not just total DV01.

## Equity factor risk model

```
r_it = α_i + Σ_k β_ik · f_kt + ε_it
```

5-factor PCA on 30-stock universe (756 days):
- PC1 explains 42% of cross-sectional variance (market factor)
- Average R² = 0.52 (52% systematic, 48% idiosyncratic)
- Condition number: 30 → factor model is well-conditioned for portfolio optimisation

## Usage

```python
from pca_factor_analysis import pca, generate_yield_curve_data, equity_pca_risk_model

# Yield curve PCA
yields, tenor_labels, tenors_yr = generate_yield_curve_data(T=2520)
result = pca(yields, n_components=5, feature_names=tenor_labels)
print(result.explained_variance_ratio[:3])   # [0.986, 0.013, 0.0002]

# Project a new curve onto factor space
new_curve = yields[-1:]
factor_scores = result.factor_score(new_curve)  # [PC1_score, PC2_score, ...]

# Equity factor model
eq_model = equity_pca_risk_model(stock_returns, n_factors=5)
# Access: betas, R2 per stock, idiosyncratic vols, reconstructed covariance
```

## References

- Litterman, R. & Scheinkman, J. (1991). Common Factors Affecting Bond Returns. *JFI* 1(1).
- Connor, G. & Korajczyk, R. (1988). Risk and Return in an Equilibrium APT. *JFE* 21(2).
- Hull, J.C. (2022). *Options, Futures and Other Derivatives*, Ch. 7.
