# Copula Models for Joint Risk

Gaussian and Student-t copula implementations for credit portfolio loss simulation, multi-asset VaR, and joint tail-risk analysis. Demonstrates the exact failure mode of the Li (2000) Gaussian copula that contributed to the 2008 CDO crisis.

## The core idea

Correlation measures linear co-movement. Copulas model the full **dependence structure** — including tail dependence, the probability that extremes happen simultaneously.

**Sklar's Theorem (1959):** Any joint distribution can be decomposed as:
```
F(x₁,...,xₙ) = C(F₁(x₁), ..., Fₙ(xₙ))
```
The copula C captures dependence *separately* from the marginals Fᵢ. You can combine any marginals with any dependence structure.

## Gaussian vs Student-t: the critical difference

| Property | Gaussian Copula | Student-t Copula |
|----------|----------------|------------------|
| Tail dependence λ_U | **0 always** | **> 0** (positive) |
| Joint extremes | Independent in tails | Cluster in tails |
| CDO pricing (2000–2007) | Standard (Li 2000) | Not used |
| Credit risk post-2008 | Abandoned | Industry standard |

The **tail dependence coefficient** for a bivariate t-copula:
```
λ = 2·T_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
```

For ρ=0.70: Gaussian λ=0, t(ν=3) λ=0.45, t(ν=10) λ=0.19.

## Results

```
Credit Portfolio Loss Simulation (100 obligors, total EAD $131M)
Avg PD: 3.2%, Avg LGD: 51%, sector-correlated portfolio

              Gaussian        t(ν=5)        t(ν=3)
VaR 99%        $10.9M         $19.4M        $23.6M   ← 2.2× Gaussian
VaR 99.9%      $17.3M         $32.4M        $37.5M   ← 2.2× Gaussian
ES 99%         $13.5M         $24.9M        $30.0M

Mean loss is identical — tails differ massively.
```

This is precisely the underestimation of joint losses that caused catastrophic CDO tranche mispricing in 2008.

## Usage

```python
import numpy as np
from copula_models import GaussianCopula, StudentTCopula, CopulaPortfolioLoss

# Bivariate t-copula with ρ=0.7, ν=5
corr = np.array([[1.0, 0.7], [0.7, 1.0]])
t_cop = StudentTCopula(corr, nu=5)

# Tail dependence
lam = t_cop.tail_dependence(rho=0.7)
print(f"Tail dependence λ = {lam:.4f}")   # 0.343

# Sample from copula (returns uniform[0,1] marginals)
samples = t_cop.sample(n=100_000)

# Credit portfolio loss
model = CopulaPortfolioLoss(
    corr=sector_corr,
    pd=pd_array,
    lgd=lgd_array,
    ead=ead_array,
    nu=5   # None → Gaussian
)
results = model.simulate(n=100_000)
print(f"99% VaR: ${results['var_99']/1e6:.1f}M")
print(f"99.9% VaR: ${results['var_999']/1e6:.1f}M")
```

## References

- Li, D.X. (2000). On Default Correlation: A Copula Function Approach. *JFI* 9(4).
- McNeil, A.J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management*. Princeton UP, Ch. 5–7.
- Hull, J.C. & White, A. (2004). Valuation of a CDO. *Journal of Derivatives* 12(1).
