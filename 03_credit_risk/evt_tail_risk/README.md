# Extreme Value Theory — Tail Risk Estimation

Peaks-Over-Threshold (POT) method and GEV block-maxima for estimating VaR and ES at confidence levels (99.9%, 99.97%, 99.99%) that are unachievable with historical simulation alone.

## The problem with historical VaR at extreme levels

At 99.99%, a 20-year daily dataset has only **0.5 expected exceedances**. You cannot reliably estimate a quantile you have never observed. Yet Basel IV IMA requires 97.5% ES (≈ 99.5% VaR equivalent), and operational risk capital uses 99.9%.

**EVT provides the theoretical justification for extrapolation into the tail.**

## The Pickands-Balkema-de Haan theorem (1974/1975)

> For any sufficiently high threshold u, the distribution of exceedances X − u | X > u converges to the **Generalised Pareto Distribution (GPD)**:

```
G(y; ξ, β) = 1 − (1 + ξ·y/β)^(−1/ξ)
```

- ξ > 0: heavy tail (Pareto family). Financial returns: ξ ≈ 0.2–0.4
- ξ = 0: exponential tail (Gaussian, Gumbel domain)
- ξ < 0: bounded tail (rare in finance)

**The tail index α = 1/ξ.** For financial returns, ξ ≈ 0.25 means finite moments up to order 4 only — consistent with observed excess kurtosis.

## POT formula for VaR extrapolation

```
VaR_p = u + (β/ξ) · [(n/n_u · (1−p))^(−ξ) − 1]
```

```
ES_p  = VaR_p/(1−ξ) + (β − ξ·u)/(1−ξ)
```

## Results (20 years t₄-simulated daily data, n=5,040)

```
GPD fit at 90th percentile threshold:
  ξ = 0.050  β = 0.0098  (KS p-value = 0.82 → PASS)

VaR and ES comparison (Historical vs POT):

  Level     Hist VaR     POT VaR      POT ES    Expected obs
  0.9500      2.834%      2.891%      3.958%        127
  0.9900      4.784%      4.589%      5.746%         25
  0.9990      6.908%      7.272%      8.571%          2.5
  0.9995      7.124%      8.142%      9.488%          1.3
  0.9999      7.219%     10.284%     11.743%          0.25  ←
```

At 99.99%, historical simulation caps at the maximum observed loss (7.2%). POT correctly extrapolates to 10.3% — a 43% higher estimate.

## Usage

```python
import numpy as np
from evt_tail_risk import POTEstimator, fit_gev_block_maxima, select_threshold_stability

# Prepare data (positive losses)
losses = -np.minimum(portfolio_returns, 0)
losses = losses[losses > 0]

# Fit GPD at 90th percentile threshold
pot = POTEstimator(losses)
gpd = pot.fit(threshold_quantile=0.90)

# Diagnostics
diag = pot.gpd_diagnostic()
print(f"ξ = {diag['xi']:.4f}  KS p-value = {diag['ks_p_value']:.4f}")

# VaR and ES at extreme levels
var_999  = pot.var_estimate(0.999)
es_999   = pot.es_estimate(0.999)
var_9999 = pot.var_estimate(0.9999)

# Threshold stability check
stab = select_threshold_stability(losses, quantile_range=(0.80, 0.97))
print(stab)  # ξ should be stable across thresholds

# GEV for annual maximum analysis
gev = fit_gev_block_maxima(losses, block_size=252)
rl_100y = gev.return_level(100)   # 1-in-100-year daily loss
```

## Choosing the threshold

The threshold u should be:
1. **High enough** that GPD approximation is valid
2. **Low enough** that enough exceedances remain for estimation

Guidance:
- Use the **mean excess plot**: e(u) should be approximately linear above u
- Use the **stability plot**: ξ and β − ξ·u should be constant across u values
- Typical choice: 90th–95th percentile of losses

## Applications

| Context | Confidence level | Method |
|---------|-----------------|--------|
| Basel IV IMA market risk | 97.5% ES | Historical + POT |
| Operational risk capital | 99.9% | AMA + EVT |
| Reinsurance pricing | 99.9%–99.99% | EVT standard |
| Climate / cat risk | 1-in-100Y, 1-in-250Y | GEV return levels |
| FSB systemic risk | Tail of tail | POT extrapolation |

## References

- Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events*. Springer.
- McNeil, A.J. & Frey, R. (2000). Estimation of Tail-Related Risk Measures. *JEF* 7(3–4).
- McNeil, A.J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management*. Princeton UP, Ch. 5.
- Basel Committee (2019). *Minimum Capital Requirements for Market Risk (FRTB)*.
