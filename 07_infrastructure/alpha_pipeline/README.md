# Alpha Research Pipeline

End-to-end signal evaluation infrastructure: signal processing, IC decay analysis, autocorrelation/turnover, multiple testing correction, and a go/no-go production checklist. The workflow used at systematic hedge funds before deploying a new factor.

## The seven deadly sins it guards against

1. **Lookahead bias** — walk-forward IC computation only uses past data
2. **Overfitting** — t-statistic test + Bonferroni/BH multiple testing correction
3. **Narrative fallacy** — production checklist requires quantitative criteria
4. **High turnover costs** — explicit signal autocorrelation and turnover estimate
5. **Small sample** — minimum observation requirements enforced

## Key metrics

| Metric | Formula | Production threshold |
|--------|---------|---------------------|
| IC | Spearman(signal, return) | > 0.02 |
| IR | IC_mean / IC_std | > 0.3 |
| Hit rate | % periods IC > 0 | > 52% |
| t-statistic | IC_mean / (IC_std/√T) | > 2.0 |
| Turnover | 1 − AutoCorr(1d) | < 80% |

## Results (AR(0.9) value signal, 2Y daily, 100 stocks)

```
IC Mean:   0.0303   ✓  (target > 0.02)
IR:        0.3143   ✓  (target > 0.3)
Hit Rate:  61.63%   ✓  (target > 52%)
t-stat:    7.048    ✓  (p = 0.000)
1d AutoCorr: 0.884  → 11.7% daily turnover  ✓

Production Checklist: GO (7/8 checks passed)
```

Multiple testing correction for 20 signals:
- Naïve (p<0.05): finds 3 significant (inflated)
- Benjamini-Hochberg: correctly identifies 1 (controls FDR at 5%)

## Usage

```python
from alpha_pipeline import process_signal, ic_decay_profile, AlphaSignal
from alpha_pipeline import signal_autocorrelation, benjamini_hochberg_correction

# Process raw signal
processed = process_signal(raw_signal, winsorise_pct=0.02, method="zscore")

# IC decay analysis
decay_df = ic_decay_profile(signals_matrix, returns_matrix, max_horizon=20)

# Full signal object with production checklist
signal = AlphaSignal(name="My Factor", ic_series=ic_arr, decay_df=decay_df,
                     autocorr_df=ac_df, universe_size=500, avg_adv_usd=50e6)
print(signal.summary())   # full checklist output
```

## References
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley, Ch. 3, 12.
- Grinold, R.C. & Kahn, R.N. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.
- Harvey, C.R. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1).
