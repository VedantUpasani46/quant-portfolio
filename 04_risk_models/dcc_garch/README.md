# DCC-GARCH: Dynamic Conditional Correlation

Time-varying correlation matrix for multi-asset portfolios. Captures correlation spikes during crises (ρ: 0.3 → 0.7).

## Key results
```
Correlation regime shift (simulated):
  Normal (days 1-630):   mean ρ = 0.28
  Crisis (days 631-1260): mean ρ = 0.65  ← spike
  DCC persistence: α+β = 0.9976 (very high)

Static vol understates 95th percentile risk by 12.2%
```

## References
- Engle (2002). Dynamic Conditional Correlation. *J. Bus & Econ Statistics* 20(3).
