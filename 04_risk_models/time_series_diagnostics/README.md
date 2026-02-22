# Time Series Diagnostics

ACF/PACF, Ljung-Box Q-test, Durbin-Watson, ADF unit root, and Lo-MacKinlay variance ratio test.

## Key results
```
GARCH(1,1) simulation (T=1260):
  Raw returns ACF: all < 0.06 (insignificant) → efficient ✓
  Squared returns ACF: 0.19 at lag 1, all *** → ARCH effects ✓
  Ljung-Box Q(10) on returns: p=0.55 (no autocorr) ✓
  Ljung-Box Q(10) on r²:      p=0.000 (ARCH!) ✓

ADF:
  Log prices: t=0.15, p=0.50 → unit root ✓
  Returns:    t=-25.4, p=0.01 → stationary ✓

Signal (AC=0.7):
  DW = 0.62 (persistent signal confirmed)
```

## References
- Ljung & Box (1978). *Biometrika* 65(2).
- Lo & MacKinlay (1988). *RFS* 1(1).
- Dickey & Fuller (1979). *JASA* 74(366).
