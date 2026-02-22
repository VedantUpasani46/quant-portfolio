# Liquidity Risk Measurement

Amihud illiquidity ratio, Roll bid-ask spread estimator, Liquidity-adjusted VaR, and optimal liquidation schedule.

## Key results
```
Amihud ranks (AAPL most liquid → MICRO least):
  AAPL: ILLIQ=0.000375, MICRO: ILLIQ=1.803

LVaR for $50M in MICRO-cap (ADV=$5M, spread=80bps):
  Base VaR:    $5.2M
  ELC (½ spr): $0.2M
  ILC (impact):$2.3M
  LVaR:        $7.7M  ← 47% above base VaR

Time to liquidate $50M in SMALL-cap at 20% ADV: 9 days
Total impact cost: $1.1M (220bps)
```

## References
- Amihud (2002). Illiquidity and Stock Returns. *J. Financial Markets* 5(1).
- Roll (1984). Implicit Measure of Bid-Ask Spread. *JF* 39(4).
- Bangia et al. (1999). Liquidity-Adjusted VaR.
