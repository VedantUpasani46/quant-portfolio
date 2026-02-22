# Kelly Criterion and Optimal Leverage

Optimal bet sizing for single and multi-asset portfolios. Includes discrete Kelly, continuous Kelly, fractional Kelly tradeoff, and multi-asset (matrix) Kelly.

## Key results
```
Discrete (p=55% coin): f* = 10% of bankroll ✓

Continuous (Sharpe=0.85):
  f* = μ/σ² = 3.56× leverage
  Half Kelly (κ=0.5): 75% of max growth, much smaller drawdowns

Fractional Kelly tradeoff:
  κ=0.50: 75% of max growth
  κ=1.00: 100% growth, median MDD = -79%
  κ=2.00: 0% growth, ruin 42%  ← overbetting kills wealth

Multi-asset Kelly = fully-levered tangency portfolio
  (Markowitz answers WHAT, Kelly answers HOW MUCH)
```

## References
- Kelly (1956). A New Interpretation of Information Rate. *Bell Sys. Tech. J.*
- Thorp (1997). The Kelly Criterion in Blackjack and the Stock Market.
