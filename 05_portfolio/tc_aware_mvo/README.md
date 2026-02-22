# Transaction Cost–Aware Portfolio Optimization

Garleanu-Pedersen (2013) framework: optimal portfolio lies between current and Markowitz target. Trade a fraction of the gap, not all at once.

## Key results
```
5 large-cap stocks:
  Turnover to Markowitz target: 25.5%
  Turnover to TC-aware optimal: 8.3%  ← 3× less turnover

GP policy (how much to trade):
  λ=10:   trade 36% of gap
  λ=50:   trade 10% of gap
  λ=200:  trade 2.8% of gap
  (higher λ = higher TC = trade slower)
```

## References
- Garleanu & Pedersen (2013). Dynamic Trading with Predictable Returns. *JF* 68(6).
