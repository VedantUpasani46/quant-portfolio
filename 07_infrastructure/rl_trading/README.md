# Reinforcement Learning for Trading

Q-Learning MDP for single-asset trading, optimal stopping via backward induction, and honest benchmarking against buy-and-hold.

## Key results
```
Q-Learning (T=800 train, 460 test days):
  Sharpe: 1.06 (vs buy-and-hold 2.10)
  Total P&L: 0.459 (vs B&H 0.907)
  Agent learns: momentum → hold, negative → buy (contrarian)

Optimal stopping (21-day horizon):
  Optimal value: $99.57
  Mean stopping day: 0.6 (sell early on upward drift)
  98.4% of paths stopped early
```

## References
- Mnih et al. (2015). Human-Level Control through Deep RL. *Nature* 518.
- Hambly et al. (2023). Recent Advances in RL in Finance. *Math Finance* 33(3).
