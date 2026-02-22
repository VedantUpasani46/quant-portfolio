# Almgren-Chriss Optimal Execution

Computes the optimal trade trajectory minimising implementation shortfall plus risk. Benchmarks TWAP, VWAP, and Almgren-Chriss optimal against each other via Monte Carlo simulation.

## The core trade-off

A trader must buy X shares over [0, T]. Two competing costs:
- **Trade fast**: large market impact (temporary + permanent)  
- **Trade slowly**: large timing risk (price drifts against you)

The optimal schedule balances these via a risk-aversion parameter λ:
```
min  E[IS] + λ · Var[IS]
```

## The Almgren-Chriss solution

Closed-form optimal inventory trajectory:
```
x*(t) = X · sinh(κ(T−t)) / sinh(κT)
```

where κ² = λσ²/η (trade-off between timing risk and impact cost).

- **κ → 0** (λ small, risk-tolerant): uniform TWAP-like trajectory
- **κ → ∞** (λ large, risk-averse): front-load aggressively

## Results

```
Order: 100K shares, 2% of ADV, $10M notional, 25% annual vol

Strategy        IS (bps)   IS ($K)     Var ($²)   Front-25%
────────────────────────────────────────────────────────────
TWAP               6.29    $6.29K      840,978      24.87%
VWAP               6.45    $6.45K      813,878      28.02%
AC (λ=1e-6)        6.29    $6.29K      840,939      24.87%  ← near-TWAP (low risk)
AC (λ=1e-5)        6.29    $6.29K      840,590      24.89%  ← front-loads slightly
```

For a liquid 2%-of-ADV order, the optimal solution approaches TWAP — impact is too small to warrant aggressive front-loading.

## Usage

```python
from optimal_execution import MarketParams, almgren_chriss_trajectory, simulate_execution

params = MarketParams(S0=100, sigma_annual=0.25, ADV=5_000_000, spread_bps=2)

# Optimal trajectory for 100K shares over 1 day, minute-by-minute
trades = almgren_chriss_trajectory(X=100_000, T=1.0, N=390, params=params, lam=1e-6)

# Simulate with market impact (Monte Carlo)
result = simulate_execution(trades, params, n_paths=1000)
print(f"IS: {result.is_bps:.2f} bps  Var: ${result.realised_variance:,.0f}")

# Execution efficient frontier
from optimal_execution import execution_frontier
frontier = execution_frontier(X=100_000, T=1.0, N=390, params=params)
```

## References
- Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. *Journal of Risk* 3(2).
- Bertsimas, D. & Lo, A. (1998). Optimal Control of Execution Costs. *Journal of Financial Markets* 1(1).
