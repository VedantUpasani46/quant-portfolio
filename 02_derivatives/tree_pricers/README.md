# Binomial & Trinomial Tree Option Pricers

Cox-Ross-Rubinstein (1979) binomial tree and Kamrad-Ritchken (1991) trinomial tree implementations for pricing **European and American options**. Trees are the standard approach for American options — no closed form exists for American puts.

## Why trees when we have BSM?

BSM is European only — it cannot handle early exercise. The key insight of trees:

> At each node, compare **hold** (continuation value) vs **exercise** (intrinsic value). The American option value is `max(continuation, exercise)`.

This backward induction is exact in the discrete-time limit and converges to the correct continuous-time American option price as N → ∞.

## CRR Binomial Tree

At each step dt = T/N:

| Parameter | Formula | Meaning |
|-----------|---------|---------|
| u | `exp(σ√dt)` | Up factor |
| d | `1/u` | Down factor (ensures recombination) |
| p | `(exp((r-q)dt) - d) / (u-d)` | Risk-neutral up probability |

Recombination: `S·u·d = S` means the tree has O(N²) nodes, not O(2^N).

```
          S·u²
        S·u
      S     S·u·d = S
        S·d
          S·d²
```

## Trinomial Tree (Kamrad-Ritchken)

Three branches: up, middle, down.

| Branch | Factor | Probability |
|--------|--------|-------------|
| Up     | u = exp(λσ√dt) | p_u = 1/(2λ²) + drift |
| Middle | 1               | p_m = 1 - 1/λ²        |
| Down   | d = 1/u         | p_d = 1/(2λ²) - drift  |

Default λ=√3 ensures valid (0,1) probabilities. The trinomial tree is equivalent to an explicit finite-difference scheme.

## Results

```
Benchmark: BSM European Call = 10.450584

N=1000 Binomial European Call:   10.448584  (error: 2.00e-03)
N=500  Trinomial European Call:  10.446603  (error: 3.98e-03)

N=1000 American Put:   6.089595
N=500  American Put:   6.086648
Early exercise premium: 0.516069  (American > European)

Deep ITM American Put (S=70, K=100):
  Intrinsic: 30.0000  →  American: 30.0000  (correctly exercises immediately)
```

## Usage

```python
from tree_pricers import CRRBinomialTree, TrinomialTree

# American put (no closed form — trees are the standard approach)
tree = CRRBinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0, n_steps=1000)
result = tree.price(option_type='put', american=True)
print(result.summary())
# Price: 6.089595  |  Delta: -0.498  |  Early exercise: True

# European call (verify against BSM)
bsm_check = CRRBinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.20, n_steps=1000)
result_eu = bsm_check.price(option_type='call', american=False)
# Price: 10.448584  (BSM: 10.450584, error: 0.002)

# Trinomial — faster convergence per step
tri = TrinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.20, n_steps=500)
result_tri = tri.price(option_type='put', american=True)
print(result_tri.summary())
```

### Convergence study

```python
from tree_pricers import convergence_study

convergence_study(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0,
                  option_type='call', american=False,
                  step_counts=[10, 50, 100, 500, 1000],
                  bsm_price=10.450584)
```

## American vs European — when early exercise is optimal

For a **put**:
- If S falls far below K, the value of waiting < intrinsic value
- Holding costs you carry (you earn rf on K if exercised)
- Early exercise is always eventually optimal for deep ITM puts

For a **call** (no dividends):
- Never early exercise — time value always positive
- American call = European call (no dividend)
- With dividends: early exercise just before ex-div date can be optimal

## Convergence properties

| Method | Convergence Rate | Notes |
|--------|-----------------|-------|
| Binomial | O(1/N) with oscillation | Oscillates as N increases |
| Trinomial | Smoother O(1/N) | Better per unit computation |
| Richardson extrapolation | O(1/N²) | Combine N and 2N for higher order |

## References

- Cox, J.C., Ross, S.A. & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *JFE*, 7(3).
- Boyle, P.P. (1988). A Lattice Framework for Option Pricing. *JFQA*, 23(1).
- Kamrad, B. & Ritchken, P. (1991). Multinomial Approximating Models. *Management Science*, 37(12).
- Hull, J.C. (2022). *Options, Futures and Other Derivatives*, 11th ed., Ch. 21.
