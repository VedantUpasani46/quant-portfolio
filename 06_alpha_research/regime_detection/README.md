# Market Regime Detection — Gaussian HMM

Hidden Markov Model for identifying latent market regimes (bull, bear, crisis) implemented entirely from scratch: Baum-Welch EM estimation, forward-backward algorithm, and Viterbi decoding.

## The model

```
Observed:  r_t ~ N(μ_{S_t}, σ²_{S_t})         [Gaussian emission]
Hidden:    P(S_t = j | S_{t-1} = i) = A_{ij}   [Markov transitions]
```

Three regimes for equities:
- **Crisis**: very negative return, high vol (−61% ann, σ=30%)
- **Bear**: mild return, moderate vol (+6.6% ann, σ=11%)  
- **Bull**: positive return, normal vol (+8.3% ann, σ=19%)

## Results (5-year simulated daily data)

```
Regime Detection Accuracy: 98.2%

Transition Matrix:
            →Crisis   →Bear   →Bull
Crisis      0.480     0.000   0.520   (very transient)
Bear        0.004     0.996   0.000   (very persistent: ~262 days avg)
Bull        0.172     0.011   0.817   (persistent: ~5.5 days avg)
```

## The Baum-Welch algorithm

**E-step** (forward-backward):
```
γ_t(k) = P(S_t=k | X_{1:T}) ∝ α_t(k) · β_t(k)
```

**M-step** (parameter update):
```
μ_k = Σ_t γ_t(k)·X_t / Σ_t γ_t(k)
A_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
```

Implemented in numerically stable log-space to avoid underflow on long sequences.

## Usage

```python
from regime_detection import GaussianHMM

hmm = GaussianHMM(n_states=3, n_iter=100, n_init=10)
hmm.fit(returns)

states   = hmm.predict(returns)        # Viterbi: most likely state sequence
probs    = hmm.predict_proba(returns)  # P(S_t=k | all data)
summary  = hmm.regime_summary(["Crisis", "Bear", "Bull"])

# Regime-conditional strategy
equity_alloc = {0: 0.10, 1: 0.40, 2: 0.80}  # crisis, bear, bull
current_state = states[-1]
current_equity = equity_alloc[current_state]
```

## References

- Hamilton, J.D. (1989). A New Approach to Nonstationary Time Series. *Econometrica* 57(2).
- Ang, A. & Bekaert, G. (2002). International Asset Allocation with Regime Shifts. *RFS* 15(4).
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 17.
