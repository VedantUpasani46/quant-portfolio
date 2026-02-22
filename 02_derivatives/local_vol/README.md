# Dupire Local Volatility Model

Extracts the **Dupire (1994) local volatility surface** σ_loc(S,t) from a market implied vol surface. Local vol is the unique continuous diffusion consistent with any given set of vanilla option prices — it is the cornerstone of exotic option pricing at every major structured products desk.

## The core idea

BSM prices all options with a **single constant vol**. The market instead shows a vol *surface*: different (K, T) pairs have different implied vols. Dupire asked:

> Is there a diffusion `dS = r·S·dt + σ(S,t)·S·dW` that is consistent with **all** market option prices simultaneously?

Answer: **yes, and it's unique.** The function σ_loc(S,t) is completely determined by the market implied vol surface via Dupire's formula.

## Dupire's formula

Derived from the forward Kolmogorov (Fokker-Planck) PDE:

```
σ_loc²(K,T) =  ∂C/∂T + (r-q)K·∂C/∂K + q·C
               ─────────────────────────────
                      ½ · K² · ∂²C/∂K²
```

where C(K,T) is the **market call price surface** (not a model price).

The denominator `½K²·∂²C/∂K²` is the (risk-neutral, discounted) probability density of S_T = K — the Breeden-Litzenberger (1978) formula. If it's negative anywhere, the surface has butterfly arbitrage.

## Implementation

```python
from dupire_local_vol import ImpliedVolSurface, DupireLocalVol, generate_market_surface

# Build from market data
surface = ImpliedVolSurface(
    strikes=strikes,       # array of strikes
    maturities=maturities, # array of maturities
    implied_vols=vols,     # 2D array (n_maturities × n_strikes)
    S0=100, r=0.05, q=0.0
)

dupire = DupireLocalVol(surface)

# Local vol at a single point
lv = dupire.local_vol(K=95, T=0.5)
print(f"Local vol: {lv:.4%}")

# Full surface
local_vol_surface = dupire.local_vol_surface(
    strikes=np.linspace(80, 120, 9),
    maturities=np.array([0.25, 0.5, 1.0, 2.0])
)
```

## Key properties

**1. Local vol vs implied vol at ATM:**
```
σ_loc(F_T, T) ≈ σ_impl(F_T, T) + T · ∂σ_impl/∂T
```
Local vol equals implied vol only for a perfectly flat surface.

**2. Skew amplification:**
```
σ_loc skew ≈ 2 × σ_impl skew
```
The local vol surface has roughly twice the skew of the implied vol surface. This is a key fact for model validation interviews.

**3. Forward smile:**
Local vol's biggest limitation — it predicts a **flat forward smile** (future smile is flat regardless of the current smile shape). Observed in cliquets and autocall pricing.

**4. Arbitrage-free conditions:**
- **Butterfly**: `∂²C/∂K² ≥ 0` (non-negative density)
- **Calendar spread**: `∂(σ²·T)/∂T ≥ 0` (total variance increasing in T)

## Why this matters for interviews

Model validation roles at Tier-1 banks will ask:

- *"What is the relationship between local vol and implied vol at ATM?"*
  → `σ_loc ≈ σ_impl + T·∂σ/∂T`
  
- *"What is Dupire's formula and where does it come from?"*
  → Forward Kolmogorov PDE applied to call prices
  
- *"What are the limitations of local vol?"*
  → Flat forward smile; poor for exotics sensitive to future vol dynamics
  
- *"How does local-stochastic vol (LSV) improve on plain local vol?"*
  → Combines Heston dynamics with a Dupire-style leverage function

## References

- Dupire, B. (1994). *Pricing with a Smile*. Risk Magazine, 7(1), 18–20.
- Gatheral, J. (2006). *The Volatility Surface*. Wiley, Ch. 1–2.
- Derman, E. & Kani, I. (1994). *Riding on a Smile*. Risk Magazine, 7(2), 32–39.
- Breeden, D. & Litzenberger, R. (1978). Prices of State-Contingent Claims. *J. Business*, 51(4).
