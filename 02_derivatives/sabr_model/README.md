# SABR Stochastic Volatility Model

The SABR (Stochastic Alpha Beta Rho) model — Hagan et al. (2002) — is the **industry standard for volatility surfaces** in interest rate derivatives. Every major bank uses SABR to price swaptions, caps, and floors. This is a from-scratch implementation of the full Hagan formula with calibration and no-arbitrage diagnostics.

## Why SABR is everywhere in rates

BSM prices options with a single flat vol. In practice, swaption vol surfaces are:
- **Skewed**: out-of-the-money receivers (low strike) trade richer than payers (high strike)
- **Curved**: the surface has a U-shape (smile) across strikes

SABR solves this with a **closed-form approximate implied vol formula** — fast enough to calibrate in real time, accurate enough for production use.

## The 4 parameters

| Param | Role | Effect |
|-------|------|--------|
| **α** (alpha) | ATM vol level | Shifts the surface up/down |
| **β** (beta)  | CEV backbone ∈ [0,1] | β=0: normal SABR (zero/neg rates); β=1: log-normal |
| **ρ** (rho)   | Spot-vol correlation | Negative → downward skew |
| **ν** (nu)    | Vol-of-vol | Larger → more pronounced smile curvature |

## Choosing β by asset class

| Asset Class | Typical β | Reason |
|-------------|-----------|--------|
| Interest rates (EUR, GBP) | β = 0 | Normal SABR handles near-zero rates |
| Interest rates (USD) | β = 0.5 | Common market convention |
| FX options | β = 1 | Log-normal backbone standard |
| Equity options | β = 1 | Heston preferred, but SABR used |

## Hagan formula (Eq. 2.17b)

For F ≠ K, the approximate BSM implied vol is:

```
σ(F,K) ≈ [α / (FK)^((1-β)/2) · (...)] · [z/χ(z)] · [1 + (...) · T]
```

where:
- `z = (ν/α) · (FK)^((1-β)/2) · ln(F/K)`
- `χ(z) = ln[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]`

The ATM limit (F=K) requires L'Hôpital → handled separately.

## Installation

```bash
pip install numpy scipy
```

## Usage

```python
from sabr_model import SABRParams, sabr_implied_vol, sabr_option_price, SABRCalibrator

# Typical swaption market parameters (rates context, β=0.5)
params = SABRParams(alpha=0.025, beta=0.5, rho=-0.30, nu=0.45)

F = 0.04   # 4% forward rate
K = 0.035  # 3.5% strike (OTM receiver)
T = 1.0    # 1-year expiry

# Implied vol
iv = sabr_implied_vol(F, K, T, params)
print(f"SABR implied vol: {iv:.4%}")    # e.g. 23.46%

# Option price (SABR vol fed into BSM)
price = sabr_option_price(F, K, T, r=0.0, params=params, option_type='put')
print(f"Receiver swaption: {price:.6f}")
```

### Calibrate to market vol strip

```python
import numpy as np
from sabr_model import SABRCalibrator

F = 0.04
T = 1.0
strikes     = F * np.array([0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30])
market_vols = np.array([0.245, 0.228, 0.213, 0.200, 0.195, 0.198, 0.208])

calibrator = SABRCalibrator(F=F, T=T, strikes=strikes, market_vols=market_vols, beta=0.5)
fitted = calibrator.calibrate()
print(fitted)
```

### Effect of ρ on skew

```python
for rho in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    p = SABRParams(alpha=0.02, beta=0.5, rho=rho, nu=0.40)
    v_otm = sabr_implied_vol(F, F*0.80, T, p)   # OTM receiver
    v_atm = sabr_implied_vol(F, F,      T, p)   # ATM
    v_itm = sabr_implied_vol(F, F*1.20, T, p)   # OTM payer
    print(f"ρ={rho:+.2f}  80%: {v_otm:.3%}  ATM: {v_atm:.3%}  120%: {v_itm:.3%}")
```

## Running the demo

```bash
python sabr_model.py
```

Shows: vol smile across moneyness, effect of ρ on skew, effect of ν on curvature, perfect parameter recovery on calibration.

## References

- Hagan, P.S. et al. (2002). *Managing Smile Risk*. Wilmott Magazine, 84–108.
- West, G. (2005). Calibration of the SABR Model in Illiquid Markets. *AMF*, 12(4).
- Hull, J.C. (2022). *Options, Futures and Other Derivatives*, 11th ed., Ch. 27.
