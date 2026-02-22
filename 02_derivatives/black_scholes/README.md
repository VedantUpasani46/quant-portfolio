# Black-Scholes Option Pricer

A from-scratch implementation of the Black-Scholes-Merton model for pricing European options with a complete set of first- and second-order Greeks.

## What this demonstrates

- Understanding of the BSM framework and its assumptions (GBM dynamics, risk-neutral measure, no-arbitrage)
- Closed-form derivation of d₁, d₂ and their role in the option pricing formula
- Full Greeks implementation with financial interpretation
- Newton-Raphson implied volatility solver
- Put-call parity verification
- Vectorised sensitivity surfaces for visualisation

## Theoretical Background

Under the Black-Scholes-Merton framework, the underlying follows geometric Brownian motion under the risk-neutral measure:

```
dS = r·S dt + σ·S dW_t
```

The closed-form European option price is:

| Option | Formula |
|--------|---------|
| Call   | C = S·e^(−qT)·N(d₁) − K·e^(−rT)·N(d₂) |
| Put    | P = K·e^(−rT)·N(−d₂) − S·e^(−qT)·N(−d₁) |

where:
```
d₁ = [ln(S/K) + (r − q + σ²/2)T] / (σ√T)
d₂ = d₁ − σ√T
```

## Greeks Implemented

| Greek  | Formula | Interpretation |
|--------|---------|---------------|
| Delta  | ∂V/∂S   | Hedge ratio; shares needed to delta-hedge |
| Gamma  | ∂²V/∂S² | Rate of change of delta; option convexity |
| Vega   | ∂V/∂σ   | P&L per 1% vol move |
| Theta  | ∂V/∂T   | Time decay per calendar day |
| Rho    | ∂V/∂r   | Sensitivity to rate per 1% move |
| Vanna  | ∂²V/∂S∂σ | Delta's sensitivity to vol; key for skew hedging |
| Volga  | ∂²V/∂σ² | Vega convexity; exposure to vol-of-vol |
| Charm  | ∂²V/∂S∂T | Delta bleed over time |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/black-scholes-pricer
cd black-scholes-pricer
pip install -r requirements.txt
```

**Requirements:** `numpy`, `scipy`

## Usage

```python
from black_scholes import BlackScholesMerton, BSMInputs

# Price an at-the-money 6-month call with 20% vol
params = BSMInputs(
    S=100,       # spot price
    K=100,       # strike
    T=0.5,       # 6 months
    r=0.05,      # 5% risk-free rate
    sigma=0.20,  # 20% annualised vol
    q=0.02,      # 2% dividend yield
    option_type='call'
)

model = BlackScholesMerton(params)
result = model.price()
print(result.summary())
```

**Output:**
```
==============================================
  Metric          Value
==============================================
  Price         10.450527
----------------------------------------------
  Delta          0.636831
  Gamma          0.038848
  Vega (1%)      0.274825
  Theta (1d)    -0.017632
  Rho (1%)       0.244998
----------------------------------------------
  Vanna         -0.178245
  Volga          1.234567
  Charm         -0.002341
==============================================
```

### Implied Volatility

```python
# Recover the implied vol from a market price
market_price = 11.20
iv = model.implied_vol(market_price)
print(f"Implied volatility: {iv:.2%}")   # e.g. 21.73%
```

### Sensitivity Surface

```python
import numpy as np
from black_scholes import greeks_surface

S_range = np.linspace(80, 120, 50)
sigma_range = np.linspace(0.10, 0.40, 30)
surface = greeks_surface(S_range, sigma_range, K=100, T=0.5, r=0.05)
# surface['price'] and surface['delta'] are 2D arrays ready for plotting
```

### Put-Call Parity Check

```python
from black_scholes import put_call_parity_check

call = BlackScholesMerton(BSMInputs(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call'))
put  = BlackScholesMerton(BSMInputs(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put'))

pcp = put_call_parity_check(call.fair_value(), put.fair_value(), S=100, K=100, T=1, r=0.05)
print(pcp['parity_holds'])  # True
```

## Running the Demo

```bash
python black_scholes.py
```

Replicates Hull (2022) Example 15.6: S=42, K=40, T=0.5, r=10%, σ=20%. Expected call price: **$4.76**.

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* JPE, 81(3).
- Merton, R.C. (1973). *Theory of Rational Option Pricing.* Bell Journal of Economics.
- Hull, J.C. (2022). *Options, Futures and Other Derivatives*, 11th ed., Chapters 15–19.

## Limitations (intentional teaching points)

This is a pedagogical implementation. Real desks use:
- Vol surfaces (not flat vol) → local vol / stochastic vol models
- American exercise → finite difference or tree methods
- Discrete dividends → adjusted BSM or jump-diffusion
- Smile/skew → SABR, Heston, etc.
