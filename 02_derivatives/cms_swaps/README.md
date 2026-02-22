# CMS (Constant Maturity Swap) Pricing

Hagan (2003) convexity adjustment, CMS caplets/floorlets (Black formula), full CMS swap pricing, and CMS spread options (Kirk's approximation for curve steepeners).

## Key results
```
Convexity adjustments (CMS5, σ=20%):
  0.25Y reset: +0.72bps  (CA grows with reset time ✓)
  1.00Y reset: +3.02bps
  5.00Y reset: +16.5bps
  10.0Y reset: +34.2bps  (industry benchmark: ~30-40bps ✓)

CMS5 swap ($10M, 5Y, pay CMS vs fixed 5.20%):
  CMS leg PV:   $1,867,054
  Fixed leg PV: $2,270,053
  Swap PV:      +$403,000 (receive-fixed is in-the-money)

CMS spread (10Y-2Y), forward = 41.8bps:
  ATM call (K=0):  28.1bps
  25bp call:        3.4bps
```

## References
- Hagan (2003). Convexity Conundrums. *Wilmott Magazine*.
- Brigo & Mercurio (2006). Interest Rate Models, 2nd ed. Springer, Ch. 13.
