# FRA, Bond Futures, CTD, and Convexity Adjustment

FRA pricing from zero curve, bond futures cheapest-to-deliver analysis, and the Hull-White convexity adjustment between futures and forward yields.

## Key results
```
At-money 3×6 FRA (K=4.4344%): PV = $0.00 ✓

CTD Bond (5.000% Nov 2033): lowest gross basis = 59.22

Convexity adjustment (futures yield > forward yield):
  2Y expiry on 10Y bond: -32.9bp  (Hull textbook: ~30bp)
  → Misestimating this on $1B 10Y swap book = $10M error
```

## References
- Hull (2022). *Options, Futures and Other Derivatives*, Ch. 6.
- Burghardt et al. (1994). *The Treasury Bond Basis*.
