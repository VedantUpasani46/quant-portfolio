# Multi-Curve OIS Discounting Framework

Post-2008 dual-curve swap pricing: OIS (risk-free) for discounting, LIBOR for projection. LIBOR-OIS spread impacts FRA and IRS prices by 12-28bp depending on tenor.

## Key result
```
IRS Par Rates (dual vs single curve):
  1Y: dual=4.7330%, single=4.4485%, spread=28.44bp
  5Y: dual=4.8035%, single=4.5963%, spread=20.72bp
 10Y: dual=4.8584%, single=4.6884%, spread=17.01bp

$10M 5Y payer: dual-curve PV differs by $91,745 vs single-curve
```

## References
- Bianchetti (2010). Two Curves, One Price. *Risk Magazine*.
- Mercurio (2010). A LIBOR Market Model with a Stochastic Basis.
