# CVA / XVA: Counterparty Credit Risk

Market value of counterparty default risk on a 5Y IRS. Implements CVA (counterparty), DVA (own credit), and FVA (funding cost).

## Key result
```
5Y payer IRS ($10M, CDS=150bp counterparty):
  CVA:  $7,546  (7.55bp)
  DVA:    $413  (0.41bp)  — own credit benefit
  FVA: -$3,168  (funding cost)
  Total XVA: $10,300 → reduces risk-free PV by $10,300

CVA scales with CDS spread:
  CDS 50bp  → CVA 2.6bp
  CDS 150bp → CVA 7.5bp  ← BBB counterparty
  CDS 500bp → CVA 22.5bp ← distressed
```

## References
- Gregory (2015). *The xVA Challenge*. Wiley.
- Basel Committee (2011). Basel III CCR Rules.
