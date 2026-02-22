# Key Rate Duration (KRD) Analysis

Measures bond and portfolio sensitivity to yield changes at specific tenor buckets. Unlike DV01 (parallel shift), KRD shows WHERE on the curve your risk is concentrated.

## Key result
```
10Y 5% Bond ($1M face):
  10Y bucket: $638 (82.4%)   ← overwhelmingly in 10Y bucket
   7Y bucket: $62  (8.0%)
   5Y bucket: $39  (5.0%)
  Total DV01: $774  | KRD sum = $774 (1.000× total ✓)

5Y Zero-Coupon Bond:
   5Y bucket: $387 (100.0%)  ← all risk at one point ✓
```

## References
- Ho, T.S.Y. (1992). Key Rate Durations. *Journal of Fixed Income* 2(2).
- Tuckman & Serrat (2012). *Fixed Income Securities*, 3rd ed.
