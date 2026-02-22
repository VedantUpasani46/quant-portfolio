# Quant Portfolio 🏦

A production-grade quantitative finance library spanning derivatives pricing,
risk models, portfolio optimisation, alpha research, and execution algorithms.

Built from scratch in pure Python + NumPy/SciPy — no black-box wrappers.
Every module includes mathematical derivations, benchmark validation,
and interview-ready explanations of the key results.

---

## Structure

```
quant-portfolio/
├── 01_fixed_income/        Bond analytics, duration, multi-curve OIS
├── 02_derivatives/         Options, Heston, SABR, local vol, CMS swaps
├── 03_credit_risk/         Merton model, CVA/XVA, copulas, EVT
├── 04_risk_models/         GARCH, DCC-GARCH, VaR, liquidity risk
├── 05_portfolio/           MVO, Kelly criterion, TC-aware optimisation
├── 06_alpha_research/      Factor models, ML, NLP sentiment, RL
└── 07_infrastructure/      Backtesting, execution, risk engine, P&L
```

---

## Modules

### 01 — Fixed Income

| Module | Description |
|--------|-------------|
| `fixed_income/` | Bond analytics, duration, convexity, DV01 |
| `yield_curve/` | Zero curve bootstrapping from market instruments |
| `key_rate_durations/` | Bucketed DV01 — hedging twist and butterfly risk |
| `multicurve_ois/` | Post-2008 dual-curve OIS discounting (LIBOR-OIS spread) |
| `fra_futures/` | FRA pricing, bond futures, CTD identification, convexity adj |

### 02 — Derivatives Pricing

| Module | Description |
|--------|-------------|
| `black_scholes/` | BSM closed-form, full Greeks, put-call parity |
| `binomial_tree/` | CRR binomial — American and European |
| `monte_carlo/` | MC pricer with antithetic variates and control variates |
| `heston_model/` | Heston stochastic vol — characteristic function, calibration |
| `sabr_model/` | SABR vol smile — Hagan approximation, swaption vols |
| `local_vol/` | Dupire local vol surface from implied vol grid |
| `tree_pricers/` | Trinomial trees and Black-Derman-Toy interest rate model |
| `interest_rate_models/` | Hull-White, CIR, Vasicek — calibration and simulation |
| `cms_swaps/` | CMS pricing — Hagan (2003) convexity adj, spread options |

### 03 — Credit Risk

| Module | Description |
|--------|-------------|
| `credit_risk/` | CDS pricing, survival probabilities, hazard rates |
| `merton_model/` | Structural credit model — equity as call on assets |
| `cva_xva/` | CVA/DVA/FVA — Monte Carlo expected exposure profiles |
| `evt_tail_risk/` | Extreme Value Theory — GEV, GPD, POT, tail VaR |
| `copula_models/` | Gaussian/t-copula, CDO tranching, joint tail dependence |
| `stress_testing/` | Historical and hypothetical stress scenarios |

### 04 — Risk Models

| Module | Description |
|--------|-------------|
| `garch/` | GARCH(1,1) MLE — vol forecasting, news impact curve |
| `dcc_garch/` | Dynamic Conditional Correlation — time-varying correlation matrix |
| `var_calculator/` | VaR: historical simulation, parametric, Monte Carlo |
| `liquidity_risk/` | Amihud illiquidity, Roll spread, LVaR, liquidation schedule |
| `bootstrap_ci/` | Bootstrap CIs — Sharpe, alpha, VaR; block bootstrap for TS |
| `time_series_diagnostics/` | ACF, Ljung-Box Q-test, ADF, Lo-MacKinlay variance ratio |

### 05 — Portfolio Optimisation

| Module | Description |
|--------|-------------|
| `mean_variance/` | Markowitz MVO — efficient frontier, Sharpe maximisation |
| `mean_variance_full/` | Extended MVO with constraints and Black-Litterman |
| `kelly_criterion/` | Kelly optimal leverage — fractional Kelly, multi-asset |
| `tc_aware_mvo/` | Garleanu-Pedersen TC-aware optimisation — GP policy |

### 06 — Alpha Research

| Module | Description |
|--------|-------------|
| `fama_french/` | FF3/FF5 factor model — rolling betas, alpha extraction |
| `pairs_trading/` | Pairs trading — cointegration test, Kalman filter spread |
| `pca_factors/` | PCA factor analysis — statistical risk factors |
| `regime_detection/` | HMM regime detection — bull/bear/crisis states |
| `ml_return_prediction/` | ML cross-sectional alpha — Lasso, Ridge, random forest |
| `gradient_boosting/` | XGBoost/LightGBM alpha — walk-forward CV, IC analysis |
| `neural_networks/` | Neural net return prediction — Gu/Kelly/Xiu (2020) |
| `nlp_sentiment/` | NLP earnings call sentiment — Loughran-McDonald lexicon |
| `robust_stats/` | Huber regression, Theil-Sen, MCD covariance |

### 07 — Infrastructure

| Module | Description |
|--------|-------------|
| `backtesting/` | Event-driven backtesting engine with transaction costs |
| `optimal_execution/` | Almgren-Chriss execution — TWAP, VWAP, IS strategies |
| `options_risk/` | Real-time options risk engine — Greeks, scenario P&L |
| `pnl_attribution/` | P&L attribution — Brinson, factor decomp, Greeks PnL |
| `alpha_pipeline/` | End-to-end alpha pipeline — signal → portfolio → execution |
| `rl_trading/` | Reinforcement learning trading — Q-learning, optimal stopping |

---

## Requirements

```bash
pip install numpy scipy pandas scikit-learn matplotlib
```

All modules run independently with no external data dependencies —
synthetic data is generated internally for reproducibility.

---

## Running a Module

```bash
python 01_fixed_income/key_rate_durations/key_rate_durations.py
python 02_derivatives/heston_model/heston_model.py
python 06_alpha_research/gradient_boosting/xgboost_alpha.py
```

Each module prints full results to stdout, including:
- Mathematical derivations and formulas
- Numerical results validated against industry benchmarks
- Key interview Q&A at the end

---

## Key Benchmarks

| Module | Metric | Value |
|--------|--------|-------|
| Heston calibration | Vol smile RMSE | < 0.5 vol points |
| SABR | Hagan approx error | < 1bp at ATM |
| CMS convexity adj | 10Y reset | ~34bps (vs ~30-40bps market) |
| GARCH(1,1) | Persistence α+β | 0.97 (typical equities) |
| DCC correlation | Regime shift detection | 0.28 → 0.65 captured |
| Kelly criterion | f* coin (p=55%) | 10% exactly |
| XGBoost IC | Walk-forward mean | 0.12 |
| NLP sentiment IC | Earnings calls | 0.34 |

---

## References

A selection of the academic papers implemented:

- Black & Scholes (1973). *JoPoliE* — options pricing
- Heston (1993). *RFS* — stochastic volatility
- Hagan et al. (2002). *Wilmott* — SABR model
- Hagan (2003). *Wilmott* — CMS convexity adjustment
- Engle (2002). *JBES* — DCC-GARCH
- Garleanu & Pedersen (2013). *JF* — TC-aware optimisation
- Gu, Kelly & Xiu (2020). *RFS* — neural nets for asset pricing
- Loughran & McDonald (2011). *JF* — NLP for finance
- Almgren & Chriss (2001). *JoRisk* — optimal execution
- Merton (1974). *JF* — structural credit model
- Kelly (1956). *Bell Sys Tech J* — optimal growth criterion
- Amihud (2002). *JFM* — illiquidity and stock returns
