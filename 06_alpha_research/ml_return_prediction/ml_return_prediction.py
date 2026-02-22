"""
Machine Learning for Return Prediction
========================================
Regularised linear models (Ridge, Lasso, ElasticNet) applied to
cross-sectional return prediction. Implements the core elements of
quantitative equity research at hedge funds and systematic managers.

Why regularisation?
  In a universe of N stocks with K features, the OLS estimator
    β̂ = (X'X)⁻¹X'y
  suffers from:
  1. Overfitting: small T/K ratio → huge estimation error
  2. Multicollinearity: correlated features → unstable β̂
  3. Curse of dimensionality: prediction error ↑ with K

  Regularisation adds a penalty to the loss function to shrink β̂:
    Ridge (L2):     min ||y - Xβ||² + λ||β||²
    Lasso (L1):     min ||y - Xβ||² + λ||β||₁   (sparse β̂; feature selection)
    ElasticNet:     min ||y - Xβ||² + λ₁||β||₁ + λ₂||β||²

Regularisation paths:
  As λ increases → β̂ → 0 (more shrinkage, less variance, more bias)
  As λ → 0       → β̂ = OLS (no shrinkage)
  Optimal λ chosen via cross-validation or information criterion.

Walk-forward (rolling) validation:
  In financial time series, the standard k-fold CV is WRONG:
  it leaks future information into training.
  Instead: train on [t₀, t], predict at [t+1, t+h], roll forward.

Key metrics for return predictors:
  Information Coefficient (IC): Spearman rank correlation between
    predicted and realised returns. IC > 0.05 is considered good.
  Information Ratio (IR):  IC / std(IC) — consistency of prediction.
  Hit Rate: fraction of periods with positive IC.

References:
  - Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. JRSS-B.
  - Zou, H. & Hastie, T. (2005). Regularization via Elastic Net. JRSS-B.
  - Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Ch. 7, 12.
  - Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning.
    Review of Financial Studies 33(5), 2223–2273.
  - Chincarini, L. & Kim, D. (2006). Quantitative Equity Portfolio Management. McGraw-Hill.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(returns: np.ndarray, n_stocks: int, T: int,
                     seed: int = 42) -> np.ndarray:
    """
    Construct a realistic cross-sectional feature matrix.

    Features (all cross-sectionally z-scored at each time step):
      1. Momentum (12-1): 12-month return excluding last month (Jegadeesh 1993)
      2. Short-term reversal: 1-month return (Jegadeesh 1990)
      3. Volatility: 60-day realised vol (negatively predicts returns)
      4. Beta: market beta (CAPM)
      5. Idiosyncratic vol: vol of residuals from market regression
      6. Skewness: return skewness (Harvey & Siddique 2000)

    Returns feature matrix X of shape (n_stocks, n_features) for a given t.
    """
    rng = np.random.default_rng(seed)
    # returns: (T, n_stocks)
    features = {}

    # Momentum (use cumulative returns over first 11 months, skip last 1)
    if T > 274:  # need at least 252+22 days for momentum
        mom_window = min(T - 22, 252)
        momentum = np.sum(returns[-mom_window-22:-22, :], axis=0)
        features["momentum"] = momentum

    # Short-term reversal (last 22 trading days)
    if T >= 22:
        st_reversal = np.sum(returns[-22:, :], axis=0)
        features["st_reversal"] = -st_reversal  # negative sign: reversal

    # Volatility (last 60 days, annualised)
    if T >= 60:
        vol_window = returns[-60:, :]
        realised_vol = np.std(vol_window, axis=0) * math.sqrt(252)
        features["volatility"] = -realised_vol   # negative sign: high vol → low returns

    # Beta (CAPM): regress each stock on market return
    market = returns.mean(axis=1)  # equal-weighted market
    if T >= 60:
        betas = []
        for i in range(n_stocks):
            cov_im = np.cov(returns[-60:, i], market[-60:])[0, 1]
            var_m = np.var(market[-60:])
            betas.append(cov_im / var_m if var_m > 0 else 1.0)
        features["beta"] = -np.array(betas)  # low-beta anomaly

    # Idiosyncratic vol (residual vol after removing market factor)
    if T >= 60:
        market_60 = market[-60:]
        idio_vols = []
        for i in range(n_stocks):
            y = returns[-60:, i]
            beta_i = np.cov(y, market_60)[0, 1] / max(np.var(market_60), 1e-10)
            residuals = y - beta_i * market_60
            idio_vols.append(np.std(residuals) * math.sqrt(252))
        features["idio_vol"] = -np.array(idio_vols)

    # Return skewness (Harvey & Siddique: negative skew premium)
    if T >= 60:
        skews = []
        for i in range(n_stocks):
            r = returns[-60:, i]
            if r.std() > 0:
                skews.append(float(pd.Series(r).skew()))
            else:
                skews.append(0.0)
        features["skewness"] = -np.array(skews)  # negative skew → higher expected return

    # Ensure all 6 features are present (pad with zeros if insufficient data)
    all_feature_names = ["momentum", "st_reversal", "volatility", "beta", "idio_vol", "skewness"]
    for fname in all_feature_names:
        if fname not in features:
            features[fname] = np.zeros(n_stocks)
    # Keep consistent ordering
    features = {k: features[k] for k in all_feature_names}

    if not features:
        return np.zeros((n_stocks, 1))

    X = np.column_stack(list(features.values()))

    # Cross-sectional z-score each feature
    for j in range(X.shape[1]):
        col = X[:, j]
        std = col.std()
        if std > 0:
            X[:, j] = (col - col.mean()) / std

    return X, list(features.keys())


# ---------------------------------------------------------------------------
# Walk-forward return prediction
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Results from walk-forward ML prediction."""
    model_name: str
    ic_series: np.ndarray         # IC per period
    ic_mean: float
    ic_std: float
    ir: float                     # Information Ratio = IC_mean / IC_std
    hit_rate: float               # fraction of periods with IC > 0
    top_decile_returns: np.ndarray
    bottom_decile_returns: np.ndarray
    long_short_spread: float      # top decile mean - bottom decile mean
    alpha_bps_ann: float          # annualised alpha in bps
    feature_importances: Optional[dict] = None  # for Lasso/Ridge


class MLReturnPredictor:
    """
    Walk-forward return prediction using regularised linear models.

    The prediction framework (cross-sectional):
      1. At each rebalance date t:
           a. Compute features X_t from returns up to t
           b. Fit model on (X_{t-lookback}, ..., X_{t-1}) → y (next-period return)
           c. Predict ŷ_t = X_t · β̂
           d. Go long top decile, short bottom decile
      2. Measure IC, IR, hit rate across all periods

    Usage
    -----
    >>> predictor = MLReturnPredictor(returns, model_type='lasso')
    >>> result = predictor.run()
    >>> print(result.ic_mean, result.ir)
    """

    def __init__(
        self,
        returns: np.ndarray,   # shape (T, n_stocks)
        model_type: str = "lasso",
        alpha: float = 1e-3,   # regularisation strength
        train_window: int = 252,
        prediction_horizon: int = 21,  # 1-month ahead
        min_train_periods: int = 126,
    ):
        self.returns = returns
        self.model_type = model_type
        self.alpha = alpha
        self.train_window = train_window
        self.pred_horizon = prediction_horizon
        self.min_train_periods = min_train_periods
        self.T, self.n_stocks = returns.shape

    def _make_model(self):
        if self.model_type == "ridge":
            return Ridge(alpha=self.alpha, fit_intercept=True)
        elif self.model_type == "lasso":
            return Lasso(alpha=self.alpha, fit_intercept=True, max_iter=5000)
        elif self.model_type == "elasticnet":
            return ElasticNet(alpha=self.alpha, l1_ratio=0.5, max_iter=5000)
        elif self.model_type == "ols":
            return Ridge(alpha=1e-8)  # Ridge with tiny λ ≈ OLS
        raise ValueError(f"Unknown model: {self.model_type}")

    def _compute_ic(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Spearman IC between predicted ranks and realised return ranks."""
        if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
            return 0.0
        ic, _ = spearmanr(y_pred, y_true)
        return float(ic) if not math.isnan(ic) else 0.0

    def run(self) -> PredictionResult:
        """
        Execute the walk-forward prediction and collect metrics.
        """
        R = self.returns
        T, n = R.shape
        H = self.pred_horizon

        ic_series = []
        top_decile = []
        bottom_decile = []
        all_importances = []

        step = max(H, 21)  # rebalance monthly

        for t in range(self.min_train_periods + H, T - H, step):
            # Training window
            t_start = max(0, t - self.train_window)
            R_train = R[t_start:t]
            if len(R_train) < self.min_train_periods:
                continue

            # Build training dataset:
            # For each sub-period s in training window, features at s → returns at s+H
            X_train_list, y_train_list = [], []
            for s in range(H + 60, len(R_train) - H, step):
                X_s, _ = compute_features(R_train[:s], n, s)
                y_s = R_train[s:s + H].sum(axis=0)   # H-period ahead return
                X_train_list.append(X_s)
                y_train_list.append(y_s)

            if not X_train_list:
                continue

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)

            # Winsorise targets at 3σ
            y_std = y_train.std()
            y_train = np.clip(y_train, -3 * y_std, 3 * y_std)

            # Fit model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            model = self._make_model()
            try:
                model.fit(X_scaled, y_train)
            except Exception:
                continue

            # Predict at t
            X_pred, feat_names = compute_features(R[:t], n, t)
            X_pred_scaled = scaler.transform(X_pred)
            scores = model.predict(X_pred_scaled)

            # Realised returns over next H periods
            y_true = R[t:t + H].sum(axis=0)

            # IC
            ic = self._compute_ic(scores, y_true)
            ic_series.append(ic)

            # Top/bottom decile returns
            n_decile = max(n // 10, 1)
            rank_idx = np.argsort(scores)
            top_ret = y_true[rank_idx[-n_decile:]].mean()
            bot_ret = y_true[rank_idx[:n_decile]].mean()
            top_decile.append(top_ret)
            bottom_decile.append(bot_ret)

            # Feature importances (via absolute coefficients)
            if hasattr(model, "coef_") and feat_names:
                importances = dict(zip(feat_names, np.abs(model.coef_)))
                all_importances.append(importances)

        if not ic_series:
            raise RuntimeError("No predictions generated — check parameters.")

        ic_arr = np.array(ic_series)
        top_arr = np.array(top_decile)
        bot_arr = np.array(bottom_decile)
        spread = float((top_arr - bot_arr).mean())
        periods_per_year = 252 / self.pred_horizon
        alpha_ann = spread * periods_per_year * 10000  # bps

        # Average feature importances
        feat_imp = None
        if all_importances:
            all_keys = set().union(*[d.keys() for d in all_importances])
            feat_imp = {k: float(np.mean([d.get(k, 0) for d in all_importances]))
                        for k in all_keys}

        return PredictionResult(
            model_name=self.model_type,
            ic_series=ic_arr,
            ic_mean=float(ic_arr.mean()),
            ic_std=float(ic_arr.std()),
            ir=float(ic_arr.mean() / ic_arr.std()) if ic_arr.std() > 0 else 0.0,
            hit_rate=float((ic_arr > 0).mean()),
            top_decile_returns=top_arr,
            bottom_decile_returns=bot_arr,
            long_short_spread=spread,
            alpha_bps_ann=alpha_ann,
            feature_importances=feat_imp,
        )


# ---------------------------------------------------------------------------
# Cross-validation for λ selection
# ---------------------------------------------------------------------------

def cross_validate_alpha(
    returns: np.ndarray,
    model_type: str = "lasso",
    alpha_grid: Optional[np.ndarray] = None,
    n_splits: int = 5,
) -> dict:
    """
    Walk-forward cross-validation to select the optimal regularisation
    parameter λ (alpha in sklearn).

    Uses TimeSeriesSplit to avoid lookahead: each fold trains on past
    data only and validates on the immediately following period.
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(-5, 0, 10)

    T, n = returns.shape
    H = 21

    # Build a flat feature-return dataset for CV
    X_all, y_all = [], []
    for t in range(H + 60, T - H, 21):
        X_t, _ = compute_features(returns[:t], n, t)
        y_t = returns[t:t + H].sum(axis=0)
        X_all.append(X_t)
        y_all.append(y_t)

    if len(X_all) < n_splits + 1:
        return {"best_alpha": alpha_grid[len(alpha_grid) // 2], "cv_scores": {}}

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = {}

    for alpha in alpha_grid:
        fold_ics = []
        for train_idx, val_idx in tscv.split(X_scaled):
            if len(train_idx) < 10 or len(val_idx) < 5:
                continue
            if model_type == "ridge":
                m = Ridge(alpha=alpha)
            elif model_type == "lasso":
                m = Lasso(alpha=alpha, max_iter=3000)
            else:
                m = ElasticNet(alpha=alpha, max_iter=3000)

            m.fit(X_scaled[train_idx], y[train_idx])
            y_pred = m.predict(X_scaled[val_idx])
            ic, _ = spearmanr(y_pred, y[val_idx])
            fold_ics.append(float(ic) if not math.isnan(ic) else 0.0)

        cv_scores[alpha] = float(np.mean(fold_ics)) if fold_ics else 0.0

    best_alpha = max(cv_scores, key=cv_scores.get)
    return {"best_alpha": best_alpha, "cv_scores": cv_scores}


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  Machine Learning for Return Prediction")
    print("  Ridge, Lasso, ElasticNet with Walk-Forward Validation")
    print("═" * 66)

    # Simulate cross-sectional returns with embedded predictable structure
    rng = np.random.default_rng(0)
    T, n_stocks = 756, 50   # 3 years daily, 50 stocks

    # Realistic returns: heavy idiosyncratic noise dwarfs signal
    # Signal-to-noise ratio calibrated to real markets (IC ≈ 0.05–0.12)
    market_return = rng.standard_normal(T) * 0.01

    # Small persistent cross-sectional signal (weak momentum/mean-reversion)
    factor_loadings = rng.standard_normal((n_stocks, 3)) * 0.3
    factor_returns = np.zeros((T, 3))
    factor_returns[:, 0] = rng.standard_normal(T) * 0.003   # momentum-like
    factor_returns[:, 1] = rng.standard_normal(T) * 0.002   # reversal-like  
    factor_returns[:, 2] = rng.standard_normal(T) * 0.001   # vol signal

    # Dominant idiosyncratic noise (realistic: ~80% of total variance)
    returns = (market_return[:, None]
               + factor_returns @ factor_loadings.T * 0.2   # weak signal
               + rng.standard_normal((T, n_stocks)) * 0.018)  # strong noise

    print(f"\n  Dataset: {T} days × {n_stocks} stocks")

    # ── Model comparison ──────────────────────────────────────────
    print(f"\n── Walk-Forward Model Comparison (α=0.001) ──")
    print(f"\n  {'Model':<14} {'IC Mean':>10} {'IC Std':>10} {'IR':>10} {'Hit Rate':>10} {'Alpha (bps/yr)':>16}")
    print("  " + "─" * 72)

    results = {}
    for mtype in ["ols", "ridge", "lasso", "elasticnet"]:
        predictor = MLReturnPredictor(returns, model_type=mtype, alpha=1e-3,
                                       train_window=252, prediction_horizon=21)
        try:
            res = predictor.run()
            results[mtype] = res
            print(f"  {mtype:<14} {res.ic_mean:>10.4f} {res.ic_std:>10.4f} "
                  f"{res.ir:>10.4f} {res.hit_rate:>10.2%} {res.alpha_bps_ann:>16.1f}")
        except Exception as e:
            print(f"  {mtype:<14}  Error: {e}")

    # ── Feature importances from Lasso ────────────────────────────
    if "lasso" in results and results["lasso"].feature_importances:
        print(f"\n── Lasso Feature Importances (|coefficient| averaged across periods) ──")
        fi = results["lasso"].feature_importances
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        for fname, imp in sorted_fi:
            bar = "█" * int(imp / max(fi.values()) * 20)
            print(f"  {fname:<20} {imp:>8.4f}  {bar}")

    # ── Regularisation path ───────────────────────────────────────
    print(f"\n── Lasso Regularisation Path (IC vs λ) ──")
    alphas_test = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    print(f"\n  {'λ':>10} {'IC Mean':>10} {'Hit Rate':>10} {'N nonzero β':>14}")
    print("  " + "─" * 50)
    for a in alphas_test:
        pred = MLReturnPredictor(returns, model_type="lasso", alpha=a,
                                  train_window=252, prediction_horizon=21)
        try:
            r = pred.run()
            # Estimate nonzero: check last fitted model's sparsity
            print(f"  {a:>10.0e} {r.ic_mean:>10.4f} {r.hit_rate:>10.2%}   {'(varies)':>14}")
        except Exception:
            pass
