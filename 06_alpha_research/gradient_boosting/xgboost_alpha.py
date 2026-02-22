"""
Gradient Boosting for Alpha Generation
========================================
XGBoost and LightGBM are the standard tools at every systematic fund for
return prediction. They dominate Kaggle, dominate production alpha models
at Citadel, Two Sigma, Man AHL, and Millennium.

Why gradient boosting beats linear models for alpha:
  - Captures non-linear interactions without manual feature engineering
  - Robust to outliers (tree splits are rank-based)
  - Handles mixed-scale features (no normalization needed)
  - Built-in feature importance (which signals matter most)
  - Regularization prevents overfitting (max_depth, min_child_weight, λ)

The boosting algorithm (Friedman 2001):
  
  Start with F₀(x) = mean(y)
  For m = 1 to M:
    1. Compute pseudo-residuals: rᵢ = −∂L(yᵢ, F(xᵢ))/∂F(xᵢ)
    2. Fit tree hₘ(x) to residuals rᵢ
    3. Update: Fₘ(x) = Fₘ₋₁(x) + η·hₘ(x)
  
  Final model: F(x) = Σₘ η·hₘ(x)

XGBoost improvements over gradient boosting:
  - Second-order Taylor expansion (Newton boosting)
  - L1/L2 regularization on leaf weights
  - Column subsampling (like random forest)
  - Sparsity-aware split finding
  - Parallel tree construction

LightGBM improvements:
  - Leaf-wise growth (vs level-wise) → faster, more accurate
  - Gradient-based One-Side Sampling (GOSS)
  - Exclusive Feature Bundling (EFB) for high-dimensional data
  - Categorical feature support (no need for one-hot encoding)

Feature engineering for return prediction:
  - Technical indicators: RSI, MACD, Bollinger bands
  - Momentum: past returns at various horizons
  - Volume features: VWAP deviation, volume surge
  - Sector/industry dummies
  - Macro factors: VIX, term spread, credit spread
  - Fundamental ratios (if available): P/E, P/B, ROE

Walk-forward validation (critical):
  Train on [t-252, t-21], predict t+1 to t+5
  Roll forward monthly, never use future data

This demo:
  Since XGBoost/LightGBM aren't available in this environment, we use
  sklearn's GradientBoostingRegressor which implements the same core algorithm.
  The concepts, feature engineering, and validation methodology are identical.

References:
  - Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
    Annals of Statistics 29(5).
  - Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
  - Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS.
  - Gu, S., Kelly, B., Xiu, D. (2020). Empirical Asset Pricing via Machine Learning.
    Review of Financial Studies 33(5). [Neural nets beat GBM, but GBM beats everything else]
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(prices: pd.DataFrame, volumes: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute technical and momentum features for each stock.
    
    Features:
      - Momentum: returns over 1d, 5d, 21d, 63d, 126d, 252d
      - RSI (14-day)
      - Price/SMA ratio (20d, 50d, 200d)
      - Volatility (21d realized vol)
      - Volume features (if provided)
    
    Returns DataFrame with columns = features, index = (date, ticker).
    """
    features_list = []
    
    for ticker in prices.columns:
        px = prices[ticker].dropna()
        df = pd.DataFrame(index=px.index)
        df['ticker'] = ticker
        
        # Returns at various horizons
        for h in [1, 5, 21, 63, 126, 252]:
            df[f'ret_{h}d'] = px.pct_change(h)
        
        # RSI (14-day)
        delta = px.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Price / SMA ratios
        for window in [20, 50, 200]:
            sma = px.rolling(window).mean()
            df[f'px_sma_{window}'] = px / sma - 1
        
        # Realized volatility (21-day)
        df['vol_21d'] = px.pct_change().rolling(21).std() * np.sqrt(252)
        
        # Volume features (if available)
        if volumes is not None and ticker in volumes.columns:
            vol = volumes[ticker]
            df['volume_surge'] = vol / vol.rolling(21).mean() - 1
            df['volume_trend'] = vol.pct_change(21)
        
        features_list.append(df)
    
    features = pd.concat(features_list).sort_index()
    return features


def winsorize(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize at percentiles to remove extreme outliers."""
    lower_bound = x.quantile(lower)
    upper_bound = x.quantile(upper)
    return x.clip(lower_bound, upper_bound)


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Walk-forward validation for time-series data.
    Train on expanding window, predict next period.
    """
    
    def __init__(
        self,
        train_days: int = 252,
        test_days: int = 21,
        gap_days: int = 1,  # gap to avoid look-ahead
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.gap_days = gap_days
    
    def split(self, dates: pd.DatetimeIndex):
        """Generate train/test indices for walk-forward validation."""
        dates = pd.Series(range(len(dates)), index=dates)
        splits = []
        
        start = self.train_days
        while start + self.gap_days + self.test_days <= len(dates):
            train_idx = dates.iloc[start - self.train_days:start].values
            test_idx = dates.iloc[start + self.gap_days:start + self.gap_days + self.test_days].values
            splits.append((train_idx, test_idx))
            start += self.test_days
        
        return splits


# ---------------------------------------------------------------------------
# Gradient boosting alpha model
# ---------------------------------------------------------------------------

class GradientBoostingAlpha:
    """
    Gradient boosting model for cross-sectional return prediction.
    
    Mimics XGBoost/LightGBM workflow using sklearn's GradientBoostingRegressor.
    In production: replace with xgboost.XGBRegressor or lightgbm.LGBMRegressor.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 50,
        subsample: float = 0.8,
        max_features: float = 0.8,
    ):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            max_features=max_features,
            random_state=42,
        )
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit gradient boosting model."""
        self.feature_names = X.columns.tolist()
        self.model.fit(X.values, y.values)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict returns."""
        return self.model.predict(X.values)
    
    def feature_importance(self) -> pd.DataFrame:
        """Feature importance from tree splits."""
        imp = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': imp,
        }).sort_values('importance', ascending=False)
        return df


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 68)
    print("  Gradient Boosting for Alpha Generation")
    print("  XGBoost/LightGBM methodology (using sklearn GradientBoosting)")
    print("═" * 68)
    
    # Simulate stock universe
    np.random.seed(42)
    T = 1260  # 5 years daily
    N = 50    # 50 stocks
    
    dates = pd.date_range('2019-01-01', periods=T, freq='D')
    tickers = [f'STOCK_{i:02d}' for i in range(N)]
    
    # Simulate prices with momentum and mean-reversion
    prices = pd.DataFrame(100 * np.exp(np.cumsum(np.random.randn(T, N) * 0.02, axis=0)),
                          index=dates, columns=tickers)
    volumes = pd.DataFrame(np.random.lognormal(15, 0.5, (T, N)),
                           index=dates, columns=tickers)
    
    print(f"\n  Simulated universe: {N} stocks, {T} days ({dates[0].date()} to {dates[-1].date()})")
    
    # Compute features
    print(f"\n── Feature Engineering ──")
    
    # Build features and target in a simpler way
    data_list = []
    for ticker in tickers:
        px = prices[ticker].dropna()
        
        # Features
        features = {}
        for h in [1, 5, 21, 63, 126, 252]:
            features[f'ret_{h}d'] = px.pct_change(h)
        
        # RSI
        delta = px.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Price/SMA
        for window in [20, 50, 200]:
            sma = px.rolling(window).mean()
            features[f'px_sma_{window}'] = px / sma - 1
        
        # Vol
        features['vol_21d'] = px.pct_change().rolling(21).std() * np.sqrt(252)
        
        # Volume
        vol = volumes[ticker]
        features['volume_surge'] = vol / vol.rolling(21).mean() - 1
        
        # Target: 5-day forward return
        features['target'] = px.pct_change(5).shift(-5)
        features['ticker'] = ticker
        
        df = pd.DataFrame(features, index=px.index)
        data_list.append(df)
    
    data = pd.concat(data_list).dropna()
    data = data.reset_index().rename(columns={'index': 'date'})
    
    feature_cols = [c for c in data.columns if c not in ['ticker', 'target', 'date']]
    print(f"  Features: {len(feature_cols)}")
    print(f"  Top 5: {feature_cols[:5]}")
    print(f"  Sample size: {len(data):,} (date, stock) pairs")
    
    # Walk-forward validation
    print(f"\n── Walk-Forward Cross-Validation ──")
    validator = WalkForwardValidator(train_days=252, test_days=21)
    dates_unique = data.index.get_level_values(0).unique()
    
    splits = validator.split(dates_unique)
    print(f"  Train window: 252 days")
    print(f"  Test window:  21 days")
    print(f"  Number of folds: {len(splits)}")
    
    # Train and evaluate
    results = []
    all_predictions = []
    all_actuals = []
    
    for fold, (train_idx, test_idx) in enumerate(splits[:5]):  # first 5 folds for speed
        train_mask = data.index.isin(train_idx)
        test_mask  = data.index.isin(test_idx)
        
        train_data = data[train_mask]
        test_data  = data[test_mask]
        
        X_train, y_train = train_data[feature_cols], train_data['target']
        X_test, y_test   = test_data[feature_cols], test_data['target']
        
        # Fit model
        model = GradientBoostingAlpha(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            min_samples_leaf=30, subsample=0.8
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        ic, _ = spearmanr(y_pred, y_test)
        
        results.append({
            'fold': fold + 1,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'mse': mse,
            'r2': r2,
            'ic': ic,
        })
        
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
    
    results_df = pd.DataFrame(results)
    
    print(f"\n  {'Fold':>6} {'N_train':>10} {'N_test':>8} {'IC':>8} {'R²':>8}")
    print("  " + "─" * 44)
    for _, row in results_df.head(5).iterrows():
        print(f"  {row['fold']:>6} {row['n_train']:>10.0f} "
              f"{row['n_test']:>8.0f} {row['ic']:>8.4f} {row['r2']:>8.4f}")
    print("  " + "─" * 44)
    print(f"  {'Mean':>6} {results_df['n_train'].mean():>10.0f} "
          f"{results_df['n_test'].mean():>8.0f} "
          f"{results_df['ic'].mean():>8.4f} {results_df['r2'].mean():>8.4f}")
    
    # Feature importance (from last fold)
    print(f"\n── Feature Importance (from final fold) ──")
    importance = model.feature_importance()
    print(f"\n  {'Feature':>20} {'Importance':>12}")
    print("  " + "─" * 34)
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row['importance'] * 200)
        print(f"  {row['feature']:>20} {row['importance']:>12.4f}  {bar}")
    
    # Information Coefficient analysis
    print(f"\n── Information Coefficient (IC) Distribution ──")
    ic_mean = results_df['ic'].mean()
    ic_std = results_df['ic'].std()
    ic_tstat = ic_mean / (ic_std / np.sqrt(len(results_df)))
    
    print(f"  Mean IC:     {ic_mean:.4f}")
    print(f"  Std IC:      {ic_std:.4f}")
    print(f"  t-statistic: {ic_tstat:.4f}  (target > 2.0 for significance)")
    print(f"  IC > 0:      {(results_df['ic'] > 0).sum()} / {len(results_df)} folds "
          f"({(results_df['ic'] > 0).mean():.1%})")
    
    # Quintile analysis
    print(f"\n── Quintile Analysis (all out-of-sample predictions) ──")
    df_pred = pd.DataFrame({
        'predicted': all_predictions,
        'actual': all_actuals,
    })
    df_pred['quintile'] = pd.qcut(df_pred['predicted'], 5, labels=False, duplicates='drop')
    
    quintile_stats = df_pred.groupby('quintile')['actual'].agg(['mean', 'std', 'count'])
    quintile_stats.index = quintile_stats.index + 1  # 1-indexed
    
    print(f"\n  {'Quintile':>10} {'Mean Ret':>12} {'Std':>10} {'Count':>8}")
    print("  " + "─" * 44)
    for q, row in quintile_stats.iterrows():
        print(f"  {q:>10} {row['mean']:>12.4%} {row['std']:>10.4%} {row['count']:>8.0f}")
    
    spread = quintile_stats.loc[5, 'mean'] - quintile_stats.loc[1, 'mean']
    tstat_spread = spread / (quintile_stats.loc[5, 'std'] / np.sqrt(quintile_stats.loc[5, 'count']))
    print(f"\n  Q5 - Q1 spread:  {spread:.4%}  (annualised: {spread * 252/5:.2%})")
    print(f"  t-statistic:     {tstat_spread:.4f}")
    
    print(f"""
── XGBoost/LightGBM in Production ──

  This demo uses sklearn's GradientBoostingRegressor for illustration.
  In production at systematic funds, use:

  1. XGBoost (xgboost.XGBRegressor):
     params = {{
         'max_depth': 3,
         'learning_rate': 0.05,
         'n_estimators': 200,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'reg_alpha': 0.1,     # L1 regularization
         'reg_lambda': 1.0,    # L2 regularization
         'min_child_weight': 10,
         'objective': 'reg:squarederror',
     }}

  2. LightGBM (lightgbm.LGBMRegressor):
     params = {{
         'max_depth': 3,
         'learning_rate': 0.05,
         'n_estimators': 200,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'reg_alpha': 0.1,
         'reg_lambda': 1.0,
         'min_child_samples': 50,
         'boosting_type': 'gbdt',
         'objective': 'regression',
     }}

  Hyperparameter tuning:
    - Use Bayesian optimization (optuna, hyperopt)
    - Optimize on IC, not MSE (IC is what matters for alpha)
    - Prevent overfitting: early stopping with validation set

  Why GBM beats linear models (Lasso/Ridge):
    - Captures non-linearities (momentum × volatility interaction)
    - Robust to outliers (tree splits are rank-based)
    - No manual interaction terms needed
    - Gu/Kelly/Xiu (2020): GBM R² = 0.38%, linear = 0.26%

  Interview question (Two Sigma, Citadel):
  Q: "Your XGBoost model has IC=0.05 in backtest, IC=0.01 live. Why?"
  A: "Overfitting. Solutions: (1) stronger regularization (max_depth=2),
      (2) larger min_child_samples, (3) early stopping on validation IC,
      (4) ensemble multiple models with different random seeds."
    """)
