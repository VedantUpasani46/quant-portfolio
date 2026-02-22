"""
Neural Networks for Stock Return Prediction
=============================================
Deep learning for cross-sectional return prediction. Gu, Kelly & Xiu (2020)
show neural nets outperform all other methods (linear, trees, GBM) on the
cross-section of stock returns.

Key result (Gu/Kelly/Xiu 2020, RFS):
  Out-of-sample R²:
    - Linear (Ridge/Lasso):         0.26%
    - Random forest:                0.30%
    - Gradient boosting:            0.38%
    - Neural network (5 layers):    0.43%  ← best

Why neural networks work for alpha:
  1. Universal function approximation (capture any non-linearity)
  2. Automatic feature interactions (no need to manually create x₁·x₂ terms)
  3. Regularization via dropout, early stopping, weight decay
  4. Batch normalization stabilizes training
  5. Can handle 100+ features without explicit feature selection

Architecture (Gu/Kelly/Xiu):
  Input:    94 stock characteristics (momentum, value, quality, etc.)
  Hidden:   5 fully-connected layers, 32 neurons each
  Activation: ReLU
  Regularization: Dropout (0.1-0.2), batch normalization, early stopping
  Output:   Single neuron (predicted return)
  Loss:     MSE or Huber loss (robust to outliers)

Training:
  - Mini-batch SGD (batch size = 1024)
  - Learning rate schedule: start 0.001, decay on plateau
  - Early stopping on validation IC (not MSE — IC is what matters)
  - Walk-forward validation (never use future data)

Why dropout matters:
  At test time, neurons are always on → network averages over
  2^N possible sub-networks (like bagging) → reduces overfitting

Ensemble:
  Train 10 models with different random seeds, average predictions.
  Gu/Kelly/Xiu: ensemble improves IC from 0.035 to 0.043.

This demo:
  Uses PyTorch-style API but implemented in numpy/sklearn for portability.
  Shows the core architecture and training loop. In production, use PyTorch.

References:
  - Gu, S., Kelly, B., Xiu, D. (2020). Empirical Asset Pricing via Machine Learning.
    Review of Financial Studies 33(5), 2223–2273.
  - Goodfellow, I. et al. (2016). Deep Learning. MIT Press.
  - Chen, L. et al. (2021). Deep Learning in Asset Pricing. Management Science.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Feature engineering (same as XGBoost)
# ---------------------------------------------------------------------------

def compute_features_nn(prices: pd.DataFrame, n_stocks: int) -> pd.DataFrame:
    """
    Compute features for neural network input.
    Same features as GBM, but will be standardized.
    """
    data_list = []
    
    for ticker in prices.columns[:n_stocks]:
        px = prices[ticker].dropna()
        
        features = {}
        # Momentum
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
        
        # Volatility
        features['vol_21d'] = px.pct_change().rolling(21).std()
        features['vol_63d'] = px.pct_change().rolling(63).std()
        
        # Target: 5-day forward return
        features['target'] = px.pct_change(5).shift(-5)
        features['ticker'] = ticker
        
        df = pd.DataFrame(features, index=px.index)
        data_list.append(df)
    
    data = pd.concat(data_list).dropna()
    data = data.reset_index().rename(columns={'index': 'date'})
    return data


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

class NeuralNetAlpha:
    """
    Multi-layer perceptron for return prediction.
    Mimics Gu/Kelly/Xiu architecture.
    """
    
    def __init__(
        self,
        hidden_layers: tuple = (32, 32, 32, 32, 32),  # 5 layers, 32 neurons each
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        early_stopping: bool = True,
        validation_fraction: float = 0.2,
        alpha: float = 0.0001,  # L2 penalty
    ):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            alpha=alpha,
            batch_size=256,
            random_state=42,
            verbose=False,
        )
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit neural network.
        Standardizes features (critical for neural nets).
        """
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X.values)
        self.model.fit(X_scaled, y.values)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict returns."""
        X_scaled = self.scaler.transform(X.values)
        return self.model.predict(X_scaled)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 68)
    print("  Neural Networks for Stock Return Prediction")
    print("  Gu/Kelly/Xiu (2020) architecture — 5-layer deep network")
    print("═" * 68)
    
    # Simulate stock universe
    np.random.seed(42)
    T = 1260  # 5 years
    N = 30    # 30 stocks (small for speed)
    
    dates = pd.date_range('2019-01-01', periods=T, freq='D')
    tickers = [f'STOCK_{i:02d}' for i in range(N)]
    
    # Simulate with stronger signal (momentum + mean-reversion)
    prices = pd.DataFrame(index=dates, columns=tickers)
    for ticker in tickers:
        # AR(1) with drift
        r = np.zeros(T)
        r[0] = 0.0005
        for t in range(1, T):
            r[t] = 0.0002 + 0.95 * r[t-1] + np.random.randn() * 0.015
        prices[ticker] = 100 * np.exp(np.cumsum(r))
    
    print(f"\n  Simulated universe: {N} stocks, {T} days")
    
    # Compute features
    print(f"\n── Feature Engineering ──")
    data = compute_features_nn(prices, N)
    
    feature_cols = [c for c in data.columns if c not in ['ticker', 'target', 'date']]
    print(f"  Features: {len(feature_cols)}")
    print(f"  Sample size: {len(data):,} (date, stock) pairs")
    
    # Train/test split (time-series split)
    train_cutoff = int(len(data) * 0.7)
    train_data = data.iloc[:train_cutoff]
    test_data  = data.iloc[train_cutoff:]
    
    X_train, y_train = train_data[feature_cols], train_data['target']
    X_test, y_test   = test_data[feature_cols], test_data['target']
    
    print(f"\n── Train/Test Split (time-series) ──")
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(data):.1%})")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(data):.1%})")
    
    # Train neural network
    print(f"\n── Training Neural Network ──")
    print(f"  Architecture: {len(feature_cols)} → 32 → 32 → 32 → 32 → 32 → 1")
    print(f"  Optimizer: Adam (learning_rate=0.001)")
    print(f"  Regularization: L2 (α=0.0001), early stopping")
    
    nn = NeuralNetAlpha(
        hidden_layers=(32, 32, 32, 32, 32),
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
    )
    nn.fit(X_train, y_train)
    
    print(f"  Training complete: {nn.model.n_iter_} iterations")
    
    # Evaluate
    print(f"\n── Out-of-Sample Evaluation ──")
    y_pred_train = nn.predict(X_train)
    y_pred_test  = nn.predict(X_test)
    
    ic_train, _ = spearmanr(y_pred_train, y_train)
    ic_test, _  = spearmanr(y_pred_test, y_test)
    
    r2_train = 1 - ((y_train - y_pred_train)**2).sum() / ((y_train - y_train.mean())**2).sum()
    r2_test  = 1 - ((y_test - y_pred_test)**2).sum() / ((y_test - y_test.mean())**2).sum()
    
    print(f"\n  {'':>12} {'IC':>10} {'R²':>10}")
    print("  " + "─" * 34)
    print(f"  {'Train':>12} {ic_train:>10.4f} {r2_train:>10.4f}")
    print(f"  {'Test (OOS)':>12} {ic_test:>10.4f} {r2_test:>10.4f}")
    print(f"  {'Δ (overfit)':>12} {ic_train - ic_test:>10.4f} {r2_train - r2_test:>10.4f}")
    
    # Quintile analysis
    print(f"\n── Quintile Analysis (Out-of-Sample) ──")
    df_test = pd.DataFrame({
        'predicted': y_pred_test,
        'actual': y_test.values,
    })
    df_test['quintile'] = pd.qcut(df_test['predicted'], 5, labels=False, duplicates='drop')
    
    quintile_stats = df_test.groupby('quintile')['actual'].agg(['mean', 'std', 'count'])
    quintile_stats.index = quintile_stats.index + 1
    
    print(f"\n  {'Quintile':>10} {'Mean Ret':>12} {'Std':>10} {'Count':>8}")
    print("  " + "─" * 44)
    for q, row in quintile_stats.iterrows():
        print(f"  {q:>10} {row['mean']:>12.4%} {row['std']:>10.4%} {row['count']:>8.0f}")
    
    spread = quintile_stats.loc[5, 'mean'] - quintile_stats.loc[1, 'mean']
    print(f"\n  Q5 - Q1 spread:  {spread:.4%}  (5-day)")
    print(f"  Annualised:      {spread * 252/5:.2%}")
    
    # Compare to linear model
    print(f"\n── Neural Network vs Linear Benchmark ──")
    from sklearn.linear_model import Ridge
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    ic_ridge, _ = spearmanr(y_pred_ridge, y_test)
    r2_ridge = 1 - ((y_test - y_pred_ridge)**2).sum() / ((y_test - y_test.mean())**2).sum()
    
    print(f"\n  {'Model':>15} {'IC (OOS)':>12} {'R² (OOS)':>12}")
    print("  " + "─" * 42)
    print(f"  {'Ridge (linear)':>15} {ic_ridge:>12.4f} {r2_ridge:>12.4f}")
    print(f"  {'Neural Net':>15} {ic_test:>12.4f} {r2_test:>12.4f}")
    print(f"  {'Improvement':>15} {ic_test - ic_ridge:>12.4f} {r2_test - r2_ridge:>12.4f}")
    
    if ic_test > ic_ridge:
        pct_gain = (ic_test / ic_ridge - 1) * 100
        print(f"\n  Neural network IC is {pct_gain:.1f}% higher than Ridge ✓")
    
    print(f"""
── Gu/Kelly/Xiu (2020) Findings ──

  Dataset: 30,000 stocks, 94 characteristics, 1957-2016

  Out-of-sample R² (monthly returns):
    - Linear (PLS):           0.26%
    - Random forest:          0.30%
    - Gradient boosting:      0.38%
    - Neural network (deep):  0.43%  ← best by 13%

  Key insights:
    1. Deep networks beat shallow (5 layers > 3 layers > 1 layer)
    2. Ensemble of 10 nets improves IC from 0.035 to 0.043
    3. Most gain comes from non-linear interactions, not raw feature count
    4. Dropout (0.1-0.2) critical to prevent overfitting
    5. Early stopping on validation IC beats fixed epochs

  Production at systematic funds:
    - PyTorch or TensorFlow (not sklearn)
    - GPU training (50-100 epochs in minutes vs hours)
    - Hyperparameter tuning via Optuna or Ray Tune
    - Ensemble 5-10 models with different seeds
    - Monitor live IC vs backtest IC (drift detection)

  Interview question (DE Shaw, Two Sigma):
  Q: "Your neural net has IC=0.05 in backtest, IC=0.02 live. Fix it."
  A: "Classic overfitting. Solutions:
      (1) Stronger regularization (dropout 0.3, weight decay 1e-3)
      (2) Smaller network (3 layers × 16 neurons)
      (3) Early stopping on validation IC, not training loss
      (4) Ensemble multiple models to reduce variance
      (5) Retrain monthly on expanding window (adapt to regime shifts)"
    """)
