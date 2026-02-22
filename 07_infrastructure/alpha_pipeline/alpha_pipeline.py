"""
Alpha Research Pipeline
=========================
End-to-end infrastructure for taking a raw signal from idea to
production-grade alpha factor. Used at systematic hedge funds
(Two Sigma, DE Shaw, Renaissance, AQR, Man AHL) to evaluate
whether a new signal is genuinely predictive before deploying capital.

Pipeline stages:
  1. Signal generation    — raw factor computation
  2. Signal processing    — rank normalisation, z-scoring, winsorisation
  3. Decay analysis       — how fast does predictive power decay?
  4. IC time series       — Information Coefficient per period
  5. Turnover analysis    — how much does the signal change period-to-period?
  6. Capacity estimation  — maximum AUM before signal degrades
  7. Portfolio construction — translate signal into weights
  8. Production checklist  — go/no-go criteria

The seven deadly sins of alpha research (Lopez de Prado 2018):
  1. Survivorship bias     — only backtesting on stocks that survived
  2. Lookahead bias        — using future information in past predictions
  3. Narrative fallacy     — finding a reason after the signal works
  4. Multiple testing      — testing 100 signals, finding 5 that work by chance
  5. Transaction costs     — ignoring the cost of trading the signal
  6. Overfitting           — complex model that fits history but not future
  7. Small sample          — claiming significance on 20 observations

The Bonferroni correction for multiple testing:
  If you test K signals at significance α, you expect α·K false positives.
  Adjusted threshold: α* = α / K   (very conservative)
  Or use Benjamini-Hochberg FDR control (less conservative).

Information Decay:
  IC(h) = Spearman(signal_t, return_{t+h})
  A good signal has IC(1) > 0 and IC decaying to 0 by horizon h*.
  If IC stays high → signal is slow-moving (low turnover, easier to trade).
  If IC falls fast → signal requires high turnover (high cost).

References:
  - Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
  - Grinold, R.C. & Kahn, R.N. (2000). Active Portfolio Management, 2nd ed. McGraw-Hill.
  - Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via ML. RFS 33(5).
  - Harvey, C.R. & Liu, Y. (2015). Backtesting. Journal of Portfolio Management.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, rankdata


# ---------------------------------------------------------------------------
# Signal processing utilities
# ---------------------------------------------------------------------------

def cross_sectional_rank(x: np.ndarray) -> np.ndarray:
    """
    Rank normalise to [-0.5, +0.5] cross-sectionally.
    Rank i → (rank_i - 1) / (N - 1) - 0.5
    Used to make signal scale-invariant.
    """
    r = rankdata(x, method="average")
    N = len(x)
    if N <= 1:
        return np.zeros_like(x)
    return (r - 1) / (N - 1) - 0.5


def cross_sectional_zscore(x: np.ndarray) -> np.ndarray:
    """Standardise cross-sectionally: (x - mean) / std."""
    std = x.std()
    if std < 1e-10:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def winsorise(x: np.ndarray, pct: float = 0.01) -> np.ndarray:
    """Winsorise at pct and (1-pct) quantiles to remove outliers."""
    lo, hi = np.percentile(x, pct * 100), np.percentile(x, (1 - pct) * 100)
    return np.clip(x, lo, hi)


def neutralise(signal: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """
    Remove the group mean from the signal (sector/country neutralisation).
    After neutralisation, within each group the signal sums to zero.
    """
    out = signal.copy()
    for g in np.unique(groups):
        mask = groups == g
        out[mask] -= out[mask].mean()
    return out


def process_signal(
    raw: np.ndarray,
    winsorise_pct: float = 0.02,
    groups: Optional[np.ndarray] = None,
    method: str = "zscore",   # "zscore" or "rank"
) -> np.ndarray:
    """
    Standard signal processing pipeline:
      1. Winsorise (remove extreme outliers)
      2. Sector/group neutralise (optional)
      3. Cross-sectional standardise (rank or z-score)
    """
    x = winsorise(raw, winsorise_pct)
    if groups is not None:
        x = neutralise(x, groups)
    if method == "rank":
        return cross_sectional_rank(x)
    return cross_sectional_zscore(x)


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def information_coefficient(signal: np.ndarray, returns: np.ndarray) -> float:
    """
    Spearman rank IC between signal and realised returns.
    Spearman is preferred over Pearson: robust to outliers,
    measures monotonic (not just linear) association.
    Returns float in [-1, +1].
    """
    if len(signal) < 5 or np.std(signal) < 1e-10 or np.std(returns) < 1e-10:
        return 0.0
    ic, _ = spearmanr(signal, returns)
    return float(ic) if not math.isnan(ic) else 0.0


def ic_t_statistic(ic_series: np.ndarray) -> dict:
    """
    Test the null hypothesis H₀: E[IC] = 0.
    t = IC_mean / (IC_std / √T)
    """
    n = len(ic_series)
    mean = ic_series.mean()
    std = ic_series.std(ddof=1)
    if std < 1e-10:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False}
    t_stat = mean / (std / math.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "significant_bonferroni_100": p_value < 0.05 / 100,  # 100-signal multiple testing
    }


# ---------------------------------------------------------------------------
# Information decay analysis
# ---------------------------------------------------------------------------

def ic_decay_profile(
    signals: np.ndarray,   # (T, N) signal matrix
    returns: np.ndarray,   # (T, N) return matrix
    max_horizon: int = 20,
    horizon_step: int = 1,
) -> pd.DataFrame:
    """
    Compute IC at horizons h = 1, 2, ..., max_horizon.
    Shows how predictive power decays over time.

    IC(h) = Spearman(signal_t, return_{t:t+h})

    Returns DataFrame with columns: horizon, ic_mean, ic_std, ir, t_stat, n_obs.
    """
    T, N = signals.shape
    records = []

    for h in range(1, max_horizon + 1, horizon_step):
        ic_vals = []
        for t in range(T - h):
            sig_t = signals[t]
            # Forward h-period return
            ret_th = returns[t:t+h].sum(axis=0)
            ic = information_coefficient(sig_t, ret_th)
            ic_vals.append(ic)

        ic_arr = np.array(ic_vals)
        n = len(ic_arr)
        mean_ic = ic_arr.mean()
        std_ic = ic_arr.std(ddof=1)
        ir = mean_ic / std_ic if std_ic > 0 else 0.0
        t_stat = mean_ic / (std_ic / math.sqrt(n)) if std_ic > 0 else 0.0

        records.append({
            "horizon":  h,
            "ic_mean":  mean_ic,
            "ic_std":   std_ic,
            "ir":       ir,
            "t_stat":   t_stat,
            "n_obs":    n,
            "pct_positive": (ic_arr > 0).mean(),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Turnover analysis
# ---------------------------------------------------------------------------

def signal_autocorrelation(
    signals: np.ndarray,   # (T, N)
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    Cross-sectional autocorrelation of the signal at lags 1..max_lag.
    AC(lag) = E_t[Corr(signal_t, signal_{t-lag})]

    High AC → slow-moving signal → low turnover → cheap to trade.
    Low AC  → fast signal → high turnover → requires tight bid-ask.
    """
    T, N = signals.shape
    records = []
    for lag in range(1, max_lag + 1):
        acs = []
        for t in range(lag, T):
            r = spearmanr(signals[t], signals[t - lag])[0]
            if not math.isnan(r):
                acs.append(r)
        mean_ac = np.mean(acs) if acs else 0.0
        records.append({"lag": lag, "autocorrelation": mean_ac})
    return pd.DataFrame(records)


def expected_turnover(signal_autocorr: float) -> float:
    """
    Approximate one-way turnover given signal autocorrelation.
    Turnover ≈ 1 - AC  (higher AC → lower turnover)
    This is an approximation; actual turnover depends on weight scaling.
    """
    return max(0.0, 1 - signal_autocorr)


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------

def benjamini_hochberg_correction(
    p_values: np.ndarray,
    fdr: float = 0.05,
) -> dict:
    """
    Benjamini-Hochberg FDR correction for multiple hypothesis testing.
    Less conservative than Bonferroni while controlling expected false discovery rate.

    Algorithm:
      1. Sort p-values p_(1) ≤ ... ≤ p_(m)
      2. Find largest k such that p_(k) ≤ k/m · FDR
      3. Reject all H₀_(i) for i ≤ k

    Returns: dict with sorted p-values, BH threshold, and which are significant.
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    bh_threshold = np.arange(1, m + 1) / m * fdr
    significant_mask = sorted_p <= bh_threshold
    # Find the last True (all signals with rank ≤ last_k are significant)
    if significant_mask.any():
        last_k = np.where(significant_mask)[0][-1]
        is_significant = np.zeros(m, dtype=bool)
        is_significant[:last_k + 1] = True
    else:
        is_significant = np.zeros(m, dtype=bool)

    # Map back to original order
    result_sig = np.zeros(m, dtype=bool)
    result_sig[sorted_idx] = is_significant

    return {
        "n_signals": m,
        "n_significant_bh": is_significant.sum(),
        "n_significant_bonferroni": (p_values < fdr / m).sum(),
        "bh_significant_mask": result_sig,
        "most_significant_p": float(sorted_p[0]),
    }


# ---------------------------------------------------------------------------
# Capacity estimation
# ---------------------------------------------------------------------------

def capacity_estimate(
    ic_mean: float,
    ic_decay_days: int,
    universe_size: int,
    avg_adv_usd: float,
    participation_rate: float = 0.05,
    target_ir: float = 0.5,
) -> dict:
    """
    Estimate the maximum AUM a signal can support before it degrades.

    Approach (simplified Grinold-Kahn):
      ICIR = IC_mean × √(universe_size) / IC_std
      Capacity ≈ min(liquidity-based, signal-degradation-based)

    Liquidity constraint: max trade ≤ participation_rate × ADV
      Capacity_liquidity = participation_rate × avg_ADV × N_stocks × horizon

    Signal constraint: larger AUM → larger trades → price impact erodes IC
      Conservative: capacity where marginal cost ≈ IC × price
    """
    # Liquidity-based capacity
    cap_liquidity = participation_rate * avg_adv_usd * universe_size * ic_decay_days

    # Rough alpha per unit (IC × vol proxy)
    annual_alpha_bps = ic_mean * 252 / ic_decay_days * 10000 * 0.15  # rough

    return {
        "liquidity_capacity_usd": cap_liquidity,
        "estimated_annual_alpha_bps": annual_alpha_bps,
        "ic_mean": ic_mean,
        "ic_decay_days": ic_decay_days,
        "universe_size": universe_size,
        "note": "Liquidity capacity is an upper bound; factor in impact costs.",
    }


# ---------------------------------------------------------------------------
# Production readiness checklist
# ---------------------------------------------------------------------------

@dataclass
class AlphaSignal:
    """
    Container for a fully evaluated alpha signal, ready for review.
    """
    name: str
    ic_series: np.ndarray
    decay_df: pd.DataFrame
    autocorr_df: pd.DataFrame
    universe_size: int
    avg_adv_usd: float

    @property
    def ic_mean(self) -> float:
        return float(self.ic_series.mean())

    @property
    def ic_std(self) -> float:
        return float(self.ic_series.std(ddof=1))

    @property
    def ir(self) -> float:
        return self.ic_mean / self.ic_std if self.ic_std > 1e-10 else 0.0

    @property
    def hit_rate(self) -> float:
        return float((self.ic_series > 0).mean())

    @property
    def halflife(self) -> Optional[float]:
        """Decay half-life: horizon where IC drops to 50% of IC(1)."""
        if self.decay_df.empty:
            return None
        ic1 = self.decay_df.iloc[0]["ic_mean"]
        if abs(ic1) < 1e-6:
            return None
        target = ic1 / 2
        crossing = self.decay_df[self.decay_df["ic_mean"] <= target]
        return float(crossing.iloc[0]["horizon"]) if not crossing.empty else None

    @property
    def signal_autocorrelation_1d(self) -> float:
        if self.autocorr_df.empty:
            return 0.0
        return float(self.autocorr_df.iloc[0]["autocorrelation"])

    def production_checklist(self) -> dict:
        """
        Go/no-go criteria for deploying a signal to production.
        Based on Lopez de Prado (2018) and industry practice.
        """
        tstat = ic_t_statistic(self.ic_series)
        hl = self.halflife
        ac = self.signal_autocorrelation_1d
        turnover = expected_turnover(ac)
        cap = capacity_estimate(
            self.ic_mean, hl or 5, self.universe_size, self.avg_adv_usd
        )

        checks = {
            "IC > 0.02":              (self.ic_mean > 0.02, self.ic_mean),
            "IR > 0.3":               (self.ir > 0.3, self.ir),
            "Hit rate > 52%":         (self.hit_rate > 0.52, self.hit_rate),
            "t-stat > 2.0":           (abs(tstat["t_stat"]) > 2.0, tstat["t_stat"]),
            "t-stat significant":     (tstat["significant"], tstat["p_value"]),
            "Signal has decay":       (hl is not None and hl < 20, hl),
            "Turnover < 80%":         (turnover < 0.80, turnover),
            "Capacity > $100M":       (cap["liquidity_capacity_usd"] > 1e8,
                                       cap["liquidity_capacity_usd"] / 1e6),
        }

        n_pass = sum(1 for v, _ in checks.values() if v)
        return {
            "signal_name": self.name,
            "checks": checks,
            "n_pass": n_pass,
            "n_total": len(checks),
            "go_decision": n_pass >= 6,
            "recommendation": "GO" if n_pass >= 6 else "NO-GO" if n_pass < 4 else "REVIEW",
        }

    def summary(self) -> str:
        checklist = self.production_checklist()
        lines = [
            f"\n{'═'*60}",
            f"  Alpha Signal: {self.name}",
            f"{'═'*60}",
            f"  IC Mean:     {self.ic_mean:>10.4f}",
            f"  IC Std:      {self.ic_std:>10.4f}",
            f"  IR:          {self.ir:>10.4f}",
            f"  Hit Rate:    {self.hit_rate:>10.2%}",
            f"  Half-life:   {self.halflife or 'N/A':>10} days",
            f"  1d AutoCorr: {self.signal_autocorrelation_1d:>10.4f}",
            f"\n── Production Checklist ──",
        ]
        for check_name, (passed, value) in checklist["checks"].items():
            status = "✓ PASS" if passed else "✗ FAIL"
            if isinstance(value, float):
                val_str = f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}"
            else:
                val_str = str(value)
            lines.append(f"  {status}  {check_name:<30} ({val_str})")
        lines.append(f"\n  Result: {checklist['recommendation']} "
                     f"({checklist['n_pass']}/{checklist['n_total']} checks passed)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 60)
    print("  Alpha Research Pipeline")
    print("  Signal evaluation: decay, IC, turnover, production checklist")
    print("═" * 60)

    rng = np.random.default_rng(42)
    T, N = 504, 100   # 2 years, 100 stocks

    # Simulate a persistent slow-decaying signal (e.g. value factor)
    # True signal: AR(0.95) cross-sectional scores
    raw_signal = np.zeros((T, N))
    raw_signal[0] = rng.standard_normal(N)
    for t in range(1, T):
        raw_signal[t] = 0.90 * raw_signal[t-1] + rng.standard_normal(N) * 0.4

    # Returns: weakly predictable from signal with noise
    IC_true = 0.055
    returns = np.zeros((T, N))
    for t in range(T - 1):
        signal_t = cross_sectional_zscore(raw_signal[t])
        noise = rng.standard_normal(N) * 0.015
        returns[t] = IC_true * signal_t * 0.010 + noise

    # ── Signal processing ─────────────────────────────────────────
    print(f"\n── Signal Processing ──")
    processed = np.array([process_signal(raw_signal[t], method="zscore")
                           for t in range(T)])
    print(f"  Raw signal stats:       mean={raw_signal.mean():.2f}, std={raw_signal.std():.2f}")
    print(f"  Processed signal stats: mean={processed.mean():.4f}, std={processed.std():.4f}")

    # ── IC time series ────────────────────────────────────────────
    print(f"\n── IC Time Series (1-period ahead) ──")
    ic_vals = [information_coefficient(processed[t], returns[t+1])
               for t in range(T - 1)]
    ic_arr = np.array(ic_vals)
    tstat = ic_t_statistic(ic_arr)
    print(f"  IC mean:   {ic_arr.mean():.4f}  (target > 0.02)")
    print(f"  IC std:    {ic_arr.std():.4f}")
    print(f"  IR:        {ic_arr.mean()/ic_arr.std():.4f}  (target > 0.3)")
    print(f"  Hit rate:  {(ic_arr > 0).mean():.2%}  (target > 52%)")
    print(f"  t-stat:    {tstat['t_stat']:.4f}  (p={tstat['p_value']:.4f})")

    # ── IC decay ──────────────────────────────────────────────────
    print(f"\n── IC Decay Profile (slow decay → higher half-life) ──")
    decay = ic_decay_profile(processed, returns, max_horizon=15, horizon_step=1)
    print(f"\n  {'Horizon':>8} {'IC Mean':>10} {'IR':>10} {'t-stat':>10} {'% Pos':>8}")
    print("  " + "─" * 50)
    for _, row in decay.iterrows():
        print(f"  {row['horizon']:>8.0f}d {row['ic_mean']:>10.4f} "
              f"{row['ir']:>10.4f} {row['t_stat']:>10.4f} "
              f"{row['pct_positive']:>8.2%}")

    # ── Signal autocorrelation ────────────────────────────────────
    print(f"\n── Signal Autocorrelation (persistence = low turnover) ──")
    ac_df = signal_autocorrelation(processed, max_lag=5)
    print(f"\n  {'Lag':>6} {'Autocorr':>12} {'Implied Turnover':>18}")
    print("  " + "─" * 40)
    for _, row in ac_df.iterrows():
        to = expected_turnover(row["autocorrelation"])
        print(f"  {row['lag']:>6.0f}d {row['autocorrelation']:>12.4f} {to:>18.2%}")

    # ── Multiple testing ──────────────────────────────────────────
    print(f"\n── Multiple Testing Correction (simulating 20 signals) ──")
    # Simulate 20 signals: 3 real, 17 noise
    p_vals_real  = np.array([0.001, 0.008, 0.023])
    p_vals_noise = rng.uniform(0.05, 0.95, 17)
    all_p = np.concatenate([p_vals_real, p_vals_noise])
    bh = benjamini_hochberg_correction(all_p, fdr=0.05)
    print(f"  20 signals tested at FDR=5%:")
    print(f"  BH significant:         {bh['n_significant_bh']} signals")
    print(f"  Bonferroni significant: {bh['n_significant_bonferroni']} signals")
    print(f"  Naïve significant (p<0.05): {(all_p < 0.05).sum()} signals")
    print(f"  (3 true positives; naive inflates to {(all_p < 0.05).sum()})")

    # ── Production checklist ──────────────────────────────────────
    signal = AlphaSignal(
        name="Value/Reversal (AR0.9)",
        ic_series=ic_arr,
        decay_df=decay,
        autocorr_df=ac_df,
        universe_size=N,
        avg_adv_usd=50e6,
    )
    print(signal.summary())
