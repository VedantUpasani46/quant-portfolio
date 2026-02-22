"""
Market Regime Detection via Hidden Markov Models
===================================================
A Hidden Markov Model (HMM) for identifying latent market regimes
(bull, bear, crisis) from observed return sequences.

The HMM framework:
  - Hidden states S_t ∈ {1, ..., K}: regimes (bull, bear, volatile)
  - Observable emissions X_t | S_t = k ~ N(μ_k, σ_k²)
  - Transition matrix A: A_{ij} = P(S_t = j | S_{t-1} = i)
  - Initial distribution π: π_k = P(S_0 = k)

The Expectation-Maximisation (Baum-Welch) algorithm:
  E-step: compute P(S_t = k | X_{1:T}) via forward-backward algorithm
  M-step: update (μ_k, σ_k, A, π) to maximise expected log-likelihood

Applications:
  - Regime-conditional VaR and portfolio allocation
  - Dynamic risk budgeting (reduce equity exposure in bear regime)
  - Macro strategy timing (overweight bonds in flight-to-quality regime)
  - Credit risk: cycle-dependent default probabilities
  - Central bank macro surveillance (detect turning points)

The three-regime model for equities:
  Regime 1 (Bull):    High mean return, low volatility
  Regime 2 (Bear):    Low/negative mean return, moderate volatility
  Regime 3 (Crisis):  Very negative mean return, very high volatility

Regime persistence:
  The diagonal of A tells you how sticky each regime is.
  A_{kk} = 0.98 → average duration = 1/(1-0.98) = 50 periods.
  Real data: bull regimes last ~18-24 months, crises ~3-6 months.

Viterbi algorithm:
  Finds the most likely sequence of hidden states:
  S*_{1:T} = argmax P(S_{1:T} | X_{1:T})

References:
  - Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary
    Time Series. Econometrica 57(2), 357–384.
  - Baum, L. et al. (1970). A Maximization Technique for HMM. Ann. Math. Stat.
  - Ang, A. & Bekaert, G. (2002). International Asset Allocation with Regime Shifts.
    Review of Financial Studies 15(4), 1137–1187.
  - Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Ch. 17.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Gaussian HMM implementation from scratch
# ---------------------------------------------------------------------------

@dataclass
class HMMParams:
    """
    Parameters of a Gaussian Hidden Markov Model.

    Attributes
    ----------
    mu    : (K,) mean of emissions in each state
    sigma : (K,) std of emissions in each state
    A     : (K, K) transition matrix; A[i,j] = P(next=j | current=i)
    pi    : (K,) initial state probabilities
    K     : int   number of states
    """
    mu: np.ndarray
    sigma: np.ndarray
    A: np.ndarray
    pi: np.ndarray

    @property
    def K(self) -> int:
        return len(self.mu)

    def emission_log_prob(self, x: float) -> np.ndarray:
        """Log P(X_t = x | S_t = k) for all k. Shape (K,)."""
        return norm.logpdf(x, loc=self.mu, scale=np.maximum(self.sigma, 1e-6))

    def emission_log_probs(self, X: np.ndarray) -> np.ndarray:
        """Log emission probabilities for entire sequence. Shape (T, K)."""
        return np.column_stack([
            norm.logpdf(X, loc=self.mu[k], scale=max(self.sigma[k], 1e-6))
            for k in range(self.K)
        ])


class GaussianHMM:
    """
    Gaussian Hidden Markov Model with Baum-Welch EM estimation.

    Fitted via maximum likelihood using the Baum-Welch algorithm:
    an EM algorithm tailored to sequential latent variable models.

    Usage
    -----
    >>> hmm = GaussianHMM(n_states=3, n_iter=200)
    >>> hmm.fit(returns)
    >>> regimes = hmm.predict(returns)  # most likely state per period
    >>> probs = hmm.predict_proba(returns)  # posterior state probabilities
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200,
                 tol: float = 1e-6, n_init: int = 10, seed: int = 42):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.seed = seed
        self.params_: Optional[HMMParams] = None
        self.log_likelihood_: float = -np.inf

    # ── Forward algorithm ──────────────────────────────────────────

    def _forward(self, log_emission: np.ndarray,
                 params: HMMParams) -> tuple[np.ndarray, float]:
        """
        Forward algorithm: compute α_t(k) = P(X_{1:t}, S_t=k).
        Uses log-sum-exp for numerical stability.

        Returns (log_alpha: shape (T, K), log_likelihood: float).
        """
        T, K = log_emission.shape
        log_A = np.log(np.maximum(params.A, 1e-300))
        log_pi = np.log(np.maximum(params.pi, 1e-300))

        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_emission[0]

        for t in range(1, T):
            # log α_t(j) = log P(x_t | s_t=j) + log Σ_i α_{t-1}(i)·A_{ij}
            for j in range(K):
                log_alpha[t, j] = log_emission[t, j] + _logsumexp(log_alpha[t-1] + log_A[:, j])

        log_lik = _logsumexp(log_alpha[-1])
        return log_alpha, log_lik

    # ── Backward algorithm ────────────────────────────────────────

    def _backward(self, log_emission: np.ndarray,
                  params: HMMParams) -> np.ndarray:
        """
        Backward algorithm: compute β_t(k) = P(X_{t+1:T} | S_t=k).
        Returns log_beta: shape (T, K).
        """
        T, K = log_emission.shape
        log_A = np.log(np.maximum(params.A, 1e-300))
        log_beta = np.zeros((T, K))
        # Initialise: β_T(k) = 1 → log_beta[T-1] = 0

        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[t, i] = _logsumexp(
                    log_A[i] + log_emission[t+1] + log_beta[t+1]
                )
        return log_beta

    # ── E-step: compute responsibilities ──────────────────────────

    def _e_step(self, X: np.ndarray, params: HMMParams) -> tuple:
        """
        E-step: compute
          γ_t(k) = P(S_t = k | X_{1:T})   (state posterior)
          ξ_t(i,j) = P(S_t=i, S_{t+1}=j | X_{1:T})   (transition posterior)

        Returns (gamma, xi, log_likelihood).
        """
        log_emission = params.emission_log_probs(X)
        log_alpha, log_lik = self._forward(log_emission, params)
        log_beta = self._backward(log_emission, params)
        log_A = np.log(np.maximum(params.A, 1e-300))

        T, K = log_alpha.shape

        # γ_t(k) (normalised)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp_axis(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # ξ_t(i,j): shape (T-1, K, K)
        xi = np.zeros((T-1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = (log_alpha[t, i] + log_A[i, j] +
                                   log_emission[t+1, j] + log_beta[t+1, j])
            # Normalise
            xi[t] = np.exp(xi[t] - _logsumexp(xi[t].ravel()))

        return gamma, xi, log_lik

    # ── M-step: update parameters ─────────────────────────────────

    def _m_step(self, X: np.ndarray, gamma: np.ndarray,
                xi: np.ndarray) -> HMMParams:
        """
        M-step: update (μ, σ, A, π) to maximise expected log-likelihood.

        π_k = γ_0(k)
        A_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
        μ_k = Σ_t γ_t(k)·X_t / Σ_t γ_t(k)
        σ²_k = Σ_t γ_t(k)·(X_t - μ_k)² / Σ_t γ_t(k)
        """
        K = gamma.shape[1]

        pi = gamma[0] / gamma[0].sum()
        A = xi.sum(axis=0)
        A /= A.sum(axis=1, keepdims=True) + 1e-300

        N_k = gamma.sum(axis=0)  # expected number of steps in state k
        mu = (gamma * X[:, None]).sum(axis=0) / (N_k + 1e-300)
        sigma = np.sqrt(
            (gamma * (X[:, None] - mu[None, :]) ** 2).sum(axis=0) / (N_k + 1e-300)
        )
        sigma = np.maximum(sigma, 1e-4)  # prevent degenerate solutions

        return HMMParams(mu=mu, sigma=sigma, A=A, pi=pi)

    # ── Random initialisation ─────────────────────────────────────

    def _random_init(self, X: np.ndarray, rng: np.random.Generator) -> HMMParams:
        """Initialise from data with k-means-style centres + noise."""
        T = len(X)
        # Random cluster assignment
        assignments = rng.integers(0, self.K, size=T)
        mu = np.array([X[assignments == k].mean() if (assignments == k).any()
                       else X.mean() + rng.normal(0, X.std() * 0.1)
                       for k in range(self.K)])
        sigma = np.full(self.K, X.std())

        # Dirichlet transition matrix (sparse diagonal)
        A = rng.dirichlet(np.ones(self.K) * 0.5, size=self.K)
        # Increase diagonal (encourage persistence)
        A = A * 0.2 + np.eye(self.K) * 0.8
        A /= A.sum(axis=1, keepdims=True)

        pi = rng.dirichlet(np.ones(self.K))
        return HMMParams(mu=mu, sigma=sigma, A=A, pi=pi)

    # ── Fit ───────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fit the HMM via Baum-Welch (EM) with multiple random restarts.
        Keeps the solution with highest log-likelihood.
        """
        rng = np.random.default_rng(self.seed)
        best_params = None
        best_ll = -np.inf

        for init_idx in range(self.n_init):
            params = self._random_init(X, rng)
            prev_ll = -np.inf

            for iteration in range(self.n_iter):
                gamma, xi, log_lik = self._e_step(X, params)
                params = self._m_step(X, gamma, xi)

                if abs(log_lik - prev_ll) < self.tol:
                    break
                prev_ll = log_lik

            if log_lik > best_ll:
                best_ll = log_lik
                best_params = params

        self.params_ = best_params
        self.log_likelihood_ = best_ll

        # Sort states by mean return (ascending: crisis, bear, bull)
        order = np.argsort(self.params_.mu)
        self.params_ = HMMParams(
            mu=self.params_.mu[order],
            sigma=self.params_.sigma[order],
            A=self.params_.A[np.ix_(order, order)],
            pi=self.params_.pi[order],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities γ_t(k). Shape (T, K)."""
        if self.params_ is None:
            raise RuntimeError("Fit the model first.")
        gamma, _, _ = self._e_step(X, self.params_)
        return gamma

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding: return most likely hidden state sequence.
        Returns array of state indices (0 to K-1). Shape (T,).
        """
        if self.params_ is None:
            raise RuntimeError("Fit the model first.")
        T = len(X)
        K = self.K
        params = self.params_
        log_emission = params.emission_log_probs(X)
        log_A = np.log(np.maximum(params.A, 1e-300))

        # Viterbi trellis
        viterbi = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        viterbi[0] = np.log(np.maximum(params.pi, 1e-300)) + log_emission[0]

        for t in range(1, T):
            for j in range(K):
                trans = viterbi[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(trans)
                viterbi[t, j] = np.max(trans) + log_emission[t, j]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def regime_summary(self, labels: list = None) -> pd.DataFrame:
        """Summary of fitted regime parameters."""
        if self.params_ is None:
            raise RuntimeError("Fit first.")
        p = self.params_
        if labels is None:
            labels = [f"State {k}" for k in range(self.K)]
        rows = []
        for k in range(self.K):
            rows.append({
                "State": labels[k],
                "Mean (ann %)": p.mu[k] * 252 * 100,
                "Vol (ann %)": p.sigma[k] * math.sqrt(252) * 100,
                "Sharpe (ann)": (p.mu[k] * 252) / (p.sigma[k] * math.sqrt(252)) if p.sigma[k] > 0 else 0,
                "Avg Duration (days)": 1 / max(1 - p.A[k, k], 1e-6),
                "Stationary Prob": p.pi[k],
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Log-sum-exp utilities
# ---------------------------------------------------------------------------

def _logsumexp(log_x: np.ndarray) -> float:
    """Numerically stable log(sum(exp(log_x)))."""
    m = log_x.max()
    if m == -np.inf:
        return -np.inf
    return m + math.log(np.exp(log_x - m).sum())


def _logsumexp_axis(log_x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp along an axis."""
    m = log_x.max(axis=axis, keepdims=True)
    result = m + np.log(np.exp(log_x - m).sum(axis=axis, keepdims=True))
    if not keepdims:
        result = result.squeeze(axis=axis)
    return result


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  Market Regime Detection — Gaussian Hidden Markov Model")
    print("  Bull / Bear / Crisis regime identification via Baum-Welch EM")
    print("═" * 66)

    # Simulate returns with 3 known regimes
    rng = np.random.default_rng(0)
    T = 1260  # 5 years

    # True regimes (ground truth for validation)
    true_regimes = np.zeros(T, dtype=int)
    # Bull: 60% of time
    bull_intervals  = [(0, 300), (500, 700), (800, 1050), (1150, 1260)]
    # Bear: 25% of time
    bear_intervals  = [(300, 500), (700, 800), (1050, 1150)]
    # Crisis: 15% of time — scattered short episodes

    for a, b in bull_intervals:
        true_regimes[a:b] = 2    # bull = state 2 (highest mean)
    for a, b in bear_intervals:
        true_regimes[a:b] = 1    # bear = state 1

    # Regime parameters (daily):
    regime_params_true = {
        0: {"mu": -0.0020, "sigma": 0.028},  # crisis
        1: {"mu": -0.0003, "sigma": 0.014},  # bear
        2: {"mu": +0.0006, "sigma": 0.007},  # bull
    }

    returns = np.array([
        rng.normal(regime_params_true[true_regimes[t]]["mu"],
                   regime_params_true[true_regimes[t]]["sigma"])
        for t in range(T)
    ])

    # ── Fit HMM ───────────────────────────────────────────────────
    print(f"\n  Fitting 3-state Gaussian HMM on {T} daily returns...")
    hmm = GaussianHMM(n_states=3, n_iter=100, n_init=10, seed=42)
    hmm.fit(returns)

    labels = ["Crisis", "Bear", "Bull"]
    summary = hmm.regime_summary(labels)

    print(f"\n── Fitted Regime Parameters ──")
    print(f"\n  {'Regime':<10} {'Mean (ann)':>12} {'Vol (ann)':>12} {'Sharpe':>8} {'Avg Days':>10}")
    print("  " + "─" * 56)
    for _, row in summary.iterrows():
        print(f"  {row['State']:<10} {row['Mean (ann %)']:>11.2f}%"
              f" {row['Vol (ann %)']:>11.2f}%"
              f" {row['Sharpe (ann)']:>8.3f}"
              f" {row['Avg Duration (days)']:>10.1f}")

    # ── Transition matrix ─────────────────────────────────────────
    print(f"\n── Transition Matrix A[i,j] = P(next=j | current=i) ──")
    A = hmm.params_.A
    print(f"\n  {'':>10}", end="")
    for lbl in labels:
        print(f"  {'→'+lbl:>12}", end="")
    print()
    print("  " + "─" * 46)
    for i, lbl in enumerate(labels):
        print(f"  {lbl:<10}", end="")
        for j in range(3):
            print(f"  {A[i,j]:>12.4f}", end="")
        print()

    # ── Posterior probabilities and predicted states ───────────────
    predicted_states = hmm.predict(returns)
    probs = hmm.predict_proba(returns)

    # Accuracy (permutation-invariant: find best label mapping)
    from itertools import permutations
    best_acc = 0.0
    for perm in permutations(range(3)):
        mapped = np.array([perm[s] for s in predicted_states])
        acc = float((mapped == true_regimes).mean())
        if acc > best_acc:
            best_acc = acc
    print(f"\n── Regime Detection Accuracy ──")
    print(f"  Fraction of days correctly classified: {best_acc:.4f}")

    # ── Regime statistics ──────────────────────────────────────────
    print(f"\n── Regime-Conditional Return Statistics ──")
    print(f"\n  {'Regime':<12} {'Days':>6} {'Ann Return':>12} {'Ann Vol':>10} {'Sharpe':>8}")
    print("  " + "─" * 52)
    regime_labels_sorted = ["Crisis (0)", "Bear (1)", "Bull (2)"]
    for k, rname in enumerate(regime_labels_sorted):
        mask = (predicted_states == k)
        if mask.sum() > 0:
            r_k = returns[mask]
            ann_ret = r_k.mean() * 252 * 100
            ann_vol = r_k.std() * math.sqrt(252) * 100
            sr = (r_k.mean() * 252) / (r_k.std() * math.sqrt(252)) if r_k.std() > 0 else 0
            print(f"  {rname:<12} {mask.sum():>6} {ann_ret:>11.2f}% {ann_vol:>9.2f}% {sr:>8.3f}")

    # ── Regime-conditional allocation ─────────────────────────────
    print(f"\n── Regime-Conditional Portfolio Allocation ──")
    print(f"""
  A practical regime-switching strategy:

  Regime     Equity  Bonds   Cash   Rationale
  ─────────────────────────────────────────────────────
  Bull       80%     15%     5%     Risk-on: max equity
  Bear       40%     50%     10%    Defensive: overweight bonds
  Crisis     10%     60%     30%    Flight to quality

  Expected benefit: capture bull market upside while reducing drawdown
  in bear/crisis periods. Estimated Sharpe improvement: +0.2 to +0.4
  vs static 60/40 (Ang & Bekaert 2002).
    """)
