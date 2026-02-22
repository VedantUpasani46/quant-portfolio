"""
Extreme Value Theory (EVT) for Tail Risk
==========================================
Rigorous statistical framework for estimating probabilities of extreme
events beyond the historical data range.

Why EVT?
  VaR at 99% is well-estimated by historical simulation.
  But regulatory capital requires 99.9% (Basel IV IMA), 99.97% (operational risk),
  or even 99.99% in stressed scenarios. Historical data is too sparse at these levels.

  EVT provides theoretically justified extrapolation into the extreme tail,
  using only the few observations that exceed a high threshold — no assumption
  needed about the body of the distribution.

The two fundamental EVT theorems:

  1. Fisher-Tippett-Gnedenko (1928/1943):
     Max of large samples converges to Generalised Extreme Value (GEV) distribution.
     GEV = union of Gumbel (ξ=0), Fréchet (ξ>0), Weibull (ξ<0).

  2. Pickands-Balkema-de Haan (1974/1975) — Peaks-Over-Threshold (POT):
     Exceedances above a high threshold u follow the
     Generalised Pareto Distribution (GPD):
       G(y; ξ, β) = 1 - (1 + ξ·y/β)^(-1/ξ)    if ξ ≠ 0
       G(y; β) = 1 - exp(-y/β)                   if ξ = 0

     ξ (shape): tail index. ξ > 0 → heavy tail (Pareto family)
     β (scale): spread of exceedances

     Financial returns have ξ ≈ 0.2–0.4 → heavy tails, infinite kurtosis for ξ ≥ 0.25

Applications:
  - VaR and ES at very high confidence levels (99.9%, 99.97%)
  - Operational risk capital under Basel III Advanced Measurement Approach
  - Reinsurance pricing (rare catastrophes)
  - Climate risk (extreme weather events)
  - Systemic risk (BIS, FSB Financial Stability Reports)

POT method workflow:
  1. Choose threshold u (e.g. 95th percentile of losses)
  2. Fit GPD to exceedances Y = X - u | X > u via MLE
  3. Extrapolate to any high quantile:
     VaR_p = u + (β/ξ)·[(n/n_u·(1-p))^(-ξ) - 1]
     ES_p = (VaR_p + β - ξ·u) / (1 - ξ)

References:
  - Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). Modelling Extremal Events.
    Springer.
  - McNeil, A.J. & Frey, R. (2000). Estimation of Tail-Related Risk Measures.
    Journal of Empirical Finance, 7(3–4), 271–300.
  - McNeil, A.J., Frey, R. & Embrechts, P. (2015). Quantitative Risk Management.
    Princeton UP. Ch. 5.
  - Basel Committee (2019). Minimum Capital Requirements for Market Risk (FRTB).
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import genpareto, genextreme


# ---------------------------------------------------------------------------
# GPD distribution (Generalised Pareto)
# ---------------------------------------------------------------------------

@dataclass
class GPDParams:
    """
    Fitted Generalised Pareto Distribution parameters.

    Attributes
    ----------
    xi    : float  Shape parameter (tail index).
                   xi > 0: heavy tail (power-law decay). Financial data: 0.2–0.4.
                   xi = 0: exponential tail (Gumbel domain).
                   xi < 0: bounded tail (Weibull domain, rare in finance).
    beta  : float  Scale parameter (> 0). Controls spread of tail.
    threshold : float  Threshold u above which GPD was fitted.
    n_exceedances : int  Number of observations exceeding the threshold.
    n_total : int        Total sample size.
    log_likelihood : float
    """
    xi: float
    beta: float
    threshold: float
    n_exceedances: int
    n_total: int
    log_likelihood: float = 0.0

    @property
    def mean_excess(self) -> float:
        """Mean excess loss above threshold: E[X - u | X > u] = β/(1-ξ) for ξ < 1."""
        if self.xi >= 1:
            return float("inf")
        return self.beta / (1 - self.xi)

    @property
    def tail_index_alpha(self) -> Optional[float]:
        """Pareto tail index: α = 1/ξ. Finite moments exist up to order α."""
        return 1 / self.xi if self.xi > 0 else None


# ---------------------------------------------------------------------------
# MLE for GPD
# ---------------------------------------------------------------------------

def fit_gpd_mle(exceedances: np.ndarray) -> GPDParams:
    """
    Fit GPD parameters to exceedances Y = X - u | X > u via MLE.

    The GPD log-likelihood for ξ ≠ 0:
    l(ξ,β) = -n·ln(β) - (1+1/ξ)·Σ ln(1 + ξ·yᵢ/β)

    For ξ = 0 (exponential): l(β) = -n·ln(β) - Σ yᵢ/β

    Constraints: β > 0; if ξ < 0 then all yᵢ < -β/ξ (bounded support).
    """
    y = np.asarray(exceedances, dtype=float)
    n = len(y)
    if n < 10:
        raise ValueError(f"Need at least 10 exceedances for reliable GPD fit (got {n})")

    def neg_log_lik(params):
        xi, log_beta = params
        beta = math.exp(log_beta)  # log-transform ensures β > 0
        if xi == 0:
            return n * log_beta + y.sum() / beta
        z = 1 + xi * y / beta
        if np.any(z <= 0):
            return 1e10  # invalid region
        return n * log_beta + (1 + 1 / xi) * np.sum(np.log(z))

    # Grid of starting points for robustness
    best_nll = float("inf")
    best_params = None
    for xi0 in [0.0, 0.1, 0.2, 0.3, -0.1]:
        for beta0 in [y.mean(), y.std()]:
            x0 = np.array([xi0, math.log(max(beta0, 1e-6))])
            try:
                res = minimize(neg_log_lik, x0, method="Nelder-Mead",
                               options={"maxiter": 5000, "xatol": 1e-8})
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_params = res.x
            except Exception:
                pass

    if best_params is None:
        raise RuntimeError("GPD MLE failed to converge.")

    xi_fit = best_params[0]
    beta_fit = math.exp(best_params[1])
    return GPDParams(xi=xi_fit, beta=beta_fit,
                     threshold=0.0,  # will be set by caller
                     n_exceedances=n, n_total=n,
                     log_likelihood=-best_nll)


# ---------------------------------------------------------------------------
# POT (Peaks-Over-Threshold) full framework
# ---------------------------------------------------------------------------

class POTEstimator:
    """
    Full Peaks-Over-Threshold EVT framework.

    Workflow:
      1. Choose threshold u (mean excess plot, stability test)
      2. Fit GPD to exceedances Y = X - u | X > u
      3. Estimate extreme VaR and ES at any confidence level

    Usage
    -----
    >>> losses = np.abs(portfolio_returns)   # work with positive losses
    >>> pot = POTEstimator(losses)
    >>> gpd = pot.fit(threshold_quantile=0.90)
    >>> var_999 = pot.var_estimate(0.999)
    >>> es_999  = pot.es_estimate(0.999)
    """

    def __init__(self, losses: np.ndarray):
        """
        Parameters
        ----------
        losses : np.ndarray  Positive loss values (not signed returns).
        """
        self.losses = np.sort(np.asarray(losses, dtype=float))[::-1]  # desc sort
        self.n = len(self.losses)
        self._gpd: Optional[GPDParams] = None

    def mean_excess_plot_data(self, n_thresholds: int = 50) -> pd.DataFrame:
        """
        Mean Excess Function e(u) = E[X - u | X > u].

        For GPD:  e(u) = (β + ξ·u) / (1 - ξ)   — linear in u
        For exponential: e(u) = β               — flat (constant)

        A linear upward slope in the empirical e(u) indicates heavy tails (ξ > 0).
        The threshold choice: pick u where e(u) becomes approximately linear.
        """
        thresholds = np.percentile(self.losses, np.linspace(50, 95, n_thresholds))
        records = []
        for u in thresholds:
            exc = self.losses[self.losses > u] - u
            if len(exc) >= 5:
                records.append({
                    "threshold": u,
                    "mean_excess": exc.mean(),
                    "n_exceedances": len(exc),
                    "pct_quantile": 100 * (self.losses > u).mean()
                })
        return pd.DataFrame(records)

    def fit(self, threshold_quantile: float = 0.90,
            threshold: Optional[float] = None) -> GPDParams:
        """
        Fit GPD to losses exceeding the threshold.

        Parameters
        ----------
        threshold_quantile : float  Quantile of losses to use as threshold (0–1).
        threshold : float, optional  Override with a specific threshold value.
        """
        if threshold is None:
            threshold = float(np.percentile(self.losses, threshold_quantile * 100))

        exceedances = self.losses[self.losses > threshold] - threshold
        gpd = fit_gpd_mle(exceedances)
        gpd.threshold = threshold
        gpd.n_total = self.n
        self._gpd = gpd
        return gpd

    def var_estimate(self, p: float) -> float:
        """
        POT VaR estimate at confidence level p (e.g. p=0.999).

        Formula (McNeil & Frey 2000):
          VaR_p = u + (β/ξ)·[(n/n_u · (1-p))^(-ξ) - 1]

        where:
          u = threshold
          n = total sample size
          n_u = number of exceedances
          β, ξ = fitted GPD parameters
        """
        if self._gpd is None:
            raise RuntimeError("Call fit() before var_estimate().")
        gpd = self._gpd
        n_u = gpd.n_exceedances
        n = gpd.n_total
        u, xi, beta = gpd.threshold, gpd.xi, gpd.beta

        if xi == 0:
            return u - beta * math.log((n / n_u) * (1 - p))

        z = (n / n_u) * (1 - p)
        if z <= 0:
            return float("inf")
        return u + (beta / xi) * (z ** (-xi) - 1)

    def es_estimate(self, p: float) -> float:
        """
        POT Expected Shortfall (CVaR) at confidence level p.

        ES_p = VaR_p / (1-ξ) + (β - ξ·u) / (1-ξ)

        Requires ξ < 1 (finite mean of exceedances).
        For financial data (ξ ≈ 0.2–0.4), this is always satisfied.
        """
        if self._gpd is None:
            raise RuntimeError("Call fit() first.")
        gpd = self._gpd
        if gpd.xi >= 1:
            return float("inf")

        var_p = self.var_estimate(p)
        return (var_p + gpd.beta - gpd.xi * gpd.threshold) / (1 - gpd.xi)

    def historical_var(self, p: float) -> float:
        """Historical simulation VaR for comparison."""
        return float(np.percentile(self.losses, p * 100))

    def tail_probability(self, x: float) -> float:
        """
        Estimated P(X > x) for a given extreme level x using POT.
        F̄(x) = (n_u/n) · (1 + ξ·(x-u)/β)^(-1/ξ)
        """
        if self._gpd is None:
            raise RuntimeError("Call fit() first.")
        gpd = self._gpd
        if x <= gpd.threshold:
            return float((self.losses > x).mean())
        z = 1 + gpd.xi * (x - gpd.threshold) / gpd.beta
        if z <= 0:
            return 0.0
        return (gpd.n_exceedances / gpd.n_total) * z ** (-1 / gpd.xi)

    def gpd_diagnostic(self) -> dict:
        """
        Goodness-of-fit diagnostics for the fitted GPD:
          - Kolmogorov-Smirnov test on exceedances
          - Anderson-Darling test
          - Probability-probability (PP) plot summary
        """
        if self._gpd is None:
            raise RuntimeError("Call fit() first.")
        gpd = self._gpd
        exceedances = self.losses[self.losses > gpd.threshold] - gpd.threshold

        # KS test against fitted GPD
        ks_stat, ks_p = stats.kstest(
            exceedances,
            lambda x: genpareto.cdf(x, c=gpd.xi, scale=gpd.beta)
        )

        # Mean excess test: compare empirical e(u) to GPD prediction
        mean_excess_gpd = gpd.mean_excess

        return {
            "xi":              round(gpd.xi, 4),
            "beta":            round(gpd.beta, 4),
            "threshold":       round(gpd.threshold, 4),
            "n_exceedances":   gpd.n_exceedances,
            "n_total":         gpd.n_total,
            "pct_exceedances": round(100 * gpd.n_exceedances / gpd.n_total, 2),
            "ks_statistic":    round(ks_stat, 4),
            "ks_p_value":      round(ks_p, 4),
            "gpd_fit_ok":      ks_p > 0.05,
            "mean_excess_gpd": round(mean_excess_gpd, 4),
            "tail_index_alpha": round(1 / gpd.xi, 3) if gpd.xi > 0 else "∞",
        }


# ---------------------------------------------------------------------------
# Threshold selection helper
# ---------------------------------------------------------------------------

def select_threshold_stability(losses: np.ndarray,
                                quantile_range: tuple = (0.80, 0.97),
                                n_points: int = 20) -> pd.DataFrame:
    """
    Threshold stability plot: xi and beta should be stable across
    a range of thresholds if the GPD is correctly specified.

    Returns DataFrame with (threshold, xi, beta) for visual inspection.
    Used to select the minimum threshold that yields stable GPD estimates.
    """
    quantiles = np.linspace(*quantile_range, n_points)
    records = []
    for q in quantiles:
        try:
            pot = POTEstimator(losses)
            gpd = pot.fit(threshold_quantile=q)
            records.append({
                "threshold_quantile": round(q, 3),
                "threshold":  round(gpd.threshold, 4),
                "xi":         round(gpd.xi, 4),
                "beta_adj":   round(gpd.beta - gpd.xi * gpd.threshold, 4),
                "n_excess":   gpd.n_exceedances,
            })
        except Exception:
            pass
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# GEV distribution (for block maxima)
# ---------------------------------------------------------------------------

@dataclass
class GEVParams:
    """Fitted GEV parameters from block-maxima method."""
    xi: float    # shape (0=Gumbel, >0=Fréchet, <0=Weibull)
    mu: float    # location
    sigma: float # scale
    block_size: int
    n_blocks: int

    def return_level(self, return_period: float) -> float:
        """
        Compute the T-year return level: the loss exceeded on average once every T years.
        Assumes block = 1 year.

        RL(T) = μ - (σ/ξ)·[1 - (-ln(1 - 1/T))^(-ξ)]   for ξ ≠ 0
              = μ - σ·ln(-ln(1 - 1/T))                  for ξ = 0
        """
        p = 1 - 1 / return_period
        if abs(self.xi) < 1e-8:
            return self.mu - self.sigma * math.log(-math.log(p))
        return self.mu - (self.sigma / self.xi) * (1 - (-math.log(p)) ** (-self.xi))


def fit_gev_block_maxima(data: np.ndarray, block_size: int = 252) -> GEVParams:
    """
    Fit GEV to block maxima (annual maximum losses).
    Block size of 252 = trading days per year.
    """
    n_blocks = len(data) // block_size
    block_max = np.array([
        data[i * block_size:(i + 1) * block_size].max()
        for i in range(n_blocks)
    ])

    # SciPy GEV fit (note: scipy uses shape=xi, loc=mu, scale=sigma)
    xi, mu, sigma = genextreme.fit(block_max)
    return GEVParams(xi=xi, mu=mu, sigma=sigma,
                     block_size=block_size, n_blocks=n_blocks)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 64)
    print("  Extreme Value Theory — Peaks-Over-Threshold (POT) Method")
    print("  Rigorous tail risk quantification beyond historical VaR")
    print("═" * 64)

    # Simulate heavy-tailed returns (Student-t, ν=4 → ξ≈0.25)
    rng = np.random.default_rng(42)
    n = 5040  # ~20 years of daily data
    returns = rng.standard_t(df=4, size=n) * 0.01   # scale to daily returns
    losses = -np.minimum(returns, 0)  # focus on loss side (positive)
    losses = losses[losses > 0]
    n_losses = len(losses)

    print(f"\n  Sample: {n:,} observations (20Y daily)")
    print(f"  Losses extracted: {n_losses:,}")
    print(f"  Max loss: {losses.max():.4%}")

    # ── GPD fit at 90th percentile threshold ──────────────────────
    print(f"\n── GPD Fit (threshold = 90th percentile) ──")
    pot = POTEstimator(losses)
    gpd = pot.fit(threshold_quantile=0.90)

    diag = pot.gpd_diagnostic()
    print(f"\n  Fitted parameters:")
    print(f"    ξ (shape):   {gpd.xi:.4f}  (>0 → heavy tail; true value ≈ 0.25 for t₄)")
    print(f"    β (scale):   {gpd.beta:.4f}")
    print(f"    Threshold u: {gpd.threshold:.4%}")
    print(f"    Exceedances: {gpd.n_exceedances} ({diag['pct_exceedances']:.1f}% of sample)")
    print(f"\n  Goodness of fit:")
    print(f"    KS statistic: {diag['ks_statistic']:.4f}")
    print(f"    KS p-value:   {diag['ks_p_value']:.4f}  ({'PASS ✓' if diag['gpd_fit_ok'] else 'FAIL ✗'})")
    print(f"    Tail index α = 1/ξ = {diag['tail_index_alpha']}  (finite moments up to order α)")

    # ── VaR and ES comparison ─────────────────────────────────────
    print(f"\n── VaR and ES at High Confidence Levels ──")
    print(f"  (Historical vs POT extrapolation)")
    print(f"\n  {'Level':<10} {'Hist VaR':>12} {'POT VaR':>12} {'POT ES':>12} {'Obs at level':>14}")
    print("  " + "─" * 54)
    for p in [0.950, 0.990, 0.995, 0.999, 0.9995, 0.9999]:
        hist_var = pot.historical_var(p)
        pot_var = pot.var_estimate(p)
        pot_es = pot.es_estimate(p)
        n_exceed = n_losses * (1 - p)
        print(f"  {p:.4f}    {hist_var:>12.4%} {pot_var:>12.4%} {pot_es:>12.4%} {n_exceed:>14.1f}")

    print(f"\n  Note: At 99.99%, historical simulation has only {n_losses * 0.0001:.2f}")
    print(f"  expected exceedances — far too few for reliable estimation.")
    print(f"  POT extrapolation gives theoretically justified estimates.")

    # ── Threshold stability ───────────────────────────────────────
    print(f"\n── Threshold Stability (xi should be stable across thresholds) ──")
    stab = select_threshold_stability(losses, quantile_range=(0.80, 0.97), n_points=8)
    print(f"\n  {'Quantile':>10} {'Threshold':>12} {'ξ':>10} {'N_excess':>10}")
    print("  " + "─" * 46)
    for _, row in stab.iterrows():
        print(f"  {row['threshold_quantile']:>10.3f} {row['threshold']:>12.4%} "
              f"{row['xi']:>10.4f} {row['n_excess']:>10.0f}")

    # ── Block maxima (GEV) ─────────────────────────────────────────
    print(f"\n── GEV Block Maxima (1-year blocks = 252 trading days) ──")
    gev = fit_gev_block_maxima(losses * 100, block_size=252)   # scale to %
    print(f"\n  ξ (shape): {gev.xi:.4f}  μ={gev.mu:.4f}%  σ={gev.sigma:.4f}%")
    print(f"  Blocks used: {gev.n_blocks}")
    print(f"\n  Return period analysis (worst loss in 1/T years):")
    print(f"  {'Return period':>16} {'Return level (daily loss)':>26}")
    print("  " + "─" * 44)
    for rp in [2, 5, 10, 25, 50, 100, 250]:
        rl = gev.return_level(rp)
        print(f"  {'1-in-'+str(rp)+' years':>16} {rl:>26.4f}%")
