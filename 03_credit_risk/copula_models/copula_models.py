"""
Copula Models for Joint Risk Modelling
========================================
Implements Gaussian and Student-t copulas for modelling joint distributions
in multi-asset portfolios and credit risk applications.

Why copulas?
  Correlation describes linear dependence between two variables — but
  financial returns exhibit non-linear dependence, especially in tails.
  Two assets can have low linear correlation yet crash together
  (tail dependence). The 2008 financial crisis exposed the fatal flaw
  of assuming Gaussian copulas for CDO tranches (Li 2000).

Sklar's Theorem (1959):
  Any joint distribution F(x₁,...,xₙ) can be written as:
    F(x₁,...,xₙ) = C(F₁(x₁), ..., Fₙ(xₙ))
  where C is a copula (a joint distribution on [0,1]ⁿ with uniform margins)
  and Fᵢ are the marginal CDFs.

  The copula C captures the DEPENDENCE STRUCTURE separately from the marginals.
  This separation is the key insight: you can choose any marginals and any
  dependence structure independently.

Gaussian copula:
  C_Gauss(u₁,...,uₙ; Σ) = Φₙ(Φ⁻¹(u₁), ..., Φ⁻¹(uₙ); Σ)
  - Zero tail dependence: extreme events occur independently
  - Fully described by the correlation matrix Σ
  - The Li (2000) model used in CDO pricing: notorious for underestimating
    joint tail losses in 2008

Student-t copula:
  C_t(u₁,...,uₙ; Σ, ν) = Tₙ(T_ν⁻¹(u₁), ..., T_ν⁻¹(uₙ); Σ, ν)
  - Positive upper and lower tail dependence: λ_L = λ_U = 2·T_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
  - Explicitly models extreme co-movement (lower ν → fatter joint tails)
  - Standard in credit and CCR models post-2008

Tail dependence coefficient λ_U:
  λ_U = lim_{u→1} P(U₂ > u | U₁ > u)
  Gaussian: λ_U = 0 (independent extremes)
  Student-t: λ_U > 0 (extremes cluster)

Applications:
  - Credit portfolio loss distributions (CreditMetrics, Basel III IMA)
  - Counterparty credit risk (CVA)
  - Multi-asset VaR (captures non-linear dependence)
  - Stress testing: P(both assets crash)

References:
  - Sklar, A. (1959). Fonctions de répartition à n dimensions.
  - Li, D.X. (2000). On Default Correlation: A Copula Function Approach. JFI.
  - McNeil, A.J., Frey, R. & Embrechts, P. (2015). Quantitative Risk Management.
    Princeton UP. Ch. 5–7.
  - Hull, J.C. & White, A. (2004). Valuation of a CDO. JD 12(1), 8–23.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, t as t_dist, kendalltau, spearmanr


# ---------------------------------------------------------------------------
# Correlation estimation
# ---------------------------------------------------------------------------

def pearson_correlation(data: np.ndarray) -> np.ndarray:
    """Standard Pearson (linear) correlation matrix."""
    return np.corrcoef(data.T)


def spearman_correlation(data: np.ndarray) -> np.ndarray:
    """
    Spearman rank correlation matrix.
    Robust to outliers; measures monotonic (not just linear) dependence.
    Used to estimate copula parameters from empirical data.
    """
    n = data.shape[1]
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(data[:, i], data[:, j])
            corr[i, j] = corr[j, i] = rho
    return corr


def kendall_to_pearson(tau: float) -> float:
    """
    Convert Kendall's τ to the Pearson correlation of a Gaussian copula.
    Relationship: ρ = sin(π·τ/2)   (Lindskog et al. 2003)

    Used when fitting copula parameters from rank correlations (more robust
    to marginal distribution assumptions than Pearson).
    """
    return math.sin(math.pi * tau / 2)


def spearman_to_pearson_gaussian(rho_s: float) -> float:
    """
    Convert Spearman's ρ_s to Pearson's ρ for a Gaussian copula.
    Relationship: ρ = 2·sin(π·ρ_s/6)
    """
    return 2 * math.sin(math.pi * rho_s / 6)


# ---------------------------------------------------------------------------
# Gaussian Copula
# ---------------------------------------------------------------------------

class GaussianCopula:
    """
    Multivariate Gaussian copula sampler and density evaluator.

    The Gaussian copula maps uniform marginals through the inverse normal CDF,
    applies multivariate normal dependence, then maps back.

    Simulation algorithm:
      1. Generate Z ~ N(0, Σ) via Cholesky decomposition
      2. Transform to uniform: Uᵢ = Φ(Zᵢ)
      3. Apply desired marginal: Xᵢ = F_i⁻¹(Uᵢ)

    Usage
    -----
    >>> cop = GaussianCopula(corr_matrix)
    >>> uniforms = cop.sample(n=10000)        # shape (10000, n_assets)
    >>> log_lik = cop.log_density(uniforms)   # for MLE
    """

    def __init__(self, corr: np.ndarray):
        """
        Parameters
        ----------
        corr : np.ndarray   Correlation matrix (n × n), positive definite.
        """
        self.corr = np.asarray(corr)
        self.n = corr.shape[0]
        # Cholesky decomposition for simulation
        try:
            self.L = np.linalg.cholesky(self.corr)
        except np.linalg.LinAlgError:
            # Add small regularisation if not positive definite
            reg = self.corr + np.eye(self.n) * 1e-6
            self.L = np.linalg.cholesky(reg)

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        """
        Draw n samples from the Gaussian copula.

        Returns
        -------
        np.ndarray of shape (n, n_assets) with values in (0, 1).
        Each column is marginally Uniform(0,1); jointly Gaussian-dependent.
        """
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, self.n))          # iid normals
        X = Z @ self.L.T                               # introduce correlation
        U = norm.cdf(X)                                # map to [0,1]
        return U

    def log_density(self, u: np.ndarray) -> np.ndarray:
        """
        Log-density of the Gaussian copula at each observation.

        log c(u; Σ) = -½ z'(Σ⁻¹ - I)z - ½ log|Σ|
        where z = (Φ⁻¹(u₁), ..., Φ⁻¹(uₙ))

        Parameters
        ----------
        u : np.ndarray   Shape (n_obs, n_assets), values in (0,1).
        """
        u_clipped = np.clip(u, 1e-10, 1 - 1e-10)
        z = norm.ppf(u_clipped)                       # (n_obs, n_assets)
        corr_inv = np.linalg.inv(self.corr)
        log_det = np.linalg.slogdet(self.corr)[1]

        # Vectorised: each row z_i, compute z_i'(Σ⁻¹-I)z_i
        quadratic = np.einsum('ni,ij,nj->n', z, corr_inv - np.eye(self.n), z)
        return -0.5 * (quadratic + log_det)

    def tail_dependence(self) -> float:
        """
        Upper tail dependence coefficient for the bivariate Gaussian copula.
        λ_U = 0 for ALL ρ < 1.

        This is the fundamental flaw of the Gaussian copula:
        no matter how high the correlation, extreme events occur independently.
        """
        return 0.0

    def conditional_probability(self, rho: float, threshold: float = 0.99) -> float:
        """
        P(U₂ > threshold | U₁ > threshold) for bivariate Gaussian copula.
        Demonstrates near-zero tail dependence even at high ρ.
        """
        z = norm.ppf(threshold)
        # P(Z₂ > z | Z₁ > z) for bivariate normal with correlation ρ
        # = P(Z₂ > z, Z₁ > z) / P(Z₁ > z)
        from scipy.stats import multivariate_normal
        cov = np.array([[1, rho], [rho, 1]])
        # P(Z1>z, Z2>z) = 1 - 2*Phi(z) + Phi_2(z,z)
        joint = 1 - 2*norm.cdf(z) + multivariate_normal(mean=[0,0], cov=cov).cdf([z, z])
        marginal = 1 - norm.cdf(z)
        return max(joint / marginal, 0.0) if marginal > 0 else 0.0


# ---------------------------------------------------------------------------
# Student-t Copula
# ---------------------------------------------------------------------------

class StudentTCopula:
    """
    Multivariate Student-t copula — models joint tail events.

    The t-copula has positive tail dependence: the probability that both
    assets experience extreme moves simultaneously is strictly positive.

    Simulation algorithm:
      1. Generate Z ~ Nₙ(0, Σ)
      2. Generate W ~ χ²(ν)/ν (independent of Z)
      3. T = Z / √W   →   T ~ tₙ(0, Σ, ν)
      4. Uᵢ = T_ν(Tᵢ)  (map each marginal to uniform via t-CDF)

    Tail dependence coefficient:
      λ_U = λ_L = 2·T_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))

    As ν → ∞, t-copula → Gaussian copula (λ → 0).
    As ν → 1, tail dependence → 1 (perfect co-movement in tails).
    """

    def __init__(self, corr: np.ndarray, nu: float):
        """
        Parameters
        ----------
        corr : np.ndarray   Correlation matrix.
        nu : float          Degrees of freedom (ν > 2 for finite variance).
        """
        self.corr = np.asarray(corr)
        self.nu = nu
        self.n = corr.shape[0]
        try:
            self.L = np.linalg.cholesky(self.corr)
        except np.linalg.LinAlgError:
            reg = self.corr + np.eye(self.n) * 1e-6
            self.L = np.linalg.cholesky(reg)

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        """
        Draw n samples from the Student-t copula.

        Returns np.ndarray of shape (n, n_assets) in (0,1).
        """
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, self.n))
        X = Z @ self.L.T                              # correlated normals
        W = rng.chisquare(self.nu, size=n) / self.nu  # chi-squared scaling
        T = X / np.sqrt(W[:, None])                   # t-distributed
        U = t_dist.cdf(T, df=self.nu)                 # map to [0,1]
        return U

    def tail_dependence(self, rho: float | None = None) -> float:
        """
        Upper (= lower) tail dependence coefficient.

        λ = 2·T_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))

        For a 2×2 copula with off-diagonal correlation ρ.
        For a general matrix, uses the average pairwise ρ.
        """
        if rho is None:
            # Use average off-diagonal correlation
            mask = ~np.eye(self.n, dtype=bool)
            rho = float(self.corr[mask].mean())

        if rho >= 1.0:
            return 1.0
        if rho <= -1.0:
            return 0.0

        nu = self.nu
        arg = -math.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        return 2 * t_dist.cdf(arg, df=nu + 1)

    def compare_tail_dependence(self) -> dict:
        """
        Compare tail dependence across degrees of freedom for a given ρ.
        Shows how t-copula with low ν dramatically increases joint tail risk.
        """
        rho = float(self.corr[0, 1]) if self.n >= 2 else 0.5
        results = {}
        for nu in [3, 5, 10, 20, 50, 100, "Gaussian"]:
            if nu == "Gaussian":
                results[nu] = 0.0
            else:
                arg = -math.sqrt((nu + 1) * (1 - rho) / (1 + rho))
                results[nu] = 2 * t_dist.cdf(arg, df=nu + 1)
        return results


# ---------------------------------------------------------------------------
# Copula-based portfolio loss distribution
# ---------------------------------------------------------------------------

class CopulaPortfolioLoss:
    """
    Monte Carlo simulation of portfolio loss using copulas.

    Framework (CreditMetrics / Basel II Internal Models Approach):
      1. Each obligor/asset has a marginal loss distribution F_i
      2. The copula C captures their joint dependence
      3. Simulate joint scenarios via the copula
      4. Compute portfolio loss = Σ EAD_i · LGD_i · 1{default_i}

    This is the foundation of:
      - Basel III IRB credit risk capital calculations
      - CDO tranche pricing (Gaussian copula → Li 2000)
      - CVA/DVA computation under correlated defaults

    Usage
    -----
    >>> model = CopulaPortfolioLoss(corr, pd_array, lgd_array, ead_array, nu=5)
    >>> results = model.simulate(n=100_000)
    >>> print(results['var_99'])
    """

    def __init__(
        self,
        corr: np.ndarray,
        pd: np.ndarray,       # probability of default per obligor
        lgd: np.ndarray,      # loss given default (recovery rate = 1 - LGD)
        ead: np.ndarray,      # exposure at default ($)
        nu: float | None = None,  # None → Gaussian copula; float → t-copula
    ):
        self.corr = corr
        self.pd = np.asarray(pd)
        self.lgd = np.asarray(lgd)
        self.ead = np.asarray(ead)
        self.nu = nu
        self.n_obligors = len(pd)

        if nu is None:
            self.copula = GaussianCopula(corr)
        else:
            self.copula = StudentTCopula(corr, nu)

    def simulate(self, n: int = 100_000, seed: int = 42) -> dict:
        """
        Simulate the portfolio loss distribution.

        For each scenario:
          - Draw joint uniforms U from the copula
          - Asset i defaults if U_i ≤ PD_i (i.e. default threshold = Φ⁻¹(PD_i))
          - Portfolio loss = Σ EAD_i · LGD_i · 1{U_i ≤ PD_i}

        Returns dict with loss distribution statistics and percentiles.
        """
        U = self.copula.sample(n, seed=seed)         # (n, n_obligors)
        defaults = U <= self.pd[None, :]              # (n, n_obligors) bool
        losses = (defaults * self.lgd[None, :] * self.ead[None, :]).sum(axis=1)

        total_ead = float(self.ead.sum())
        return {
            "mean_loss":         float(losses.mean()),
            "std_loss":          float(losses.std()),
            "var_95":            float(np.percentile(losses, 95)),
            "var_99":            float(np.percentile(losses, 99)),
            "var_999":           float(np.percentile(losses, 99.9)),
            "es_99":             float(losses[losses >= np.percentile(losses, 99)].mean()),
            "max_loss":          float(losses.max()),
            "total_ead":         total_ead,
            "var_99_pct_ead":    float(np.percentile(losses, 99)) / total_ead,
            "default_rate_mean": float(defaults.mean()),
            "loss_series":       losses,  # full series for histogram
        }


# ---------------------------------------------------------------------------
# Copula calibration via MLE
# ---------------------------------------------------------------------------

class CopulaMLE:
    """
    Fit Gaussian or Student-t copula parameters via Maximum Likelihood.

    The two-step (IFM) method:
      Step 1: Fit marginal distributions F_i separately (e.g. empirical CDF)
      Step 2: Transform data to pseudo-uniforms u_i = F_i(x_i)
      Step 3: Maximise copula log-likelihood over (Σ, ν)

    This separates the marginal and dependence estimation problems.
    """

    def __init__(self, data: np.ndarray):
        """
        Parameters
        ----------
        data : np.ndarray   Shape (n_obs, n_vars). Raw data (returns, spreads, etc.)
        """
        self.data = data
        self.n_obs, self.n_vars = data.shape
        # Transform to pseudo-uniforms via empirical CDF (rank-based)
        self.uniforms = self._empirical_cdf_transform(data)

    @staticmethod
    def _empirical_cdf_transform(data: np.ndarray) -> np.ndarray:
        """
        Transform each column to pseudo-uniforms via empirical CDF.
        Uses the scaled rank: u_i = rank(x_i) / (n+1)
        (the +1 avoids 0 and 1 which cause -∞ in norm.ppf)
        """
        n = data.shape[0]
        uniforms = np.zeros_like(data)
        for j in range(data.shape[1]):
            ranks = stats.rankdata(data[:, j])
            uniforms[:, j] = ranks / (n + 1)
        return uniforms

    def fit_gaussian(self) -> np.ndarray:
        """
        Fit Gaussian copula: MLE of correlation matrix Σ.

        For Gaussian copula, the MLE of Σ is simply the sample correlation
        of the normal scores z_i = Φ⁻¹(u_i).

        Returns the fitted correlation matrix.
        """
        z = norm.ppf(np.clip(self.uniforms, 1e-8, 1 - 1e-8))
        return np.corrcoef(z.T)

    def fit_student_t(self, nu_grid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """
        Fit Student-t copula: grid search over ν, MLE of Σ for each ν.

        Returns (correlation_matrix, degrees_of_freedom).
        """
        if nu_grid is None:
            nu_grid = np.array([3, 4, 5, 7, 10, 15, 20, 30, 50])

        u = np.clip(self.uniforms, 1e-8, 1 - 1e-8)
        best_ll = -np.inf
        best_corr, best_nu = None, None

        for nu in nu_grid:
            t_scores = t_dist.ppf(u, df=nu)            # transform to t-scores
            corr = np.corrcoef(t_scores.T)             # MLE corr at this ν

            # Compute log-likelihood of t-copula at (corr, ν)
            cop = StudentTCopula(corr, nu)
            log_lik = cop._log_likelihood(u)
            if log_lik > best_ll:
                best_ll = log_lik
                best_corr = corr
                best_nu = float(nu)

        return best_corr, best_nu


# Attach log_likelihood to StudentTCopula
def _t_copula_log_likelihood(self, u: np.ndarray) -> float:
    """Log-likelihood of the t-copula (for calibration)."""
    u = np.clip(u, 1e-10, 1 - 1e-10)
    t_scores = t_dist.ppf(u, df=self.nu)
    n, d = u.shape
    corr_inv = np.linalg.inv(self.corr)
    log_det = np.linalg.slogdet(self.corr)[1]

    # Multivariate t log-density contribution
    ll = 0.0
    nu = self.nu
    for i in range(n):
        z = t_scores[i]
        quad = float(z @ corr_inv @ z)
        # Multivariate t density
        ll += (
            math.lgamma((nu + d) / 2) - math.lgamma(nu / 2)
            - (d / 2) * math.log(nu * math.pi) - 0.5 * log_det
            - ((nu + d) / 2) * math.log(1 + quad / nu)
        )
        # Subtract sum of univariate t log-densities (copula density = joint - marginals)
        for j in range(d):
            ll -= math.log(max(t_dist.pdf(z[j], df=nu), 1e-300))

    return ll


StudentTCopula._log_likelihood = _t_copula_log_likelihood


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  Copula Models — Gaussian and Student-t")
    print("  Joint risk modelling for credit portfolios and multi-asset VaR")
    print("═" * 66)

    rng = np.random.default_rng(0)

    # ── Tail dependence: Gaussian vs t-copula ──────────────────────
    rho = 0.70
    corr_2x2 = np.array([[1.0, rho], [rho, 1.0]])

    print(f"\n── Tail Dependence Comparison (ρ = {rho}) ──")
    print(f"  Gaussian copula: λ_U = 0.000  (extremes ALWAYS independent)")
    print(f"\n  Student-t copula tail dependence by degrees of freedom:")
    print(f"  {'ν':>8} {'λ_U':>12} {'Interpretation'}")
    print("  " + "─" * 50)
    for nu, interp in [(3,   "Heavy tails — high co-crash risk"),
                       (5,   "Standard t — credit risk models"),
                       (10,  "Moderate tail dependence"),
                       (20,  "Near-Gaussian tails"),
                       (100, "Essentially Gaussian")]:
        t_cop = StudentTCopula(corr_2x2, nu)
        lam = t_cop.tail_dependence(rho)
        print(f"  {nu:>8} {lam:>12.6f}  {interp}")

    # ── Gaussian copula conditional probability ─────────────────────
    print(f"\n── Gaussian Copula: P(U₂ > 99% | U₁ > 99%) by ρ ──")
    print(f"  (Shows that even ρ=0.99, Gaussian gives near-zero joint tail prob)")
    g_cop = GaussianCopula(corr_2x2)
    print(f"  {'ρ':>6}  {'P(both > 99%)':>16}")
    print("  " + "─" * 26)
    for rho_v in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        p = g_cop.conditional_probability(rho_v, threshold=0.99)
        print(f"  {rho_v:>6.2f}  {p:>16.6f}")

    # ── Credit portfolio loss simulation ───────────────────────────
    print(f"\n── Credit Portfolio Loss Simulation (100 obligors) ──")
    n_obl = 100
    # Construct block correlation: 5 sectors with ρ=0.3 within, ρ=0.1 between
    sector_corr = np.full((n_obl, n_obl), 0.10)
    for s in range(5):
        idx = slice(s * 20, (s + 1) * 20)
        sector_corr[idx, idx] = 0.30
    np.fill_diagonal(sector_corr, 1.0)

    pd_arr  = rng.uniform(0.01, 0.05, n_obl)   # PD: 1%–5%
    lgd_arr = rng.uniform(0.40, 0.60, n_obl)   # LGD: 40%–60%
    ead_arr = rng.uniform(0.5e6, 2e6, n_obl)   # EAD: $0.5M–$2M

    print(f"\n  Portfolio: {n_obl} obligors, total EAD = ${ead_arr.sum()/1e6:.1f}M")
    print(f"  Avg PD: {pd_arr.mean():.2%}  Avg LGD: {lgd_arr.mean():.2%}")

    print(f"\n  {'Metric':<30}  {'Gaussian':>14}  {'t(ν=5)':>14}  {'t(ν=3)':>14}")
    print("  " + "─" * 76)

    results = {}
    for label, nu in [("Gaussian", None), ("t(ν=5)", 5), ("t(ν=3)", 3)]:
        model = CopulaPortfolioLoss(sector_corr, pd_arr, lgd_arr, ead_arr, nu=nu)
        res = model.simulate(n=100_000)
        results[label] = res

    for metric in ["var_95", "var_99", "var_999", "es_99", "mean_loss"]:
        row = f"  {metric:<30}"
        for label in ["Gaussian", "t(ν=5)", "t(ν=3)"]:
            val = results[label][metric]
            row += f"  ${val/1e6:>12.3f}M"
        print(row)

    print(f"\n  Key insight: t(ν=3) 99.9% VaR is significantly higher than Gaussian")
    print(f"  This is the 'correlation underestimation' problem that caused 2008 CDO losses")

    # ── Copula calibration ─────────────────────────────────────────
    print(f"\n── MLE Copula Calibration ──")
    # Generate data from known t-copula
    true_rho, true_nu = 0.65, 6.0
    true_corr = np.array([[1.0, true_rho], [true_rho, 1.0]])
    t_generator = StudentTCopula(true_corr, true_nu)
    data_u = t_generator.sample(2000)
    # Convert to synthetic returns via normal quantile
    data = norm.ppf(data_u)

    mle = CopulaMLE(data)
    fitted_gauss_corr = mle.fit_gaussian()
    fitted_t_corr, fitted_nu = mle.fit_student_t(np.array([3, 4, 5, 6, 7, 8, 10]))

    print(f"  True: ρ={true_rho}, ν={true_nu}")
    print(f"  Fitted Gaussian ρ: {fitted_gauss_corr[0,1]:.4f}")
    print(f"  Fitted t-copula:   ρ={fitted_t_corr[0,1]:.4f}, ν={fitted_nu:.1f}")
