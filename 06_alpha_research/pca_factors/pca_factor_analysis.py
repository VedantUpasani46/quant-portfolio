"""
PCA Factor Analysis for Yield Curves and Equity Returns
==========================================================
Principal Component Analysis extracts orthogonal latent factors that
explain the maximum variance of a dataset. In finance, PCA is used:

  1. Yield curve analysis (the "level/slope/curvature" decomposition)
     PC1 ≈ parallel shift (explains ~85% of rate variance)
     PC2 ≈ slope/steepening (explains ~10%)
     PC3 ≈ curvature/butterfly (explains ~3%)

  2. Equity factor extraction
     PC1 ≈ market factor (beta)
     PC2 ≈ sector rotation
     PC3 ≈ size/style factor

  3. Risk factor reduction for large portfolios (compress N assets into K factors)

  4. Covariance matrix regularisation (use K-factor model instead of full Σ)

The PCA decomposition:
  X = F · L' + E
  where F = factor scores (T × K), L = loadings (N × K), E = idiosyncratic

  Loadings L = eigenvectors of X'X (sorted by eigenvalue)
  Scores F = X · L

Explained variance ratio:
  R²_k = λ_k / Σλ_j   (fraction explained by PC k)

Yield curve interpretation:
  PC1 (level): all tenors move together → parallel shift
  PC2 (slope): short-end moves opposite to long-end → steepening/flattening
  PC3 (curvature): middle moves opposite to wings → butterfly

Risk factor model:
  DV01_hedge_ratio[tenor] = L[tenor, 1:3]   (hedge using PCs not individual tenors)

References:
  - Litterman, R. & Scheinkman, J. (1991). Common Factors Affecting Bond Returns.
    JFI 1(1), 54–61. (the seminal yield curve PCA paper)
  - Connor, G. & Korajczyk, R. (1988). Risk and Return in an Equilibrium APT.
    JFE 21(2), 255–289.
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 7 (yield curve risk).
  - Lopez de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 3.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh


# ---------------------------------------------------------------------------
# Core PCA implementation (from scratch — no sklearn for educational clarity)
# ---------------------------------------------------------------------------

@dataclass
class PCAResult:
    """Result of PCA decomposition."""
    loadings: np.ndarray          # shape (n_features, n_components)
    scores: np.ndarray            # shape (n_obs, n_components)
    eigenvalues: np.ndarray       # shape (n_components,)
    explained_variance_ratio: np.ndarray
    cumulative_explained: np.ndarray
    feature_names: list
    component_names: list
    mean: np.ndarray              # mean used for centering

    def reconstruct(self, n_components: int = None) -> np.ndarray:
        """Reconstruct data using top n_components PCs."""
        k = n_components or len(self.eigenvalues)
        return self.scores[:, :k] @ self.loadings[:, :k].T + self.mean

    def factor_score(self, new_data: np.ndarray, n_components: int = None) -> np.ndarray:
        """Project new data into the PC space."""
        k = n_components or self.loadings.shape[1]
        return (new_data - self.mean) @ self.loadings[:, :k]


def pca(X: np.ndarray, n_components: int = None,
        feature_names: list = None) -> PCAResult:
    """
    Compute PCA via eigendecomposition of the covariance matrix.

    Steps:
      1. Centre the data: X̃ = X - μ
      2. Compute covariance matrix: C = X̃'X̃ / (T-1)
      3. Eigendecompose: C = V·Λ·V'
      4. Sort by descending eigenvalue
      5. Scores: F = X̃ · V_k (project onto top-k eigenvectors)

    Note: for large N, it's more efficient to use the SVD of X̃.
    For a yield curve (N ≤ 30), eigendecomposition of C is fine.
    """
    T, N = X.shape
    mean = X.mean(axis=0)
    X_c = X - mean  # centre

    # Covariance matrix
    C = (X_c.T @ X_c) / (T - 1)

    # Eigendecomposition (eigh for symmetric matrices — more stable than eig)
    eigvals, eigvecs = eigh(C)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Ensure non-negative eigenvalues (numerical precision)
    eigvals = np.maximum(eigvals, 0)

    k = n_components or N
    k = min(k, N, T)

    # Sign convention: largest absolute loading positive
    for j in range(k):
        if eigvecs[np.argmax(np.abs(eigvecs[:, j])), j] < 0:
            eigvecs[:, j] *= -1

    explained = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained)

    scores = X_c @ eigvecs[:, :k]

    comp_names = [f"PC{i+1}" for i in range(k)]

    return PCAResult(
        loadings=eigvecs[:, :k],
        scores=scores,
        eigenvalues=eigvals[:k],
        explained_variance_ratio=explained[:k],
        cumulative_explained=cumulative[:k],
        feature_names=feature_names or [f"F{i}" for i in range(N)],
        component_names=comp_names,
        mean=mean,
    )


# ---------------------------------------------------------------------------
# Yield curve PCA
# ---------------------------------------------------------------------------

def generate_yield_curve_data(T: int = 2520, seed: int = 42) -> tuple:
    """
    Generate a realistic synthetic yield curve dataset.

    Models the yield curve as driven by 3 principal components:
      Level: all rates rise/fall together (AR(1) process)
      Slope: short rates vs long rates (mean-reverting)
      Curvature: butterfly (oscillating)

    Returns (yields: shape (T, N), tenors: list of tenor labels in years).
    """
    rng = np.random.default_rng(seed)
    tenors_yr = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])

    # True factors: AR(1) processes
    level  = np.zeros(T)
    slope  = np.zeros(T)
    curve  = np.zeros(T)

    for t in range(1, T):
        level[t]  = 0.995 * level[t-1]  + rng.normal(0, 0.005)  # slow-moving level
        slope[t]  = 0.980 * slope[t-1]  + rng.normal(0, 0.003)  # slope
        curve[t]  = 0.960 * curve[t-1]  + rng.normal(0, 0.002)  # curvature

    # True factor loadings (Litterman-Scheinkman 1991 shape)
    n_tenors = len(tenors_yr)
    # Level: flat loading
    L1 = np.ones(n_tenors) / math.sqrt(n_tenors)
    # Slope: negative for short, positive for long
    slope_raw = (tenors_yr - tenors_yr.mean()) / tenors_yr.std()
    L2 = slope_raw / np.linalg.norm(slope_raw)
    # Curvature: U-shape (positive at short/long, negative at mid)
    curv_raw = (tenors_yr - tenors_yr.mean()) ** 2
    curv_raw = curv_raw - curv_raw.mean()
    L3 = curv_raw / np.linalg.norm(curv_raw)

    # Loadings matrix (N × 3)
    L = np.column_stack([L1, L2, L3])

    # Factor amplitudes (level dominates)
    factors = np.column_stack([level * 2.5, slope * 0.8, curve * 0.3])

    # Yield curve = factor model + small idiosyncratic noise
    yields = factors @ L.T + rng.normal(0, 0.001, (T, n_tenors))

    # Add realistic level: ~3% base + factor variation
    yields += 0.03
    yields = np.maximum(yields, -0.05)  # allow slight negative rates

    tenor_labels = [f"{int(t*12)}M" if t < 1 else f"{int(t)}Y" for t in tenors_yr]
    return yields, tenor_labels, tenors_yr


def interpret_yield_curve_pcs(result: PCAResult, tenors_yr: np.ndarray) -> dict:
    """
    Assign economic labels to yield curve PCs based on loading shapes.

    PC1 (Level): all loadings same sign and similar magnitude
    PC2 (Slope): loadings monotonically increasing with tenor
    PC3 (Curvature): loadings U-shaped (pos at short/long, neg at mid)
    """
    interpretations = {}
    for k in range(min(3, result.loadings.shape[1])):
        loading = result.loadings[:, k]
        # Level: small standard deviation of loadings
        loading_std = loading.std() / abs(loading).mean()
        # Slope: correlation with tenor (monotonic increase)
        slope_corr = float(np.corrcoef(tenors_yr, loading)[0, 1])
        # Curvature: correlation with |tenor - median|² (U-shape)
        u_shape = (tenors_yr - np.median(tenors_yr)) ** 2
        curv_corr = float(np.corrcoef(u_shape, loading ** 2)[0, 1])

        if loading_std < 0.3:
            label = "Level (parallel shift)"
        elif abs(slope_corr) > 0.8:
            label = "Slope (steepening/flattening)"
        else:
            label = "Curvature (butterfly)"

        interpretations[f"PC{k+1}"] = {
            "label": label,
            "explained_variance": result.explained_variance_ratio[k],
        }
    return interpretations


# ---------------------------------------------------------------------------
# Equity PCA for factor risk model
# ---------------------------------------------------------------------------

def equity_pca_risk_model(
    returns: np.ndarray,
    n_factors: int = 5,
    asset_names: list = None,
) -> dict:
    """
    Extract statistical risk factors from equity returns via PCA.

    Factor risk model:
      r_it = α_i + Σ_k β_ik · f_kt + ε_it

    where f_k = PC scores, β_ik = factor loadings.

    Applications:
      - Portfolio risk decomposition
      - Hedge ratio calculation (beta-neutral, factor-neutral)
      - Risk attribution: systematic vs idiosyncratic
      - Covariance estimation: lower-noise K-factor model
    """
    T, N = returns.shape
    result = pca(returns, n_components=n_factors, feature_names=asset_names)

    # Factor betas: regression of each stock on factor scores
    # β_i = (F'F)⁻¹ F'r_i   (OLS, factors are already orthogonal)
    F = result.scores  # (T, K) — already orthogonal
    betas = np.zeros((N, n_factors))
    alphas = np.zeros(N)
    R2s = np.zeros(N)
    idio_vols = np.zeros(N)

    for i in range(N):
        y = returns[:, i]
        # OLS: since F are orthogonal, β_k = Cov(r_i, f_k) / Var(f_k)
        for k in range(n_factors):
            cov_ik = np.cov(y, F[:, k])[0, 1]
            var_k = np.var(F[:, k])
            betas[i, k] = cov_ik / var_k if var_k > 0 else 0.0

        y_hat = F @ betas[i] + np.mean(y)
        residuals = y - y_hat
        alphas[i] = np.mean(residuals)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        R2s[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        idio_vols[i] = np.std(residuals) * math.sqrt(252)

    # Factor covariance (diagonal since factors are orthogonal)
    factor_vols = np.std(F, axis=0) * math.sqrt(252)
    factor_cov = np.diag(factor_vols ** 2)

    # Reconstructed covariance: Σ ≈ B·Λ_F·B' + D (idiosyncratic diagonal)
    B = betas
    D = np.diag(idio_vols ** 2)
    cov_factor_model = B @ factor_cov @ B.T + D

    return {
        "pca_result": result,
        "betas": betas,
        "alphas": alphas,
        "r2_per_stock": R2s,
        "idio_vols": idio_vols,
        "factor_cov": factor_cov,
        "factor_vols": factor_vols,
        "reconstructed_cov": cov_factor_model,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 66)
    print("  PCA Factor Analysis: Yield Curves & Equity Returns")
    print("  Level/Slope/Curvature decomposition + Factor Risk Model")
    print("═" * 66)

    # ── Yield Curve PCA ────────────────────────────────────────────
    print(f"\n── Yield Curve PCA (Litterman-Scheinkman 1991) ──")
    yields, tenor_labels, tenors_yr = generate_yield_curve_data(T=2520)
    result_yc = pca(yields, n_components=5, feature_names=tenor_labels)
    interp = interpret_yield_curve_pcs(result_yc, tenors_yr)

    print(f"\n  Explained Variance by PC (10Y daily yield data, {len(tenor_labels)} tenors):")
    print(f"\n  {'PC':<8} {'Expl Var':>10} {'Cumul':>10} {'Interpretation'}")
    print("  " + "─" * 56)
    for k in range(5):
        pc_name = f"PC{k+1}"
        ev = result_yc.explained_variance_ratio[k]
        cum = result_yc.cumulative_explained[k]
        interp_str = interp.get(pc_name, {}).get("label", "")
        print(f"  {pc_name:<8} {ev:>10.4%} {cum:>10.4%}  {interp_str}")

    print(f"\n  Factor Loadings (how each tenor responds to each PC):")
    print(f"\n  {'Tenor':<8}", end="")
    for k in range(3):
        print(f" {'PC'+str(k+1)+' loading':>14}", end="")
    print()
    print("  " + "─" * 50)
    for i, tenor in enumerate(tenor_labels):
        print(f"  {tenor:<8}", end="")
        for k in range(3):
            loading = result_yc.loadings[i, k]
            print(f" {loading:>14.4f}", end="")
        print()

    print(f"""
  Economic interpretation:
    PC1 (Level):     All tenors load ~equally positive.
                     When PC1 factor rises → whole curve shifts up.
                     Explains ~{result_yc.explained_variance_ratio[0]:.1%} of yield variance.

    PC2 (Slope):     Short-end negative, long-end positive.
                     When PC2 rises → curve steepens (2s10s widens).
                     Captures Fed policy vs long-term growth expectations.

    PC3 (Curvature): Mid-curve loads opposite to wings.
                     'Butterfly' risk: 5Y moves vs 2Y+10Y.
                     Important for mortgage and structured products desks.

  Practical use:
    A DV01-neutral rate trade can still have PC1 risk.
    Hedging: match PC1, PC2, PC3 sensitivities, not just DV01.
    """)

    # ── Equity PCA Factor Model ────────────────────────────────────
    print(f"── Equity PCA Factor Risk Model ──")
    rng = np.random.default_rng(42)
    T, N = 756, 30
    market = rng.standard_normal(T) * 0.01
    sector_factor = rng.standard_normal(T) * 0.005
    stock_returns = (market[:, None] * (0.8 + rng.uniform(0.2, 0.6, N)) +
                     sector_factor[:, None] * rng.choice([-1, 1], N) * 0.3 +
                     rng.standard_normal((T, N)) * 0.015)

    asset_names = [f"STK_{i:02d}" for i in range(N)]
    eq_pca = equity_pca_risk_model(stock_returns, n_factors=5, asset_names=asset_names)

    print(f"\n  {N}-stock universe, {T} trading days, 5 statistical factors")
    print(f"\n  Factor volatilities (annualised):")
    for k, fvol in enumerate(eq_pca["factor_vols"]):
        ev = eq_pca["pca_result"].explained_variance_ratio[k]
        print(f"    PC{k+1}: {fvol:.4%}/yr  ({ev:.2%} of cross-sectional variance)")

    print(f"\n  Average R² (systematic var explained by 5 factors): "
          f"{eq_pca['r2_per_stock'].mean():.4f}")
    print(f"  Average idiosyncratic vol (annualised): "
          f"{eq_pca['idio_vols'].mean():.4%}")

    # Covariance reconstruction quality
    cov_sample = np.cov(stock_returns.T) * 252
    cov_factor = eq_pca["reconstructed_cov"]
    frob_err = np.linalg.norm(cov_factor - cov_sample, "fro") / np.linalg.norm(cov_sample, "fro")
    cn_sample = np.linalg.cond(cov_sample)
    cn_factor = np.linalg.cond(cov_factor)
    print(f"\n  Covariance matrix comparison:")
    print(f"    Sample covariance condition number:       {cn_sample:>10.1f}")
    print(f"    5-factor model condition number:          {cn_factor:>10.1f}")
    print(f"    Frobenius distance from sample:           {frob_err:>10.4f}")
    print(f"  Factor model significantly improves matrix conditioning.")
