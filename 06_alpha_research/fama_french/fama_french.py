"""
Multi-Factor Model (Fama-French Framework)
============================================
Implements a factor model for equity returns in the Fama-French tradition:

  r_i,t - rf_t = α_i + β_{i,MKT}·(r_m - rf)_t
                     + β_{i,SMB}·SMB_t
                     + β_{i,HML}·HML_t
                     + β_{i,MOM}·MOM_t  (Carhart 4-factor)
                     + ε_{i,t}

Factors:
  MKT (Market): Excess return of the market over the risk-free rate.
                Measures systematic / undiversifiable risk (CAPM beta).
  SMB (Small Minus Big): Return of small-cap minus large-cap stocks.
                         Captures size premium (Fama & French 1992).
  HML (High Minus Low): Return of high B/M minus low B/M stocks.
                        Captures value premium.
  MOM (Momentum): 12-1 month return spread (Jegadeesh & Titman 1993).
                  Carhart (1997) adds this as a 4th factor.

Interpretation:
  α (Jensen's Alpha): risk-adjusted excess return above what the factor model predicts.
                      Positive α = manager skill or mispricing.
  β_MKT:  market sensitivity (CAPM beta). β=1.2 → 20% more sensitive than market.
  β_SMB:  positive → tilted toward small caps.
  β_HML:  positive → tilted toward value stocks.
  β_MOM:  positive → momentum exposure.

Applications:
  - Performance attribution (how much of the return is β vs α?)
  - Risk decomposition (factor vs idiosyncratic risk)
  - Portfolio construction with factor constraints

References:
  - Fama, E.F. & French, K.R. (1993). Common Risk Factors. JFE 33(1), 3–56.
  - Carhart, M.M. (1997). On Persistence in Mutual Fund Performance. JF 52(1).
  - Sharpe, W.F. (1964). Capital Asset Prices. JF 19(3), 425–442.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FactorModelResult:
    """
    Results from a factor model regression on a single asset or portfolio.

    Attributes
    ----------
    alpha : float        Jensen's alpha (annualised).
    alpha_t_stat : float t-statistic on alpha (> 2 → statistically significant).
    alpha_p_val : float  p-value on alpha (< 0.05 → reject null of zero alpha).
    betas : dict         Factor exposures: {factor_name: beta_value}
    r_squared : float    Proportion of variance explained by factors.
    tracking_error : float  Annualised idiosyncratic volatility (residual std).
    information_ratio : float  alpha / tracking_error (annualised).
    factor_contributions : dict  Share of total variance from each factor.
    """
    alpha: float
    alpha_t_stat: float
    alpha_p_val: float
    betas: dict
    t_stats: dict
    p_values: dict
    r_squared: float
    adj_r_squared: float
    tracking_error: float
    information_ratio: float
    factor_contributions: dict
    n_obs: int

    def summary(self, asset_name: str = "Asset") -> str:
        lines = [
            "=" * 58,
            f"  Factor Model Results: {asset_name}",
            "=" * 58,
            f"  {'Factor':<16} {'Beta':>10} {'t-stat':>10} {'p-value':>10}",
            "─" * 58,
            f"  {'Alpha (ann.)':<16} {self.alpha:>10.4%} {self.alpha_t_stat:>10.3f} {self.alpha_p_val:>10.4f}",
        ]
        for fname, beta in self.betas.items():
            t = self.t_stats.get(fname, 0)
            p = self.p_values.get(fname, 1)
            sig = " **" if p < 0.05 else (" *" if p < 0.10 else "")
            lines.append(f"  {fname:<16} {beta:>10.4f} {t:>10.3f} {p:>10.4f}{sig}")
        lines.extend([
            "─" * 58,
            f"  {'R²':<28} {self.r_squared:>10.4f}",
            f"  {'Adj. R²':<28} {self.adj_r_squared:>10.4f}",
            f"  {'Tracking Error (ann.)':<28} {self.tracking_error:>10.4%}",
            f"  {'Information Ratio':<28} {self.information_ratio:>10.4f}",
            f"  {'Observations':<28} {self.n_obs:>10,}",
            "=" * 58,
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factor Model
# ---------------------------------------------------------------------------

class FamaFrenchModel:
    """
    Multi-factor return attribution and risk decomposition.

    Supports any set of factors; defaults to Fama-French 3-factor + Carhart momentum.

    Usage
    -----
    >>> model = FamaFrenchModel(factors_df, factors=['MKT', 'SMB', 'HML', 'MOM'])
    >>> result = model.fit(asset_returns)
    >>> print(result.summary("MSFT"))
    """

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        factors: list[str] | None = None,
    ):
        """
        Parameters
        ----------
        factor_returns : pd.DataFrame
            DataFrame with factor return series as columns.
            Must include an 'RF' column for the risk-free rate.
            Index should be dates.
        factors : list[str], optional
            Factor names to use from factor_returns. Defaults to MKT/SMB/HML/MOM.
        """
        self.factor_returns = factor_returns.copy()
        if factors is None:
            available = [c for c in ["MKT", "SMB", "HML", "MOM"] if c in factor_returns.columns]
            self.factors = available
        else:
            self.factors = factors

    # ------------------------------------------------------------------
    # OLS regression
    # ------------------------------------------------------------------

    def fit(self, asset_returns: pd.Series, ann_factor: int = 252) -> FactorModelResult:
        """
        Fit the factor model to an asset's excess return series.

        r_excess = α + β · F + ε

        Parameters
        ----------
        asset_returns : pd.Series
            Gross daily returns (NOT excess; RF is subtracted internally).
        ann_factor : int
            Trading days per year (for annualising alpha and vol).

        Returns
        -------
        FactorModelResult with full regression output.
        """
        # Align on common dates
        rf = self.factor_returns.get("RF", pd.Series(0, index=self.factor_returns.index))
        excess_r = asset_returns.sub(rf, fill_value=0).dropna()

        F = self.factor_returns[self.factors].copy()
        data = pd.concat([excess_r.rename("excess_r"), F], axis=1).dropna()

        if len(data) < len(self.factors) + 10:
            raise ValueError(f"Insufficient overlapping observations: {len(data)}")

        y = data["excess_r"].values
        X_raw = data[self.factors].values
        X = np.column_stack([np.ones(len(y)), X_raw])  # prepend constant (alpha)

        # OLS: β = (X'X)^{-1} X'y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        coef = XtX_inv @ X.T @ y
        alpha_daily = coef[0]
        betas = dict(zip(self.factors, coef[1:]))

        # Residuals and diagnostics
        y_hat = X @ coef
        residuals = y - y_hat
        n, k = len(y), len(coef)
        sigma2 = np.sum(residuals**2) / (n - k)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))

        t_stats_all = coef / se
        p_vals_all = 2 * (1 - stats.t.cdf(np.abs(t_stats_all), df=n - k))

        # R²
        ss_tot = np.sum((y - y.mean())**2)
        ss_res = np.sum(residuals**2)
        r2 = 1 - ss_res / ss_tot
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

        # Annualise
        alpha_ann = alpha_daily * ann_factor
        tracking_error = float(residuals.std() * math.sqrt(ann_factor))
        ir = alpha_ann / tracking_error if tracking_error > 0 else 0.0

        # Factor variance contributions
        factor_var = {}
        total_var = float(np.var(y))
        for i, fname in enumerate(self.factors):
            factor_var[fname] = round(float(betas[fname]**2 * np.var(data[fname].values) / total_var), 4)
        factor_var["idiosyncratic"] = round(1 - sum(factor_var.values()), 4)

        return FactorModelResult(
            alpha=alpha_ann,
            alpha_t_stat=float(t_stats_all[0]),
            alpha_p_val=float(p_vals_all[0]),
            betas=betas,
            t_stats={f: float(t_stats_all[i + 1]) for i, f in enumerate(self.factors)},
            p_values={f: float(p_vals_all[i + 1]) for i, f in enumerate(self.factors)},
            r_squared=float(r2),
            adj_r_squared=float(adj_r2),
            tracking_error=tracking_error,
            information_ratio=ir,
            factor_contributions=factor_var,
            n_obs=n,
        )

    def rolling_betas(self, asset_returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """
        Compute rolling factor betas using a fixed window.
        Useful for detecting time-variation in factor exposures.

        Returns DataFrame with one column per factor plus 'alpha'.
        """
        rf = self.factor_returns.get("RF", pd.Series(0, index=self.factor_returns.index))
        excess_r = asset_returns.sub(rf, fill_value=0).dropna()
        F = self.factor_returns[self.factors].copy()
        data = pd.concat([excess_r.rename("excess_r"), F], axis=1).dropna()

        results = []
        for i in range(window, len(data) + 1):
            chunk = data.iloc[i - window:i]
            y = chunk["excess_r"].values
            X = np.column_stack([np.ones(len(y)), chunk[self.factors].values])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                row = {"date": data.index[i - 1], "alpha_daily": coef[0]}
                row.update(dict(zip(self.factors, coef[1:])))
                results.append(row)
            except Exception:
                pass

        return pd.DataFrame(results).set_index("date") if results else pd.DataFrame()

    def portfolio_decomposition(self, portfolio_returns: dict[str, pd.Series],
                                 weights: dict[str, float]) -> pd.DataFrame:
        """
        Decompose a weighted portfolio's factor exposures.

        Portfolio factor exposure = Σ_i w_i · β_{i,factor}
        This is the weighted average of individual asset exposures.
        """
        rows = []
        for name, ret in portfolio_returns.items():
            try:
                res = self.fit(ret)
                row = {"asset": name, "weight": weights.get(name, 0),
                       "alpha_ann": res.alpha, "IR": res.information_ratio}
                row.update({f"beta_{f}": b for f, b in res.betas.items()})
                rows.append(row)
            except Exception:
                pass

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("asset")

        # Portfolio-level exposures (weighted average)
        total_w = sum(weights.values())
        port_row = {"weight": total_w, "alpha_ann": 0.0, "IR": 0.0}
        for fname in self.factors:
            col = f"beta_{fname}"
            if col in df.columns:
                port_row[col] = float((df[col] * df["weight"]).sum() / total_w)
        port_row["alpha_ann"] = float((df["alpha_ann"] * df["weight"]).sum() / total_w)
        df.loc["PORTFOLIO"] = port_row
        return df


# ---------------------------------------------------------------------------
# Synthetic factor data generator
# ---------------------------------------------------------------------------

def generate_synthetic_factors(n: int = 1260, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Fama-French-like factor returns for testing.
    Calibrated to approximate historical US factor moments.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)

    # Approximate historical annualised means and daily vols
    # (MKT: 8% ann, SMB: 2%, HML: 3%, MOM: 5%, RF: 4% ann)
    MKT = rng.normal(0.08 / 252, 0.012, n)
    SMB = rng.normal(0.02 / 252, 0.006, n)
    HML = rng.normal(0.03 / 252, 0.007, n)
    MOM = rng.normal(0.05 / 252, 0.010, n)
    RF  = np.full(n, 0.04 / 252)  # 4% annual risk-free rate

    return pd.DataFrame({
        "MKT": MKT, "SMB": SMB, "HML": HML, "MOM": MOM, "RF": RF
    }, index=dates)


def generate_synthetic_asset(factors_df: pd.DataFrame, betas: dict,
                               alpha_daily: float = 0.0002, seed: int = 42) -> pd.Series:
    """Generate an asset with known factor loadings (for testing parameter recovery)."""
    rng = np.random.default_rng(seed)
    n = len(factors_df)
    idio_vol = 0.008  # daily idiosyncratic vol

    r = alpha_daily + factors_df.get("RF", 0).values
    for fname, beta in betas.items():
        if fname in factors_df.columns:
            r = r + beta * factors_df[fname].values
    r += rng.normal(0, idio_vol, n)
    return pd.Series(r, index=factors_df.index, name="asset")


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 60)
    print("  Fama-French 4-Factor Model")
    print("═" * 60)

    # Generate synthetic data with known true betas
    TRUE_BETAS = {"MKT": 1.15, "SMB": 0.30, "HML": -0.20, "MOM": 0.10}
    TRUE_ALPHA_DAILY = 0.0002   # ~5% annual

    factors = generate_synthetic_factors(n=1260)
    asset = generate_synthetic_asset(factors, TRUE_BETAS, TRUE_ALPHA_DAILY)

    model = FamaFrenchModel(factors, factors=["MKT", "SMB", "HML", "MOM"])
    result = model.fit(asset)
    print("\n" + result.summary("Synthetic Asset"))

    print("\n── Parameter Recovery ──")
    print(f"  True alpha (ann): {TRUE_ALPHA_DAILY * 252:.4%}  |  Fitted: {result.alpha:.4%}")
    print(f"\n  {'Factor':<8} {'True Beta':>12} {'Fitted Beta':>14} {'Sig (5%)?':>12}")
    print("  " + "─" * 50)
    for f, true_b in TRUE_BETAS.items():
        fitted = result.betas[f]
        sig = "Yes" if result.p_values[f] < 0.05 else "No"
        print(f"  {f:<8} {true_b:>12.4f} {fitted:>14.4f} {sig:>12}")

    print("\n── Variance Decomposition ──")
    for source, contribution in result.factor_contributions.items():
        print(f"  {source:<18}: {contribution:.2%} of total return variance")
