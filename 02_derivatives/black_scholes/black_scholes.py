"""
Black-Scholes Option Pricing Model
===================================
Prices European call and put options under the Black-Scholes-Merton framework
and computes all first- and second-order Greeks.

Black-Scholes assumptions:
  - Underlying follows geometric Brownian motion: dS = μS dt + σS dW
  - Constant risk-free rate and volatility
  - No dividends (extended here to continuous dividend yield q)
  - No transaction costs; continuous trading
  - European-style exercise only

References:
  - Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
    Journal of Political Economy, 81(3), 637–654.
  - Merton, R.C. (1973). Theory of Rational Option Pricing. Bell Journal of Economics.
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, 11th ed., Chapters 15–19.
"""

import math
from dataclasses import dataclass, field
from typing import Literal
from scipy.stats import norm
import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BSMInputs:
    """
    All parameters needed for Black-Scholes-Merton valuation.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to expiry in years (e.g. 0.25 for 3 months).
    r : float
        Continuously compounded risk-free rate (e.g. 0.05 for 5%).
    sigma : float
        Annualised implied / historical volatility (e.g. 0.20 for 20%).
    q : float
        Continuous dividend yield (default 0). Use 0 for non-dividend-paying stocks.
    option_type : str
        'call' or 'put'.
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    option_type: Literal["call", "put"] = "call"

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"Spot price S must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike K must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry T must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{self.option_type}'")


@dataclass
class BSMResult:
    """
    Output container: fair value and all Greeks.

    Greeks
    ------
    price  : Fair value of the option.
    delta  : ∂V/∂S  — sensitivity to spot price.
    gamma  : ∂²V/∂S² — rate of change of delta.
    vega   : ∂V/∂σ  — sensitivity to volatility (per 1% move).
    theta  : ∂V/∂T  — time decay (per calendar day).
    rho    : ∂V/∂r  — sensitivity to risk-free rate (per 1% move).
    vanna  : ∂²V/∂S∂σ — cross-sensitivity of delta to vol.
    volga  : ∂²V/∂σ² — convexity of option price w.r.t. vol (also called vomma).
    charm  : ∂²V/∂S∂T — rate of change of delta over time.
    """
    price: float
    delta: float
    gamma: float
    vega: float       # per 1% change in vol
    theta: float      # per calendar day
    rho: float        # per 1% change in rate
    vanna: float
    volga: float
    charm: float
    d1: float = field(repr=False)
    d2: float = field(repr=False)

    def summary(self) -> str:
        lines = [
            "=" * 46,
            f"  {'Metric':<12}  {'Value':>12}",
            "=" * 46,
            f"  {'Price':<12}  {self.price:>12.6f}",
            "-" * 46,
            f"  {'Delta':<12}  {self.delta:>12.6f}",
            f"  {'Gamma':<12}  {self.gamma:>12.6f}",
            f"  {'Vega (1%)':<12}  {self.vega:>12.6f}",
            f"  {'Theta (1d)':<12}  {self.theta:>12.6f}",
            f"  {'Rho (1%)':<12}  {self.rho:>12.6f}",
            "-" * 46,
            f"  {'Vanna':<12}  {self.vanna:>12.6f}",
            f"  {'Volga':<12}  {self.volga:>12.6f}",
            f"  {'Charm':<12}  {self.charm:>12.6f}",
            "=" * 46,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core BSM engine
# ---------------------------------------------------------------------------

class BlackScholesMerton:
    """
    Black-Scholes-Merton option pricer with Greeks.

    Usage
    -----
    >>> params = BSMInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
    >>> model = BlackScholesMerton(params)
    >>> result = model.price()
    >>> print(result.summary())
    """

    def __init__(self, params: BSMInputs):
        self.p = params
        self._d1, self._d2 = self._compute_d1_d2()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_d1_d2(self) -> tuple[float, float]:
        """
        Compute the d1 and d2 terms central to BSM.

        d1 = [ ln(S/K) + (r - q + σ²/2)T ] / (σ√T)
        d2 = d1 - σ√T
        """
        p = self.p
        sigma_sqrt_T = p.sigma * math.sqrt(p.T)
        d1 = (math.log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma ** 2) * p.T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        return d1, d2

    @property
    def _phi_d1(self) -> float:
        """Standard normal PDF evaluated at d1: φ(d1)."""
        return norm.pdf(self._d1)

    @property
    def _Nd1(self) -> float:
        """Standard normal CDF at d1: N(d1)."""
        return norm.cdf(self._d1)

    @property
    def _Nd2(self) -> float:
        """Standard normal CDF at d2: N(d2)."""
        return norm.cdf(self._d2)

    # ------------------------------------------------------------------
    # Fair value
    # ------------------------------------------------------------------

    def fair_value(self) -> float:
        """
        BSM closed-form option price.

        Call: C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
        Put:  P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
        """
        p = self.p
        disc_q = math.exp(-p.q * p.T)
        disc_r = math.exp(-p.r * p.T)

        if p.option_type == "call":
            return p.S * disc_q * self._Nd1 - p.K * disc_r * self._Nd2
        else:
            return p.K * disc_r * norm.cdf(-self._d2) - p.S * disc_q * norm.cdf(-self._d1)

    # ------------------------------------------------------------------
    # First-order Greeks
    # ------------------------------------------------------------------

    def delta(self) -> float:
        """
        Delta = ∂V/∂S

        Call: e^(-qT)·N(d1)
        Put:  e^(-qT)·(N(d1) - 1)

        Interpretation: the hedge ratio — how many shares needed to delta-hedge one option.
        """
        disc_q = math.exp(-self.p.q * self.p.T)
        if self.p.option_type == "call":
            return disc_q * self._Nd1
        else:
            return disc_q * (self._Nd1 - 1.0)

    def gamma(self) -> float:
        """
        Gamma = ∂²V/∂S²  (same for calls and puts by symmetry)

        Γ = e^(-qT)·φ(d1) / (S·σ·√T)

        Interpretation: rate of change of delta; measures convexity of the option price.
        """
        p = self.p
        return (math.exp(-p.q * p.T) * self._phi_d1) / (p.S * p.sigma * math.sqrt(p.T))

    def vega(self) -> float:
        """
        Vega = ∂V/∂σ  (same for calls and puts)

        ν = S·e^(-qT)·φ(d1)·√T

        Returned per 1% change in volatility (divide raw by 100).
        """
        p = self.p
        raw = p.S * math.exp(-p.q * p.T) * self._phi_d1 * math.sqrt(p.T)
        return raw / 100.0

    def theta(self) -> float:
        """
        Theta = ∂V/∂T  (time decay per calendar day)

        Call Θ = -[S·e^(-qT)·φ(d1)·σ / (2√T)] - r·K·e^(-rT)·N(d2) + q·S·e^(-qT)·N(d1)
        Put  Θ = -[S·e^(-qT)·φ(d1)·σ / (2√T)] + r·K·e^(-rT)·N(-d2) - q·S·e^(-qT)·N(-d1)

        Divided by 365 to express as per-day decay.
        """
        p = self.p
        disc_q = math.exp(-p.q * p.T)
        disc_r = math.exp(-p.r * p.T)
        common = -(p.S * disc_q * self._phi_d1 * p.sigma) / (2.0 * math.sqrt(p.T))

        if p.option_type == "call":
            raw = common - p.r * p.K * disc_r * self._Nd2 + p.q * p.S * disc_q * self._Nd1
        else:
            raw = common + p.r * p.K * disc_r * norm.cdf(-self._d2) - p.q * p.S * disc_q * norm.cdf(-self._d1)
        return raw / 365.0

    def rho(self) -> float:
        """
        Rho = ∂V/∂r  (per 1% change in the risk-free rate)

        Call ρ =  K·T·e^(-rT)·N(d2)
        Put  ρ = -K·T·e^(-rT)·N(-d2)
        """
        p = self.p
        disc_r = math.exp(-p.r * p.T)
        if p.option_type == "call":
            raw = p.K * p.T * disc_r * self._Nd2
        else:
            raw = -p.K * p.T * disc_r * norm.cdf(-self._d2)
        return raw / 100.0

    # ------------------------------------------------------------------
    # Second-order / cross Greeks
    # ------------------------------------------------------------------

    def vanna(self) -> float:
        """
        Vanna = ∂²V/∂S∂σ  (sensitivity of delta to volatility)

        Vanna = -e^(-qT)·φ(d1)·d2 / σ

        Used in vol surface construction and hedging books with skew exposure.
        """
        p = self.p
        return -(math.exp(-p.q * p.T) * self._phi_d1 * self._d2) / p.sigma

    def volga(self) -> float:
        """
        Volga (Vomma) = ∂²V/∂σ²  (convexity w.r.t. volatility)

        Volga = S·e^(-qT)·φ(d1)·√T·d1·d2 / σ

        Positive volga means the option benefits from vol-of-vol.
        """
        p = self.p
        return (p.S * math.exp(-p.q * p.T) * self._phi_d1 * math.sqrt(p.T) * self._d1 * self._d2) / p.sigma

    def charm(self) -> float:
        """
        Charm = ∂²V/∂S∂T  (rate of change of delta over time; 'delta bleed')

        Call charm = e^(-qT)·[φ(d1)·(2(r-q)T - d2·σ√T) / (2T·σ√T) + q·N(d1)]
        Put  charm uses N(d1) - 1 in final term.

        Measures how delta drifts purely due to the passage of time.
        """
        p = self.p
        disc_q = math.exp(-p.q * p.T)
        sigma_sqrt_T = p.sigma * math.sqrt(p.T)
        numerator = 2 * (p.r - p.q) * p.T - self._d2 * sigma_sqrt_T
        common = disc_q * self._phi_d1 * numerator / (2 * p.T * sigma_sqrt_T)

        if p.option_type == "call":
            return common - p.q * disc_q * self._Nd1
        else:
            return common + p.q * disc_q * norm.cdf(-self._d1)

    # ------------------------------------------------------------------
    # Implied volatility (Newton-Raphson)
    # ------------------------------------------------------------------

    def implied_vol(self, market_price: float, tol: float = 1e-8, max_iter: int = 200) -> float:
        """
        Find the implied volatility σ* such that BSM(σ*) = market_price.

        Uses Newton-Raphson iteration with vega as the derivative.

        Parameters
        ----------
        market_price : float
            Observed market price of the option.
        tol : float
            Convergence tolerance on the price difference.
        max_iter : int
            Maximum Newton iterations.

        Returns
        -------
        float
            Implied volatility (annualised).

        Raises
        ------
        ValueError
            If the algorithm does not converge.
        """
        # Initial guess: Brenner-Subrahmanyam approximation
        sigma = math.sqrt(2 * math.pi / self.p.T) * (market_price / self.p.S)
        sigma = max(0.001, min(sigma, 5.0))  # clamp to sensible range

        p_orig = self.p
        for i in range(max_iter):
            trial = BSMInputs(
                S=p_orig.S, K=p_orig.K, T=p_orig.T,
                r=p_orig.r, sigma=sigma, q=p_orig.q,
                option_type=p_orig.option_type,
            )
            model = BlackScholesMerton(trial)
            price_diff = model.fair_value() - market_price
            vega_val = model.vega() * 100  # convert back to raw vega

            if abs(vega_val) < 1e-12:
                raise ValueError("Vega near zero — implied vol not solvable at this point.")

            sigma_new = sigma - price_diff / vega_val
            sigma_new = max(0.001, sigma_new)

            if abs(sigma_new - sigma) < tol:
                return sigma_new
            sigma = sigma_new

        raise ValueError(f"Implied vol did not converge after {max_iter} iterations.")

    # ------------------------------------------------------------------
    # Full pricing bundle
    # ------------------------------------------------------------------

    def price(self) -> BSMResult:
        """Compute and return the full result: fair value + all Greeks."""
        return BSMResult(
            price=self.fair_value(),
            delta=self.delta(),
            gamma=self.gamma(),
            vega=self.vega(),
            theta=self.theta(),
            rho=self.rho(),
            vanna=self.vanna(),
            volga=self.volga(),
            charm=self.charm(),
            d1=self._d1,
            d2=self._d2,
        )


# ---------------------------------------------------------------------------
# Put-Call Parity checker
# ---------------------------------------------------------------------------

def put_call_parity_check(
    call_price: float, put_price: float, S: float, K: float, T: float, r: float, q: float = 0.0
) -> dict:
    """
    Verify put-call parity: C - P = S·e^(-qT) - K·e^(-rT)

    Any deviation indicates an arbitrage opportunity (ignoring transaction costs).
    """
    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    return {
        "lhs (C - P)": round(lhs, 6),
        "rhs (S·e^{-qT} - K·e^{-rT})": round(rhs, 6),
        "difference": round(lhs - rhs, 8),
        "parity_holds": abs(lhs - rhs) < 1e-6,
    }


# ---------------------------------------------------------------------------
# Sensitivity surface (vectorised)
# ---------------------------------------------------------------------------

def greeks_surface(
    S_range: np.ndarray,
    sigma_range: np.ndarray,
    K: float, T: float, r: float, q: float = 0.0, option_type: str = "call",
) -> dict[str, np.ndarray]:
    """
    Compute price and delta across a grid of (S, σ) values.
    Useful for visualising the option's sensitivity landscape.

    Returns dict of 2-D arrays: 'price' and 'delta', shape (len(S_range), len(sigma_range)).
    """
    prices = np.zeros((len(S_range), len(sigma_range)))
    deltas = np.zeros_like(prices)

    for i, S in enumerate(S_range):
        for j, sigma in enumerate(sigma_range):
            params = BSMInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
            m = BlackScholesMerton(params)
            prices[i, j] = m.fair_value()
            deltas[i, j] = m.delta()

    return {"price": prices, "delta": deltas}


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n── Example 1: ATM 1-year call (Hull Example 15.6) ──")
    params = BSMInputs(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
    model = BlackScholesMerton(params)
    result = model.price()
    print(result.summary())

    print("\n── Example 2: ATM put with same params ──")
    params_put = BSMInputs(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="put")
    result_put = BlackScholesMerton(params_put).price()
    print(result_put.summary())

    print("\n── Put-Call Parity Check ──")
    pcp = put_call_parity_check(result.price, result_put.price, S=42, K=40, T=0.5, r=0.10)
    for k, v in pcp.items():
        print(f"  {k}: {v}")

    print("\n── Example 3: Implied Volatility Recovery ──")
    market_price = 4.76   # hypothetical observed market price
    iv = model.implied_vol(market_price)
    print(f"  Market price: {market_price}")
    print(f"  Implied vol:  {iv:.4%}")
