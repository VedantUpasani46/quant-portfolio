"""
Binomial and Trinomial Tree Option Pricers
============================================
Lattice methods for pricing European and American options.

Why trees?
  The Black-Scholes formula prices EUROPEAN options only (exercise at expiry).
  AMERICAN options (exercise at any time) have no general closed form.
  Trees solve this by working backwards through a discrete-time lattice,
  checking at each node whether early exercise is optimal.

  American put ≥ European put (early exercise can be optimal when deeply ITM)
  American call = European call (for non-dividend-paying stocks)
  American call > European call (when there are dividends)

CRR Binomial Tree (Cox, Ross & Rubinstein 1979):
  At each step of size dt = T/N:
    u = e^(σ√dt)    (up factor)
    d = 1/u         (down factor, ensures recombination)
    p = (e^((r-q)dt) - d) / (u - d)   (risk-neutral probability of up move)

  Terminal stock prices form a recombining binomial lattice:
    S_{i,j} = S₀ · u^j · d^(i-j)   at step i with j up-moves

  Backward induction:
    At expiry T: V_{N,j} = payoff(S_{N,j})
    Before T:   V_{i,j} = max(early_exercise, e^(-r·dt)·[p·V_{i+1,j+1} + (1-p)·V_{i+1,j}])

  The second argument in max() is the early exercise value:
    Call: max(S_{i,j} - K, 0)
    Put:  max(K - S_{i,j}, 0)

Trinomial Tree (Boyle 1988):
  Three branches at each node: up, middle, down.
  Middle branch allows staying at the same price level.
  More accurate per step than binomial — equivalent to a finer binomial grid.

    u = e^(λσ√dt),  d = 1/u,  m = 1  (λ typically √3 for stability)
    p_u = [(r-q-σ²/2)dt/σ√dt + 1/λ²]/2 + λ²/(6λ²)
    Wait — standard Kamrad-Ritchken (1991) formulation used here.

Convergence:
  Binomial: O(1/N) convergence to BSM (with oscillations for American)
  Trinomial: smoother convergence, comparable accuracy at ~N/2 steps

References:
  - Cox, J.C., Ross, S.A. & Rubinstein, M. (1979). Option Pricing: A Simplified Approach.
    Journal of Financial Economics, 7(3), 229–263.
  - Boyle, P.P. (1988). A Lattice Framework for Option Pricing. JFQA, 23(1), 1–12.
  - Kamrad, B. & Ritchken, P. (1991). Multinomial Approximating Models for Options
    with k State Variables. Management Science, 37(12), 1640–1652.
  - Hull, J.C. (2022). Options, Futures and Other Derivatives, Ch. 21.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TreeResult:
    """
    Result from a lattice option pricing calculation.

    Attributes
    ----------
    price : float           Option fair value.
    delta : float           Delta (∂V/∂S), estimated from the tree.
    gamma : float           Gamma (∂²V/∂S²), estimated from the tree.
    theta : float           Theta (∂V/∂t per day), estimated from the tree.
    tree_type : str         'binomial' or 'trinomial'.
    n_steps : int           Number of time steps used.
    early_exercise_flag : bool  True if at least one node exercised early.
    """
    price: float
    delta: float
    gamma: float
    theta: float
    tree_type: str
    n_steps: int
    early_exercise_flag: bool

    def summary(self) -> str:
        return (
            f"  Tree type : {self.tree_type} ({self.n_steps} steps)\n"
            f"  Price     : {self.price:.6f}\n"
            f"  Delta     : {self.delta:.6f}\n"
            f"  Gamma     : {self.gamma:.6f}\n"
            f"  Theta/day : {self.theta:.6f}\n"
            f"  Early exercise occurred: {self.early_exercise_flag}"
        )


# ---------------------------------------------------------------------------
# Payoff functions
# ---------------------------------------------------------------------------

def call_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(S - K, 0.0)

def put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S, 0.0)


# ---------------------------------------------------------------------------
# CRR Binomial Tree
# ---------------------------------------------------------------------------

class CRRBinomialTree:
    """
    Cox-Ross-Rubinstein (1979) binomial tree for European and American options.

    Recombining lattice: at step i with j up-moves, S_{i,j} = S₀·u^j·d^(i-j).
    This ensures the tree recombines (S·u·d = S), limiting nodes to O(N²).

    Usage
    -----
    >>> tree = CRRBinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.20, n_steps=500)
    >>> result = tree.price(option_type='put', american=True)
    >>> print(result.summary())
    """

    def __init__(
        self,
        S: float, K: float, T: float,
        r: float, sigma: float,
        q: float = 0.0,
        n_steps: int = 500,
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.N = n_steps

        # CRR parameters
        self.dt = T / n_steps
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = 1.0 / self.u                          # ensures recombination
        self.discount = math.exp(-r * self.dt)
        # Risk-neutral up-probability
        self.p = (math.exp((r - q) * self.dt) - self.d) / (self.u - self.d)

        if not (0 < self.p < 1):
            raise ValueError(
                f"Risk-neutral probability p={self.p:.4f} outside (0,1). "
                f"Check that σ²·dt > (r-q)²·dt² (stability condition)."
            )

    def price(
        self,
        option_type: Literal["call", "put"] = "put",
        american: bool = True,
    ) -> TreeResult:
        """
        Price the option via backward induction through the binomial tree.

        Parameters
        ----------
        option_type : 'call' or 'put'
        american : bool
            True → American (check early exercise at each node)
            False → European (no early exercise)

        Returns
        -------
        TreeResult with price and Greeks estimated from the tree.
        """
        N = self.N
        payoff_fn = call_payoff if option_type == "call" else put_payoff

        # ── Terminal stock prices ─────────────────────────────────────
        # S_{N,j} = S₀ · u^j · d^(N-j) = S₀ · d^N · (u/d)^j
        # More numerically stable using powers of u and d
        S_T = self.S * self.d ** N * (self.u / self.d) ** np.arange(N + 1)

        # ── Terminal option values ────────────────────────────────────
        V = payoff_fn(S_T, self.K)

        # ── Backward induction ────────────────────────────────────────
        early_exercise_flag = False
        p, q_prob = self.p, 1.0 - self.p

        for i in range(N - 1, -1, -1):
            # Stock prices at step i: S_{i,j} = S₀ · u^j · d^(i-j)
            S_i = self.S * self.d ** i * (self.u / self.d) ** np.arange(i + 1)

            # Continuation value (discounted expected payoff)
            V_cont = self.discount * (p * V[1:i + 2] + q_prob * V[:i + 1])

            if american:
                # Early exercise value
                V_ex = payoff_fn(S_i, self.K)
                # Hold the option only if continuation > exercise
                V_new = np.where(V_ex > V_cont, V_ex, V_cont)
                if np.any(V_ex > V_cont):
                    early_exercise_flag = True
                V = V_new
            else:
                V = V_cont

        option_price = float(V[0])

        # ── Greeks from tree (estimated at root nodes) ─────────────
        # One step from root: prices S·u and S·d
        S_u = self.S * self.u
        S_d = self.S * self.d
        # Two steps from root (for gamma)
        S_uu = self.S * self.u ** 2
        S_ud = self.S            # S·u·d = S (recombining!)
        S_dd = self.S * self.d ** 2

        # We need V at nodes 1 and 2 steps from root
        # Re-run a short 2-step tree for the same option
        if N >= 2:
            # 1-step values (V_u at S·u, V_d at S·d)
            V2_term = payoff_fn(
                np.array([S_uu, S_ud, S_dd]), self.K
            )
            V1_u_cont = self.discount * (self.p * V2_term[0] + (1 - self.p) * V2_term[1])
            V1_d_cont = self.discount * (self.p * V2_term[1] + (1 - self.p) * V2_term[2])

            if american:
                V1_u = max(V1_u_cont, payoff_fn(np.array([S_u]), self.K)[0])
                V1_d = max(V1_d_cont, payoff_fn(np.array([S_d]), self.K)[0])
            else:
                V1_u, V1_d = V1_u_cont, V1_d_cont

            dS = S_u - S_d
            delta = (V1_u - V1_d) / dS

            # Gamma: second derivative at the root
            V0_approx = self.discount * (self.p * V1_u + (1 - self.p) * V1_d)
            h = 0.5 * (S_uu - S_dd)
            gamma = ((V1_u - V0_approx) / (S_u - self.S) -
                     (V0_approx - V1_d) / (self.S - S_d)) / (0.5 * (S_u - S_d))

            # Theta: price difference per calendar day (dt in years)
            theta_per_year = (V0_approx - option_price) / self.dt
            theta_per_day = theta_per_year / 365.0
        else:
            delta = gamma = theta_per_day = float("nan")

        return TreeResult(
            price=option_price,
            delta=delta,
            gamma=gamma,
            theta=theta_per_day,
            tree_type="CRR Binomial",
            n_steps=self.N,
            early_exercise_flag=early_exercise_flag,
        )


# ---------------------------------------------------------------------------
# Trinomial Tree (Kamrad-Ritchken)
# ---------------------------------------------------------------------------

class TrinomialTree:
    """
    Kamrad-Ritchken (1991) trinomial tree for European and American options.

    Three branches at each node:
      Up:     S → S·u   with probability p_u
      Middle: S → S     with probability p_m
      Down:   S → S/u   with probability p_d

    The stretch parameter λ (default √3) ensures p_u, p_m, p_d ∈ (0,1).

    Trinomial trees:
      - Converge faster than binomial for the same number of steps
      - Equivalent to an explicit finite-difference scheme
      - Preferred for barrier options (place barrier on the lattice)
      - Used for American options in many commercial systems

    Kamrad-Ritchken probabilities:
      u = exp(λ·σ·√dt)
      p_u = 1/(2λ²) + (r - q - σ²/2)·√dt / (2λσ)
      p_d = 1/(2λ²) - (r - q - σ²/2)·√dt / (2λσ)
      p_m = 1 - 1/λ²
    """

    def __init__(
        self,
        S: float, K: float, T: float,
        r: float, sigma: float,
        q: float = 0.0,
        n_steps: int = 300,
        lam: float = math.sqrt(3),   # stretch parameter
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.N = n_steps
        self.lam = lam

        dt = T / n_steps
        self.dt = dt
        self.u = math.exp(lam * sigma * math.sqrt(dt))
        self.d = 1.0 / self.u
        self.discount = math.exp(-r * dt)

        # Risk-neutral probabilities (Kamrad-Ritchken)
        drift_term = (r - q - 0.5 * sigma ** 2) * math.sqrt(dt) / (lam * sigma)
        self.p_u = 1.0 / (2 * lam ** 2) + drift_term / 2
        self.p_d = 1.0 / (2 * lam ** 2) - drift_term / 2
        self.p_m = 1.0 - 1.0 / lam ** 2

        if not (0 < self.p_u < 1 and 0 < self.p_d < 1 and 0 < self.p_m < 1):
            raise ValueError(
                f"Trinomial probabilities out of (0,1): "
                f"p_u={self.p_u:.4f}, p_m={self.p_m:.4f}, p_d={self.p_d:.4f}"
            )

    def price(
        self,
        option_type: Literal["call", "put"] = "put",
        american: bool = True,
    ) -> TreeResult:
        """
        Price via backward induction on the trinomial tree.

        The trinomial tree at step i has 2i+1 nodes.
        Stock price at node j (j ∈ {-i, ..., i}) is S₀·u^j.
        """
        N = self.N
        payoff_fn = call_payoff if option_type == "call" else put_payoff

        # Terminal nodes: 2N+1 nodes, indexed from -N to +N
        n_terminal = 2 * N + 1
        j_range = np.arange(-N, N + 1)
        S_T = self.S * (self.u ** j_range)
        V = payoff_fn(S_T, self.K)

        early_exercise_flag = False
        p_u, p_m, p_d = self.p_u, self.p_m, self.p_d
        V_step1 = None  # store values at step i=1 for Greeks

        # Backward induction: at step i, there are 2i+1 nodes
        for i in range(N - 1, -1, -1):
            n_nodes = 2 * i + 1
            j_i = np.arange(-i, i + 1)
            S_i = self.S * (self.u ** j_i)

            # Continuation: V[j] comes from V_up[j+1], V_mid[j], V_down[j-1]
            # In the current V array (length 2(i+1)+1 = 2i+3):
            # node j at step i links to j+1, j, j-1 at step i+1
            # Offset: at step i+1, node j has index j + (i+1) in V array
            idx_up = np.arange(2, 2 * n_nodes + 2, 2)    # ... wait, cleaner:
            # At step i, node j_i[k] links upward to index (j_i[k]+1) + (i+1) = j_i[k]+i+2
            # links middle to j_i[k] + (i+1) = j_i[k]+i+1
            # links downward to j_i[k]-1 + (i+1) = j_i[k]+i

            k = np.arange(n_nodes)
            idx_up_arr   = (j_i + 1) + (i + 1)   # = k + 2
            idx_mid_arr  = j_i + (i + 1)           # = k + 1
            idx_down_arr = (j_i - 1) + (i + 1)    # = k

            V_cont = self.discount * (
                p_u * V[idx_up_arr] +
                p_m * V[idx_mid_arr] +
                p_d * V[idx_down_arr]
            )

            if american:
                V_ex = payoff_fn(S_i, self.K)
                V_new = np.where(V_ex > V_cont, V_ex, V_cont)
                if np.any(V_ex > V_cont):
                    early_exercise_flag = True
                V = V_new
            else:
                V = V_cont
            if i == 1:
                V_step1 = V.copy()

        option_price = float(V[0])

        # Greeks: finite difference on root values
        if N >= 1:
            # Use second step from root
            V1_up   = V_step1[2] if V_step1 is not None and len(V_step1) > 2 else float("nan")
            V1_down = V_step1[0] if V_step1 is not None and len(V_step1) > 2 else float("nan")
            S_up = self.S * self.u
            S_down = self.S * self.d
            delta = (V1_up - V1_down) / (S_up - S_down)

            # Simple gamma approximation
            gamma_approx = (V1_up + V1_down - 2 * option_price) / ((S_up - self.S) ** 2)
            # Theta: small dt forward
            theta = -(p_u * V1_up + p_m * option_price + p_d * V1_down - option_price) / (self.dt * 365)
        else:
            delta = gamma_approx = theta = float("nan")

        return TreeResult(
            price=option_price,
            delta=delta,
            gamma=gamma_approx,
            theta=theta,
            tree_type="Trinomial (Kamrad-Ritchken)",
            n_steps=self.N,
            early_exercise_flag=early_exercise_flag,
        )


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

def convergence_study(
    S: float, K: float, T: float, r: float, sigma: float, q: float,
    option_type: str, american: bool,
    step_counts: list[int],
    bsm_price: float | None = None,
) -> None:
    """
    Show how binomial and trinomial prices converge as N increases.
    Compares to BSM (European) or known benchmark (American).
    """
    print(f"\n  {'N':>6} {'Binomial':>12} {'Trinomial':>12}", end="")
    if bsm_price is not None:
        print(f"  {'BSM':>10}  {'Bin Error':>10}  {'Tri Error':>10}")
    else:
        print()
    print("  " + "─" * (70 if bsm_price else 35))

    for N in step_counts:
        try:
            bin_tree = CRRBinomialTree(S, K, T, r, sigma, q, n_steps=N)
            bin_res = bin_tree.price(option_type, american)
            bin_p = bin_res.price

            tri_tree = TrinomialTree(S, K, T, r, sigma, q, n_steps=max(N // 2, 10))
            tri_res = tri_tree.price(option_type, american)
            tri_p = tri_res.price

            line = f"  {N:>6} {bin_p:>12.6f} {tri_p:>12.6f}"
            if bsm_price is not None:
                line += (f"  {bsm_price:>10.6f}"
                         f"  {abs(bin_p - bsm_price):>10.6f}"
                         f"  {abs(tri_p - bsm_price):>10.6f}")
            print(line)
        except Exception as e:
            print(f"  {N:>6}  Error: {e}")


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from scipy.stats import norm as sp_norm

    def bsm_european(S, K, T, r, sigma, q=0.0, option_type="call"):
        d1 = (math.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        if option_type == "call":
            return S*math.exp(-q*T)*sp_norm.cdf(d1) - K*math.exp(-r*T)*sp_norm.cdf(d2)
        return K*math.exp(-r*T)*sp_norm.cdf(-d2) - S*math.exp(-q*T)*sp_norm.cdf(-d1)

    S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.0

    print("═" * 64)
    print("  Binomial & Trinomial Tree Option Pricers")
    print("═" * 64)

    # ── European call: compare to BSM ──────────────────────────────
    bsm_call = bsm_european(S, K, T, r, sigma, q, "call")
    bsm_put  = bsm_european(S, K, T, r, sigma, q, "put")

    print(f"\n  Benchmark: BSM European Call = {bsm_call:.6f}  Put = {bsm_put:.6f}")

    bin_eu = CRRBinomialTree(S, K, T, r, sigma, q, n_steps=1000).price("call", american=False)
    tri_eu = TrinomialTree(S, K, T, r, sigma, q, n_steps=500).price("call", american=False)

    print(f"\n── European Call (N=1000 binomial / N=500 trinomial) ──")
    print(f"\n  Binomial:")
    print(bin_eu.summary())
    print(f"\n  Trinomial:")
    print(tri_eu.summary())
    print(f"\n  BSM Call:    {bsm_call:.6f}")
    print(f"  Bin Error:   {abs(bin_eu.price - bsm_call):.2e}")
    print(f"  Tri Error:   {abs(tri_eu.price - bsm_call):.2e}")

    # ── American put: no closed form, compare bin vs tri ──────────
    bin_am_put = CRRBinomialTree(S, K, T, r, sigma, q, n_steps=1000).price("put", american=True)
    tri_am_put = TrinomialTree(S, K, T, r, sigma, q, n_steps=500).price("put", american=True)

    print(f"\n── American Put (S=K=100, T=1Y, r=5%, σ=20%) ──")
    print(f"\n  Binomial:")
    print(bin_am_put.summary())
    print(f"\n  Trinomial:")
    print(tri_am_put.summary())
    print(f"\n  Early exercise premium: {bin_am_put.price - bsm_put:.6f}")
    print(f"  (American put > European put by this amount)")

    # ── Convergence study ─────────────────────────────────────────
    print(f"\n── European Call Convergence (comparing to BSM) ──")
    convergence_study(S, K, T, r, sigma, q, "call", american=False,
                      step_counts=[10, 25, 50, 100, 250, 500, 1000],
                      bsm_price=bsm_call)

    print(f"\n── American Put Convergence (bin vs tri) ──")
    convergence_study(S, K, T, r, sigma, q, "put", american=True,
                      step_counts=[50, 100, 200, 500, 1000])

    # ── Deep ITM American put — early exercise boundary ──────────
    print(f"\n── Deep ITM American Put (S=70, K=100) — should exercise immediately ──")
    deep_itm = CRRBinomialTree(70, 100, T, r, sigma, q, n_steps=500)
    result_deep = deep_itm.price("put", american=True)
    intrinsic = max(100 - 70, 0)
    print(f"  Intrinsic value:  {intrinsic:.4f}")
    print(f"  American put:     {result_deep.price:.4f}")
    print(f"  European put:     {bsm_european(70, 100, T, r, sigma, q, 'put'):.4f}")
    print(f"  Early exercise occurred in tree: {result_deep.early_exercise_flag}")
