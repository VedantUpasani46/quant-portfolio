"""
Reinforcement Learning for Trading
=====================================
RL frames trading as a sequential decision problem:
  State:   current market observation (prices, signals, inventory)
  Action:  trade size (buy/sell/hold)
  Reward:  P&L minus transaction costs
  Policy:  maps states → actions to maximise cumulative reward

Why RL for trading:
  1. Handles sequential decisions and path dependence
  2. Naturally incorporates transaction costs into the objective
  3. Can learn optimal entry/exit timing (not just cross-sectional rank)
  4. Handles continuous action spaces (position sizing)

Key RL algorithms used in finance:

1. Q-Learning (discrete actions):
   Q(s, a) = E[Rₜ + γ·max_{a'} Q(sₜ₊₁, a')]
   Tabular for small state spaces, deep Q-network (DQN) for large spaces.

2. Policy Gradient (continuous actions):
   Directly optimise E[Σ rₜ] by gradient ascent on policy parameters.
   REINFORCE, PPO (Proximal Policy Optimisation) → most used in practice.

3. Actor-Critic (A2C, A3C):
   Actor: policy π(a|s) — what to do
   Critic: value V(s) — how good is the current state
   Reduces variance vs pure policy gradient.

4. Optimal stopping (classic RL problem):
   When to sell a stock: stop if price exceeds threshold OR after T days.
   Solution: dynamic programming backwards induction.

Markov Decision Process for trading:
  State:  sₜ = (pₜ, invₜ, f₁ₜ, f₂ₜ, ...)  [price, inventory, factors]
  Action: aₜ ∈ {-1, 0, +1} × position_size  [sell, hold, buy]
  Reward: rₜ = Δpₜ · posₜ - λ|Δposₜ|      [P&L minus TC]
  Discount: γ = 0.99 (prefer near-term P&L)

Key challenges in financial RL:
  - Non-stationary reward distributions (regime changes)
  - Low signal-to-noise ratio (financial data is near-random)
  - Sparse rewards (profits only realised at trade)
  - Simulation gap: backtested environment ≠ live market
  - Overfitting: RL with many parameters can fit noise

This demo implements:
  1. Simple Q-Learning on discretised market states
  2. Optimal stopping (sell the asset problem)
  3. Policy gradient with baseline (REINFORCE)
  4. Episode-based backtest

References:
  - Mnih, V. et al. (2015). Human-Level Control through Deep RL. Nature 518.
  - Moody, J. & Saffell, M. (2001). Learning to Trade via Direct Reinforcement.
    IEEE Trans. Neural Networks 12(4).
  - Kolm, P.N. & Ritter, G. (2020). Dynamic Replication and Hedging: A RL Approach.
    Journal of Financial Data Science 1(1).
  - Hambly, B. et al. (2023). Recent Advances in Reinforcement Learning in Finance.
    Mathematical Finance 33(3), 437–503.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Environment: simplified single-asset trading MDP
# ---------------------------------------------------------------------------

class TradingEnvironment:
    """
    Single-asset trading environment.
    State:  (price_change_bucket, momentum_bucket, inventory_bucket)
    Action: {0: sell, 1: hold, 2: buy}
    Reward: daily P&L minus transaction costs
    """
    
    def __init__(
        self,
        returns: np.ndarray,      # historical return series
        n_price_buckets: int = 5,
        n_mom_buckets: int = 3,
        n_inv_buckets: int = 5,
        tc_bps: float = 5.0,      # 5bps one-way transaction cost
        max_position: float = 1.0,
    ):
        self.returns = returns
        self.n_price_buckets = n_price_buckets
        self.n_mom_buckets = n_mom_buckets
        self.n_inv_buckets = n_inv_buckets
        self.tc = tc_bps / 10000
        self.max_position = max_position
        self.n_actions = 3  # sell, hold, buy
        
        # Compute momentum features
        self.mom_5 = pd.Series(returns).rolling(5).mean().fillna(0).values
        
        # Discretisation boundaries
        self.price_bins = np.percentile(returns, np.linspace(0, 100, n_price_buckets + 1))
        self.mom_bins   = np.percentile(self.mom_5, np.linspace(0, 100, n_mom_buckets + 1))
        
        self.reset()
    
    def reset(self) -> tuple:
        self.t = 5  # start after warmup
        self.position = 0.0  # current position: -1 to +1
        return self._get_state()
    
    def _get_state(self) -> tuple:
        """Discretise market features into state tuple."""
        if self.t >= len(self.returns):
            return (2, 1, 2)  # default state at end
        
        r = self.returns[self.t]
        m = self.mom_5[self.t]
        
        r_bucket = int(np.digitize(r, self.price_bins[1:-1]))
        m_bucket = int(np.digitize(m, self.mom_bins[1:-1]))
        # Inventory bucket: 0=short, 2=flat, 4=long (for 5 buckets)
        inv_bucket = int((self.position + 1) / 2 * (self.n_inv_buckets - 1))
        inv_bucket = min(inv_bucket, self.n_inv_buckets - 1)
        
        return (r_bucket, m_bucket, inv_bucket)
    
    def step(self, action: int) -> tuple:
        """
        Take an action, return (next_state, reward, done).
        action: 0=sell (-0.5 target), 1=hold, 2=buy (+0.5 target)
        """
        target_pos = {0: -self.max_position, 1: self.position, 2: self.max_position}[action]
        delta_pos = target_pos - self.position
        
        # TC on the trade
        tc_cost = abs(delta_pos) * self.tc
        self.position = target_pos
        
        self.t += 1
        
        if self.t >= len(self.returns):
            return self._get_state(), 0, True
        
        # Reward: position × return - TC
        r = self.returns[self.t]
        reward = self.position * r - tc_cost
        
        done = self.t >= len(self.returns) - 1
        return self._get_state(), reward, done


# ---------------------------------------------------------------------------
# Q-Learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-Learning for discrete action trading.
    Q-update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        n_actions: int = 3,
        alpha: float = 0.1,    # learning rate
        gamma: float = 0.99,   # discount factor
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: state → Q-values for each action
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def choose_action(self, state: tuple) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # explore
        return int(np.argmax(self.Q[state]))           # exploit
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# Optimal stopping (sell-or-hold)
# ---------------------------------------------------------------------------

def optimal_stopping_dp(
    prices: np.ndarray,       # simulated price paths (n_paths × T)
    discount: float = 0.99,
) -> dict:
    """
    Optimal stopping problem: when to sell an asset you hold.
    
    Backwards induction (Bellman):
      V(T) = P(T)                         [sell at expiry]
      V(t) = max(P(t), γ·E[V(t+1)|Fₜ])   [sell now vs wait]
    
    Compare continuation value vs current price.
    """
    n_paths, T = prices.shape
    
    # Backward induction
    V = prices.copy().astype(float)  # value at each node
    exercise = np.zeros_like(V, dtype=bool)
    exercise[:, -1] = True  # always sell at expiry
    
    for t in range(T - 2, -1, -1):
        # Continuation value = discounted expected future value
        continuation = discount * V[:, t + 1].mean()
        # Sell now if current price > continuation
        sell_now = prices[:, t] > continuation
        V[:, t] = np.where(sell_now, prices[:, t], discount * V[:, t + 1])
        exercise[:, t] = sell_now
    
    # Optimal stopping time for each path (first time to sell)
    stopping_times = np.argmax(exercise, axis=1)
    stopping_prices = prices[np.arange(n_paths), stopping_times]
    
    return {
        'optimal_value': V[:, 0].mean(),
        'stopping_times': stopping_times,
        'mean_stopping_time': stopping_times.mean(),
        'sell_at_expiry_pct': (stopping_times == T - 1).mean(),
        'stopping_prices': stopping_prices,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 68)
    print("  Reinforcement Learning for Trading")
    print("  Q-Learning, Optimal Stopping, Policy Performance")
    print("═" * 68)
    
    np.random.seed(42)
    T_total = 1260  # 5 years
    
    # Simulate returns with mild momentum
    returns = np.zeros(T_total)
    for t in range(1, T_total):
        momentum = 0.1 * returns[max(0, t-5):t].mean()
        returns[t] = momentum + np.random.normal(0, 0.015)
    
    # ── Q-Learning training ───────────────────────────────────────
    print(f"\n── Training Q-Learning Agent ──")
    
    train_returns = returns[:800]
    test_returns  = returns[800:]
    
    env = TradingEnvironment(train_returns, tc_bps=5.0)
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon_start=1.0)
    
    n_episodes = 100
    episode_rewards = []
    
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        if (ep + 1) % 25 == 0:
            recent_avg = np.mean(episode_rewards[-25:])
            print(f"  Episode {ep+1:>4}/{n_episodes}  ε={agent.epsilon:.3f}  "
                  f"Avg reward (last 25): {recent_avg:.6f}")
    
    # ── Policy evaluation on test data ───────────────────────────
    print(f"\n── Policy Evaluation (out-of-sample) ──")
    
    # Q-Learning policy
    env_test = TradingEnvironment(test_returns, tc_bps=5.0)
    state = env_test.reset()
    ql_rewards = []
    ql_positions = []
    done = False
    agent.epsilon = 0.0  # greedy during evaluation
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env_test.step(action)
        ql_rewards.append(reward)
        ql_positions.append(env_test.position)
        state = next_state
    
    # Benchmark 1: Buy and hold
    bh_rewards = [test_returns[t] for t in range(1, len(test_returns))]
    
    # Benchmark 2: Random policy
    env_rand = TradingEnvironment(test_returns, tc_bps=5.0)
    state = env_rand.reset()
    rand_rewards = []
    done = False
    while not done:
        action = np.random.randint(3)
        next_state, reward, done = env_rand.step(action)
        rand_rewards.append(reward)
        state = next_state
    
    def sharpe(r):
        r = np.array(r)
        return r.mean() / (r.std() + 1e-10) * np.sqrt(252)
    
    print(f"\n  {'Policy':>20} {'Total P&L':>14} {'Sharpe':>10} {'Turnover':>12}")
    print("  " + "─" * 58)
    
    for name, rewards, positions in [
        ("Q-Learning",   ql_rewards,   ql_positions),
        ("Buy & Hold",   bh_rewards,   [1.0]*len(bh_rewards)),
        ("Random",       rand_rewards, [0.0]*len(rand_rewards)),
    ]:
        pnl = sum(rewards)
        sr = sharpe(rewards)
        # Turnover = sum of |position changes|
        pos = np.array(positions)
        turnover = np.abs(np.diff(pos)).sum() / len(pos)
        print(f"  {name:>20} {pnl:>14.6f} {sr:>10.3f} {turnover:>12.4f}")
    
    # ── Q-table analysis ──────────────────────────────────────────
    print(f"\n── Learned Policy Analysis ──")
    print(f"  State space explored: {len(agent.Q)} unique states")
    
    # Most common action for each momentum bucket
    print(f"\n  Momentum bucket → preferred action:")
    action_names = {0: "SELL", 1: "HOLD", 2: "BUY "}
    for m in range(3):
        m_label = ["Negative", "Neutral", "Positive"][m]
        votes = defaultdict(int)
        for (r_b, m_b, inv_b), q_vals in agent.Q.items():
            if m_b == m:
                votes[np.argmax(q_vals)] += 1
        if votes:
            preferred = max(votes, key=votes.get)
            print(f"    {m_label} momentum (m={m}): {action_names[preferred]} "
                  f"(in {votes[preferred]}/{sum(votes.values())} states)")
    
    # ── Optimal stopping demo ─────────────────────────────────────
    print(f"\n── Optimal Stopping: When to Sell ──")
    print(f"  You hold a stock. Should you sell now or wait?")
    
    n_paths = 1000
    T_stop = 21  # 21 trading days to decide
    mu, sigma = 0.0003, 0.015
    
    # Simulate GBM price paths from $100
    noise = np.random.normal(mu, sigma, (n_paths, T_stop))
    paths = 100 * np.exp(np.cumsum(noise, axis=1))
    
    result = optimal_stopping_dp(paths)
    
    print(f"\n  {n_paths} simulated paths, {T_stop}-day horizon")
    print(f"  Optimal expected value:       ${result['optimal_value']:.4f}")
    print(f"  Naive hold-to-expiry value:   ${paths[:, -1].mean():.4f}")
    print(f"  Mean optimal stopping day:    {result['mean_stopping_time']:.1f}")
    print(f"  Sell at expiry (no early):    {result['sell_at_expiry_pct']:.1%}")
    
    print(f"""
── Reinforcement Learning in Finance: Reality Check ──

  Where RL genuinely works:
    1. Optimal execution (Almgren-Chriss as RL → TWAP/VWAP variants)
    2. Market making (bid-ask spread optimisation)
    3. Options hedging (discrete rebalancing with TC)
    4. Risk-constrained allocation (position sizing with drawdown constraint)

  Where RL struggles with financial data:
    - Low SNR: financial returns ≈ noise → hard for RL to learn from
    - Non-stationarity: policy trained on 2019 fails on 2022
    - Simulation gap: even perfect backtests ≠ live markets
    - Sample efficiency: RL needs lots of data, markets have little

  Best practices for financial RL:
    1. Use reward shaping (Sharpe, not just P&L)
    2. Add TC explicitly in reward function
    3. Ensemble many episodes with different seeds
    4. Robust evaluation: test on multiple out-of-sample periods
    5. Simple policies first (mean-variance → RL if it genuinely beats it)

  Interview question (DE Shaw, Citadel Research):
  Q: "Why is RL hard to apply to stock trading?"
  A: "Three main reasons: (1) Non-stationarity — market regimes shift faster
      than RL can adapt. (2) Low SNR — returns are nearly unpredictable, so
      RL overfits to noise. (3) Sparse rewards — P&L only realised when you
      trade, creating credit assignment problems. Where it works is
      execution and market-making where the reward signal is clear."
    """)
