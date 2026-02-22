"""
Event-Driven Backtesting Framework
=====================================
A modular, reusable backtesting engine with:
  - Event-driven architecture (avoids lookahead bias)
  - Realistic transaction cost and slippage modelling
  - Position sizing (Kelly criterion, fixed fractional, volatility-targeted)
  - Full performance analytics (Sharpe, Sortino, Calmar, max drawdown, etc.)
  - Walk-forward validation support

Why event-driven?
  Vector-based backtests are fast but prone to lookahead bias —
  accidentally using future prices to make past decisions.
  Event-driven processing enforces strict time ordering: each decision
  is made only with information available at that moment.

Architecture:
  DataHandler → Signal → OrderManager → Portfolio → PerformanceAnalytics
                                ↑
                    (no information flows backward)

References:
  - Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
    Ch. 11 (Backtesting through Cross-Validation), Ch. 14 (Backtest Statistics)
  - Chan, E.P. (2013). Algorithmic Trading. Wiley. Ch. 3.
  - Sharpe, W.F. (1994). The Sharpe Ratio. JPM 21(1), 49–58.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class MarketEvent:
    """New price data available."""
    timestamp: pd.Timestamp
    prices: dict[str, float]  # {ticker: close_price}
    volumes: dict[str, float] = field(default_factory=dict)


@dataclass
class SignalEvent:
    """Trading signal from a strategy."""
    timestamp: pd.Timestamp
    ticker: str
    direction: Literal["LONG", "SHORT", "FLAT"]
    strength: float = 1.0   # signal strength 0–1; used for position sizing


@dataclass
class OrderEvent:
    """Order to be sent to the execution engine."""
    timestamp: pd.Timestamp
    ticker: str
    order_type: Literal["MARKET", "LIMIT"]
    quantity: float          # positive = buy, negative = sell
    price: Optional[float] = None  # for limit orders


@dataclass
class FillEvent:
    """Confirmation of executed order."""
    timestamp: pd.Timestamp
    ticker: str
    quantity: float
    fill_price: float
    commission: float


# ---------------------------------------------------------------------------
# Abstract strategy interface
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """
    Abstract base class for all strategies.

    Every strategy must implement `generate_signals()` which takes
    the current market data and returns a list of SignalEvents.

    The strategy MUST NOT access any data beyond the current timestamp
    — enforced by the framework architecture.
    """

    @abstractmethod
    def generate_signals(self, event: MarketEvent, portfolio: "Portfolio") -> list[SignalEvent]:
        """
        Given new market data, decide which signals to generate.

        Parameters
        ----------
        event : MarketEvent    Current prices (no future data).
        portfolio : Portfolio  Current portfolio state.

        Returns
        -------
        list of SignalEvent (may be empty if no signals today).
        """
        pass


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Position sizing methods.

    Kelly Criterion: maximises expected log-wealth.
      f* = (μ - rf) / σ²  (continuous Kelly)
      In practice, use half-Kelly for robustness.

    Volatility Targeting: sizes positions to achieve a target daily vol.
      shares = (portfolio_value * target_vol) / (price * asset_vol)

    Fixed Fractional: risk a fixed fraction of capital per trade.
    """

    @staticmethod
    def volatility_target(
        price: float, portfolio_value: float,
        asset_vol_daily: float, target_vol_annual: float = 0.10,
        ann_factor: int = 252,
    ) -> float:
        """
        Position size to achieve target portfolio volatility.

        shares = (V * σ_target_daily) / (P * σ_asset_daily)
        where σ_target_daily = target_vol_annual / √ann_factor
        """
        if price <= 0 or asset_vol_daily <= 0:
            return 0.0
        target_vol_daily = target_vol_annual / math.sqrt(ann_factor)
        notional = portfolio_value * target_vol_daily / asset_vol_daily
        return notional / price

    @staticmethod
    def half_kelly(
        edge: float, odds: float, portfolio_value: float, price: float
    ) -> float:
        """
        Half-Kelly fraction: f = 0.5 * (edge/odds)
        edge = probability of win - probability of loss
        odds = average win / average loss
        """
        f_kelly = edge / odds
        fraction = 0.5 * f_kelly  # half Kelly for robustness
        return portfolio_value * fraction / price

    @staticmethod
    def fixed_fractional(
        risk_pct: float, portfolio_value: float,
        price: float, stop_distance: float,
    ) -> float:
        """
        Risk risk_pct% of capital, with stop_distance as max loss per share.
        shares = (portfolio_value * risk_pct) / stop_distance
        """
        if stop_distance <= 0:
            return 0.0
        dollar_risk = portfolio_value * risk_pct
        return dollar_risk / stop_distance


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------

class TransactionCostModel:
    """
    Realistic transaction cost modelling.

    Components:
      1. Commission: flat fee per share or per trade
      2. Bid-ask spread: half-spread paid on each trade
      3. Market impact: price moves against you for large orders
         Linear model: impact = market_impact_bps * (order_size / ADV)
    """

    def __init__(
        self,
        commission_per_share: float = 0.005,   # $0.005/share (IB-like)
        bid_ask_spread_bps: float = 5.0,       # 5bps half-spread
        market_impact_bps: float = 2.0,        # linear impact param
    ):
        self.comm = commission_per_share
        self.half_spread = bid_ask_spread_bps / 20000   # half-spread as fraction
        self.impact = market_impact_bps / 10000

    def total_cost(self, price: float, quantity: float,
                   adv: float = 1e6) -> tuple[float, float]:
        """
        Compute fill price and commission for an order.

        Returns (fill_price, commission_dollars)
        """
        # Bid-ask: buy at ask, sell at bid
        spread_adj = self.half_spread if quantity > 0 else -self.half_spread
        # Market impact: scaled by order/ADV ratio
        impact_adj = self.impact * (abs(quantity) * price / adv) * (1 if quantity > 0 else -1)

        fill_price = price * (1 + spread_adj + impact_adj)
        commission = abs(quantity) * self.comm
        return fill_price, commission


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """
    Tracks cash, positions, P&L, and NAV over time.
    """

    def __init__(self, initial_capital: float, cost_model: TransactionCostModel):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, float] = {}       # {ticker: shares}
        self.cost_basis: dict[str, float] = {}      # {ticker: avg_cost}
        self.cost_model = cost_model
        self.nav_history: list[dict] = []
        self.trade_history: list[FillEvent] = []

    def execute_order(self, order: OrderEvent, market_price: float,
                      adv: float = 1e6) -> FillEvent:
        """
        Execute an order, deducting transaction costs from cash.
        Returns a FillEvent confirming the execution.
        """
        fill_price, commission = self.cost_model.total_cost(
            market_price, order.quantity, adv
        )
        trade_value = order.quantity * fill_price + commission

        self.cash -= trade_value

        # Update position
        ticker = order.ticker
        old_pos = self.positions.get(ticker, 0.0)
        new_pos = old_pos + order.quantity

        if abs(new_pos) < 1e-8:
            self.positions.pop(ticker, None)
            self.cost_basis.pop(ticker, None)
        else:
            self.positions[ticker] = new_pos
            # Update cost basis (FIFO approximation)
            if old_pos == 0 or (old_pos > 0) == (order.quantity > 0):
                total_cost = self.cost_basis.get(ticker, 0) * old_pos + fill_price * order.quantity
                self.cost_basis[ticker] = total_cost / new_pos
            else:
                self.cost_basis[ticker] = fill_price

        fill = FillEvent(
            timestamp=order.timestamp,
            ticker=ticker,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
        )
        self.trade_history.append(fill)
        return fill

    def mark_to_market(self, timestamp: pd.Timestamp, prices: dict[str, float]) -> float:
        """Compute current NAV = cash + market value of positions."""
        mkt_value = sum(
            self.positions.get(t, 0) * prices.get(t, 0)
            for t in self.positions
        )
        nav = self.cash + mkt_value
        self.nav_history.append({"timestamp": timestamp, "nav": nav, "cash": self.cash,
                                   "mkt_value": mkt_value})
        return nav

    @property
    def nav_series(self) -> pd.Series:
        if not self.nav_history:
            return pd.Series()
        df = pd.DataFrame(self.nav_history).set_index("timestamp")
        return df["nav"]


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

class PerformanceAnalytics:
    """
    Industry-standard performance and risk metrics.

    All return-based metrics are annualised assuming daily data.
    """

    def __init__(self, nav_series: pd.Series, rf_rate: float = 0.04,
                 ann_factor: int = 252):
        self.nav = nav_series.dropna()
        self.rf = rf_rate
        self.ann = ann_factor
        self.returns = self.nav.pct_change().dropna()

    def total_return(self) -> float:
        return (self.nav.iloc[-1] / self.nav.iloc[0]) - 1

    def annualised_return(self) -> float:
        n = len(self.nav)
        return (1 + self.total_return()) ** (self.ann / n) - 1

    def annualised_volatility(self) -> float:
        return float(self.returns.std() * math.sqrt(self.ann))

    def sharpe_ratio(self) -> float:
        """
        Sharpe = (R_ann - rf) / σ_ann
        The ratio of excess return per unit of total risk.
        """
        excess = self.annualised_return() - self.rf
        vol = self.annualised_volatility()
        return excess / vol if vol > 0 else 0.0

    def sortino_ratio(self) -> float:
        """
        Sortino = (R_ann - rf) / σ_downside
        Only penalises downside volatility (returns below rf/ann_factor).
        """
        daily_rf = (1 + self.rf) ** (1 / self.ann) - 1
        downside = self.returns[self.returns < daily_rf]
        if len(downside) == 0:
            return float("inf")
        downside_vol = float(downside.std() * math.sqrt(self.ann))
        excess = self.annualised_return() - self.rf
        return excess / downside_vol if downside_vol > 0 else 0.0

    def max_drawdown(self) -> float:
        """
        Maximum drawdown: largest peak-to-trough decline.
        MDD = min_t [(NAV_t - max_{s≤t} NAV_s) / max_{s≤t} NAV_s]
        """
        running_max = self.nav.expanding().max()
        drawdowns = (self.nav - running_max) / running_max
        return float(drawdowns.min())

    def calmar_ratio(self) -> float:
        """
        Calmar = Annualised Return / |Max Drawdown|
        Favoured for evaluating risk-adjusted return in drawdown terms.
        """
        mdd = abs(self.max_drawdown())
        return self.annualised_return() / mdd if mdd > 0 else float("inf")

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Omega = E[max(R-L, 0)] / E[max(L-R, 0)]
        where L is the threshold return.
        Generalises Sharpe by capturing all moments of the distribution.
        """
        excess = self.returns - threshold / self.ann
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        return gains / losses if losses > 0 else float("inf")

    def var_95(self) -> float:
        """1-day 95% Value at Risk (historical simulation)."""
        return float(np.percentile(self.returns, 5))

    def cvar_95(self) -> float:
        """1-day 95% Expected Shortfall (CVaR)."""
        q = self.var_95()
        tail = self.returns[self.returns <= q]
        return float(tail.mean()) if len(tail) > 0 else q

    def hit_rate(self) -> float:
        """Fraction of days with positive returns."""
        return float((self.returns > 0).mean())

    def full_report(self) -> str:
        lines = [
            "=" * 52,
            "  Strategy Performance Report",
            "=" * 52,
            f"  {'Total Return':<30} {self.total_return():>10.4%}",
            f"  {'Annualised Return':<30} {self.annualised_return():>10.4%}",
            f"  {'Annualised Volatility':<30} {self.annualised_volatility():>10.4%}",
            "─" * 52,
            f"  {'Sharpe Ratio':<30} {self.sharpe_ratio():>10.4f}",
            f"  {'Sortino Ratio':<30} {self.sortino_ratio():>10.4f}",
            f"  {'Calmar Ratio':<30} {self.calmar_ratio():>10.4f}",
            f"  {'Omega Ratio':<30} {self.omega_ratio():>10.4f}",
            "─" * 52,
            f"  {'Max Drawdown':<30} {self.max_drawdown():>10.4%}",
            f"  {'1-Day 95% VaR':<30} {self.var_95():>10.4%}",
            f"  {'1-Day 95% CVaR':<30} {self.cvar_95():>10.4%}",
            f"  {'Hit Rate':<30} {self.hit_rate():>10.4%}",
            "─" * 52,
            f"  {'Observations':<30} {len(self.returns):>10,}",
            "=" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven backtesting engine.

    Processes market data one day at a time in strict chronological order.
    No future data can leak into strategy decisions.

    Usage
    -----
    >>> strategy = MyStrategy(lookback=20)
    >>> engine = BacktestEngine(strategy, prices_df, initial_capital=1_000_000)
    >>> result = engine.run()
    >>> print(result.full_report())
    """

    def __init__(
        self,
        strategy: Strategy,
        price_data: pd.DataFrame,       # columns = tickers, index = dates
        initial_capital: float = 1_000_000,
        commission_per_share: float = 0.005,
        bid_ask_bps: float = 5.0,
    ):
        self.strategy = strategy
        self.price_data = price_data.copy()
        cost_model = TransactionCostModel(commission_per_share, bid_ask_bps)
        self.portfolio = Portfolio(initial_capital, cost_model)

    def run(self) -> PerformanceAnalytics:
        """
        Run the backtest. Returns a PerformanceAnalytics object.
        """
        tickers = list(self.price_data.columns)

        for timestamp, row in self.price_data.iterrows():
            prices = row.to_dict()
            # Remove NaN prices
            prices = {t: p for t, p in prices.items() if not math.isnan(p)}
            if not prices:
                continue

            # 1. Create market event
            market_event = MarketEvent(timestamp=timestamp, prices=prices)

            # 2. Mark portfolio to market
            self.portfolio.mark_to_market(timestamp, prices)

            # 3. Strategy generates signals
            signals = self.strategy.generate_signals(market_event, self.portfolio)

            # 4. Convert signals to orders
            for signal in signals:
                if signal.ticker not in prices:
                    continue

                price = prices[signal.ticker]
                nav = self.portfolio.nav_series.iloc[-1] if len(self.portfolio.nav_history) > 0 else self.portfolio.initial_capital

                if signal.direction == "FLAT":
                    # Close position
                    current_pos = self.portfolio.positions.get(signal.ticker, 0)
                    if abs(current_pos) > 0:
                        order = OrderEvent(timestamp, signal.ticker, "MARKET", -current_pos)
                        self.portfolio.execute_order(order, price)
                elif signal.direction in ("LONG", "SHORT"):
                    sign = 1 if signal.direction == "LONG" else -1
                    # Simple position sizing: 5% of NAV per position
                    quantity = sign * signal.strength * nav * 0.05 / price
                    order = OrderEvent(timestamp, signal.ticker, "MARKET", quantity)
                    self.portfolio.execute_order(order, price)

        return PerformanceAnalytics(self.portfolio.nav_series)


# ---------------------------------------------------------------------------
# Example strategy: Simple Moving Average Crossover
# ---------------------------------------------------------------------------

class SMACrossover(Strategy):
    """
    Simple Moving Average crossover strategy.

    Signal:
      LONG  when fast_MA crosses above slow_MA (golden cross)
      FLAT  when fast_MA crosses below slow_MA (death cross)

    This is a classic trend-following strategy, simple enough to serve
    as a framework demonstration. In practice, momentum strategies
    require more sophisticated signal processing.
    """

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow
        self._prices: dict[str, list[float]] = {}

    def generate_signals(self, event: MarketEvent, portfolio: "Portfolio") -> list[SignalEvent]:
        signals = []
        for ticker, price in event.prices.items():
            if ticker not in self._prices:
                self._prices[ticker] = []
            self._prices[ticker].append(price)

            history = self._prices[ticker]
            if len(history) < self.slow:
                continue

            fast_ma = np.mean(history[-self.fast:])
            slow_ma = np.mean(history[-self.slow:])
            fast_ma_prev = np.mean(history[-self.fast - 1:-1])
            slow_ma_prev = np.mean(history[-self.slow - 1:-1])

            # Detect crossover
            if fast_ma > slow_ma and fast_ma_prev <= slow_ma_prev:
                signals.append(SignalEvent(event.timestamp, ticker, "LONG"))
            elif fast_ma < slow_ma and fast_ma_prev >= slow_ma_prev:
                signals.append(SignalEvent(event.timestamp, ticker, "FLAT"))

        return signals


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate synthetic trending price series
    rng = np.random.default_rng(42)
    n = 756  # ~3 years
    dates = pd.bdate_range("2022-01-01", periods=n)

    # Two synthetic assets with mild trending behaviour
    r1 = rng.normal(0.0004, 0.012, n)
    r2 = rng.normal(0.0003, 0.015, n)
    prices = pd.DataFrame({
        "ASSET_A": 100 * np.exp(np.cumsum(r1)),
        "ASSET_B": 80 * np.exp(np.cumsum(r2)),
    }, index=dates)

    print("═" * 60)
    print("  Event-Driven Backtesting Framework")
    print("  Strategy: 20/50-day SMA Crossover")
    print("  Assets: ASSET_A, ASSET_B")
    print("  Period: 3 years  |  Capital: $1,000,000")
    print("═" * 60)

    strategy = SMACrossover(fast=20, slow=50)
    engine = BacktestEngine(
        strategy=strategy,
        price_data=prices,
        initial_capital=1_000_000,
        commission_per_share=0.005,
        bid_ask_bps=5.0,
    )
    perf = engine.run()
    print("\n" + perf.full_report())
    print(f"\n  Trades executed: {len(engine.portfolio.trade_history):,}")
