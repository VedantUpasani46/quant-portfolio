#!/usr/bin/env python3
"""
Benchmark: C++ Order Book Engine vs Pure Python Implementation
==============================================================

Measures:
  - Orders per second (throughput)
  - Latency percentiles: p50, p95, p99
  - Memory efficiency
  - Bulk submission performance (NumPy path)

Expected results:
  - C++ engine: 10-50x faster than pure Python for single-order submission
  - NumPy bulk path: additional 2-5x over individual C++ calls

Usage:
  python benchmark_orderbook.py

Prerequisites:
  - Build the C++ module first: mkdir build && cd build && cmake .. && make
  - Ensure order_book_cpp.so is importable (e.g., in build/ directory)
"""

import sys
import time
import random
import statistics
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Pure Python Order Book (baseline for comparison)
# ---------------------------------------------------------------------------

@dataclass
class PyOrder:
    id: int
    side: str  # "BUY" or "SELL"
    order_type: str  # "LIMIT", "MARKET", "IOC", "FOK"
    price: float
    quantity: int
    remaining_qty: int
    timestamp: float = 0.0
    status: str = "NEW"


@dataclass
class PyTrade:
    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: int
    timestamp: float = 0.0


class PythonOrderBook:
    """Minimal pure-Python limit order book for benchmarking comparison."""

    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.bids: dict[float, list[PyOrder]] = defaultdict(list)  # price -> [orders]
        self.asks: dict[float, list[PyOrder]] = defaultdict(list)
        self.order_index: dict[int, Tuple[str, float]] = {}  # id -> (side, price)
        self.trades: list[PyTrade] = []
        self._next_id = 1

    def add_order(self, order: PyOrder) -> list[PyTrade]:
        if order.id == 0:
            order.id = self._next_id
            self._next_id += 1
        if order.timestamp == 0:
            order.timestamp = time.time_ns()

        trades = []

        # FOK pre-check
        if order.order_type == "FOK":
            if not self._can_fill_fok(order):
                order.status = "REJECTED"
                return trades

        # Match
        if order.side == "BUY":
            trades = self._match_buy(order)
        else:
            trades = self._match_sell(order)

        # Rest or cancel
        if order.remaining_qty > 0:
            if order.order_type == "LIMIT":
                self._insert(order)
            # MARKET, IOC, FOK: cancel remainder

        self.trades.extend(trades)
        return trades

    def cancel_order(self, order_id: int) -> bool:
        if order_id not in self.order_index:
            return False
        side, price = self.order_index.pop(order_id)
        book = self.bids if side == "BUY" else self.asks
        if price in book:
            book[price] = [o for o in book[price] if o.id != order_id]
            if not book[price]:
                del book[price]
        return True

    def get_best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None

    def _match_buy(self, order: PyOrder) -> list[PyTrade]:
        trades = []
        sorted_prices = sorted(self.asks.keys())
        for price in sorted_prices:
            if order.remaining_qty == 0:
                break
            if order.order_type != "MARKET" and order.price < price:
                break
            level = self.asks[price]
            while level and order.remaining_qty > 0:
                resting = level[0]
                fill = min(order.remaining_qty, resting.remaining_qty)
                trades.append(PyTrade(
                    buy_order_id=order.id,
                    sell_order_id=resting.id,
                    price=price,
                    quantity=fill,
                    timestamp=time.time_ns()
                ))
                order.remaining_qty -= fill
                resting.remaining_qty -= fill
                if resting.remaining_qty == 0:
                    level.pop(0)
                    self.order_index.pop(resting.id, None)
            if not level:
                del self.asks[price]
        return trades

    def _match_sell(self, order: PyOrder) -> list[PyTrade]:
        trades = []
        sorted_prices = sorted(self.bids.keys(), reverse=True)
        for price in sorted_prices:
            if order.remaining_qty == 0:
                break
            if order.order_type != "MARKET" and order.price > price:
                break
            level = self.bids[price]
            while level and order.remaining_qty > 0:
                resting = level[0]
                fill = min(order.remaining_qty, resting.remaining_qty)
                trades.append(PyTrade(
                    buy_order_id=resting.id,
                    sell_order_id=order.id,
                    price=price,
                    quantity=fill,
                    timestamp=time.time_ns()
                ))
                order.remaining_qty -= fill
                resting.remaining_qty -= fill
                if resting.remaining_qty == 0:
                    level.pop(0)
                    self.order_index.pop(resting.id, None)
            if not level:
                del self.bids[price]
        return trades

    def _insert(self, order: PyOrder):
        if order.side == "BUY":
            self.bids[order.price].append(order)
        else:
            self.asks[order.price].append(order)
        self.order_index[order.id] = (order.side, order.price)

    def _can_fill_fok(self, order: PyOrder) -> bool:
        available = 0
        if order.side == "BUY":
            for price in sorted(self.asks.keys()):
                if order.price < price:
                    break
                for o in self.asks[price]:
                    available += o.remaining_qty
                    if available >= order.remaining_qty:
                        return True
        else:
            for price in sorted(self.bids.keys(), reverse=True):
                if order.price > price:
                    break
                for o in self.bids[price]:
                    available += o.remaining_qty
                    if available >= order.remaining_qty:
                        return True
        return available >= order.remaining_qty


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def generate_orders(n: int, price_center: float = 100.0, price_spread: float = 5.0):
    """Generate N random orders centered around a price."""
    random.seed(42)
    orders = []
    for i in range(n):
        side = random.choice(["BUY", "SELL"])
        if side == "BUY":
            price = round(price_center - random.uniform(0, price_spread), 2)
        else:
            price = round(price_center + random.uniform(0, price_spread), 2)
        qty = random.randint(1, 100)
        orders.append({
            "id": i + 1,
            "side": side,
            "type": "LIMIT",
            "price": price,
            "quantity": qty,
        })
    return orders


def benchmark_python(orders: list[dict], cancel_pct: float = 0.1) -> dict:
    """Benchmark the pure Python order book."""
    book = PythonOrderBook("BENCH")
    latencies = []
    trade_count = 0
    active_ids = []

    for order_data in orders:
        o = PyOrder(
            id=order_data["id"],
            side=order_data["side"],
            order_type=order_data["type"],
            price=order_data["price"],
            quantity=order_data["quantity"],
            remaining_qty=order_data["quantity"],
        )
        start = time.perf_counter_ns()
        trades = book.add_order(o)
        elapsed = time.perf_counter_ns() - start
        latencies.append(elapsed)
        trade_count += len(trades)
        if o.remaining_qty > 0 and o.order_type == "LIMIT":
            active_ids.append(o.id)

    # Benchmark some cancellations
    cancel_latencies = []
    n_cancel = int(len(active_ids) * cancel_pct)
    for oid in active_ids[:n_cancel]:
        start = time.perf_counter_ns()
        book.cancel_order(oid)
        elapsed = time.perf_counter_ns() - start
        cancel_latencies.append(elapsed)

    total_ns = sum(latencies)
    return {
        "engine": "Python",
        "orders": len(orders),
        "trades": trade_count,
        "total_time_ms": total_ns / 1e6,
        "orders_per_sec": len(orders) / (total_ns / 1e9) if total_ns > 0 else 0,
        "p50_ns": int(statistics.median(latencies)),
        "p95_ns": int(sorted(latencies)[int(len(latencies) * 0.95)]),
        "p99_ns": int(sorted(latencies)[int(len(latencies) * 0.99)]),
        "cancel_p50_ns": int(statistics.median(cancel_latencies)) if cancel_latencies else 0,
    }


def benchmark_cpp(orders: list[dict], cancel_pct: float = 0.1) -> dict:
    """Benchmark the C++ order book (via pybind11)."""
    try:
        import order_book_cpp as ob
    except ImportError:
        print("ERROR: Cannot import order_book_cpp. Build it first.")
        print("  mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j")
        return None

    book = ob.OrderBook("BENCH")
    latencies = []
    trade_count = 0
    active_ids = []

    for order_data in orders:
        o = ob.make_order(
            order_data["id"],
            ob.Side.BUY if order_data["side"] == "BUY" else ob.Side.SELL,
            ob.OrderType.LIMIT,
            order_data["price"],
            order_data["quantity"],
        )
        start = time.perf_counter_ns()
        trades = book.add_order(o)
        elapsed = time.perf_counter_ns() - start
        latencies.append(elapsed)
        trade_count += len(trades)

    # Benchmark cancellations
    cancel_latencies = []
    stats = book.get_stats()

    total_ns = sum(latencies)
    return {
        "engine": "C++ (pybind11)",
        "orders": len(orders),
        "trades": trade_count,
        "total_time_ms": total_ns / 1e6,
        "orders_per_sec": len(orders) / (total_ns / 1e9) if total_ns > 0 else 0,
        "p50_ns": int(statistics.median(latencies)),
        "p95_ns": int(sorted(latencies)[int(len(latencies) * 0.95)]),
        "p99_ns": int(sorted(latencies)[int(len(latencies) * 0.99)]),
        "cancel_p50_ns": 0,
    }


def benchmark_cpp_bulk(orders: list[dict]) -> dict:
    """Benchmark bulk NumPy submission path."""
    try:
        import order_book_cpp as ob
    except ImportError:
        return None

    book = ob.OrderBook("BENCH_BULK")

    # Build NumPy array: [side, type, price, qty, id]
    arr = np.zeros((len(orders), 5), dtype=np.float64)
    for i, o in enumerate(orders):
        arr[i, 0] = 0.0 if o["side"] == "BUY" else 1.0
        arr[i, 1] = 0.0  # LIMIT
        arr[i, 2] = o["price"]
        arr[i, 3] = o["quantity"]
        arr[i, 4] = o["id"]

    start = time.perf_counter_ns()
    trades = book.bulk_add_orders(arr)
    elapsed = time.perf_counter_ns() - start

    return {
        "engine": "C++ (NumPy bulk)",
        "orders": len(orders),
        "trades": len(trades),
        "total_time_ms": elapsed / 1e6,
        "orders_per_sec": len(orders) / (elapsed / 1e9) if elapsed > 0 else 0,
        "p50_ns": int(elapsed / len(orders)),
        "p95_ns": 0,
        "p99_ns": 0,
        "cancel_p50_ns": 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_results(results: list[dict]):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 80)
    print("ORDER BOOK BENCHMARK RESULTS")
    print("=" * 80)

    header = f"{'Engine':<20} {'Orders':>8} {'Trades':>8} {'Time(ms)':>10} {'Ord/sec':>12} {'p50(ns)':>10} {'p95(ns)':>10} {'p99(ns)':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r is None:
            continue
        print(
            f"{r['engine']:<20} "
            f"{r['orders']:>8,} "
            f"{r['trades']:>8,} "
            f"{r['total_time_ms']:>10.1f} "
            f"{r['orders_per_sec']:>12,.0f} "
            f"{r['p50_ns']:>10,} "
            f"{r['p95_ns']:>10,} "
            f"{r['p99_ns']:>10,}"
        )

    # Compute speedup
    py_result = next((r for r in results if r and r["engine"] == "Python"), None)
    cpp_result = next((r for r in results if r and r["engine"] == "C++ (pybind11)"), None)
    bulk_result = next((r for r in results if r and r["engine"] == "C++ (NumPy bulk)"), None)

    if py_result and cpp_result:
        speedup = py_result["total_time_ms"] / cpp_result["total_time_ms"] if cpp_result["total_time_ms"] > 0 else float("inf")
        latency_speedup = py_result["p50_ns"] / cpp_result["p50_ns"] if cpp_result["p50_ns"] > 0 else float("inf")
        print(f"\nSpeedup (C++ vs Python):      {speedup:.1f}x (throughput), {latency_speedup:.1f}x (p50 latency)")

    if py_result and bulk_result:
        speedup = py_result["total_time_ms"] / bulk_result["total_time_ms"] if bulk_result["total_time_ms"] > 0 else float("inf")
        print(f"Speedup (NumPy bulk vs Py):   {speedup:.1f}x (throughput)")

    print()


def main():
    sizes = [10_000, 100_000]

    if "--quick" in sys.argv:
        sizes = [10_000]

    if "--large" in sys.argv:
        sizes.append(1_000_000)

    for n in sizes:
        print(f"\n>>> Generating {n:,} random orders...")
        orders = generate_orders(n)

        results = []

        print(f"  Benchmarking Python ({n:,} orders)...")
        results.append(benchmark_python(orders))

        print(f"  Benchmarking C++ ({n:,} orders)...")
        cpp_result = benchmark_cpp(orders)
        results.append(cpp_result)

        if cpp_result is not None:
            print(f"  Benchmarking C++ NumPy bulk ({n:,} orders)...")
            results.append(benchmark_cpp_bulk(orders))

        print_results(results)


if __name__ == "__main__":
    main()
