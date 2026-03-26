# C++ Limit Order Book Engine

A production-quality **price-time priority matching engine** implemented in modern C++17, with Python bindings via pybind11. Designed to demonstrate systems-level C++ competency for quantitative trading roles.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OrderBook ("AAPL")                          │
│                                                                     │
│  ┌─────────────────────────┐   ┌─────────────────────────────────┐ │
│  │      BID SIDE           │   │         ASK SIDE                │ │
│  │  std::map<Price,Level>  │   │   std::map<Price,Level>         │ │
│  │  (greater<> = desc)     │   │   (less<> = asc)                │ │
│  │                         │   │                                 │ │
│  │  101.50 ─► [O4,O7,O9]  │   │  102.00 ─► [O2,O5]             │ │
│  │  101.00 ─► [O1,O3]     │   │  102.50 ─► [O6]                │ │
│  │  100.50 ─► [O8]        │   │  103.00 ─► [O10,O11,O12]       │ │
│  │                         │   │                                 │ │
│  │  ▲ best bid = begin()   │   │  ▲ best ask = begin()          │ │
│  └─────────────────────────┘   └─────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────┐                          │
│  │  order_index_: unordered_map         │  O(1) lookup by ID      │
│  │  OrderId ──► (Side, Price)           │  for cancel/modify      │
│  └──────────────────────────────────────┘                          │
│                                                                     │
│  Level = std::deque<Order>   ◄── FIFO queue (time priority)        │
│  O(1) push_back / pop_front                                        │
└─────────────────────────────────────────────────────────────────────┘

Matching Flow:
  ┌──────────┐     ┌───────────┐     ┌──────────┐     ┌───────────┐
  │ Incoming  │────►│ Validate  │────►│  Match   │────►│ Rest or   │
  │  Order    │     │  Order    │     │ (Phase1) │     │  Cancel   │
  └──────────┘     └───────────┘     └──────────┘     │ (Phase2)  │
                                          │            └───────────┘
                                          ▼
                                    ┌──────────┐
                                    │  Trades  │
                                    │ (output) │
                                    └──────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **Order Types** | Limit, Market, IOC (Immediate-or-Cancel), FOK (Fill-or-Kill) |
| **Matching** | Price-time priority (FIFO at each price level) |
| **Operations** | Add, Cancel, Modify (cancel-and-replace) |
| **Market Data** | L2 (aggregated) and L3 (order-by-order) depth snapshots |
| **Analytics** | VWAP, spread, mid price, total quantity, statistics |
| **Python** | Full API via pybind11, NumPy bulk submission, NumPy trade export |
| **Performance** | O(log P) insertion, O(1) best bid/ask, cache-friendly deque |

## Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `add_order` | O(log P + M) | P = price levels, M = matched levels |
| `cancel_order` | O(log P + Q) | Q = orders at price level |
| `get_best_bid/ask` | O(1) amortized | `map::begin()` |
| `get_depth` | O(D × Q_avg) | D = depth levels requested |
| `modify_order` | O(log P + M) | Cancel + re-add |

## Build Instructions

### Prerequisites

- C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.15+
- Python 3.8+ (for bindings)
- pybind11 (auto-fetched if not installed)

### Quick Start

```bash
# Install pybind11 (optional — CMake will fetch it if missing)
pip install pybind11

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)

# Run tests
ctest --output-on-failure

# Run benchmark
cd .. && python benchmark/benchmark_orderbook.py
```

### Build Options

```bash
cmake .. -DBUILD_TESTS=OFF       # Skip unit tests
cmake .. -DBUILD_PYTHON=OFF      # Skip Python bindings
cmake .. -DCMAKE_BUILD_TYPE=Debug # Debug build with sanitizers
```

### Using the Python Module

After building, the `order_book_cpp` shared library will be in the `build/` directory:

```bash
cd build
python -c "import order_book_cpp; print(order_book_cpp.__version__)"
```

## API Documentation

### Python API

```python
import order_book_cpp as ob

# Create a book
book = ob.OrderBook("AAPL")

# Create and submit orders
order = ob.make_order(
    id=1,
    side=ob.Side.BUY,
    type=ob.OrderType.LIMIT,
    price=150.0,
    quantity=100
)
trades = book.add_order(order)

# Cancel an order
success = book.cancel_order(order_id=1)

# Modify an order (cancel-and-replace)
trades = book.modify_order(order_id=2, new_price=151.0, new_qty=200)

# Market data
best_bid = book.get_best_bid()    # Optional[float]
best_ask = book.get_best_ask()    # Optional[float]
spread   = book.get_spread()      # Optional[float]
mid      = book.get_mid_price()   # Optional[float]
vwap     = book.get_vwap(last_n=100)  # Optional[float]

# Depth snapshots
bids, asks = book.get_depth(levels=10)       # L2 (aggregated)
bids, asks = book.get_l3_depth(levels=10)    # L3 (order-by-order)

# NumPy integration
import numpy as np
trades_array = book.trades_to_numpy()         # (N, 5) float64
bids_np, asks_np = book.depth_to_numpy(10)    # (N, 3) float64 each

# Bulk order submission
orders = np.array([
    [0, 0, 150.0, 100, 1],   # [side, type, price, qty, id]
    [1, 0, 155.0, 50,  2],
], dtype=np.float64)
trades = book.bulk_add_orders(orders)

# Statistics
stats = book.get_stats()
print(f"Trades: {stats.trade_count}, Volume: {stats.total_volume}")
```

### C++ API

```cpp
#include "order_book.hpp"

lob::OrderBook book("AAPL");

// Submit a limit buy
lob::Order order;
order.id            = 1;
order.side          = lob::Side::BUY;
order.type          = lob::OrderType::LIMIT;
order.price         = 150.0;
order.quantity       = 100;
order.remaining_qty = 100;

auto trades = book.add_order(std::move(order));

// Query the book
auto best_bid = book.get_best_bid();  // std::optional<double>
auto [bids, asks] = book.get_depth(10);
auto vwap = book.get_vwap(100);
```

## Benchmark Results (Placeholder)

Run `python benchmark/benchmark_orderbook.py` after building. Expected results on a modern desktop:

```
================================================================================
ORDER BOOK BENCHMARK RESULTS
================================================================================
Engine               Orders   Trades   Time(ms)      Ord/sec    p50(ns)    p95(ns)    p99(ns)
----------------------------------------------------------------------------------------------
Python               100,000   XX,XXX      XXX.X      XXX,XXX      X,XXX     XX,XXX     XX,XXX
C++ (pybind11)       100,000   XX,XXX       XX.X    X,XXX,XXX        XXX      X,XXX      X,XXX
C++ (NumPy bulk)     100,000   XX,XXX        X.X   XX,XXX,XXX        XXX          0          0

Speedup (C++ vs Python):      10-50x (throughput), 10-30x (p50 latency)
Speedup (NumPy bulk vs Py):   50-200x (throughput)
```

## Design Decisions

### Why `std::map` over `std::unordered_map` for price levels?

Ordered traversal is essential: we iterate from best price outward during matching. `std::map` (red-black tree) provides O(log n) insert/find with ordered iteration. An `std::unordered_map` would require sorting keys before each match pass.

### Why `std::deque` over `std::vector` for order queues?

`std::deque` provides O(1) `push_back` (new orders) and O(1) `pop_front` (fills), which are the two primary operations. `std::vector` has O(n) `pop_front` (requires shifting all elements).

### Why not a lock-free design?

A lock-free order book (e.g., using compare-and-swap on atomic pointers) adds significant complexity without benefit in single-threaded benchmarks. The code documents where locks would go for a threaded design. For production HFT, a single-threaded sequencer with lock-free input/output queues (Disruptor pattern) is the preferred approach.

### Cache-friendliness

- `Order` struct is 64 bytes (one cache line on most architectures).
- `std::deque` stores elements in contiguous chunks, providing better cache locality than `std::list`.
- The `order_index_` hash map provides O(1) average cancel/modify without scanning levels.

## Thread-Safety Notes

The implementation is **single-threaded by design**. For concurrent access:

1. **Simple**: `std::shared_mutex` on the book. Writers (add/cancel/modify) take exclusive lock; readers (get_depth, etc.) take shared lock.
2. **Better**: Per-side locks (`bid_mutex_`, `ask_mutex_`) reduce contention.
3. **Best (HFT)**: Single-threaded sequencer thread owns the book. Input orders arrive via a lock-free SPSC queue; output trades/market data published via another lock-free queue.

All public methods are annotated with `LOCK_POINT` comments indicating where synchronization would be added.

## Testing

The test suite (Catch2) covers:

- Empty book behavior
- Limit order insertion and priority
- Price-time priority matching
- Partial fills (aggressive and passive)
- Market orders (full fill, partial fill, empty book)
- IOC orders (full, partial, no match)
- FOK orders (full, reject insufficient, reject no cross, multi-level)
- Order cancellation (existing, nonexistent, one-of-many, already-filled)
- Order modification (qty decrease preserves priority, price change re-matches)
- L2/L3 depth snapshots
- VWAP calculation
- Statistics tracking
- Edge cases (zero qty, negative price, crossed book, exact fill, 1000 levels)

```bash
cd build && ctest --output-on-failure -V
```

## File Structure

```
cpp_engine/
├── CMakeLists.txt              # Build system (CMake 3.15+)
├── README.md                   # This file
├── include/
│   └── order_book.hpp          # Header: types, enums, OrderBook class
├── src/
│   ├── order_book.cpp          # Full matching engine implementation
│   └── bindings.cpp            # pybind11 Python bindings
├── tests/
│   └── test_order_book.cpp     # Catch2 unit tests
└── benchmark/
    └── benchmark_orderbook.py  # Python benchmark (C++ vs Python)
```

## License

Part of the quant-portfolio project. MIT License.
