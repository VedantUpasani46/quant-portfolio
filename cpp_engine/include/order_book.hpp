/**
 * @file order_book.hpp
 * @brief Production-quality Limit Order Book (LOB) with price-time priority matching.
 *
 * Architecture Overview
 * =====================
 * The order book maintains two sorted sides:
 *   - Bids: std::map<Price, Level, std::greater<Price>>  (highest price first)
 *   - Asks: std::map<Price, Level>                       (lowest price first)
 *
 * Each price level holds a std::deque<Order> providing O(1) front/back access,
 * which is critical for time-priority (FIFO) matching at each price level.
 *
 * Matching Algorithm (Price-Time Priority)
 * ========================================
 * 1. An incoming BUY order matches against the ask side if order.price >= best_ask.
 * 2. An incoming SELL order matches against the bid side if order.price <= best_bid.
 * 3. Within a price level, the oldest resting order fills first (time priority).
 * 4. Partial fills reduce remaining_qty; the resting order stays in the book.
 * 5. When a resting order is fully filled, it is removed from the deque.
 * 6. When a price level's deque is empty, the level is erased from the map.
 *
 * Complexity
 * ==========
 * - add_order:    O(log P + M)  where P = price levels, M = matched levels
 * - cancel_order: O(log P + Q)  where Q = orders at the price level (deque scan)
 * - best bid/ask: O(1) amortized (map::begin)
 * - get_depth:    O(D * Q_avg)  where D = requested depth levels
 *
 * Thread-Safety Notes
 * ===================
 * This implementation is NOT thread-safe by default. For concurrent access:
 * - A std::shared_mutex on the entire book gives readers||writer semantics.
 * - For lower contention, consider per-side locks (bid_mutex_, ask_mutex_).
 * - For HFT, a lock-free SPSC design with a sequencer thread is preferred.
 * Locking points are annotated with LOCK_POINT comments throughout the code.
 *
 * @author Quant Portfolio
 * @version 1.0.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lob {

// ============================================================================
// Type aliases
// ============================================================================

using Price    = double;
using Quantity = uint64_t;
using OrderId  = uint64_t;

/** High-resolution timestamp in nanoseconds since epoch. */
using Timestamp = uint64_t;

// ============================================================================
// Enums
// ============================================================================

/** Order side: BUY places bids, SELL places asks. */
enum class Side : uint8_t {
    BUY  = 0,
    SELL = 1
};

/**
 * Order type governs matching and resting behavior.
 *
 * LIMIT:  Rests on the book if not immediately matchable at limit price.
 * MARKET: Aggressively walks the book; any unfilled qty is cancelled.
 * IOC:    Immediate-or-Cancel — fills what it can, cancels the rest.
 * FOK:    Fill-or-Kill — must fill entirely or is rejected outright.
 */
enum class OrderType : uint8_t {
    LIMIT  = 0,
    MARKET = 1,
    IOC    = 2,
    FOK    = 3
};

/** Order status tracks lifecycle state. */
enum class OrderStatus : uint8_t {
    NEW             = 0,
    PARTIALLY_FILLED = 1,
    FILLED          = 2,
    CANCELLED       = 3,
    REJECTED        = 4
};

// ============================================================================
// Data structures
// ============================================================================

/**
 * Represents a single order in the book.
 *
 * Design note: fields are ordered to minimize padding on 64-bit architectures.
 * The struct is 64 bytes — exactly one cache line on most modern CPUs.
 */
struct Order {
    OrderId   id            = 0;
    Price     price         = 0.0;
    Quantity  quantity       = 0;    ///< Original quantity
    Quantity  remaining_qty = 0;    ///< Remaining unfilled quantity
    Timestamp timestamp     = 0;    ///< Arrival time (nanoseconds since epoch)
    Side      side          = Side::BUY;
    OrderType type          = OrderType::LIMIT;
    OrderStatus status      = OrderStatus::NEW;

    /** Convenience: is this order fully filled? */
    [[nodiscard]] constexpr bool is_filled() const noexcept {
        return remaining_qty == 0;
    }

    /** Convenience: is this order still active on the book? */
    [[nodiscard]] constexpr bool is_active() const noexcept {
        return status == OrderStatus::NEW || status == OrderStatus::PARTIALLY_FILLED;
    }
};

/**
 * Represents a single trade (execution).
 *
 * Each trade is the result of matching an aggressive (incoming) order
 * against a passive (resting) order.
 */
struct Trade {
    OrderId   buy_order_id  = 0;
    OrderId   sell_order_id = 0;
    Price     price         = 0.0;    ///< Execution price (passive order's price)
    Quantity  quantity       = 0;      ///< Executed quantity
    Timestamp timestamp     = 0;      ///< Execution time
};

/**
 * A single level in the L2 market data snapshot.
 * Aggregates all orders at a given price into total quantity and order count.
 */
struct L2Level {
    Price    price      = 0.0;
    Quantity quantity   = 0;
    size_t   num_orders = 0;
};

/**
 * A single order entry in the L3 (full order-by-order) market data snapshot.
 */
struct L3Entry {
    OrderId  id            = 0;
    Price    price         = 0.0;
    Quantity remaining_qty = 0;
    Timestamp timestamp    = 0;
};

/**
 * Aggregated book statistics for monitoring and analytics.
 */
struct BookStats {
    uint64_t message_count = 0;   ///< Total messages processed (adds + cancels + modifies)
    uint64_t trade_count   = 0;   ///< Total trades executed
    uint64_t total_volume  = 0;   ///< Total quantity traded
    uint64_t order_count   = 0;   ///< Currently active orders in the book
    size_t   bid_levels    = 0;   ///< Number of distinct bid price levels
    size_t   ask_levels    = 0;   ///< Number of distinct ask price levels
};

// ============================================================================
// OrderBook class
// ============================================================================

/**
 * @class OrderBook
 * @brief A price-time priority limit order book with full matching engine.
 *
 * This class is the core of the matching engine. It accepts orders, matches
 * them according to price-time priority, and produces trades.
 *
 * Usage:
 * @code
 *     lob::OrderBook book("AAPL");
 *     auto trades = book.add_order({
 *         .id = 1, .price = 150.0, .quantity = 100, .remaining_qty = 100,
 *         .timestamp = now(), .side = Side::BUY, .type = OrderType::LIMIT
 *     });
 * @endcode
 */
class OrderBook {
public:
    // -- Construction ----------------------------------------------------------

    /**
     * Construct an order book for the given symbol.
     * @param symbol Ticker symbol (e.g., "AAPL"). Moved into storage.
     * @param enable_self_trade_prevention If true, orders from matching
     *        counterparties (same id prefix convention) are prevented.
     */
    explicit OrderBook(std::string symbol = "",
                       bool enable_self_trade_prevention = false);

    // Rule of five: default move, no copy (book state is large).
    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;
    OrderBook(OrderBook&&) noexcept = default;
    OrderBook& operator=(OrderBook&&) noexcept = default;
    ~OrderBook() = default;

    // -- Order operations ------------------------------------------------------

    /**
     * Submit a new order to the book.
     *
     * The matching engine will:
     * 1. Validate the order (reject if invalid).
     * 2. Attempt to match against the opposite side.
     * 3. For LIMIT orders, rest any unfilled remainder on the book.
     * 4. For MARKET/IOC, cancel any unfilled remainder.
     * 5. For FOK, reject entirely if full fill is not possible.
     *
     * @param order The order to submit (moved in; id and timestamp should be set).
     * @return Vector of trades generated by this order.
     * @throws std::invalid_argument if order has zero quantity or invalid price.
     *
     * LOCK_POINT: Acquire exclusive lock on the entire book here.
     */
    std::vector<Trade> add_order(Order order);

    /**
     * Cancel an existing resting order.
     *
     * @param order_id The ID of the order to cancel.
     * @return true if the order was found and cancelled, false otherwise.
     *
     * LOCK_POINT: Acquire exclusive lock on the entire book here.
     */
    bool cancel_order(OrderId order_id);

    /**
     * Modify an existing resting order's quantity and/or price.
     *
     * Semantics: cancel-and-replace. If the price changes, the order loses
     * its time priority. If only quantity decreases, priority is preserved.
     *
     * @param order_id    The order to modify.
     * @param new_price   New price (use 0.0 or same price to keep current).
     * @param new_qty     New total quantity (must be > 0, must be >= filled qty).
     * @return Vector of trades if the modification triggers a match.
     * @throws std::invalid_argument if order not found or new_qty invalid.
     *
     * LOCK_POINT: Acquire exclusive lock on the entire book here.
     */
    std::vector<Trade> modify_order(OrderId order_id, Price new_price, Quantity new_qty);

    // -- Market data queries ---------------------------------------------------

    /**
     * @return The highest bid price, or std::nullopt if no bids.
     *
     * LOCK_POINT: Acquire shared lock (read-only).
     */
    [[nodiscard]] std::optional<Price> get_best_bid() const noexcept;

    /**
     * @return The lowest ask price, or std::nullopt if no asks.
     *
     * LOCK_POINT: Acquire shared lock (read-only).
     */
    [[nodiscard]] std::optional<Price> get_best_ask() const noexcept;

    /**
     * @return Spread (best_ask - best_bid), or std::nullopt if either side empty.
     *
     * LOCK_POINT: Acquire shared lock (read-only).
     */
    [[nodiscard]] std::optional<Price> get_spread() const noexcept;

    /**
     * @return Mid price (best_bid + best_ask) / 2, or std::nullopt.
     *
     * LOCK_POINT: Acquire shared lock (read-only).
     */
    [[nodiscard]] std::optional<Price> get_mid_price() const noexcept;

    /**
     * Get Level-2 (aggregated) depth snapshot.
     *
     * @param levels Number of price levels per side (0 = all).
     * @return Pair of (bids, asks), each a vector of L2Level.
     *
     * LOCK_POINT: Acquire shared lock (read-only).
     */
    [[nodiscard]] std::pair<std::vector<L2Level>, std::vector<L2Level>>
    get_depth(size_t levels = 10) const;

    /**
     * Get Level-3 (full order-by-order) depth snapshot.
     *
     * @param levels Number of price levels per side (0 = all).
     * @return Pair of (bids, asks), each a vector of L3Entry.
     *
     * LOCK_POINT: Acquire shared lock (read-only).
     */
    [[nodiscard]] std::pair<std::vector<L3Entry>, std::vector<L3Entry>>
    get_l3_depth(size_t levels = 10) const;

    /**
     * Volume-weighted average price of recent trades.
     *
     * @param last_n Number of recent trades to consider (0 = all).
     * @return VWAP, or std::nullopt if no trades.
     */
    [[nodiscard]] std::optional<Price> get_vwap(size_t last_n = 0) const;

    /**
     * @return Total quantity resting on a given side.
     */
    [[nodiscard]] Quantity get_total_quantity(Side side) const noexcept;

    /**
     * @return Quantity available at a specific price level on a given side.
     */
    [[nodiscard]] Quantity get_quantity_at_price(Side side, Price price) const noexcept;

    // -- Accessors -------------------------------------------------------------

    [[nodiscard]] const std::string& symbol() const noexcept { return symbol_; }
    [[nodiscard]] const std::vector<Trade>& trade_history() const noexcept { return trades_; }
    [[nodiscard]] BookStats get_stats() const noexcept;

    /**
     * Look up an order by ID.
     * @return Pointer to the order (nullptr if not found). Invalidated by mutations.
     */
    [[nodiscard]] const Order* get_order(OrderId order_id) const noexcept;

    /** Reset the book to an empty state. */
    void clear() noexcept;

private:
    // -- Internal types --------------------------------------------------------

    /** A price level: a FIFO queue of orders at the same price. */
    using Level = std::deque<Order>;

    /**
     * Bid side: descending price order (highest bid = begin).
     * std::greater<Price> makes the map sort keys in descending order,
     * so map::begin() points to the highest (best) bid.
     */
    using BidMap = std::map<Price, Level, std::greater<Price>>;

    /**
     * Ask side: ascending price order (lowest ask = begin).
     * Default std::less<Price> sorts ascending, so map::begin()
     * points to the lowest (best) ask.
     */
    using AskMap = std::map<Price, Level>;

    // -- Internal matching helpers ---------------------------------------------

    /**
     * Core matching loop: match an incoming order against the opposite side.
     *
     * @tparam BookSide  BidMap or AskMap — the side to match *against*.
     * @tparam CanMatch  Lambda: (order_price, level_price) -> bool
     * @param  order     The aggressive (incoming) order (modified in place).
     * @param  book_side The passive side of the book.
     * @param  can_match Predicate determining if prices cross.
     * @param  trades    Output vector of generated trades.
     */
    template <typename BookSide, typename CanMatch>
    void match_order(Order& order, BookSide& book_side,
                     CanMatch can_match, std::vector<Trade>& trades);

    /**
     * Check if a FOK order can be fully filled without actually executing.
     *
     * @tparam BookSide  BidMap or AskMap.
     * @tparam CanMatch  Price-crossing predicate.
     * @param  order     The FOK order to check.
     * @param  book_side The opposite side.
     * @param  can_match Price predicate.
     * @return true if the full quantity is available.
     */
    template <typename BookSide, typename CanMatch>
    [[nodiscard]] bool can_fill_fok(const Order& order, const BookSide& book_side,
                                     CanMatch can_match) const;

    /** Insert a resting order into the appropriate side of the book. */
    void insert_order(Order& order);

    /** Remove an order from its price level (helper for cancel/fill). */
    void remove_order_from_level(Side side, Price price, OrderId order_id);

    /** Generate a timestamp (nanoseconds since epoch). */
    [[nodiscard]] static Timestamp now_ns() noexcept;

    /** Validate an incoming order. Throws on invalid. */
    static void validate_order(const Order& order);

    // -- Data members ----------------------------------------------------------

    std::string symbol_;

    BidMap bids_;    ///< Bid side: highest-first
    AskMap asks_;    ///< Ask side: lowest-first

    /**
     * Fast O(1) order lookup by ID.
     * Maps order_id -> (side, price) so we can locate the order in the book.
     * This avoids scanning all levels when cancelling or modifying.
     */
    std::unordered_map<OrderId, std::pair<Side, Price>> order_index_;

    /** Full trade history (append-only). */
    std::vector<Trade> trades_;

    /** Monotonic statistics counters. */
    uint64_t message_count_ = 0;
    uint64_t trade_count_   = 0;
    uint64_t total_volume_  = 0;

    /** Self-trade prevention flag. */
    bool self_trade_prevention_ = false;

    /** Next order ID for auto-assignment (if order.id == 0). */
    OrderId next_order_id_ = 1;
};

}  // namespace lob
