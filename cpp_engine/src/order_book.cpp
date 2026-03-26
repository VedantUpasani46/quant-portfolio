/**
 * @file order_book.cpp
 * @brief Implementation of the Limit Order Book matching engine.
 *
 * Matching Algorithm Details
 * ==========================
 * The matching engine operates in two phases for each incoming order:
 *
 * Phase 1 — Aggressive matching:
 *   The incoming order is compared against the best price on the opposite side.
 *   If prices cross (buy >= ask, or sell <= bid), we execute trades at the
 *   *passive* (resting) order's price. We iterate through orders at that price
 *   level in FIFO order (time priority). If the aggressive order still has
 *   remaining quantity after exhausting a price level, we move to the next
 *   price level and repeat.
 *
 * Phase 2 — Resting or cancellation:
 *   - LIMIT: any remaining quantity is inserted into the book.
 *   - MARKET/IOC: any remaining quantity is cancelled (not rested).
 *   - FOK: checked *before* Phase 1; if full fill isn't possible, rejected.
 *
 * Performance Considerations
 * ==========================
 * - std::map gives O(log P) for insert/find/erase on price levels.
 * - std::deque gives O(1) push_back (new orders) and pop_front (fills).
 * - The order_index_ unordered_map gives O(1) average lookup by order ID.
 * - Trade history is append-only (vector), cache-friendly for sequential reads.
 *
 * Thread-Safety Annotations
 * =========================
 * Every public method is annotated with LOCK_POINT comments in the header.
 * A production implementation would use:
 *   - std::shared_mutex book_mutex_;
 *   - std::unique_lock for mutations (add, cancel, modify)
 *   - std::shared_lock for queries (get_best_bid, get_depth, etc.)
 * For ultra-low-latency, a single-threaded sequencer with lock-free queues
 * feeding in/out is preferred (Disruptor pattern).
 */

#include "order_book.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace lob {

// ============================================================================
// Construction
// ============================================================================

OrderBook::OrderBook(std::string symbol, bool enable_self_trade_prevention)
    : symbol_(std::move(symbol))
    , self_trade_prevention_(enable_self_trade_prevention)
{
    // Reserve reasonable initial capacity to avoid early reallocations.
    trades_.reserve(1024);
    order_index_.reserve(4096);
}

// ============================================================================
// Timestamp utility
// ============================================================================

Timestamp OrderBook::now_ns() noexcept {
    using clock = std::chrono::high_resolution_clock;
    auto now = clock::now();
    return static_cast<Timestamp>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count()
    );
}

// ============================================================================
// Validation
// ============================================================================

void OrderBook::validate_order(const Order& order) {
    if (order.quantity == 0) {
        throw std::invalid_argument("Order quantity must be > 0");
    }
    if (order.remaining_qty == 0) {
        throw std::invalid_argument("Order remaining_qty must be > 0");
    }
    if (order.remaining_qty > order.quantity) {
        throw std::invalid_argument("remaining_qty cannot exceed quantity");
    }
    // Market orders don't need a valid price; they match at any price.
    if (order.type == OrderType::LIMIT || order.type == OrderType::IOC ||
        order.type == OrderType::FOK) {
        if (order.price <= 0.0) {
            throw std::invalid_argument("Limit/IOC/FOK order price must be > 0");
        }
    }
}

// ============================================================================
// Template: core matching loop
// ============================================================================

/**
 * match_order — the heart of the matching engine.
 *
 * Template parameters allow us to reuse the same logic for both buy and sell
 * sides without runtime branching in the inner loop.
 *
 * @tparam BookSide  BidMap (for incoming SELL) or AskMap (for incoming BUY).
 * @tparam CanMatch  A callable (Price order_price, Price level_price) -> bool.
 *
 * For a BUY order matching against asks:
 *   can_match = [](Price op, Price lp) { return op >= lp; }
 *   (buy at 100 matches ask at 99, 100; not 101)
 *
 * For a SELL order matching against bids:
 *   can_match = [](Price op, Price lp) { return op <= lp; }
 *   (sell at 100 matches bid at 100, 101; not 99)
 *
 * Market orders use price = 0 for buys or infinity for sells, ensuring
 * they match everything.
 */
template <typename BookSide, typename CanMatch>
void OrderBook::match_order(Order& order, BookSide& book_side,
                             CanMatch can_match, std::vector<Trade>& trades)
{
    // Walk price levels from best to worst.
    auto level_it = book_side.begin();

    while (level_it != book_side.end() && order.remaining_qty > 0) {
        const Price level_price = level_it->first;

        // Check if the incoming order's price crosses this level.
        // For MARKET orders, the effective price always crosses.
        const Price effective_price =
            (order.type == OrderType::MARKET)
                ? (order.side == Side::BUY
                       ? std::numeric_limits<Price>::max()
                       : 0.0)
                : order.price;

        if (!can_match(effective_price, level_price)) {
            break;  // No more matchable levels.
        }

        auto& level_deque = level_it->second;

        // Walk orders at this price level in FIFO order (time priority).
        while (!level_deque.empty() && order.remaining_qty > 0) {
            Order& resting = level_deque.front();

            // Self-trade prevention: skip if same "account" (simplified: same ID prefix).
            // In production, this would compare trader/firm IDs.
            if (self_trade_prevention_ && resting.id == order.id) {
                // In a real system, we'd skip or cancel one side.
                break;
            }

            // Determine fill quantity: minimum of both remaining quantities.
            const Quantity fill_qty = std::min(order.remaining_qty, resting.remaining_qty);

            // Execute the trade at the *resting* (passive) order's price.
            Trade trade;
            trade.price     = level_price;
            trade.quantity   = fill_qty;
            trade.timestamp = now_ns();

            if (order.side == Side::BUY) {
                trade.buy_order_id  = order.id;
                trade.sell_order_id = resting.id;
            } else {
                trade.buy_order_id  = resting.id;
                trade.sell_order_id = order.id;
            }

            trades.push_back(trade);

            // Update statistics.
            ++trade_count_;
            total_volume_ += fill_qty;

            // Reduce remaining quantities.
            order.remaining_qty   -= fill_qty;
            resting.remaining_qty -= fill_qty;

            // Update order statuses.
            if (order.remaining_qty == 0) {
                order.status = OrderStatus::FILLED;
            } else {
                order.status = OrderStatus::PARTIALLY_FILLED;
            }

            if (resting.remaining_qty == 0) {
                resting.status = OrderStatus::FILLED;
                // Remove from the order index.
                order_index_.erase(resting.id);
                // Remove filled resting order from the front of the deque.
                level_deque.pop_front();
            } else {
                resting.status = OrderStatus::PARTIALLY_FILLED;
                // Resting order stays in the deque with reduced qty.
                break;  // Move to next only if current is fully consumed.
                // Wait — this is wrong. If the aggressive order still has qty
                // and the resting order is partially filled, the resting order
                // is NOT fully consumed, so we should NOT continue matching
                // at this price level... actually we should because the resting
                // order still has remaining. Let me reconsider:
                //
                // If fill_qty = min(aggressive.remaining, resting.remaining),
                // and resting.remaining > 0 after fill, that means
                // aggressive.remaining was < resting.remaining, so
                // aggressive.remaining is now 0. The outer while loop will exit.
                //
                // So this 'break' is actually unreachable in practice, but
                // included for safety.
            }
        }

        // If the level is now empty, erase it from the map.
        if (level_deque.empty()) {
            level_it = book_side.erase(level_it);
        } else {
            ++level_it;
        }
    }
}

/**
 * can_fill_fok — check if a Fill-or-Kill order can be completely filled.
 *
 * We walk the opposite side *without mutating* to sum available quantity
 * at crossable price levels. If total >= order.remaining_qty, FOK can fill.
 */
template <typename BookSide, typename CanMatch>
bool OrderBook::can_fill_fok(const Order& order, const BookSide& book_side,
                              CanMatch can_match) const
{
    Quantity available = 0;

    for (auto it = book_side.begin(); it != book_side.end(); ++it) {
        const Price level_price = it->first;
        const Price effective_price =
            (order.type == OrderType::MARKET)
                ? (order.side == Side::BUY
                       ? std::numeric_limits<Price>::max()
                       : 0.0)
                : order.price;

        if (!can_match(effective_price, level_price)) {
            break;
        }

        for (const auto& resting : it->second) {
            available += resting.remaining_qty;
            if (available >= order.remaining_qty) {
                return true;  // Early exit — enough liquidity.
            }
        }
    }

    return available >= order.remaining_qty;
}

// ============================================================================
// add_order
// ============================================================================

std::vector<Trade> OrderBook::add_order(Order order) {
    // LOCK_POINT: std::unique_lock<std::shared_mutex> lock(book_mutex_);

    ++message_count_;

    // Auto-assign ID if not set.
    if (order.id == 0) {
        order.id = next_order_id_++;
    }

    // Auto-assign timestamp if not set.
    if (order.timestamp == 0) {
        order.timestamp = now_ns();
    }

    // Set remaining_qty if not already set.
    if (order.remaining_qty == 0) {
        order.remaining_qty = order.quantity;
    }

    // Validate.
    validate_order(order);

    std::vector<Trade> trades;
    trades.reserve(16);  // Avoid reallocations for typical match scenarios.

    // --- FOK: pre-check feasibility before attempting any matches ---
    if (order.type == OrderType::FOK) {
        bool feasible = false;
        if (order.side == Side::BUY) {
            feasible = can_fill_fok(
                order, asks_,
                [](Price op, Price lp) { return op >= lp; }
            );
        } else {
            feasible = can_fill_fok(
                order, bids_,
                [](Price op, Price lp) { return op <= lp; }
            );
        }

        if (!feasible) {
            order.status = OrderStatus::REJECTED;
            return trades;  // Empty — FOK rejected.
        }
    }

    // --- Phase 1: Aggressive matching against opposite side ---
    if (order.side == Side::BUY) {
        // Buy order matches against asks (lowest ask first).
        match_order(
            order, asks_,
            [](Price order_price, Price level_price) {
                return order_price >= level_price;
            },
            trades
        );
    } else {
        // Sell order matches against bids (highest bid first).
        match_order(
            order, bids_,
            [](Price order_price, Price level_price) {
                return order_price <= level_price;
            },
            trades
        );
    }

    // --- Phase 2: Rest or cancel the unfilled remainder ---
    if (order.remaining_qty > 0) {
        switch (order.type) {
            case OrderType::LIMIT:
                // Rest on the book.
                order.status = (order.remaining_qty < order.quantity)
                    ? OrderStatus::PARTIALLY_FILLED
                    : OrderStatus::NEW;
                insert_order(order);
                break;

            case OrderType::MARKET:
            case OrderType::IOC:
                // Cancel unfilled portion.
                order.status = (order.remaining_qty < order.quantity)
                    ? OrderStatus::PARTIALLY_FILLED
                    : OrderStatus::CANCELLED;
                break;

            case OrderType::FOK:
                // Should not reach here — FOK was pre-checked.
                // But if somehow it does, mark as rejected.
                order.status = OrderStatus::REJECTED;
                break;
        }
    }

    // Append trades to history.
    trades_.insert(trades_.end(), trades.begin(), trades.end());

    return trades;
}

// ============================================================================
// cancel_order
// ============================================================================

bool OrderBook::cancel_order(OrderId order_id) {
    // LOCK_POINT: std::unique_lock<std::shared_mutex> lock(book_mutex_);

    ++message_count_;

    auto idx_it = order_index_.find(order_id);
    if (idx_it == order_index_.end()) {
        return false;  // Order not found (already filled/cancelled).
    }

    const auto [side, price] = idx_it->second;
    remove_order_from_level(side, price, order_id);
    order_index_.erase(idx_it);

    return true;
}

// ============================================================================
// modify_order
// ============================================================================

std::vector<Trade> OrderBook::modify_order(OrderId order_id, Price new_price, Quantity new_qty) {
    // LOCK_POINT: std::unique_lock<std::shared_mutex> lock(book_mutex_);

    ++message_count_;

    auto idx_it = order_index_.find(order_id);
    if (idx_it == order_index_.end()) {
        throw std::invalid_argument("Order not found: " + std::to_string(order_id));
    }

    if (new_qty == 0) {
        throw std::invalid_argument("Modified quantity must be > 0");
    }

    const auto [side, old_price] = idx_it->second;

    // Find the existing order to get its details.
    Order old_order;
    bool found = false;

    if (side == Side::BUY) {
        auto level_it = bids_.find(old_price);
        if (level_it != bids_.end()) {
            for (auto& o : level_it->second) {
                if (o.id == order_id) {
                    old_order = o;
                    found = true;
                    break;
                }
            }
        }
    } else {
        auto level_it = asks_.find(old_price);
        if (level_it != asks_.end()) {
            for (auto& o : level_it->second) {
                if (o.id == order_id) {
                    old_order = o;
                    found = true;
                    break;
                }
            }
        }
    }

    if (!found) {
        throw std::invalid_argument("Order found in index but not in book — inconsistency");
    }

    const Quantity filled_qty = old_order.quantity - old_order.remaining_qty;
    if (new_qty < filled_qty) {
        throw std::invalid_argument(
            "New quantity (" + std::to_string(new_qty) +
            ") cannot be less than already filled (" + std::to_string(filled_qty) + ")"
        );
    }

    const Price effective_new_price = (new_price > 0.0) ? new_price : old_price;
    const bool price_changed = std::abs(effective_new_price - old_price) > 1e-12;
    const bool qty_increased = new_qty > old_order.quantity;

    // Determine if the order loses time priority.
    // Price change or quantity increase => loss of priority (cancel + re-add).
    // Quantity decrease at same price => modify in-place to preserve priority.
    if (!price_changed && !qty_increased) {
        // In-place modification: just update the quantity fields.
        // This preserves FIFO position in the deque.
        if (side == Side::BUY) {
            auto level_it = bids_.find(old_price);
            if (level_it != bids_.end()) {
                for (auto& o : level_it->second) {
                    if (o.id == order_id) {
                        o.quantity = new_qty;
                        o.remaining_qty = new_qty - filled_qty;
                        break;
                    }
                }
            }
        } else {
            auto level_it = asks_.find(old_price);
            if (level_it != asks_.end()) {
                for (auto& o : level_it->second) {
                    if (o.id == order_id) {
                        o.quantity = new_qty;
                        o.remaining_qty = new_qty - filled_qty;
                        break;
                    }
                }
            }
        }
        return {};  // No trades from an in-place qty decrease.
    }

    // Price changed or qty increased: cancel and re-add (loses time priority).
    remove_order_from_level(side, old_price, order_id);
    order_index_.erase(idx_it);

    // Submit as a new order with new timestamp.
    Order new_order;
    new_order.id            = order_id;
    new_order.side          = side;
    new_order.type          = OrderType::LIMIT;  // Modifications always become limit orders.
    new_order.price         = effective_new_price;
    new_order.quantity       = new_qty;
    new_order.remaining_qty = new_qty - filled_qty;
    new_order.timestamp     = now_ns();
    new_order.status        = OrderStatus::NEW;

    // Re-submit (this will trigger matching if the new price crosses).
    // We decrement message_count_ because add_order increments it,
    // and we already counted this as one message.
    --message_count_;
    return add_order(std::move(new_order));
}

// ============================================================================
// Market data queries
// ============================================================================

std::optional<Price> OrderBook::get_best_bid() const noexcept {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<Price> OrderBook::get_best_ask() const noexcept {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<Price> OrderBook::get_spread() const noexcept {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);
    auto bid = get_best_bid();
    auto ask = get_best_ask();
    if (!bid || !ask) return std::nullopt;
    return *ask - *bid;
}

std::optional<Price> OrderBook::get_mid_price() const noexcept {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);
    auto bid = get_best_bid();
    auto ask = get_best_ask();
    if (!bid || !ask) return std::nullopt;
    return (*bid + *ask) / 2.0;
}

std::pair<std::vector<L2Level>, std::vector<L2Level>>
OrderBook::get_depth(size_t levels) const {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);

    std::vector<L2Level> bid_levels, ask_levels;

    // Bids: iterate from best (highest) to worst.
    size_t count = 0;
    for (const auto& [price, level] : bids_) {
        if (levels > 0 && count >= levels) break;
        L2Level l2;
        l2.price = price;
        l2.num_orders = level.size();
        l2.quantity = 0;
        for (const auto& order : level) {
            l2.quantity += order.remaining_qty;
        }
        bid_levels.push_back(l2);
        ++count;
    }

    // Asks: iterate from best (lowest) to worst.
    count = 0;
    for (const auto& [price, level] : asks_) {
        if (levels > 0 && count >= levels) break;
        L2Level l2;
        l2.price = price;
        l2.num_orders = level.size();
        l2.quantity = 0;
        for (const auto& order : level) {
            l2.quantity += order.remaining_qty;
        }
        ask_levels.push_back(l2);
        ++count;
    }

    return {std::move(bid_levels), std::move(ask_levels)};
}

std::pair<std::vector<L3Entry>, std::vector<L3Entry>>
OrderBook::get_l3_depth(size_t levels) const {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);

    std::vector<L3Entry> bid_entries, ask_entries;

    size_t count = 0;
    for (const auto& [price, level] : bids_) {
        if (levels > 0 && count >= levels) break;
        for (const auto& order : level) {
            L3Entry entry;
            entry.id            = order.id;
            entry.price         = order.price;
            entry.remaining_qty = order.remaining_qty;
            entry.timestamp     = order.timestamp;
            bid_entries.push_back(entry);
        }
        ++count;
    }

    count = 0;
    for (const auto& [price, level] : asks_) {
        if (levels > 0 && count >= levels) break;
        for (const auto& order : level) {
            L3Entry entry;
            entry.id            = order.id;
            entry.price         = order.price;
            entry.remaining_qty = order.remaining_qty;
            entry.timestamp     = order.timestamp;
            ask_entries.push_back(entry);
        }
        ++count;
    }

    return {std::move(bid_entries), std::move(ask_entries)};
}

std::optional<Price> OrderBook::get_vwap(size_t last_n) const {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);

    if (trades_.empty()) return std::nullopt;

    const size_t n = (last_n == 0 || last_n > trades_.size())
                         ? trades_.size()
                         : last_n;

    double total_pv = 0.0;   // price * volume
    uint64_t total_v = 0;

    // Iterate from the most recent trades.
    for (size_t i = trades_.size() - n; i < trades_.size(); ++i) {
        const auto& t = trades_[i];
        total_pv += t.price * static_cast<double>(t.quantity);
        total_v  += t.quantity;
    }

    if (total_v == 0) return std::nullopt;
    return total_pv / static_cast<double>(total_v);
}

Quantity OrderBook::get_total_quantity(Side side) const noexcept {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);

    Quantity total = 0;
    if (side == Side::BUY) {
        for (const auto& [price, level] : bids_) {
            for (const auto& order : level) {
                total += order.remaining_qty;
            }
        }
    } else {
        for (const auto& [price, level] : asks_) {
            for (const auto& order : level) {
                total += order.remaining_qty;
            }
        }
    }
    return total;
}

Quantity OrderBook::get_quantity_at_price(Side side, Price price) const noexcept {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);

    Quantity total = 0;
    if (side == Side::BUY) {
        auto it = bids_.find(price);
        if (it != bids_.end()) {
            for (const auto& order : it->second) {
                total += order.remaining_qty;
            }
        }
    } else {
        auto it = asks_.find(price);
        if (it != asks_.end()) {
            for (const auto& order : it->second) {
                total += order.remaining_qty;
            }
        }
    }
    return total;
}

// ============================================================================
// Accessors
// ============================================================================

BookStats OrderBook::get_stats() const noexcept {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);

    BookStats stats;
    stats.message_count = message_count_;
    stats.trade_count   = trade_count_;
    stats.total_volume  = total_volume_;
    stats.order_count   = order_index_.size();
    stats.bid_levels    = bids_.size();
    stats.ask_levels    = asks_.size();
    return stats;
}

const Order* OrderBook::get_order(OrderId order_id) const noexcept {
    // LOCK_POINT: std::shared_lock<std::shared_mutex> lock(book_mutex_);

    auto idx_it = order_index_.find(order_id);
    if (idx_it == order_index_.end()) return nullptr;

    const auto& [side, price] = idx_it->second;

    if (side == Side::BUY) {
        auto level_it = bids_.find(price);
        if (level_it != bids_.end()) {
            for (const auto& order : level_it->second) {
                if (order.id == order_id) return &order;
            }
        }
    } else {
        auto level_it = asks_.find(price);
        if (level_it != asks_.end()) {
            for (const auto& order : level_it->second) {
                if (order.id == order_id) return &order;
            }
        }
    }

    return nullptr;  // Shouldn't happen if index is consistent.
}

void OrderBook::clear() noexcept {
    // LOCK_POINT: std::unique_lock<std::shared_mutex> lock(book_mutex_);

    bids_.clear();
    asks_.clear();
    order_index_.clear();
    trades_.clear();
    message_count_ = 0;
    trade_count_   = 0;
    total_volume_  = 0;
}

// ============================================================================
// Internal helpers
// ============================================================================

void OrderBook::insert_order(Order& order) {
    // Insert into the appropriate side.
    if (order.side == Side::BUY) {
        bids_[order.price].push_back(order);
    } else {
        asks_[order.price].push_back(order);
    }

    // Update the fast-lookup index.
    order_index_[order.id] = {order.side, order.price};
}

void OrderBook::remove_order_from_level(Side side, Price price, OrderId order_id) {
    if (side == Side::BUY) {
        auto level_it = bids_.find(price);
        if (level_it == bids_.end()) return;

        auto& deque = level_it->second;
        auto it = std::find_if(deque.begin(), deque.end(),
            [order_id](const Order& o) { return o.id == order_id; });

        if (it != deque.end()) {
            it->status = OrderStatus::CANCELLED;
            deque.erase(it);
        }

        // Remove empty level.
        if (deque.empty()) {
            bids_.erase(level_it);
        }
    } else {
        auto level_it = asks_.find(price);
        if (level_it == asks_.end()) return;

        auto& deque = level_it->second;
        auto it = std::find_if(deque.begin(), deque.end(),
            [order_id](const Order& o) { return o.id == order_id; });

        if (it != deque.end()) {
            it->status = OrderStatus::CANCELLED;
            deque.erase(it);
        }

        if (deque.empty()) {
            asks_.erase(level_it);
        }
    }
}

}  // namespace lob
