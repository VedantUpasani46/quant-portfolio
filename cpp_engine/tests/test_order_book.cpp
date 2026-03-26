/**
 * @file test_order_book.cpp
 * @brief Comprehensive unit tests for the Limit Order Book engine.
 *
 * Tests cover:
 *   - Basic order insertion and retrieval
 *   - Price-time priority matching
 *   - Partial fills
 *   - Market orders
 *   - IOC (Immediate-or-Cancel)
 *   - FOK (Fill-or-Kill)
 *   - Order cancellation
 *   - Order modification
 *   - Edge cases (empty book, single order, crossed book)
 *   - L2/L3 depth snapshots
 *   - VWAP calculation
 *   - Statistics tracking
 *
 * Framework: Catch2 v3
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "order_book.hpp"

using namespace lob;
using Catch::Approx;

// ============================================================================
// Helper: create orders conveniently
// ============================================================================

static OrderId next_id = 1;

static Order make_order(Side side, OrderType type, Price price, Quantity qty) {
    Order o;
    o.id            = next_id++;
    o.side          = side;
    o.type          = type;
    o.price         = price;
    o.quantity       = qty;
    o.remaining_qty = qty;
    o.timestamp     = 0;  // Auto-assigned by OrderBook
    o.status        = OrderStatus::NEW;
    return o;
}

static Order make_limit_buy(Price price, Quantity qty) {
    return make_order(Side::BUY, OrderType::LIMIT, price, qty);
}

static Order make_limit_sell(Price price, Quantity qty) {
    return make_order(Side::SELL, OrderType::LIMIT, price, qty);
}

static Order make_market_buy(Quantity qty) {
    return make_order(Side::BUY, OrderType::MARKET, 0.0, qty);
}

static Order make_market_sell(Quantity qty) {
    return make_order(Side::SELL, OrderType::MARKET, 0.0, qty);
}

static Order make_ioc_buy(Price price, Quantity qty) {
    return make_order(Side::BUY, OrderType::IOC, price, qty);
}

static Order make_ioc_sell(Price price, Quantity qty) {
    return make_order(Side::SELL, OrderType::IOC, price, qty);
}

static Order make_fok_buy(Price price, Quantity qty) {
    return make_order(Side::BUY, OrderType::FOK, price, qty);
}

static Order make_fok_sell(Price price, Quantity qty) {
    return make_order(Side::SELL, OrderType::FOK, price, qty);
}

// Reset ID counter before each test section
struct ResetIdFixture {
    ResetIdFixture() { next_id = 1; }
};

// ============================================================================
// TEST: Empty book behavior
// ============================================================================

TEST_CASE("Empty book returns no data", "[empty]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("No best bid or ask") {
        REQUIRE_FALSE(book.get_best_bid().has_value());
        REQUIRE_FALSE(book.get_best_ask().has_value());
    }

    SECTION("No spread or mid price") {
        REQUIRE_FALSE(book.get_spread().has_value());
        REQUIRE_FALSE(book.get_mid_price().has_value());
    }

    SECTION("No VWAP") {
        REQUIRE_FALSE(book.get_vwap().has_value());
    }

    SECTION("Empty depth") {
        auto [bids, asks] = book.get_depth(10);
        REQUIRE(bids.empty());
        REQUIRE(asks.empty());
    }

    SECTION("Zero stats") {
        auto stats = book.get_stats();
        REQUIRE(stats.order_count == 0);
        REQUIRE(stats.trade_count == 0);
        REQUIRE(stats.total_volume == 0);
    }

    SECTION("Cancel on empty book returns false") {
        REQUIRE_FALSE(book.cancel_order(999));
    }

    SECTION("Get nonexistent order returns nullptr") {
        REQUIRE(book.get_order(42) == nullptr);
    }
}

// ============================================================================
// TEST: Basic limit order insertion
// ============================================================================

TEST_CASE("Limit orders rest on the book", "[limit][basic]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("Single buy order") {
        auto trades = book.add_order(make_limit_buy(100.0, 50));
        REQUIRE(trades.empty());
        REQUIRE(book.get_best_bid().value() == 100.0);
        REQUIRE_FALSE(book.get_best_ask().has_value());
    }

    SECTION("Single sell order") {
        auto trades = book.add_order(make_limit_sell(105.0, 30));
        REQUIRE(trades.empty());
        REQUIRE(book.get_best_ask().value() == 105.0);
        REQUIRE_FALSE(book.get_best_bid().has_value());
    }

    SECTION("Multiple bids — best bid is highest") {
        book.add_order(make_limit_buy(100.0, 50));
        book.add_order(make_limit_buy(101.0, 30));
        book.add_order(make_limit_buy(99.0, 40));

        REQUIRE(book.get_best_bid().value() == 101.0);
    }

    SECTION("Multiple asks — best ask is lowest") {
        book.add_order(make_limit_sell(105.0, 50));
        book.add_order(make_limit_sell(103.0, 30));
        book.add_order(make_limit_sell(107.0, 40));

        REQUIRE(book.get_best_ask().value() == 103.0);
    }

    SECTION("Spread and mid price") {
        book.add_order(make_limit_buy(100.0, 50));
        book.add_order(make_limit_sell(102.0, 30));

        REQUIRE(book.get_spread().value() == Approx(2.0));
        REQUIRE(book.get_mid_price().value() == Approx(101.0));
    }
}

// ============================================================================
// TEST: Price-time priority matching
// ============================================================================

TEST_CASE("Price-time priority matching", "[matching][priority]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("Incoming sell matches best bid") {
        book.add_order(make_limit_buy(100.0, 50));  // id=1
        book.add_order(make_limit_buy(101.0, 30));  // id=2

        auto trades = book.add_order(make_limit_sell(100.0, 20));  // id=3

        REQUIRE(trades.size() == 1);
        // Should match against best bid (101.0), not 100.0
        REQUIRE(trades[0].price == 101.0);
        REQUIRE(trades[0].quantity == 20);
        REQUIRE(trades[0].buy_order_id == 2);
        REQUIRE(trades[0].sell_order_id == 3);
    }

    SECTION("Time priority: first order at same price fills first") {
        auto o1 = make_limit_buy(100.0, 30);  // id=1, arrives first
        auto o2 = make_limit_buy(100.0, 30);  // id=2, arrives second

        book.add_order(std::move(o1));
        book.add_order(std::move(o2));

        // Sell 20 at 100 — should match order 1 first (time priority)
        auto trades = book.add_order(make_limit_sell(100.0, 20));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].buy_order_id == 1);
        REQUIRE(trades[0].quantity == 20);
    }

    SECTION("Trade executes at passive (resting) price") {
        book.add_order(make_limit_sell(105.0, 50));  // id=1, resting at 105

        // Aggressive buy at 110 should execute at 105 (passive price)
        auto trades = book.add_order(make_limit_buy(110.0, 30));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].price == 105.0);
    }
}

// ============================================================================
// TEST: Partial fills
// ============================================================================

TEST_CASE("Partial fills", "[matching][partial]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("Aggressive order partially fills and rests") {
        book.add_order(make_limit_sell(100.0, 20));  // id=1

        // Buy 50 at 100 — fills 20, rests 30
        auto trades = book.add_order(make_limit_buy(100.0, 50));  // id=2

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].quantity == 20);

        // The remaining 30 should be resting on the bid side
        REQUIRE(book.get_best_bid().value() == 100.0);
        REQUIRE(book.get_quantity_at_price(Side::BUY, 100.0) == 30);
    }

    SECTION("Passive order partially fills and remains") {
        book.add_order(make_limit_sell(100.0, 50));  // id=1

        // Buy 20 at 100 — fills 20, ask has 30 remaining
        auto trades = book.add_order(make_limit_buy(100.0, 20));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].quantity == 20);
        REQUIRE(book.get_best_ask().value() == 100.0);
        REQUIRE(book.get_quantity_at_price(Side::SELL, 100.0) == 30);
    }

    SECTION("Aggressive order walks through multiple price levels") {
        book.add_order(make_limit_sell(100.0, 20));  // id=1
        book.add_order(make_limit_sell(101.0, 20));  // id=2
        book.add_order(make_limit_sell(102.0, 20));  // id=3

        // Buy 50 at 102 — fills all of 100, all of 101, 10 of 102
        auto trades = book.add_order(make_limit_buy(102.0, 50));  // id=4

        REQUIRE(trades.size() == 3);
        REQUIRE(trades[0].price == 100.0);
        REQUIRE(trades[0].quantity == 20);
        REQUIRE(trades[1].price == 101.0);
        REQUIRE(trades[1].quantity == 20);
        REQUIRE(trades[2].price == 102.0);
        REQUIRE(trades[2].quantity == 10);

        // 10 remaining at 102
        REQUIRE(book.get_best_ask().value() == 102.0);
        REQUIRE(book.get_quantity_at_price(Side::SELL, 102.0) == 10);
    }
}

// ============================================================================
// TEST: Market orders
// ============================================================================

TEST_CASE("Market orders", "[market]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("Market buy consumes all liquidity") {
        book.add_order(make_limit_sell(100.0, 20));
        book.add_order(make_limit_sell(101.0, 30));

        auto trades = book.add_order(make_market_buy(50));

        REQUIRE(trades.size() == 2);
        REQUIRE(trades[0].price == 100.0);
        REQUIRE(trades[0].quantity == 20);
        REQUIRE(trades[1].price == 101.0);
        REQUIRE(trades[1].quantity == 30);

        // Book should be empty on ask side
        REQUIRE_FALSE(book.get_best_ask().has_value());
    }

    SECTION("Market order with insufficient liquidity fills what it can") {
        book.add_order(make_limit_sell(100.0, 20));

        // Market buy for 50, but only 20 available
        auto trades = book.add_order(make_market_buy(50));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].quantity == 20);

        // Unfilled 30 is cancelled (market orders don't rest)
        REQUIRE_FALSE(book.get_best_bid().has_value());
    }

    SECTION("Market sell into bids") {
        book.add_order(make_limit_buy(100.0, 30));

        auto trades = book.add_order(make_market_sell(20));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].price == 100.0);
        REQUIRE(trades[0].quantity == 20);
    }

    SECTION("Market order on empty book produces no trades") {
        auto trades = book.add_order(make_market_buy(100));
        REQUIRE(trades.empty());
    }
}

// ============================================================================
// TEST: IOC (Immediate-or-Cancel)
// ============================================================================

TEST_CASE("IOC orders", "[ioc]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("IOC fills fully") {
        book.add_order(make_limit_sell(100.0, 50));

        auto trades = book.add_order(make_ioc_buy(100.0, 30));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].quantity == 30);
    }

    SECTION("IOC fills partially — remainder cancelled, not rested") {
        book.add_order(make_limit_sell(100.0, 20));

        auto trades = book.add_order(make_ioc_buy(100.0, 50));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].quantity == 20);

        // IOC should NOT rest
        REQUIRE_FALSE(book.get_best_bid().has_value());
    }

    SECTION("IOC with no match produces no trades and no resting order") {
        auto trades = book.add_order(make_ioc_buy(100.0, 50));
        REQUIRE(trades.empty());
        REQUIRE_FALSE(book.get_best_bid().has_value());
    }
}

// ============================================================================
// TEST: FOK (Fill-or-Kill)
// ============================================================================

TEST_CASE("FOK orders", "[fok]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("FOK fills entirely") {
        book.add_order(make_limit_sell(100.0, 50));

        auto trades = book.add_order(make_fok_buy(100.0, 30));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].quantity == 30);
    }

    SECTION("FOK rejected when insufficient liquidity") {
        book.add_order(make_limit_sell(100.0, 20));

        auto trades = book.add_order(make_fok_buy(100.0, 50));

        // No trades — order rejected
        REQUIRE(trades.empty());
        // The resting sell should be untouched
        REQUIRE(book.get_quantity_at_price(Side::SELL, 100.0) == 20);
    }

    SECTION("FOK rejected when price doesn't cross") {
        book.add_order(make_limit_sell(105.0, 50));

        auto trades = book.add_order(make_fok_buy(100.0, 30));

        REQUIRE(trades.empty());
    }

    SECTION("FOK fills across multiple levels") {
        book.add_order(make_limit_sell(100.0, 20));
        book.add_order(make_limit_sell(101.0, 30));

        auto trades = book.add_order(make_fok_buy(101.0, 50));

        REQUIRE(trades.size() == 2);
        REQUIRE(trades[0].price == 100.0);
        REQUIRE(trades[0].quantity == 20);
        REQUIRE(trades[1].price == 101.0);
        REQUIRE(trades[1].quantity == 30);
    }

    SECTION("FOK sell rejected on empty book") {
        auto trades = book.add_order(make_fok_sell(100.0, 50));
        REQUIRE(trades.empty());
    }
}

// ============================================================================
// TEST: Order cancellation
// ============================================================================

TEST_CASE("Order cancellation", "[cancel]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("Cancel existing order") {
        auto o = make_limit_buy(100.0, 50);
        OrderId oid = o.id;
        book.add_order(std::move(o));

        REQUIRE(book.cancel_order(oid));
        REQUIRE_FALSE(book.get_best_bid().has_value());
    }

    SECTION("Cancel nonexistent order returns false") {
        REQUIRE_FALSE(book.cancel_order(9999));
    }

    SECTION("Cancel one of multiple orders at same level") {
        auto o1 = make_limit_buy(100.0, 30);
        auto o2 = make_limit_buy(100.0, 20);
        OrderId id1 = o1.id;

        book.add_order(std::move(o1));
        book.add_order(std::move(o2));

        REQUIRE(book.cancel_order(id1));
        REQUIRE(book.get_quantity_at_price(Side::BUY, 100.0) == 20);
    }

    SECTION("Cannot cancel already-filled order") {
        auto sell = make_limit_sell(100.0, 30);
        OrderId sell_id = sell.id;
        book.add_order(std::move(sell));

        // Fill it completely
        book.add_order(make_limit_buy(100.0, 30));

        // Should not be cancellable
        REQUIRE_FALSE(book.cancel_order(sell_id));
    }

    SECTION("Cancel removes from order index") {
        auto o = make_limit_buy(100.0, 50);
        OrderId oid = o.id;
        book.add_order(std::move(o));

        book.cancel_order(oid);
        REQUIRE(book.get_order(oid) == nullptr);
    }
}

// ============================================================================
// TEST: Order modification
// ============================================================================

TEST_CASE("Order modification", "[modify]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("Reduce quantity preserves priority") {
        auto o1 = make_limit_buy(100.0, 50);
        auto o2 = make_limit_buy(100.0, 30);
        OrderId id1 = o1.id;

        book.add_order(std::move(o1));
        book.add_order(std::move(o2));

        // Reduce o1 from 50 to 30 — should keep time priority
        auto trades = book.modify_order(id1, 100.0, 30);
        REQUIRE(trades.empty());

        // Now sell 20 — should match o1 first (preserved priority)
        trades = book.add_order(make_limit_sell(100.0, 20));
        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].buy_order_id == id1);
    }

    SECTION("Change price triggers re-matching") {
        book.add_order(make_limit_sell(105.0, 30));  // id=1

        auto o = make_limit_buy(100.0, 50);  // id=2
        OrderId buy_id = o.id;
        book.add_order(std::move(o));

        // Modify buy to 105 — should now match the sell
        auto trades = book.modify_order(buy_id, 105.0, 50);

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].price == 105.0);
        REQUIRE(trades[0].quantity == 30);
    }

    SECTION("Modify nonexistent order throws") {
        REQUIRE_THROWS_AS(
            book.modify_order(9999, 100.0, 50),
            std::invalid_argument
        );
    }

    SECTION("Modify to zero quantity throws") {
        auto o = make_limit_buy(100.0, 50);
        OrderId oid = o.id;
        book.add_order(std::move(o));

        REQUIRE_THROWS_AS(
            book.modify_order(oid, 100.0, 0),
            std::invalid_argument
        );
    }
}

// ============================================================================
// TEST: L2/L3 depth snapshots
// ============================================================================

TEST_CASE("Depth snapshots", "[depth]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    book.add_order(make_limit_buy(100.0, 50));
    book.add_order(make_limit_buy(100.0, 30));
    book.add_order(make_limit_buy(99.0, 40));
    book.add_order(make_limit_sell(105.0, 20));
    book.add_order(make_limit_sell(106.0, 10));

    SECTION("L2 depth") {
        auto [bids, asks] = book.get_depth(10);

        REQUIRE(bids.size() == 2);
        REQUIRE(bids[0].price == 100.0);
        REQUIRE(bids[0].quantity == 80);  // 50 + 30
        REQUIRE(bids[0].num_orders == 2);
        REQUIRE(bids[1].price == 99.0);
        REQUIRE(bids[1].quantity == 40);

        REQUIRE(asks.size() == 2);
        REQUIRE(asks[0].price == 105.0);
        REQUIRE(asks[1].price == 106.0);
    }

    SECTION("L2 depth with limit") {
        auto [bids, asks] = book.get_depth(1);

        REQUIRE(bids.size() == 1);
        REQUIRE(bids[0].price == 100.0);
        REQUIRE(asks.size() == 1);
        REQUIRE(asks[0].price == 105.0);
    }

    SECTION("L3 depth shows individual orders") {
        auto [bids, asks] = book.get_l3_depth(10);

        // Two orders at 100.0, one at 99.0 = 3 bid entries
        REQUIRE(bids.size() == 3);
        REQUIRE(bids[0].price == 100.0);
        REQUIRE(bids[0].remaining_qty == 50);
        REQUIRE(bids[1].price == 100.0);
        REQUIRE(bids[1].remaining_qty == 30);
        REQUIRE(bids[2].price == 99.0);
        REQUIRE(bids[2].remaining_qty == 40);

        REQUIRE(asks.size() == 2);
    }
}

// ============================================================================
// TEST: VWAP
// ============================================================================

TEST_CASE("VWAP calculation", "[vwap]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    // Set up some trades
    book.add_order(make_limit_sell(100.0, 50));
    book.add_order(make_limit_sell(101.0, 30));

    book.add_order(make_limit_buy(101.0, 70));  // Fills: 50@100 + 20@101

    SECTION("VWAP of all trades") {
        auto vwap = book.get_vwap();
        REQUIRE(vwap.has_value());
        // (100*50 + 101*20) / (50+20) = (5000 + 2020) / 70 = 7020/70 ≈ 100.2857
        REQUIRE(vwap.value() == Approx(100.2857).margin(0.001));
    }

    SECTION("VWAP of last 1 trade") {
        auto vwap = book.get_vwap(1);
        REQUIRE(vwap.has_value());
        REQUIRE(vwap.value() == Approx(101.0));
    }
}

// ============================================================================
// TEST: Statistics
// ============================================================================

TEST_CASE("Book statistics", "[stats]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    book.add_order(make_limit_sell(100.0, 50));  // msg 1
    book.add_order(make_limit_buy(100.0, 30));   // msg 2, trade: 30 qty

    auto stats = book.get_stats();

    REQUIRE(stats.message_count == 2);
    REQUIRE(stats.trade_count == 1);
    REQUIRE(stats.total_volume == 30);
    REQUIRE(stats.order_count == 1);  // One sell order resting with 20 remaining
    REQUIRE(stats.ask_levels == 1);
    REQUIRE(stats.bid_levels == 0);
}

// ============================================================================
// TEST: Edge cases
// ============================================================================

TEST_CASE("Edge cases", "[edge]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("Order with zero quantity is rejected") {
        Order o;
        o.id = 999;
        o.side = Side::BUY;
        o.type = OrderType::LIMIT;
        o.price = 100.0;
        o.quantity = 0;
        o.remaining_qty = 0;

        REQUIRE_THROWS_AS(book.add_order(std::move(o)), std::invalid_argument);
    }

    SECTION("Limit order with zero price is rejected") {
        Order o;
        o.id = 999;
        o.side = Side::BUY;
        o.type = OrderType::LIMIT;
        o.price = 0.0;
        o.quantity = 50;
        o.remaining_qty = 50;

        REQUIRE_THROWS_AS(book.add_order(std::move(o)), std::invalid_argument);
    }

    SECTION("Negative price is rejected") {
        Order o;
        o.id = 999;
        o.side = Side::BUY;
        o.type = OrderType::LIMIT;
        o.price = -1.0;
        o.quantity = 50;
        o.remaining_qty = 50;

        REQUIRE_THROWS_AS(book.add_order(std::move(o)), std::invalid_argument);
    }

    SECTION("Crossed book resolves immediately") {
        // Post sell at 100
        book.add_order(make_limit_sell(100.0, 50));

        // Post buy at 105 (crosses the spread) — should immediately match
        auto trades = book.add_order(make_limit_buy(105.0, 30));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].price == 100.0);  // Executes at passive price
    }

    SECTION("Exact quantity match leaves no resting orders") {
        book.add_order(make_limit_sell(100.0, 50));
        auto trades = book.add_order(make_limit_buy(100.0, 50));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].quantity == 50);
        REQUIRE_FALSE(book.get_best_bid().has_value());
        REQUIRE_FALSE(book.get_best_ask().has_value());
    }

    SECTION("Clear resets everything") {
        book.add_order(make_limit_buy(100.0, 50));
        book.add_order(make_limit_sell(105.0, 30));

        book.clear();

        REQUIRE_FALSE(book.get_best_bid().has_value());
        REQUIRE_FALSE(book.get_best_ask().has_value());
        REQUIRE(book.trade_history().empty());

        auto stats = book.get_stats();
        REQUIRE(stats.message_count == 0);
    }

    SECTION("Multiple orders fill multiple resting orders at same level") {
        book.add_order(make_limit_sell(100.0, 20));  // id=1
        book.add_order(make_limit_sell(100.0, 20));  // id=2
        book.add_order(make_limit_sell(100.0, 20));  // id=3

        auto trades = book.add_order(make_limit_buy(100.0, 50));

        // Should fill: 20 from id=1, 20 from id=2, 10 from id=3
        REQUIRE(trades.size() == 3);
        REQUIRE(trades[0].quantity == 20);
        REQUIRE(trades[1].quantity == 20);
        REQUIRE(trades[2].quantity == 10);

        // 10 remaining in id=3
        REQUIRE(book.get_quantity_at_price(Side::SELL, 100.0) == 10);
    }

    SECTION("Large number of price levels") {
        // Insert 1000 different price levels
        for (int i = 0; i < 1000; ++i) {
            book.add_order(make_limit_buy(100.0 + i * 0.01, 10));
        }

        auto stats = book.get_stats();
        REQUIRE(stats.bid_levels == 1000);
        REQUIRE(stats.order_count == 1000);

        // Best bid should be the highest price
        REQUIRE(book.get_best_bid().value() == Approx(109.99));
    }
}

// ============================================================================
// TEST: Total quantity
// ============================================================================

TEST_CASE("Total quantity", "[quantity]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    book.add_order(make_limit_buy(100.0, 50));
    book.add_order(make_limit_buy(99.0, 30));
    book.add_order(make_limit_sell(105.0, 20));

    REQUIRE(book.get_total_quantity(Side::BUY) == 80);
    REQUIRE(book.get_total_quantity(Side::SELL) == 20);
}

// ============================================================================
// TEST: Symbol
// ============================================================================

TEST_CASE("Symbol property", "[symbol]") {
    OrderBook book("AAPL");
    REQUIRE(book.symbol() == "AAPL");
}

// ============================================================================
// TEST: Auto ID and timestamp assignment
// ============================================================================

TEST_CASE("Auto-assignment of id and timestamp", "[auto]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    Order o;
    o.id = 0;  // Should be auto-assigned
    o.side = Side::BUY;
    o.type = OrderType::LIMIT;
    o.price = 100.0;
    o.quantity = 50;
    o.remaining_qty = 50;
    o.timestamp = 0;  // Should be auto-assigned

    book.add_order(std::move(o));

    // The order should be findable (auto-assigned id >= 1)
    auto stats = book.get_stats();
    REQUIRE(stats.order_count == 1);
}

// ============================================================================
// TEST: Sell-side matching edge cases
// ============================================================================

TEST_CASE("Sell-side matching", "[sell][matching]") {
    ResetIdFixture fix;
    OrderBook book("TEST");

    SECTION("Sell FOK across bid levels") {
        book.add_order(make_limit_buy(101.0, 30));
        book.add_order(make_limit_buy(100.0, 20));

        auto trades = book.add_order(make_fok_sell(100.0, 50));

        REQUIRE(trades.size() == 2);
        REQUIRE(trades[0].price == 101.0);
        REQUIRE(trades[0].quantity == 30);
        REQUIRE(trades[1].price == 100.0);
        REQUIRE(trades[1].quantity == 20);
    }

    SECTION("Sell IOC partial fill") {
        book.add_order(make_limit_buy(100.0, 20));

        auto trades = book.add_order(make_ioc_sell(100.0, 50));

        REQUIRE(trades.size() == 1);
        REQUIRE(trades[0].quantity == 20);
        REQUIRE_FALSE(book.get_best_ask().has_value());  // IOC doesn't rest
    }

    SECTION("Market sell fills multiple levels") {
        book.add_order(make_limit_buy(101.0, 30));
        book.add_order(make_limit_buy(100.0, 20));

        auto trades = book.add_order(make_market_sell(40));

        REQUIRE(trades.size() == 2);
        REQUIRE(trades[0].price == 101.0);  // Best bid first
        REQUIRE(trades[0].quantity == 30);
        REQUIRE(trades[1].price == 100.0);
        REQUIRE(trades[1].quantity == 10);
    }
}
