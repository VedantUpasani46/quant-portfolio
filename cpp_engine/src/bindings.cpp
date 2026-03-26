/**
 * @file bindings.cpp
 * @brief pybind11 bindings exposing the C++ OrderBook engine to Python.
 *
 * This module creates a Python extension called `order_book_cpp` that exposes:
 *   - Enums: Side, OrderType, OrderStatus
 *   - Structs: Order, Trade, L2Level, L3Entry, BookStats
 *   - Class: OrderBook with full API
 *
 * NumPy Integration
 * =================
 * - Bulk order submission via NumPy structured arrays.
 * - Trade history export to NumPy arrays for zero-copy analytics.
 *
 * Usage from Python:
 * @code
 *     import order_book_cpp as ob
 *
 *     book = ob.OrderBook("AAPL")
 *     order = ob.Order()
 *     order.id = 1
 *     order.side = ob.Side.BUY
 *     order.type = ob.OrderType.LIMIT
 *     order.price = 150.0
 *     order.quantity = 100
 *     order.remaining_qty = 100
 *
 *     trades = book.add_order(order)
 *     print(f"Best bid: {book.get_best_bid()}")
 * @endcode
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "order_book.hpp"

#include <sstream>
#include <vector>

namespace py = pybind11;

// ============================================================================
// Helper: export trade history to NumPy structured array
// ============================================================================

/**
 * Convert the trade history vector to a NumPy structured array.
 * This provides zero-copy-style access for Python analytics code.
 *
 * The resulting array has dtype:
 *   [('buy_order_id', 'u8'), ('sell_order_id', 'u8'),
 *    ('price', 'f8'), ('quantity', 'u8'), ('timestamp', 'u8')]
 */
static py::array_t<double> trades_to_numpy(const std::vector<lob::Trade>& trades) {
    const size_t n = trades.size();

    // Create a 2D array: n rows x 5 columns
    // Columns: buy_order_id, sell_order_id, price, quantity, timestamp
    py::array_t<double> result({n, static_cast<size_t>(5)});

    auto buf = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n; ++i) {
        buf(i, 0) = static_cast<double>(trades[i].buy_order_id);
        buf(i, 1) = static_cast<double>(trades[i].sell_order_id);
        buf(i, 2) = trades[i].price;
        buf(i, 3) = static_cast<double>(trades[i].quantity);
        buf(i, 4) = static_cast<double>(trades[i].timestamp);
    }

    return result;
}

// ============================================================================
// Helper: bulk order submission from NumPy array
// ============================================================================

/**
 * Submit multiple orders at once from a NumPy array.
 *
 * Expected array shape: (N, 5) with columns:
 *   [side (0=BUY,1=SELL), type (0=LIMIT,...), price, quantity, id]
 *
 * Returns all generated trades.
 */
static std::vector<lob::Trade> bulk_add_orders(
    lob::OrderBook& book,
    py::array_t<double> orders_array)
{
    auto buf = orders_array.unchecked<2>();
    const auto n = buf.shape(0);

    if (buf.shape(1) < 5) {
        throw std::invalid_argument(
            "Expected array with at least 5 columns: "
            "[side, type, price, quantity, id]"
        );
    }

    std::vector<lob::Trade> all_trades;
    all_trades.reserve(static_cast<size_t>(n));

    for (py::ssize_t i = 0; i < n; ++i) {
        lob::Order order;
        order.side          = static_cast<lob::Side>(static_cast<uint8_t>(buf(i, 0)));
        order.type          = static_cast<lob::OrderType>(static_cast<uint8_t>(buf(i, 1)));
        order.price         = buf(i, 2);
        order.quantity       = static_cast<lob::Quantity>(buf(i, 3));
        order.remaining_qty = order.quantity;
        order.id            = static_cast<lob::OrderId>(buf(i, 4));

        auto trades = book.add_order(std::move(order));
        all_trades.insert(all_trades.end(),
                          std::make_move_iterator(trades.begin()),
                          std::make_move_iterator(trades.end()));
    }

    return all_trades;
}

// ============================================================================
// Helper: depth to numpy
// ============================================================================

static py::array_t<double> depth_side_to_numpy(const std::vector<lob::L2Level>& levels) {
    const size_t n = levels.size();
    py::array_t<double> result({n, static_cast<size_t>(3)});
    auto buf = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n; ++i) {
        buf(i, 0) = levels[i].price;
        buf(i, 1) = static_cast<double>(levels[i].quantity);
        buf(i, 2) = static_cast<double>(levels[i].num_orders);
    }
    return result;
}

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(order_book_cpp, m) {
    m.doc() = R"pbdoc(
        C++ Limit Order Book Engine
        ===========================

        A high-performance limit order book with price-time priority matching,
        exposed to Python via pybind11.

        Supports: Limit, Market, IOC, and FOK order types.
        Provides: L2/L3 market data snapshots, VWAP, spread, depth.

        Example:
            >>> import order_book_cpp as ob
            >>> book = ob.OrderBook("AAPL")
            >>> order = ob.Order()
            >>> order.id = 1
            >>> order.side = ob.Side.BUY
            >>> order.type = ob.OrderType.LIMIT
            >>> order.price = 150.0
            >>> order.quantity = 100
            >>> order.remaining_qty = 100
            >>> trades = book.add_order(order)
    )pbdoc";

    // ---- Enums ----

    py::enum_<lob::Side>(m, "Side", "Order side: BUY or SELL")
        .value("BUY",  lob::Side::BUY,  "Buy (bid) side")
        .value("SELL", lob::Side::SELL, "Sell (ask) side")
        .export_values();

    py::enum_<lob::OrderType>(m, "OrderType", "Order type determines matching behavior")
        .value("LIMIT",  lob::OrderType::LIMIT,
               "Limit order: rests on book if not immediately matched")
        .value("MARKET", lob::OrderType::MARKET,
               "Market order: matches aggressively, unfilled qty cancelled")
        .value("IOC",    lob::OrderType::IOC,
               "Immediate-or-Cancel: fills what it can, cancels rest")
        .value("FOK",    lob::OrderType::FOK,
               "Fill-or-Kill: must fill entirely or is rejected")
        .export_values();

    py::enum_<lob::OrderStatus>(m, "OrderStatus", "Order lifecycle status")
        .value("NEW",              lob::OrderStatus::NEW)
        .value("PARTIALLY_FILLED", lob::OrderStatus::PARTIALLY_FILLED)
        .value("FILLED",          lob::OrderStatus::FILLED)
        .value("CANCELLED",       lob::OrderStatus::CANCELLED)
        .value("REJECTED",        lob::OrderStatus::REJECTED)
        .export_values();

    // ---- Order struct ----

    py::class_<lob::Order>(m, "Order", "Represents a single order")
        .def(py::init<>())
        .def_readwrite("id",            &lob::Order::id,            "Unique order identifier")
        .def_readwrite("price",         &lob::Order::price,         "Limit price (0 for market)")
        .def_readwrite("quantity",       &lob::Order::quantity,       "Original quantity")
        .def_readwrite("remaining_qty", &lob::Order::remaining_qty, "Remaining unfilled quantity")
        .def_readwrite("timestamp",     &lob::Order::timestamp,     "Arrival time (ns since epoch)")
        .def_readwrite("side",          &lob::Order::side,          "BUY or SELL")
        .def_readwrite("type",          &lob::Order::type,          "LIMIT, MARKET, IOC, FOK")
        .def_readwrite("status",        &lob::Order::status,        "Current order status")
        .def("is_filled",  &lob::Order::is_filled,  "True if fully filled")
        .def("is_active",  &lob::Order::is_active,  "True if still on the book")
        .def("__repr__", [](const lob::Order& o) {
            std::ostringstream ss;
            ss << "Order(id=" << o.id
               << ", side=" << (o.side == lob::Side::BUY ? "BUY" : "SELL")
               << ", price=" << o.price
               << ", qty=" << o.quantity
               << ", remaining=" << o.remaining_qty
               << ", status=" << static_cast<int>(o.status) << ")";
            return ss.str();
        });

    // ---- Trade struct ----

    py::class_<lob::Trade>(m, "Trade", "Represents a single execution")
        .def(py::init<>())
        .def_readwrite("buy_order_id",  &lob::Trade::buy_order_id)
        .def_readwrite("sell_order_id", &lob::Trade::sell_order_id)
        .def_readwrite("price",         &lob::Trade::price)
        .def_readwrite("quantity",       &lob::Trade::quantity)
        .def_readwrite("timestamp",     &lob::Trade::timestamp)
        .def("__repr__", [](const lob::Trade& t) {
            std::ostringstream ss;
            ss << "Trade(buy=" << t.buy_order_id
               << ", sell=" << t.sell_order_id
               << ", price=" << t.price
               << ", qty=" << t.quantity << ")";
            return ss.str();
        });

    // ---- L2Level struct ----

    py::class_<lob::L2Level>(m, "L2Level", "Aggregated price level (L2 data)")
        .def(py::init<>())
        .def_readwrite("price",      &lob::L2Level::price)
        .def_readwrite("quantity",    &lob::L2Level::quantity)
        .def_readwrite("num_orders", &lob::L2Level::num_orders)
        .def("__repr__", [](const lob::L2Level& l) {
            std::ostringstream ss;
            ss << "L2Level(price=" << l.price
               << ", qty=" << l.quantity
               << ", orders=" << l.num_orders << ")";
            return ss.str();
        });

    // ---- L3Entry struct ----

    py::class_<lob::L3Entry>(m, "L3Entry", "Individual order entry (L3 data)")
        .def(py::init<>())
        .def_readwrite("id",            &lob::L3Entry::id)
        .def_readwrite("price",         &lob::L3Entry::price)
        .def_readwrite("remaining_qty", &lob::L3Entry::remaining_qty)
        .def_readwrite("timestamp",     &lob::L3Entry::timestamp)
        .def("__repr__", [](const lob::L3Entry& e) {
            std::ostringstream ss;
            ss << "L3Entry(id=" << e.id
               << ", price=" << e.price
               << ", qty=" << e.remaining_qty << ")";
            return ss.str();
        });

    // ---- BookStats struct ----

    py::class_<lob::BookStats>(m, "BookStats", "Aggregated book statistics")
        .def(py::init<>())
        .def_readwrite("message_count", &lob::BookStats::message_count)
        .def_readwrite("trade_count",   &lob::BookStats::trade_count)
        .def_readwrite("total_volume",  &lob::BookStats::total_volume)
        .def_readwrite("order_count",   &lob::BookStats::order_count)
        .def_readwrite("bid_levels",    &lob::BookStats::bid_levels)
        .def_readwrite("ask_levels",    &lob::BookStats::ask_levels)
        .def("__repr__", [](const lob::BookStats& s) {
            std::ostringstream ss;
            ss << "BookStats(msgs=" << s.message_count
               << ", trades=" << s.trade_count
               << ", volume=" << s.total_volume
               << ", orders=" << s.order_count
               << ", bid_lvls=" << s.bid_levels
               << ", ask_lvls=" << s.ask_levels << ")";
            return ss.str();
        });

    // ---- OrderBook class ----

    py::class_<lob::OrderBook>(m, "OrderBook",
        "High-performance Limit Order Book with price-time priority matching")
        .def(py::init<std::string, bool>(),
             py::arg("symbol") = "",
             py::arg("enable_self_trade_prevention") = false,
             "Create an order book for the given symbol")

        // Order operations
        .def("add_order",    &lob::OrderBook::add_order,
             py::arg("order"),
             R"pbdoc(
                 Submit a new order. Returns list of trades generated.
                 The order's id and timestamp will be auto-assigned if zero.
             )pbdoc")

        .def("cancel_order", &lob::OrderBook::cancel_order,
             py::arg("order_id"),
             "Cancel a resting order. Returns True if found and cancelled.")

        .def("modify_order", &lob::OrderBook::modify_order,
             py::arg("order_id"),
             py::arg("new_price"),
             py::arg("new_qty"),
             R"pbdoc(
                 Modify a resting order (cancel-and-replace).
                 Price change or qty increase loses time priority.
                 Returns any trades triggered by the modification.
             )pbdoc")

        // Market data
        .def("get_best_bid",  &lob::OrderBook::get_best_bid,
             "Best (highest) bid price, or None if no bids")
        .def("get_best_ask",  &lob::OrderBook::get_best_ask,
             "Best (lowest) ask price, or None if no asks")
        .def("get_spread",    &lob::OrderBook::get_spread,
             "Bid-ask spread, or None if either side empty")
        .def("get_mid_price", &lob::OrderBook::get_mid_price,
             "Mid price, or None if either side empty")
        .def("get_depth",     &lob::OrderBook::get_depth,
             py::arg("levels") = 10,
             "L2 depth snapshot: (bids, asks) as lists of L2Level")
        .def("get_l3_depth",  &lob::OrderBook::get_l3_depth,
             py::arg("levels") = 10,
             "L3 depth snapshot: (bids, asks) as lists of L3Entry")
        .def("get_vwap",      &lob::OrderBook::get_vwap,
             py::arg("last_n") = 0,
             "VWAP of last N trades (0 = all), or None")

        .def("get_total_quantity", &lob::OrderBook::get_total_quantity,
             py::arg("side"),
             "Total resting quantity on a given side")
        .def("get_quantity_at_price", &lob::OrderBook::get_quantity_at_price,
             py::arg("side"), py::arg("price"),
             "Quantity at a specific price level")

        // Accessors
        .def_property_readonly("symbol", &lob::OrderBook::symbol)
        .def("get_stats",        &lob::OrderBook::get_stats,
             "Get aggregated book statistics")
        .def("get_order",        &lob::OrderBook::get_order,
             py::arg("order_id"),
             py::return_value_policy::reference_internal,
             "Look up an order by ID (None if not found)")
        .def("trade_history",    &lob::OrderBook::trade_history,
             py::return_value_policy::reference_internal,
             "Full trade history (reference to internal vector)")
        .def("clear",            &lob::OrderBook::clear,
             "Reset the book to empty state")

        // NumPy integration
        .def("trades_to_numpy", [](const lob::OrderBook& book) {
                 return trades_to_numpy(book.trade_history());
             },
             R"pbdoc(
                 Export trade history as NumPy array (N x 5).
                 Columns: buy_order_id, sell_order_id, price, quantity, timestamp.
             )pbdoc")

        .def("bulk_add_orders", [](lob::OrderBook& book, py::array_t<double> arr) {
                 return bulk_add_orders(book, arr);
             },
             py::arg("orders"),
             R"pbdoc(
                 Submit multiple orders from a NumPy array (N x 5).
                 Columns: side (0=BUY,1=SELL), type (0-3), price, quantity, id.
             )pbdoc")

        .def("depth_to_numpy", [](const lob::OrderBook& book, size_t levels) {
                 auto [bids, asks] = book.get_depth(levels);
                 return py::make_tuple(
                     depth_side_to_numpy(bids),
                     depth_side_to_numpy(asks)
                 );
             },
             py::arg("levels") = 10,
             R"pbdoc(
                 Export L2 depth as pair of NumPy arrays (N x 3).
                 Columns: price, quantity, num_orders.
             )pbdoc")

        .def("__repr__", [](const lob::OrderBook& book) {
            auto stats = book.get_stats();
            std::ostringstream ss;
            ss << "OrderBook(symbol='" << book.symbol()
               << "', orders=" << stats.order_count
               << ", bid_levels=" << stats.bid_levels
               << ", ask_levels=" << stats.ask_levels
               << ", trades=" << stats.trade_count << ")";
            return ss.str();
        });

    // ---- Module-level helpers ----

    m.def("make_order", [](lob::OrderId id, lob::Side side, lob::OrderType type,
                           lob::Price price, lob::Quantity qty) {
        lob::Order o;
        o.id            = id;
        o.side          = side;
        o.type          = type;
        o.price         = price;
        o.quantity       = qty;
        o.remaining_qty = qty;
        o.timestamp     = 0;  // Auto-assigned by OrderBook.
        o.status        = lob::OrderStatus::NEW;
        return o;
    },
    py::arg("id"),
    py::arg("side"),
    py::arg("type"),
    py::arg("price"),
    py::arg("quantity"),
    R"pbdoc(
        Convenience factory to create an Order.

        Example:
            >>> order = ob.make_order(1, ob.Side.BUY, ob.OrderType.LIMIT, 150.0, 100)
    )pbdoc");

    // Version info
    m.attr("__version__") = "1.0.0";
}
