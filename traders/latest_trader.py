from datamodel import OrderDepth, TradingState, Order
from typing import List, Tuple, Optional, Dict, Any
from math import erf, log, sqrt

# =========================
# GRID SEARCH PARAMETERS
# =========================

DAY_0_START_TTE_DAYS = 8.0
DEFAULT_SPOT_PRICE = 5250.0
TIMESTAMP_UNITS_PER_DAY = 1_000_000
CURRENT_DAY = 0

# Velvet / voucher mean reversion
VELVET_WINDOW_SIZE = 100
VELVET_BLEND_PCT = 25
VELVET_SNIPE_EDGE = 13
VELVET_MAX_TAKE_SIZE = 12

VELVET_PASSIVE_ORDER_SIZE = 24
VELVET_PASSIVE_EDGE_MULTIPLIER = 35
VELVET_PASSIVE_RESERVE = 20
VELVET_INVENTORY_SKEW = 5.0

# Voucher-specific fair value / edge scaling
OPTION_TTE_OFFSET_US = 0
VOUCHER_USE_DYNAMIC_SPOT = True
VOUCHER_EDGE_MULTIPLIER = 1.00

# Hydrogel mean reversion
HYDROGEL_FAIR_VALUE = 10000.0
HYDROGEL_SNIPE_EDGE = 6
HYDROGEL_MAX_TAKE_SIZE = 5
HYDROGEL_PASSIVE_ORDER_SIZE = 4
HYDROGEL_PASSIVE_RESERVE = 12
HYDROGEL_PASSIVE_EDGE_MULTIPLIER = 20
HYDROGEL_INVENTORY_SKEW = 3.0

FITTED_ANNUALIZED_VOLS = {
    "VEV_4000": 0.524471,
    "VEV_4500": 0.305594,
    "VEV_5000": 0.241909,
    "VEV_5100": 0.240341,
    "VEV_5200": 0.242147,
    "VEV_5300": 0.244538,
    "VEV_5400": 0.229586,
    "VEV_5500": 0.248458,
    "VEV_6000": 0.377506,
    "VEV_6500": 0.570137,
}


def extract_strike(option_product: str) -> int:
    prefix = "VEV_"
    if not option_product.startswith(prefix):
        raise ValueError(f"Unsupported option product: {option_product}")
    return int(option_product[len(prefix):])


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_call(
    spot: float,
    strike: float,
    annualized_vol: float,
    tte_days_value: float,
) -> float:
    if spot <= 0.0:
        return 0.0

    intrinsic = max(spot - strike, 0.0)
    time_years = max(tte_days_value, 1e-8) / 365.0
    sigma_sqrt_t = max(annualized_vol * sqrt(time_years), 1e-12)

    if sigma_sqrt_t <= 1e-10:
        return intrinsic

    d1 = (log(spot / strike) + 0.5 * sigma_sqrt_t * sigma_sqrt_t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return max(intrinsic, spot * normal_cdf(d1) - strike * normal_cdf(d2))


def start_of_day_tte_days(day: int) -> float:
    return max(DAY_0_START_TTE_DAYS - day, 1e-8)


def tte_days(day: int, timestamp: int) -> float:
    return max(start_of_day_tte_days(day) - timestamp / TIMESTAMP_UNITS_PER_DAY, 1e-8)

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List


class Trader:
    LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }
    ROUND5_PREFIXES = (
        "GALAXY_SOUNDS_",
        "SLEEP_POD_",
        "MICROCHIP_",
        "PEBBLES_",
        "ROBOT_",
        "UV_VISOR_",
        "TRANSLATOR_",
        "PANEL_",
        "OXYGEN_SHAKE_",
        "SNACKPACK_",
    )
    ROUND5_LIMIT = 10
    QUOTE_SIZE = 5

    def run(self, state: TradingState):
        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            limit = self.limit_for(product)
            if limit is None:
                orders_by_product[product] = []
                continue
            position = int(state.position.get(product, 0))
            orders_by_product[product] = self.quote_both_sides(
                product,
                order_depth,
                position,
                limit,
            )

        return orders_by_product, 0, ""

    def limit_for(self, product: str):
        if product in self.LIMITS:
            return self.LIMITS[product]
        if product.startswith(self.ROUND5_PREFIXES):
            return self.ROUND5_LIMIT
        return None

    def quote_both_sides(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        limit: int,
    ) -> List[Order]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        if best_bid >= best_ask:
            return []

        if best_ask - best_bid > 1:
            bid_price = best_bid + 1
            ask_price = best_ask - 1
        else:
            bid_price = best_bid
            ask_price = best_ask

        buy_size = min(self.QUOTE_SIZE, max(0, limit - position))
        sell_size = min(self.QUOTE_SIZE, max(0, limit + position))

        orders: List[Order] = []
        spot_fv = self._velvet_spot_fair_value()

        if symbol.startswith("VEV_"):
            fv = self._voucher_fair_value(symbol, timestamp, spot_fv)
            snipe_edge = self._voucher_edge(symbol)
        else:
            fv = spot_fv
            snipe_edge = VELVET_SNIPE_EDGE

        fv -= self._inventory_skew(pos, limit, VELVET_INVENTORY_SKEW)

        for ask_px in sorted(depth.sell_orders.keys()):
            if ask_px < fv - snipe_edge:
                qty = min(-depth.sell_orders[ask_px], VELVET_MAX_TAKE_SIZE, int(limit - pos))
                if qty > 0:
                    orders.append(Order(symbol, ask_px, qty))
                    pos += qty

        for bid_px in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_px > fv + snipe_edge:
                qty = min(depth.buy_orders[bid_px], VELVET_MAX_TAKE_SIZE, int(limit + pos))
                if qty > 0:
                    orders.append(Order(symbol, bid_px, -qty))
                    pos -= qty

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else int(fv - 30)
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else int(fv + 30)
        my_bid = best_bid + 1
        my_ask = best_ask - 1

        if my_bid >= my_ask:
            my_bid = int(fv) - 1
            my_ask = int(fv) + 1

        passive_edge = snipe_edge * VELVET_PASSIVE_EDGE_MULTIPLIER / 100.0
        bid_qty = min(VELVET_PASSIVE_ORDER_SIZE, max(0, limit - pos - VELVET_PASSIVE_RESERVE))
        ask_qty = min(VELVET_PASSIVE_ORDER_SIZE, max(0, limit + pos - VELVET_PASSIVE_RESERVE))

        if bid_qty > 0 and my_bid < fv - passive_edge:
            orders.append(Order(symbol, my_bid, bid_qty))
        if ask_qty > 0 and my_ask > fv + passive_edge:
            orders.append(Order(symbol, my_ask, -ask_qty))

        if buy_size > 0:
            orders.append(Order(product, bid_price, buy_size))
        if sell_size > 0:
            orders.append(Order(product, ask_price, -sell_size))
        return orders
