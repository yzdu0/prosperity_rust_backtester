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


class Trader:
    LIMITS = {
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 200,
        "HYDROGEL_PACK": 200,
    }

    VOUCHERS = [
        "VEV_4000",
        "VEV_4500",
        "VEV_5000",
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
        "VEV_5400",
        "VEV_5500",
    ]

    STANDARD_DEVIATION = {
        "VEV_4000": 15.64,
        "VEV_4500": 15.64,
        "VEV_5000": 14.38,
        "VEV_5100": 12.75,
        "VEV_5200": 9.66,
        "VEV_5300": 6.23,
        "VEV_5400": 3.43,
        "VEV_5500": 1.74,
    }

    def __init__(self):
        self.velvetfruit_window: List[float] = [DEFAULT_SPOT_PRICE] * VELVET_WINDOW_SIZE
        self.trader_data = ""

    def update_globals(self, updates: Dict[str, Any]):
        globals().update(updates)

    def run(self, state: TradingState):
        result: dict[str, List[Order]] = {}
        self.trader_data = state.traderData or ""

        self._update_velvet_window(state)

        for symbol, depth in state.order_depths.items():
            pos = state.position.get(symbol, 0)
            limit = self.LIMITS.get(symbol, 200)

            if symbol == "HYDROGEL_PACK":
                pass
                #result[symbol] = self.hydrogel_mean_reversion(depth, pos, limit)
            elif symbol == "VELVETFRUIT_EXTRACT": #or symbol in self.VOUCHERS:
                result[symbol] = self.velvet_mean_reversion(depth, pos, limit, symbol, state.timestamp)
            elif symbol == "VEV_6000":
                result[symbol] = self._hedge(depth, pos, limit)

        return result, 0, self.trader_data

    def _update_velvet_window(self, state: TradingState) -> None:
        depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if depth is None:
            return

        best_bid, best_ask, _, _ = self._best_bid_ask(depth)
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
            self.velvetfruit_window.append(mid)
            while len(self.velvetfruit_window) > VELVET_WINDOW_SIZE:
                self.velvetfruit_window.pop(0)

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int], int, int]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        bid_vol = depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        ask_vol = -depth.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        return best_bid, best_ask, bid_vol, ask_vol

    @staticmethod
    def _inventory_skew(pos: int, limit: int, skew_at_limit: float) -> float:
        if limit <= 0:
            return 0.0
        return skew_at_limit * pos / limit

    def _hedge(self, depth: OrderDepth, pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        if pos < limit:
            orders.append(Order("VEV_6000", 1, limit - pos))
        return orders

    def _velvet_spot_fair_value(self) -> float:
        rolling_mid = sum(self.velvetfruit_window) / len(self.velvetfruit_window)
        return rolling_mid * VELVET_BLEND_PCT / 100.0 + DEFAULT_SPOT_PRICE * (1.0 - VELVET_BLEND_PCT / 100.0)

    def _voucher_fair_value(self, symbol: str, timestamp: int, spot_fv: float) -> float:
        strike = float(extract_strike(symbol))
        option_spot = spot_fv if VOUCHER_USE_DYNAMIC_SPOT else DEFAULT_SPOT_PRICE
        return black_scholes_call(
            spot=max(option_spot, 1e-8),
            strike=strike,
            annualized_vol=FITTED_ANNUALIZED_VOLS[symbol],
            tte_days_value=tte_days(CURRENT_DAY, timestamp + OPTION_TTE_OFFSET_US),
        )

    def _voucher_edge(self, symbol: str) -> int:
        edge = VELVET_SNIPE_EDGE
        scale = self.STANDARD_DEVIATION.get(symbol, self.STANDARD_DEVIATION["VEV_4000"])
        edge *= scale / self.STANDARD_DEVIATION["VEV_4000"]
        edge *= VOUCHER_EDGE_MULTIPLIER
        return max(1, int(round(edge)))

    def hydrogel_mean_reversion(self, depth: OrderDepth, pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        fv = HYDROGEL_FAIR_VALUE - self._inventory_skew(pos, limit, HYDROGEL_INVENTORY_SKEW)
        snipe_edge = HYDROGEL_SNIPE_EDGE

        for ask_px in sorted(depth.sell_orders.keys()):
            if ask_px < fv - snipe_edge:
                qty = min(-depth.sell_orders[ask_px], HYDROGEL_MAX_TAKE_SIZE, int(limit - pos))
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", ask_px, qty))
                    pos += qty

        for bid_px in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_px > fv + snipe_edge:
                qty = min(depth.buy_orders[bid_px], HYDROGEL_MAX_TAKE_SIZE, int(limit + pos))
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", bid_px, -qty))
                    pos -= qty

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else int(fv - 30)
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else int(fv + 30)
        my_bid = best_bid + 1
        my_ask = best_ask - 1

        if my_bid >= my_ask:
            my_bid = int(fv) - 1
            my_ask = int(fv) + 1

        passive_edge = snipe_edge * HYDROGEL_PASSIVE_EDGE_MULTIPLIER / 100.0
        bid_qty = min(HYDROGEL_PASSIVE_ORDER_SIZE, max(0, limit - pos - HYDROGEL_PASSIVE_RESERVE))
        ask_qty = min(HYDROGEL_PASSIVE_ORDER_SIZE, max(0, limit + pos - HYDROGEL_PASSIVE_RESERVE))

        if bid_qty > 0 and my_bid < fv - passive_edge:
            orders.append(Order("HYDROGEL_PACK", my_bid, bid_qty))
        if ask_qty > 0 and my_ask > fv + passive_edge:
            orders.append(Order("HYDROGEL_PACK", my_ask, -ask_qty))

        return orders

    def velvet_mean_reversion(
        self,
        depth: OrderDepth,
        pos: int,
        limit: int,
        symbol: str,
        timestamp: int,
    ) -> List[Order]:
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

        return orders
