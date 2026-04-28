from datamodel import OrderDepth, TradingState, Order
from typing import List, Tuple, Optional, Dict, Any
from math import erf, log, sqrt

DAY_0_START_TTE_DAYS = 8.0
DEFAULT_SPOT_PRICE = 5250.0
TIMESTAMP_UNITS_PER_DAY = 1_000_000
CURRENT_DAY = 3

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

BLEND = 5
WINDOW_SIZE = 3
SNIPE_EDGE = 15
PASSIVE_EDGE_MULTIPLIER = 20

MARK_67_WINDOW_SIZE = 10
MARK_67_SIGNAL_STRENGTH = 1

HYDROGEL_SNIPE_EDGE = 5
HYDROGEL_MAX_ORDER_SIZE = 4
HYDROGEL_PASSIVE_RESERVE = 10


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
    tte_days: float,
) -> float:
    if spot <= 0.0:
        return 0.0

    intrinsic = max(spot - strike, 0.0)
    time_years = max(tte_days, 1e-8) / 365.0
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


def estimate_extrinsic(
    day: int,
    timestamp: int,
    option_product: str,
    spot_price: float = DEFAULT_SPOT_PRICE,
) -> float:
    annualized_vol = FITTED_ANNUALIZED_VOLS[option_product]
    strike = extract_strike(option_product)
    call_value = black_scholes_call(
        spot=spot_price,
        strike=float(strike),
        annualized_vol=annualized_vol,
        tte_days=tte_days(day, timestamp),
    )
    return call_value - (spot_price - strike)


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
        self.velvetfruit_window: List[float] = [5250] * WINDOW_SIZE
        self.Mark67RecentBuyVolume = [0] * MARK_67_WINDOW_SIZE

    def update_globals(self, updates: Dict[str, Any]):
        globals().update(updates)

    def run(self, state: TradingState):
        result: dict[str, List[Order]] = {}

        self.Mark67RecentBuyVolume.append(0)
        self.Mark67RecentBuyVolume.pop(0)

        for trade in state.market_trades.get("VELVETFRUIT_EXTRACT", []):
            if trade.buyer == "Mark 67":
                self.Mark67RecentBuyVolume[-1] += trade.quantity
                #if trade.price > min(state.order_depths["VELVETFRUIT_EXTRACT"].sell_orders.keys(), default=0):
                #    self.signal = 1

        for symbol in state.order_depths:
            if symbol == "VELVETFRUIT_EXTRACT":
                depth = state.order_depths[symbol]
                best_bid, best_ask, _, _ = self._best_bid_ask(depth)
                if best_bid is not None and best_ask is not None:
                    mid = (best_bid + best_ask) / 2.0
                    self.velvetfruit_window.append(mid)
                    if len(self.velvetfruit_window) > WINDOW_SIZE:
                        self.velvetfruit_window.pop(0)

        for symbol in state.order_depths:
            pos = state.position.get(symbol, 0)
            limit = self.LIMITS.get(symbol, 200)
            depth = state.order_depths[symbol]

            if symbol == "HYDROGEL_PACK":
                result[symbol] = self._trade_hydrogel(depth, pos, limit)
            elif symbol == "VELVETFRUIT_EXTRACT": #or symbol in self.VOUCHERS:
                result[symbol] = self._market_maker(depth, pos, limit, symbol, state.timestamp)
            elif symbol == "VEV_6000":
                result[symbol] = self._hedge(depth, pos, limit)

        return result, 0, ""

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int], int, int]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        bid_vol = depth.buy_orders.get(best_bid, 0) if best_bid else 0
        ask_vol = -depth.sell_orders.get(best_ask, 0) if best_ask else 0
        return best_bid, best_ask, bid_vol, ask_vol

    def _hedge(self, depth: OrderDepth, pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        if pos < limit:
            orders.append(Order("VEV_6000", 1, limit - pos))
        return orders

    def _trade_hydrogel(self, depth: OrderDepth, pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        fv = 10000.0
        snipe_edge = HYDROGEL_SNIPE_EDGE
        max_order_size = HYDROGEL_MAX_ORDER_SIZE

        for ask_px in sorted(depth.sell_orders.keys()):
            if ask_px < fv - snipe_edge:
                qty = min(-depth.sell_orders[ask_px], max_order_size, int(limit - pos))
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", ask_px, qty))
                    pos += qty

        for bid_px in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_px > fv + snipe_edge:
                qty = min(depth.buy_orders[bid_px], max_order_size, int(limit + pos))
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", bid_px, -qty))
                    pos -= qty

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else fv - 30
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else fv + 30
        my_bid = best_bid + 1
        my_ask = best_ask - 1

        if my_bid >= my_ask:
            my_bid = int(fv) - 1
            my_ask = int(fv) + 1

        bid_qty = min(max_order_size, limit - pos - HYDROGEL_PASSIVE_RESERVE)
        ask_qty = min(max_order_size, limit + pos - HYDROGEL_PASSIVE_RESERVE)

        if bid_qty > 0 and my_bid < fv:
            orders.append(Order("HYDROGEL_PACK", my_bid, bid_qty))
        if ask_qty > 0 and my_ask > fv:
            orders.append(Order("HYDROGEL_PACK", my_ask, -ask_qty))

        return orders

    def _market_maker(
        self,
        depth: OrderDepth,
        pos: int,
        limit: int,
        symbol: str,
        timestamp: int,
    ) -> List[Order]:
        orders: List[Order] = []
        snipe_edge = SNIPE_EDGE
        fv = sum(self.velvetfruit_window) / len(self.velvetfruit_window) * BLEND / 100 + 5250 * (1 - BLEND / 100)

        #fv += (sum(self.Mark67RecentBuyVolume) / MARK67_WINDOW_SIZE) * MARK_SIGNAL_STRENGTH

        if symbol.startswith("VEV_"):
            voucher_price = int(symbol[-4:])
            fv -= voucher_price
            fv += estimate_extrinsic(CURRENT_DAY, timestamp + 1000 * 1000, symbol, 5250)
            snipe_edge = int(SNIPE_EDGE * self.STANDARD_DEVIATION.get(symbol, 10) / self.STANDARD_DEVIATION["VEV_4000"])

        Mark67Signal = (sum(self.Mark67RecentBuyVolume) / MARK_67_WINDOW_SIZE) * MARK_67_SIGNAL_STRENGTH / 10000

        fv *= (1 - Mark67Signal)

        for ask_px in sorted(depth.sell_orders.keys()):
            if ask_px < fv - snipe_edge:
                qty = min(-depth.sell_orders[ask_px], int(limit - pos))
                if qty > 0:
                    orders.append(Order(symbol, ask_px, qty))
                    pos += qty

        for bid_px in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_px > fv + snipe_edge:
                qty = min(depth.buy_orders[bid_px], int(limit + pos))
                if qty > 0:
                    orders.append(Order(symbol, bid_px, -qty))
                    pos -= qty

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else fv - 30
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else fv + 30
        my_bid = best_bid + 1
        my_ask = best_ask - 1

        if my_bid >= my_ask:
            my_bid = int(fv) - 1
            my_ask = int(fv) + 1

        bid_qty = int(min(30, limit - pos))
        ask_qty = int(min(30, limit + pos))

        if bid_qty > 0:
            if my_bid < fv - snipe_edge * PASSIVE_EDGE_MULTIPLIER / 100:
                orders.append(Order(symbol, my_bid, bid_qty))
        if ask_qty > 0:
            if my_ask > fv + snipe_edge * PASSIVE_EDGE_MULTIPLIER / 100:
                orders.append(Order(symbol, my_ask, -ask_qty))

        return orders
