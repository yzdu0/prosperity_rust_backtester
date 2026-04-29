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
        #"GALAXY_SOUNDS_",
        #"SLEEP_POD_",
        #"MICROCHIP_",
        "PEBBLES_",
        #"ROBOT_",
        #"UV_VISOR_",
        #"TRANSLATOR_",
        #"PANEL_",
        #"OXYGEN_SHAKE_",
        #"SNACKPACK_",
    )

    PEBBLES = {
        "PEBBLES_XS",
        "PEBBLES_S",
        "PEBBLES_M",
        "PEBBLES_L",
        "PEBBLES_XL",
    }
    ROUND5_LIMIT = 10
    QUOTE_SIZE = 10

    ROLLING_WINDOW_SIZE = 10
    SNIPE_EDGE = 20

    def __init__(self):
        self.rolling_windows = {product: [] for product in self.PEBBLES}

    def update_globals(self, updates: Dict[str, Any]):
        globals().update(updates)


    def run(self, state: TradingState):
        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            limit = self.limit_for(product)
            if limit is None:
                orders_by_product[product] = []
                continue
            position = int(state.position.get(product, 0))
            '''orders_by_product[product] = self.quote_both_sides(
                product,
                order_depth,
                position,
                limit,
            )'''
            orders_by_product[product] = self.market_maker(
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

    def market_maker(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        limit: int,
    ) -> List[Order]:
        orders: List[Order] = []

        #NewOrders, NewPosition = self.snipe(product, order_depth, position, limit)

        #orders += NewOrders

        #position = NewPosition

        orders += self.quote_both_sides(product, order_depth, position, limit)

        self.rolling_windows[product].append(self.mid_price(order_depth))
        if len(self.rolling_windows[product]) > self.ROLLING_WINDOW_SIZE:
            self.rolling_windows[product].pop(0)

        return orders

    def FV_rolling_window(self, product: str):
        return sum(self.rolling_windows[product]) / len(self.rolling_windows[product]) if self.rolling_windows[product] else None

    def mid_price(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
        
    def _best_bid_ask(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        bid_vol = depth.buy_orders.get(best_bid, 0) if best_bid else 0
        ask_vol = -depth.sell_orders.get(best_ask, 0) if best_ask else 0
        return best_bid, best_ask, bid_vol, ask_vol

    def snipe(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        limit: int,
    ) -> List[Order]:
        orders: List[Order] = []

        fv = self.FV_rolling_window(product)

        if fv:
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders)
                if best_bid < fv - self.SNIPE_EDGE:
                    quote = min(self.QUOTE_SIZE, limit - position)
                    orders.append(Order(product, best_bid + 1, quote))
                    position += quote

            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders)
                if best_ask > fv + self.SNIPE_EDGE:
                    quote = min(self.QUOTE_SIZE, position + limit)
                    orders.append(Order(product, best_ask - 1, -quote))
                    position -= quote
        return orders, position

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
        if buy_size > 0:
            orders.append(Order(product, bid_price, buy_size))
        if sell_size > 0:
            orders.append(Order(product, ask_price, -sell_size))
        return orders