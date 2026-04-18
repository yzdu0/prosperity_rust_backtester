from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json

# ─────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────

POSITION_LIMIT: Dict[str, int] = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

OSMIUM_FAIR_VALUE    = 10_000
OSMIUM_MM_INSIDE     = 60      # max distance from FV for passive quotes
OSMIUM_CLIP          = 10      # max qty per side per tick (≤ max observed market order)
SNIPE_POSITION_LIMIT = 40      # max net position built purely through sniping
WINDOW_SIZE          = 25      # rolling-average window for OSMIUM fair value
DEVIATION_MULTIPLIER = 2.0     # how much extra room to give when sniping based on current FV deviation from long-run FV


# ─────────────────────────────────────────
#  Trader
# ─────────────────────────────────────────

class Trader:
    """
    IMC Prosperity-compliant Trader.

    Persistent state (osmium FV window, pepper first-ask, timestep) is
    serialised into traderData so it survives even if the engine
    re-instantiates the class between ticks.
    """

    def __init__(self):
        self.osmium_window: List[float] = [10000.0] * 10
        self.best_ask_ever: Optional[float] = None
        self.timestep: int = 0

    def update_globals(self, osmium_clip, snipe_position_limit, window_size, deviation_multiplier):
        global OSMIUM_CLIP, SNIPE_POSITION_LIMIT, WINDOW_SIZE, DEVIATION_MULTIPLIER
        OSMIUM_CLIP = osmium_clip
        SNIPE_POSITION_LIMIT = snipe_position_limit
        WINDOW_SIZE = window_size
        DEVIATION_MULTIPLIER = deviation_multiplier

    # ── internal helpers ────────────────────────────────────────────────

    def _load_state(self, raw: str) -> None:
        if not raw:
            return
        try:
            d = json.loads(raw)
            if "osmium_window" in d:
                self.osmium_window = d["osmium_window"]
            if "best_ask_ever" in d:
                self.best_ask_ever = d["best_ask_ever"]
            if "timestep" in d:
                self.timestep = d["timestep"]
        except Exception:
            pass  # keep defaults on corrupt state

    def _save_state(self) -> str:
        return json.dumps({
            "osmium_window": self.osmium_window,
            "best_ask_ever": self.best_ask_ever,
            "timestep": self.timestep,
        })

    @staticmethod
    def _best_bid_ask(
        depth: OrderDepth,
    ) -> Tuple[Optional[int], Optional[int], int, int]:
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        bid_vol  = depth.buy_orders.get(best_bid, 0)   if best_bid else 0
        ask_vol  = -depth.sell_orders.get(best_ask, 0) if best_ask else 0
        return best_bid, best_ask, bid_vol, ask_vol

    # ── main entry point ────────────────────────────────────────────────

    def run(self, state: TradingState):
        self._load_state(state.traderData)

        result: Dict[str, List[Order]] = {}

        for symbol in state.order_depths:
            pos   = state.position.get(symbol, 0)
            limit = POSITION_LIMIT.get(symbol, 80)
            depth = state.order_depths[symbol]

            if symbol == "ASH_COATED_OSMIUM":
                result[symbol] = self._trade_osmium(depth, pos, limit)
            elif symbol == "INTARIAN_PEPPER_ROOT":
                result[symbol] = self._trade_root(depth, pos, limit)

        self.timestep += 1

        return result, 0, self._save_state()

    # ── INTARIAN_PEPPER_ROOT ─────────────────────────────────────────────
    # Strategy: trend-following. Lock in 80-unit long as quickly as
    # possible and hold for the entire session.  Product reliably gains
    # ~100 points per 100k-timestamp window, so any early buy beats a
    # later buy.

    def _trade_root(
        self,
        depth: OrderDepth,
        pos: int,
        limit: int,
    ) -> List[Order]:
        orders: List[Order] = []

        _, best_ask, _, _ = self._best_bid_ask(depth)

        # Anchor on the first ask ever observed so we don't chase a spike
        if self.best_ask_ever is None and best_ask is not None:
            self.best_ask_ever = best_ask

        if not depth.sell_orders or pos >= limit or self.best_ask_ever is None:
            return orders

        for ask_px in sorted(depth.sell_orders.keys()):
            # For the first 10 ticks, only buy within 1 tick of the
            # best-ever ask (avoids a bad early spike).
            # After tick 10 the product has already started its trend;
            # buy everything available unconditionally.
            if ask_px > self.best_ask_ever + 1 and self.timestep < 10:
                break

            avail = -depth.sell_orders[ask_px]
            room  = limit - pos
            qty   = min(avail, room)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ask_px, qty))
                pos += qty

        return orders

    # ── ASH_COATED_OSMIUM ────────────────────────────────────────────────
    # Strategy: mean-reversion market-making.
    # Fair value = lagging rolling average of mid prices (≈ 10 000).
    # 1. Snipe asks that are STRICTLY BELOW fair value (mean-reversion
    #    buy) and bids STRICTLY ABOVE fair value (mean-reversion sell).
    # 2. Passive market-make inside the prevailing spread.

    def _fair_osmium(self, depth: OrderDepth) -> float:
        best_bid, best_ask, _, _ = self._best_bid_ask(depth)
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
            # Update the rolling window AFTER computing current FV so the
            # estimate always lags slightly (prevents self-fulfilling loops)
            fv = sum(self.osmium_window) / len(self.osmium_window)
            self.osmium_window.append(mid)
            if len(self.osmium_window) > WINDOW_SIZE:
                self.osmium_window.pop(0)
            return fv
        return float(OSMIUM_FAIR_VALUE)

    def _trade_osmium(
        self,
        depth: OrderDepth,
        pos: int,
        limit: int,
    ) -> List[Order]:
        orders: List[Order] = []

        fv        = self._fair_osmium(depth)
        deviation = fv - OSMIUM_FAIR_VALUE   # negative = market below long-run FV

        # ── 1. Snipe: buy below fair value ─────────────────────────────
        # BUG FIX vs original: cast room/qty to int and guard against
        # negative room.  Float order quantities are engine-rejected.
        for ask_px in sorted(depth.sell_orders.keys()):
            if ask_px < fv:                          # strictly below fair → edge
                avail = -depth.sell_orders[ask_px]
                # When market is below long-run FV (deviation < 0), give
                # ourselves slightly more room to buy (mean-reversion bet).
                room = int(SNIPE_POSITION_LIMIT - pos - deviation * DEVIATION_MULTIPLIER)
                qty  = int(min(avail, max(0, room)))
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ask_px, qty))
                    pos += qty

        # ── 2. Snipe: sell above fair value ────────────────────────────
        for bid_px in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_px > fv:                          # strictly above fair → edge
                avail = depth.buy_orders[bid_px]
                room  = int(SNIPE_POSITION_LIMIT + pos + deviation * DEVIATION_MULTIPLIER)
                qty   = int(min(avail, max(0, room)))
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bid_px, -qty))
                    pos -= qty

        # ── 3. Passive market-making ────────────────────────────────────
        best_bid = max(depth.buy_orders.keys())  if depth.buy_orders  else 9960
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 10040

        my_bid = int(max(best_bid + 1, fv - OSMIUM_MM_INSIDE))
        my_ask = int(min(best_ask - 1, fv + OSMIUM_MM_INSIDE))

        # Guard: never post a crossed or zero-spread quote
        if my_bid >= my_ask:
            my_bid = int(fv) - 1
            my_ask = int(fv) + 1

        buy_room  = limit - pos
        sell_room = limit + pos
        bid_qty   = int(min(OSMIUM_CLIP, buy_room))
        ask_qty   = int(min(OSMIUM_CLIP, sell_room))

        if bid_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_bid,  bid_qty))
        if ask_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", my_ask, -ask_qty))

        return orders