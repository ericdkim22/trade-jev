"""Decision context → Jev state. Encoders are pluggable so a labeled-features arm can be A/B'd later."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from trade_jev.data import TICK, Day, ns_to_et


@dataclass
class Position:
    side: int = 0           # +1 long, -1 short, 0 flat
    entry_px: float = 0.0   # points
    entry_ns: int = 0


@dataclass
class Context:
    """Everything a policy may look at for one decision. Only rows <= `row` are visible."""
    day: Day
    t_ns: int
    row: int
    prev_rows: dict[int, int]  # lookback seconds → row index at t - lookback
    position: Position

    @property
    def book(self) -> np.ndarray:
        return self.day.book(self.row)

    @property
    def mid(self) -> float:
        return float(self.day.mid(self.row))

    def delta_since(self, lookback_s: int) -> int:
        r0 = self.prev_rows[lookback_s]
        return int(self.day.cum_delta[self.row] - (self.day.cum_delta[r0] if r0 >= 0 else 0))


def _px(ticks) -> float:
    return float(ticks) * TICK


def _position_state(ctx: Context) -> dict:
    p = ctx.position
    if p.side == 0:
        return {"side": "flat", "contracts": 0}
    # unrealized PnL marks to the side we'd exit on (bid for long, ask for short)
    bid, ask = _px(ctx.day.bid[ctx.row]), _px(ctx.day.ask[ctx.row])
    exit_px = bid if p.side > 0 else ask
    return {
        "side": "long" if p.side > 0 else "short",
        "contracts": 1,
        "entry_price": p.entry_px,
        "unrealized_points": round((exit_px - p.entry_px) * p.side, 2),
        "seconds_held": int((ctx.t_ns - p.entry_ns) / 1e9),
    }


INSTRUMENTS = {
    "NQ": "Nasdaq-100 E-mini futures, tick size 0.25, $5 per tick",
    "MNQ": "Micro E-mini Nasdaq-100 futures, tick size 0.25, $0.50 per tick",
}


def raw_l10_instrument(ctx: Context) -> str:
    return f"{ctx.day.symbol} ({INSTRUMENTS['MNQ' if ctx.day.symbol.startswith('MNQ') else 'NQ']})"


def raw_l10(ctx: Context) -> dict:
    """Raw numbers: L10 ladder + recent flow + recent mids + our position."""
    b = ctx.book
    bid_px, bid_sz, ask_px, ask_sz = b
    return {
        "instrument": raw_l10_instrument(ctx),
        "time_et": ns_to_et(ctx.t_ns),
        "position": _position_state(ctx),
        "order_book": {
            # asks listed far → near so the ladder reads top-down like a DOM
            "asks": [[_px(p), int(s)] for p, s in zip(ask_px[::-1], ask_sz[::-1])],
            "bids": [[_px(p), int(s)] for p, s in zip(bid_px, bid_sz)],
        },
        "mid_price": ctx.mid,
        "mid_price_15s_ago": float(ctx.day.mid(max(ctx.prev_rows[15], 0))),
        "mid_price_60s_ago": float(ctx.day.mid(max(ctx.prev_rows[60], 0))),
        "net_aggressor_volume_last_15s": ctx.delta_since(15),
        "net_aggressor_volume_last_60s": ctx.delta_since(60),
    }


def _imb(b, a) -> float:
    b, a = float(b), float(a)
    return round((b - a) / (b + a), 3) if b + a else 0.0


def _lean(x: float, th: float) -> str:
    return "buyers" if x > th else "sellers" if x < -th else "neither side"


def features(ctx: Context) -> dict:
    """The same moment as `raw_l10`, pre-computed and labeled: Jev reads words better than it does arithmetic."""
    bid_px, bid_sz, ask_px, ask_sz = ctx.book
    m15 = round((ctx.mid - float(ctx.day.mid(max(ctx.prev_rows[15], 0)))) / TICK, 1)
    m60 = round((ctx.mid - float(ctx.day.mid(max(ctx.prev_rows[60], 0)))) / TICK, 1)
    d15, d60 = ctx.delta_since(15), ctx.delta_since(60)
    i1, i5, i10 = _imb(bid_sz[0], ask_sz[0]), _imb(bid_sz[:5].sum(), ask_sz[:5].sum()), _imb(bid_sz.sum(), ask_sz.sum())
    move = lambda m: f"up {m:g}" if m > 0 else f"down {-m:g}" if m < 0 else "unchanged"  # noqa: E731
    return {
        "instrument": raw_l10_instrument(ctx),
        "time_et": ns_to_et(ctx.t_ns),
        "position": _position_state(ctx),
        "mid_price": ctx.mid,
        "spread_ticks": int(ask_px[0] - bid_px[0]),
        "book_imbalance": {
            "best_level": i1, "top_5_levels": i5, "all_10_levels": i10,
            "meaning": "(bid size - ask size) / (bid size + ask size): +1 = all resting size on the bid, -1 = all on the ask",
        },
        "price_change_ticks": {"last_15s": m15, "last_60s": m60},
        "net_aggressor_volume": {
            "last_15s": d15, "last_60s": d60,
            "meaning": "contracts bought at the ask minus contracts sold at the bid",
        },
        "summary": [
            f"Resting orders: the top 5 levels lean to {_lean(i5, 0.2)}, the full book to {_lean(i10, 0.2)}.",
            f"Aggressive orders in the last 60s: {_lean(d60, 0)} ({d60:+d} contracts net).",
            f"Price: {move(m15)} ticks in 15s, {move(m60)} ticks in 60s.",
        ],
    }


ENCODERS: dict[str, Callable[[Context], dict]] = {"raw_l10": raw_l10, "features": features}
LOOKBACKS = (15, 60)
