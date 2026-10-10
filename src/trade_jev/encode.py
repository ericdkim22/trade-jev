"""Decision context → Jev state. Encoders are pluggable so a labeled-features arm can be A/B'd later."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from trade_jev.data import TICK, Day, et_to_ns, ns_to_et, secs


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


def topbook(ctx: Context) -> dict:
    """Only the best bid / ask, price change and trade pressure: exact on MarketTick history, and a test of whether
    levels 2-10 add anything."""
    bid_px, bid_sz, ask_px, ask_sz = ctx.book
    f = features(ctx)
    i1 = _imb(bid_sz[0], ask_sz[0])
    return {
        "instrument": f["instrument"], "time_et": f["time_et"], "position": f["position"], "mid_price": f["mid_price"],
        "spread_ticks": f["spread_ticks"],
        "best_bid_size": int(bid_sz[0]), "best_ask_size": int(ask_sz[0]),
        "best_level_imbalance": i1,
        "price_change_ticks": f["price_change_ticks"],
        "net_aggressor_volume": f["net_aggressor_volume"],
        "summary": [
            f"Best quotes: {int(bid_sz[0])} bid vs {int(ask_sz[0])} offered, leaning to {_lean(i1, 0.2)}.",
            f["summary"][1], f["summary"][2],
        ],
    }


_SESSION: dict[int, dict] = {}  # id(day) → running session stats, extended as rows arrive


def _session(day, row: int) -> dict | None:
    """Open / high / low / opening range (first 15 min) of the mid since 09:30 ET, in ticks, up to `row`."""
    st = _SESSION.get(id(day))
    if st is None or st["day"] is not day:
        open_ns = et_to_ns(day.day, "09:30:00")
        st = _SESSION[id(day)] = {"day": day, "open_ns": open_ns, "or_end": open_ns + secs(900),
                                  "first": int(np.searchsorted(day.ts, open_ns)), "done": None,
                                  "open": None, "hi": None, "lo": None, "or_hi": None, "or_lo": None}
    if row < st["first"]:
        return None
    lo_row = st["first"] if st["done"] is None else st["done"] + 1
    if row >= lo_row:
        m = day.bid[lo_row:row + 1].astype(np.int64) + day.ask[lo_row:row + 1]  # 2 × mid, ticks
        if st["open"] is None:
            st["open"] = st["hi"] = st["lo"] = int(m[0])
        st["hi"], st["lo"] = max(st["hi"], int(m.max())), min(st["lo"], int(m.min()))
        in_or = day.ts[lo_row:row + 1] < st["or_end"]
        if in_or.any():
            st["or_hi"] = max(st["or_hi"] or -10**12, int(m[in_or].max()))
            st["or_lo"] = min(st["or_lo"] or 10**12, int(m[in_or].min()))
        st["done"] = row
    return st


def context(ctx: Context) -> dict:
    """`features` plus where price sits in the session and recent trend / volatility, in words."""
    f = features(ctx)
    day, row, t = ctx.day, ctx.row, ctx.t_ns
    mid2 = int(day.bid[row]) + int(day.ask[row])
    st = _session(day, row)

    def change(sec):
        r0 = int(day.row_at(t - secs(sec)))
        return round((mid2 - (int(day.bid[max(r0, 0)]) + int(day.ask[max(r0, 0)]))) / 2, 1)
    r5 = max(int(day.row_at(t - secs(300))), 0)
    m5 = day.bid[r5:row + 1].astype(np.int64) + day.ask[r5:row + 1]
    range5 = round(float(m5.max() - m5.min()) / 2, 1)
    minutes = (t - et_to_ns(day.day, "09:30:00")) / 60e9
    phase = ("before the open" if minutes < 0 else "first 30 minutes" if minutes < 30 else
             "morning" if minutes < 150 else "midday" if minutes < 270 else "afternoon" if minutes < 330 else "last hour")
    session = None
    words = [f"Time of day: {phase}."]
    if st and st["open"] is not None:
        rng = (st["hi"] - st["lo"]) / 2
        pos = (mid2 - st["lo"]) / (st["hi"] - st["lo"]) if st["hi"] > st["lo"] else 0.5
        session = {"change_from_open_ticks": round((mid2 - st["open"]) / 2, 1), "day_range_ticks": rng,
                   "position_in_day_range": round(pos, 2),
                   "meaning": "position_in_day_range: 0 = at the session low, 1 = at the session high"}
        where = "near the session high" if pos > 0.85 else "near the session low" if pos < 0.15 else "mid-range"
        words.append(f"Session: {'up' if mid2 >= st['open'] else 'down'} {abs(mid2 - st['open']) / 2:g} ticks from "
                     f"the 09:30 open, {where} of a {rng:g}-tick day range.")
        if st["or_hi"] is not None and t >= st["or_end"]:
            orr = "above" if mid2 > st["or_hi"] else "below" if mid2 < st["or_lo"] else "inside"
            session["opening_range"] = orr
            words.append(f"Price is {orr} the first 15 minutes' range.")
    c5, c15 = change(300), change(900)
    words.append(f"Trend: {c5:+g} ticks over 5 minutes, {c15:+g} over 15; the last 5 minutes spanned {range5:g} ticks.")
    return {**{k: v for k, v in f.items() if k != "summary"},
            "trend_ticks": {"last_5min": c5, "last_15min": c15, "range_last_5min": range5},
            "session": session,
            "summary": f["summary"] + words}


ENCODERS: dict[str, Callable[[Context], dict]] = {"raw_l10": raw_l10, "features": features, "topbook": topbook,
                                                   "context": context}
LOOKBACKS = (15, 60)
