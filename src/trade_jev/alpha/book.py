"""Rebuild the 10-level book from L2 Add / Update / Remove rows and sample it.

`Depth` is a 1-based row in the book. Feeds disagree on what Add and Remove do
to the rows below it, so two modes are supported:

    insert  Add inserts at the row and pushes the rows below down; Remove deletes
            the row and pulls the rows below up (NinjaTrader-style market depth).
    slot    Add and Update both overwrite the row; Remove clears it.

`mode="auto"` replays the first events both ways and keeps the mode whose book is
less often crossed or unsorted (`check_book`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_jev.alpha.io import ADD, ASK, BID, REMOVE

LEVELS = 10


def _replay(side, price, vol, depth, action, cut, mode: str):
    """Apply events and record the book after event index cut[k]-1 for each k.

    Returns px, sz arrays of shape (len(cut), 2, LEVELS); empty rows are NaN / 0.
    """
    books = ([], [])  # bid rows, ask rows: lists of [price, vol]
    n_out = len(cut)
    px = np.full((n_out, 2, LEVELS), np.nan)
    sz = np.zeros((n_out, 2, LEVELS))
    k = 0
    insert = mode == "insert"
    n = len(side)
    while k < n_out and cut[k] == 0:
        k += 1
    for i in range(n):
        rows = books[side[i]]
        d = depth[i] - 1
        a = action[i]
        if 0 <= d < LEVELS:
            if a == REMOVE:
                if d < len(rows):
                    if insert:
                        del rows[d]
                    else:
                        rows[d] = None
            elif a == ADD and insert:
                rows.insert(d, [price[i], vol[i]])
                if len(rows) > LEVELS:
                    del rows[LEVELS:]
            else:  # UPDATE, or ADD in slot mode
                while len(rows) <= d:
                    rows.append(None)
                rows[d] = [price[i], vol[i]]
        while k < n_out and cut[k] == i + 1:
            for s in (0, 1):
                for j, r in enumerate(books[s]):
                    if r is not None:
                        px[k, s, j] = r[0]
                        sz[k, s, j] = r[1]
            k += 1
    return px, sz


def _events(l2: pd.DataFrame):
    ev = l2[l2.type.isin([BID, ASK])]
    return (ev.ts.to_numpy(), ev.type.to_numpy().astype(np.int64).tolist(),
            ev.price.to_numpy().tolist(), ev.volume.to_numpy().tolist(),
            ev.depth.to_numpy().astype(np.int64).tolist(), ev.action.to_numpy().astype(np.int64).tolist())


def check_book(px: np.ndarray) -> float:
    """Share of snapshots that are crossed or whose levels are out of order (lower is better)."""
    bid, ask = px[:, 0], px[:, 1]
    with np.errstate(invalid="ignore"):
        crossed = bid[:, 0] >= ask[:, 0]
        bad_bid = (np.diff(bid, axis=1) > 0).any(axis=1)
        bad_ask = (np.diff(ask, axis=1) < 0).any(axis=1)
    has = ~np.isnan(bid[:, 0]) & ~np.isnan(ask[:, 0])
    if not has.any():
        return 1.0
    return float((crossed | bad_bid | bad_ask)[has].mean())


def pick_mode(l2: pd.DataFrame, n_events: int = 200_000) -> tuple[str, dict[str, float]]:
    ts, side, price, vol, depth, action = _events(l2.iloc[:n_events])
    if len(ts) == 0:
        return "insert", {}
    cut = np.unique(np.linspace(1, len(ts), num=min(len(ts), 2000)).astype(np.int64))
    scores = {m: check_book(_replay(side, price, vol, depth, action, cut, m)[0])
              for m in ("insert", "slot")}
    return min(scores, key=scores.get), scores


def snapshots(l2: pd.DataFrame, at_ns: np.ndarray, mode: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    """Book as of each time in `at_ns` (events with ts <= t applied).

    Returns (px, sz, mode) with px/sz shaped (len(at_ns), 2, LEVELS), side 0 = bid.
    """
    if mode == "auto":
        mode, _ = pick_mode(l2)
    ts, side, price, vol, depth, action = _events(l2)
    cut = np.searchsorted(ts, at_ns, side="right")
    px, sz = _replay(side, price, vol, depth, action, cut, mode)
    return px, sz, mode
