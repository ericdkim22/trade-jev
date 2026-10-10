import asyncio

import numpy as np

from trade_jev.data import LEVELS, Day, secs
from trade_jev.harness import Config, run_day
from trade_jev.policies import Decision

S = secs(1)


def make_day(bids: list[int], n_sec: int | None = None) -> Day:
    """One book row per second; ask = bid + 1 tick; full books stored for every row."""
    n = len(bids)
    bid = np.array(bids, dtype=np.int32)
    ask = bid + 1
    books = np.zeros((n, 4, LEVELS), dtype=np.int64)
    books[:, 0] = bid[:, None] - np.arange(LEVELS)
    books[:, 2] = ask[:, None] + np.arange(LEVELS)
    books[:, 1] = books[:, 3] = 5
    return Day("2026-01-02", "NQH6", np.arange(n, dtype=np.int64) * S, bid, ask,
               np.zeros(n, dtype=np.int64), np.arange(n), books)


class Script:
    name = "script"

    def __init__(self, actions: dict[int, str]):
        self.actions = actions  # second → action

    async def __call__(self, ctx):
        return Decision(self.actions.get(ctx.t_ns // S, "HOLD"))


def run(day, actions, **cfg):
    c = Config(**{"cadence_s": 1, "latency_ms": 0, "commission": 0, "stop_ticks": 20, "target_ticks": 40, **cfg})
    grid = np.arange(0, len(day.ts) - 1, dtype=np.int64) * S
    return asyncio.run(run_day(day, grid, Script(actions), c))


def test_buy_fills_at_ask_and_eod_exits_at_bid():
    day = make_day([400] * 5 + [404] * 5)
    r = run(day, {0: "BUY"})
    (t,) = r.trades
    assert t.side == 1 and t.entry_px == 401 * 0.25
    assert t.reason == "eod" and t.exit_px == 404 * 0.25
    assert t.points == 0.75 and t.pnl == 15.0


def test_reverse_closes_and_opens_at_same_touch():
    day = make_day([400, 400, 402, 402, 402, 402])
    r = run(day, {0: "BUY", 2: "SELL"})
    long_, short = r.trades
    assert (long_.side, long_.reason, long_.exit_px) == (1, "jev", 402 * 0.25)
    assert (short.side, short.entry_px, short.reason) == (-1, 402 * 0.25, "eod")


def test_buy_while_long_is_noop():
    day = make_day([400] * 6)
    r = run(day, {0: "BUY", 1: "BUY", 2: "BUY"})
    assert len(r.trades) == 1


def test_stop_fires_between_decisions_at_touch():
    # enter long @ 401 ask; stop 4 ticks → bid <= 397; gap straight to 395
    day = make_day([400, 400, 399, 395, 395, 395])
    r = run(day, {0: "BUY"}, stop_ticks=4, target_ticks=100)
    (t,) = r.trades
    assert t.reason == "stop" and t.exit_px == 395 * 0.25


def test_target_fills_at_target_price():
    day = make_day([400, 400, 390, 390, 390])
    r = run(day, {0: "SELL"}, stop_ticks=100, target_ticks=4)  # short @ 400 bid; target: ask <= 396
    (t,) = r.trades
    assert t.reason == "target" and t.exit_px == 396 * 0.25


def test_latency_uses_later_book():
    day = make_day([400, 408, 408, 408])
    c = Config(cadence_s=1, latency_ms=1000, commission=0, stop_ticks=20, target_ticks=40)
    grid = np.array([0], dtype=np.int64)
    r = asyncio.run(run_day(day, grid, Script({0: "BUY"}), c))
    assert r.trades[0].entry_px == 409 * 0.25


def test_no_lookahead_row_at():
    day = make_day([400] * 3)
    assert day.row_at(S - 1) == 0 and day.row_at(S) == 1 and day.row_at(-1) == -1


def test_zero_disables_stop_and_target():
    day = make_day([400, 400, 300, 500, 400, 400])
    r = run(day, {0: "BUY"}, stop_ticks=0, target_ticks=0)
    (t,) = r.trades
    assert t.reason == "eod"


def test_time_stop_closes_at_market_after_max_hold():
    day = make_day([400] * 10)
    r = run(day, {0: "BUY"}, max_hold_s=3)
    (t,) = r.trades
    assert t.reason == "time" and t.exit_ns == 3 * S and t.exit_px == 400 * 0.25  # long exits at the bid


def test_breakeven_stop_after_the_trade_was_in_profit():
    # long fills at ask 401; bid reaches 406 (5 ticks up), then falls back to 401 = entry → break-even exit
    day = make_day([400, 403, 406, 404, 401, 398, 390, 390])
    r = run(day, {0: "BUY"}, breakeven_ticks=5)
    (t,) = r.trades
    assert t.reason == "breakeven" and t.exit_px == 401 * 0.25 and t.points == 0
    # without it the trade stays open to the close (390 is 11 ticks below entry, inside the 20-tick stop)
    (t,) = run(day, {0: "BUY"}).trades
    assert t.reason == "eod"


def test_breakeven_stays_armed_across_incremental_checks():
    from trade_jev.harness import Book, DayResult
    day = make_day([400, 406, 405, 404, 401, 399])
    bk = Book(day, Config(stop_ticks=0, target_ticks=0, breakeven_ticks=5, commission=0), DayResult(day.day, "x"))
    bk.go(1, 0)               # long at 401 (row 0 ask)
    bk.check_exits(1)         # row 1: bid 406 → armed
    assert bk.armed and not bk.res.trades
    bk.check_exits(3)
    assert not bk.res.trades  # 405, 404 still above entry
    bk.check_exits(5)
    assert bk.res.trades[0].reason == "breakeven" and bk.res.trades[0].exit_ns == 4 * S
