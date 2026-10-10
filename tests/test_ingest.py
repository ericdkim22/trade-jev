"""CSV ingest: sniffing, book rebuild under each depth reading, trade signs, MNQ point value."""

import asyncio
import json

import numpy as np
import pytest

from trade_jev.data import TICK, et_to_ns, list_days, load_day, point_value
from trade_jev.harness import Config, run_day
from trade_jev.ingest import BookBuilder, book_ok, ingest, scan, sniff
from trade_jev.policies import Decision

DAY = "2026-06-23"
T0 = "20260623133000"  # 09:30:00 ET (UTC−4)


def ts(sec: int, us: int = 0) -> str:
    h, m, s = 13, 30 + sec // 60, sec % 60
    return f"20260623{h:02d}{m:02d}{s:02d}{us:06d}"


def l2_rows(delim=";") -> list[str]:
    """Insert-semantics feed: 10 levels each side, a buy trade that clears the ask, the book steps up."""
    rows = []
    for d in range(1, 11):
        rows.append([ts(0), 2, 0, f"{21000 - (d - 1) * TICK:.2f}", 5, d, 0])
        rows.append([ts(0), 2, 1, f"{21000.25 + (d - 1) * TICK:.2f}", 5, d, 0])
    rows += [
        [ts(1), 1, 2, "21000.25", 5, "", ""],   # buyer lifts the whole ask
        [ts(1), 2, 1, "21000.25", 0, 1, 2],     # remove ask d1 → asks shift up
        [ts(1), 2, 1, "21002.75", 7, 10, 0],    # new far ask
        [ts(1), 2, 0, "20997.75", 0, 10, 2],    # deepest bid drops off
        [ts(1), 2, 0, "21000.25", 3, 1, 0],     # new best bid at the old ask
        [ts(2), 1, 2, "21000.25", 2, "", ""],   # seller hits the bid
        [ts(2), 2, 0, "21000.25", 1, 1, 1],
        [ts(3), 1, 3, "12345", 999, "", ""],    # daily volume: ignored
    ]
    for k in range(4, 120):                      # quiet size updates for a couple of minutes
        rows.append([ts(k), 2, 0, "21000.00", 5 + k % 3, 2, 1])
    return [delim.join(map(str, r)) for r in rows]


@pytest.fixture
def folder(tmp_path):
    src = tmp_path / "transfer"
    src.mkdir()
    (src / "MNQU6_L2.csv").write_text("\n".join(l2_rows()) + "\n")
    (src / "MNQU6_L1.txt").write_text(f"{ts(0)},1,0,21000.00,5\n{ts(0)},1,1,21000.25,5\n")
    (src / "MNQU6_M1.csv").write_text(f"time,type,o,h,l,c,v\n{T0},2,21000,21001,20999.75,21000.5,40\n")
    return src


def test_sniff_kinds_and_delimiters(folder):
    kinds = {f.name: (f.kind, f.delimiter, f.header) for f in scan(folder)}
    assert kinds == {"MNQU6_L2.csv": ("l2", ";", False), "MNQU6_L1.txt": ("l1", ",", False),
                     "MNQU6_M1.csv": ("candles", ",", True)}
    assert sniff(folder / "MNQU6_L2.csv").symbol == "MNQU6"


@pytest.mark.parametrize("mode", ["insert", "set", "price"])
def test_book_builder_modes_agree_on_simple_feed(mode):
    bk = BookBuilder(mode)
    for d in range(10):
        bk.apply(0, 0, d, 100 - d, 5)
        bk.apply(1, 0, d, 101 + d, 5)
    assert book_ok(bk.levels(0), bk.levels(1))
    bk.apply(0, 1, 0, 100, 9)
    assert bk.levels(0)[0] == (100, 9) and len(bk.levels(0)) == 10


def test_insert_mode_shifts_levels():
    bk = BookBuilder("insert")
    for d in range(10):
        bk.apply(1, 0, d, 101 + d, 5)
    bk.apply(1, 2, 0, 101, 0)          # remove best → next becomes best
    assert bk.levels(1)[0] == (102, 5) and len(bk.levels(1)) == 9
    bk.apply(1, 0, 0, 101, 1)          # add at the top → others shift down
    assert [p for p, _ in bk.levels(1)][:3] == [101, 102, 103]


def test_ingest_end_to_end(folder, tmp_path):
    out = tmp_path / "csv"
    summary = ingest(folder, out)
    assert summary["l2_days"] == [DAY] and summary["l1_days"] == [] and summary["bar_days"] == [DAY]
    meta = json.loads(next(out.glob("*.json")).read_text())
    assert meta["depth_mode"] == "insert" and meta["crossed_rows"] == 0 and meta["padded_rows"] == 0
    assert (meta["buy_volume"], meta["sell_volume"]) == (5, 2)

    sym, path = list_days(out)[DAY]
    assert sym == "MNQU6" and path.name.startswith("CSV__MNQU6__mbp-10__rth__")
    grid = np.array([et_to_ns(DAY, "09:30:01"), et_to_ns(DAY, "09:30:30")], dtype=np.int64)
    day = load_day(DAY, grid, out)
    assert day.point_value == point_value("MNQ") == 2.0
    r = day.row_at(grid[0])
    assert day.bid[r] * TICK == 21000.25 and day.ask[r] * TICK == 21000.50   # stepped up after the sweep
    assert day.cum_delta[-1] == 5 - 2
    bid_px = day.book(int(r))[0]
    assert list(np.diff(bid_px)) == [-1] * 9


def test_mnq_pnl_uses_micro_point_value(folder, tmp_path):
    out = tmp_path / "csv"
    ingest(folder, out)
    grid = np.arange(et_to_ns(DAY, "09:30:05"), et_to_ns(DAY, "09:31:00"), 10**9 * 5, dtype=np.int64)
    day = load_day(DAY, grid, out)

    class BuyOnce:
        name = "buy"

        async def __call__(self, ctx):
            return Decision("BUY" if ctx.t_ns == grid[0] else "HOLD")

    res = asyncio.run(run_day(day, grid, BuyOnce(), Config(latency_ms=0, commission=0, stop_ticks=0, target_ticks=0)))
    (t,) = res.trades
    assert t.pnl == round(t.points * 2.0, 2)
