"""MarketTick MNQ CSV days (C:\\Transfer\\YYYYMMDD.csv) → trade-jev day files in data/history/, for backtests.

  python -m trade_jev.markettick convert C:\\Transfer\\20250521.csv     # one day
  python -m trade_jev.markettick sample C:\\Transfer --days 60          # weekdays spread evenly over the folder

  backtest them: TRADE_JEV_DATA=data/history python -m trade_jev.run --days ... --commission 0.62

MarketTick rows (UTC): Level 1 `ts;1;type;price;volume` (type 0 bid, 1 ask, 2 trade) and Level 2
`ts;2;type;price;volume;depth;action` (action 0 add, 1 update, 2 remove). Their Level 2 rows don't rebuild into a
consistent book by position (checked 2026-10-09: best bid/ask matched the file's own Level 1 under 3% of the time), but
keyed by price, with levels better than the Level 1 quote dropped as stale, 99.1% of rows sit within one level of their
stated depth. So: level 1 of each row is the Level 1 quote (exact), levels 2-10 come from that price-keyed book
(approximate), and trades are signed against the quote (at/above the ask = buy, at/below the bid = sell).

A row is written on every change of the best bid / ask price, and otherwise at most every 100 ms (trades in between
are summed into it), between 09:25 and 16:00 ET; the book is built from the file's start.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path

from trade_jev import ROOT
from trade_jev.data import ET, LEVELS, TICK, et_to_ns
from trade_jev.live import Recorder, front_month, rows_to_batch

OUT = ROOT / "data" / "history"
TAG = "markettick"
START_ET, END_ET = "09:25:00", "16:00:00"   # decisions run 09:30-15:55; 60 s of look-back before the first
MIN_SIZE = 100 * 2**20                      # smaller files are Sundays / holidays (no regular session)
EVERY_NS = 100_000_000                      # a row per best bid/ask change, else at most one per 100 ms


def _ns(ts: str, cache: dict) -> int:
    """'YYYYMMDDhhmmssffffff' (UTC) → epoch ns."""
    hour = ts[:10]
    base = cache.get(hour)
    if base is None:
        base = cache[hour] = calendar.timegm((int(ts[:4]), int(ts[4:6]), int(ts[6:8]), int(ts[8:10]), 0, 0)) * 10**9
    return base + (int(ts[10:12]) * 60 + int(ts[12:14])) * 10**9 + int(ts[14:20]) * 1000


def convert(path: Path, out_dir: Path = OUT) -> dict:
    """One MarketTick day → a day file. Returns stats (rows written, trades, crossed quotes)."""
    day = date(int(path.stem[:4]), int(path.stem[4:6]), int(path.stem[6:8]))
    d = day.isoformat()
    lo, hi = et_to_ns(d, START_ET), et_to_ns(d, END_ET)
    symbol = front_month(datetime.combine(day, datetime.min.time(), ET), "MNQ")
    t0 = time.time()
    bids: dict[int, int] = {}   # price in ticks → size
    asks: dict[int, int] = {}
    bb = ba = None              # Level 1 quote, ticks
    bbs = bas = 0
    delta = 0
    rows: list[tuple] = []
    rec = Recorder(d, symbol, out_dir=out_dir, tag=TAG, flush_s=1e9)
    stats = {"day": d, "symbol": symbol, "rows": 0, "trades": 0, "contracts": 0, "crossed": 0}
    cache: dict = {}
    last = 0
    top = None  # (bb, ba) of the last row written
    with path.open(newline="") as fh:
        for r in csv.reader(fh, delimiter=";"):
            if len(r) < 5 or not r[0][:8].isdigit():
                continue
            try:
                px, vol = round(float(r[3]) / TICK), int(r[4])
            except ValueError:
                continue
            if r[1] == "2":
                if len(r) < 7 or r[6] == "":
                    continue
                side = bids if r[2] == "0" else asks if r[2] == "1" else None
                if side is None:
                    continue
                if r[6] == "2":
                    side.pop(px, None)
                else:
                    side[px] = vol
                    if len(side) > 40:  # keep the 20 nearest the quote
                        for p in sorted(side, reverse=side is bids)[20:]:
                            del side[p]
                continue
            if r[1] != "1" or r[2] not in ("0", "1", "2"):
                continue
            ts = _ns(r[0], cache)
            if r[2] == "0":
                bb, bbs = px, vol
                for p in [p for p in bids if p >= bb]:
                    del bids[p]
            elif r[2] == "1":
                ba, bas = px, vol
                for p in [p for p in asks if p <= ba]:
                    del asks[p]
            elif bb is not None and ba is not None:
                delta += vol if px >= ba else -vol if px <= bb else 0
                stats["trades"] += 1
                stats["contracts"] += vol
            if bb is None or ba is None or not lo <= ts <= hi:
                if ts > hi:
                    break
                delta = 0 if ts < lo else delta
                continue
            if bb >= ba:
                stats["crossed"] += 1
                continue
            if (bb, ba) == top and ts < last + EVERY_NS:
                continue
            top = (bb, ba)
            bp = [bb] + sorted((p for p in bids if p < bb), reverse=True)[:LEVELS - 1]
            ap = [ba] + sorted(p for p in asks if p > ba)[:LEVELS - 1]
            bs = [bbs] + [bids[p] for p in bp[1:]]
            as_ = [bas] + [asks[p] for p in ap[1:]]
            while len(bp) < LEVELS:
                bp.append(bp[-1] - 1); bs.append(0)
            while len(ap) < LEVELS:
                ap.append(ap[-1] + 1); as_.append(0)
            last = ts = max(ts, last + 1)  # strictly increasing
            rows.append((ts, bp, bs, ap, as_, delta))
            delta = 0
            if len(rows) >= 100_000:
                rec.add(rows_to_batch(rows)); rec.flush(); rows = []
    if rows:
        rec.add(rows_to_batch(rows))
    stats["rows"] = rec.rows + rec.n_pending
    if stats["rows"] == 0:
        rec.pending = []
        rec.parts.rmdir()
        stats["skipped"] = "no regular-session rows"
        return stats
    rec.close()
    stats["seconds"] = round(time.time() - t0, 1)
    stats["path"] = str(rec.path)
    return stats


def sample(folder: Path, n: int) -> list[Path]:
    """`n` weekday files spread evenly over the folder (Sundays / holidays left out by size)."""
    files = sorted(p for p in folder.glob("*.csv") if p.stem.isdigit() and len(p.stem) == 8
                   and date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:8])).weekday() < 5
                   and p.stat().st_size >= MIN_SIZE)
    if n >= len(files):
        return files
    step = len(files) / n
    return [files[int(i * step + step / 2)] for i in range(n)]


def _convert_logged(path: Path) -> dict:
    out = OUT / f"GLBX.MDP3__{front_month(datetime.strptime(path.stem, '%Y%m%d').replace(tzinfo=ET), 'MNQ')}" \
                f"__mbp-10__rth__{path.stem[:4]}-{path.stem[4:6]}-{path.stem[6:8]}__{TAG}.json"
    if out.exists():
        return {"day": path.stem, "skipped": "already converted"}
    try:
        return convert(path)
    except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the batch
        return {"day": path.stem, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="MarketTick MNQ CSV → trade-jev day files (data/history/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("files", nargs="+", type=Path)
    s = sub.add_parser("sample")
    s.add_argument("folder", type=Path)
    s.add_argument("--days", type=int, default=60)
    s.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    files = args.files if args.cmd == "convert" else sample(args.folder, args.days)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=getattr(args, "workers", 1)) as ex:
        for st in ex.map(_convert_logged, files):
            print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
