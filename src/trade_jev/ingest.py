r"""Ingest: CSV tick exports (candles, L1 ticks, L2 ticks) → day files the backtest reads.

  uv run python -m trade_jev.ingest "C:\Transfer" --scan       # list what's there, convert nothing
  uv run python -m trade_jev.ingest "C:\Transfer"              # convert → data/csv/
  uv run python -m trade_jev.ingest "C:\Transfer" --symbol MNQ --depth-mode insert

Input (one record per line; header optional; delimiter ; , tab or | auto-detected; .gz ok):
  candles   YYYYMMDDhhmmss,       type, open, high, low, close, volume
  L1 ticks  YYYYMMDDhhmmssffffff, level, type, price, volume
  L2 ticks  YYYYMMDDhhmmssffffff, level, type, price, volume, depth (1-10), action (0 add, 1 update, 2 remove)
  type: 0 bid, 1 ask, 2 last trade; 3-9 are daily stats and are ignored. Timestamps are UTC.

Output: one `CSV__{symbol}__mbp-10__rth__{day}__{l2|l1}.parquet` + JSON sidecar per ET trading day,
the same layout as the Databento files (data/DATA.md), so run / replay / view work unchanged.

- L2 ticks rebuild the 10-level book. Events with the same timestamp are applied together and
  give one book row. How `depth` is meant is vendor-specific, so `--depth-mode auto` (default)
  tries three readings on the first events and keeps the one with the fewest crossed or
  out-of-order books: `insert` (add shifts deeper levels down, remove shifts them up),
  `set` (each depth is a fixed slot) and `price` (levels keyed by price; depth ignored).
- Trades (type 2) become `trade_delta`: +size at or above the best ask (buyer lifted),
  −size at or below the best bid, otherwise the tick rule.
- Days with no L2 fall back to L1 ticks: best bid/ask only, levels 2-10 padded with size 0
  (sidecar `"levels": 1`). Jev then sees an empty ladder past the touch.
- Candles go to `bars/{symbol}__{day}.json` for the dashboard's price chart.
- Rows are kept 08:30-17:00 ET (`--session`), like the Databento pull. A gap of 15+ minutes
  (daily halt, weekend) clears the book.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from trade_jev.data import CSV_DIR, ET, LEVELS, TICK

DELIMS = (";", ",", "\t", "|")
SUFFIXES = (".csv", ".txt", ".tsv", ".dat", ".csv.gz", ".txt.gz", ".gz")
COLS = {
    "candles": ["ts", "type", "open", "high", "low", "close", "volume"],
    "l1": ["ts", "level", "type", "price", "volume"],
    "l2": ["ts", "level", "type", "price", "volume", "depth", "action"],
}
BID, ASK, LAST = 0, 1, 2
ADD, UPDATE, REMOVE = 0, 1, 2
MODES = ("insert", "price", "set")  # auto tie-break order
GAP_NS = 15 * 60 * 10**9
DAY_NS = 86_400 * 10**9
FLUSH_ROWS = 250_000
SCHEMA = pa.schema([
    ("ts_event", pa.timestamp("ns", tz="UTC")),
    ("instrument_id", pa.int64()),
    ("bid_px", pa.list_(pa.float64())),
    ("bid_sz", pa.list_(pa.int64())),
    ("ask_px", pa.list_(pa.float64())),
    ("ask_sz", pa.list_(pa.int64())),
    ("trade_delta", pa.int64()),
])
_SYM = re.compile(r"(?<![A-Za-z])(MNQ|NQ)([FGHJKMNQUVXZ]\d{1,2})?(?![a-z])", re.I)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- scan

@dataclass
class CsvFile:
    path: str
    name: str
    kind: str          # candles | l1 | l2 | unknown
    delimiter: str = ","
    header: bool = False
    columns: int = 0
    size: int = 0
    first_ts: str | None = None   # UTC, ISO
    symbol: str | None = None
    ragged: bool = False          # rows with different field counts (read with pandas)
    note: str = ""


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")


def _ts_iso(digits: str) -> str | None:
    try:
        d = digits[:14]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{d[8:10]}:{d[10:12]}:{d[12:14]}Z"
    except IndexError:
        return None


def sniff(path: Path) -> CsvFile:
    """Kind, delimiter and header of one file from its first lines."""
    f = CsvFile(str(path), path.name, "unknown", size=path.stat().st_size)
    if m := _SYM.search(path.name):
        f.symbol = m.group(0).upper()
    try:
        with _open_text(path) as fh:
            lines = [ln.strip().lstrip("\ufeff") for _, ln in zip(range(200), fh)]
    except OSError as e:
        f.note = f"unreadable: {e}"
        return f
    lines = [ln for ln in lines if ln]
    if not lines:
        f.note = "empty"
        return f
    best = None
    for d in DELIMS:
        parts = [ln.split(d) for ln in lines]
        data = [p for p in parts if p[0].strip().isdigit()]
        if not data or max(len(p) for p in data) < 5:
            continue
        if best is None or len(data) > len(best[1]):
            best = (d, data, parts)
    if best is None:
        f.note = "no ; , tab or | separated rows starting with a YYYYMMDD… timestamp"
        return f
    d, data, parts = best
    f.delimiter, f.header = d, not parts[0][0].strip().isdigit()
    f.columns = max(len(p) for p in data)
    f.ragged = len({len(p) for p in data}) > 1
    ts = data[0][0].strip()
    f.first_ts = _ts_iso(ts)
    if len(ts) == 14 and f.columns >= 7:
        f.kind = "candles"
    elif len(ts) == 20 and f.columns >= 7:
        f.kind = "l2"
    elif len(ts) == 20 and f.columns >= 5:
        f.kind = "l1"
    else:
        f.note = f"unrecognised: {len(ts)}-digit timestamp, {f.columns} fields"
    return f


def scan(folder: Path) -> list[CsvFile]:
    folder = Path(folder).expanduser()
    if folder.is_file():
        return [sniff(folder)]
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.name.lower().endswith(SUFFIXES)]
    return sorted((sniff(p) for p in files), key=lambda f: (f.kind, f.first_ts or "", f.name))


def guess_symbol(files: list[CsvFile]) -> str:
    syms = [f.symbol for f in files if f.symbol]
    sym = syms[0] if syms else "MNQ"
    return re.sub(r"\W", "", sym) or "MNQ"


# ---------------------------------------------------------------- read

def _digits_to_ns(s: pa.ChunkedArray | pa.Array, micros: bool) -> np.ndarray:
    """'YYYYMMDDhhmmss[ffffff]' strings → int64 ns since epoch (UTC). Vectorised, no strptime."""
    s = pc.utf8_trim_whitespace(s)
    n = pc.cast(pc.utf8_slice_codeunits(s, 0, 14), pa.int64()).to_numpy()
    sec, n = n % 100, n // 100
    mi, n = n % 100, n // 100
    hh, n = n % 100, n // 100
    dd, n = n % 100, n // 100
    mm, yy = n % 100, n // 100
    months = (yy - 1970) * 12 + (mm - 1)
    days = months.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) + (dd - 1)
    ns = ((days * 24 + hh) * 3600 + mi * 60 + sec) * 10**9
    if micros:
        frac = pc.utf8_rpad(pc.utf8_slice_codeunits(s, 14, 20), 6, "0")
        ns = ns + pc.cast(frac, pa.int64()).to_numpy() * 1000
    return ns.astype(np.int64)


def _pandas_blocks(f: CsvFile, names: list[str], rows: int = 2_000_000):
    """Ragged files: short rows get nulls for the missing trailing fields."""
    for df in pd.read_csv(f.path, sep=f.delimiter, header=None, names=names, usecols=COLS[f.kind],
                          skiprows=1 if f.header else 0, dtype={"ts": str}, chunksize=rows,
                          on_bad_lines="skip", compression="infer"):
        yield pa.Table.from_pandas(df, preserve_index=False)


def read_batches(f: CsvFile, block_mb: int = 64):
    """Yield dicts of numpy columns per block of the file."""
    names = COLS[f.kind] + [f"_x{i}" for i in range(max(0, f.columns - len(COLS[f.kind])))]
    types = {c: pa.float64() for c in COLS[f.kind]}
    types["ts"] = pa.string()
    if f.ragged:
        reader = _pandas_blocks(f, names)
    else:
        src = pa.input_stream(f.path, compression="gzip" if f.path.lower().endswith(".gz") else None)
        reader = pacsv.open_csv(
            src,
            read_options=pacsv.ReadOptions(column_names=names, skip_rows=1 if f.header else 0,
                                           block_size=block_mb << 20),
            parse_options=pacsv.ParseOptions(delimiter=f.delimiter, invalid_row_handler=lambda r: "skip"),
            convert_options=pacsv.ConvertOptions(column_types=types, include_columns=COLS[f.kind],
                                                 strings_can_be_null=True),
        )
    for b in reader:
        b = b.cast(pa.schema([(c, types[c]) for c in COLS[f.kind]])) if f.ragged else b
        ok = pc.is_valid(b.column("ts"))
        for c in ("type", "price") if f.kind != "candles" else ("type", "close"):
            ok = pc.and_(ok, pc.is_valid(b.column(c)))
        b = b.filter(ok)
        if not b.num_rows:
            continue
        out = {"ts": _digits_to_ns(b.column("ts"), micros=f.kind != "candles")}
        for c in COLS[f.kind][1:]:
            out[c] = b.column(c).fill_null(-1).to_numpy()
        yield out


def et_split(ts: np.ndarray, session: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per row: ET day number (days since epoch) and whether it falls in the session window."""
    local = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert(ET).tz_localize(None).asi8
    tod = local % DAY_NS
    return local // DAY_NS, (tod >= session[0]) & (tod < session[1])


def _day_str(n: int) -> str:
    return str(np.datetime64(int(n), "D"))


# ---------------------------------------------------------------- book

class BookBuilder:
    """10-level book from depth events. `mode` says how `depth` addresses the ladder."""

    def __init__(self, mode: str):
        assert mode in MODES, mode
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        if self.mode == "price":
            self.sides: list = [{}, {}]
        elif self.mode == "set":
            self.sides = [[None] * LEVELS, [None] * LEVELS]
        else:
            self.sides = [[], []]

    def apply(self, side: int, action: int, depth: int, px: int, sz: int) -> None:
        """side 0 bid / 1 ask; depth 0-based; px in ticks."""
        lv = self.sides[side]
        if self.mode == "price":
            if action == REMOVE or sz <= 0:
                lv.pop(px, None)
                return
            lv[px] = sz
            other = self.sides[1 - side]  # a new level clears stale levels it crosses
            if other:
                for p in [p for p in other if (p <= px if side == BID else p >= px)]:
                    del other[p]
            if len(lv) > 3 * LEVELS:
                for p in (sorted(lv)[: len(lv) - 2 * LEVELS] if side == BID
                          else sorted(lv)[2 * LEVELS:]):
                    del lv[p]
            return
        if not 0 <= depth < LEVELS:
            return
        if self.mode == "set":
            lv[depth] = None if action == REMOVE else (px, sz)
            return
        if action == ADD:
            lv.insert(min(depth, len(lv)), (px, sz))
            del lv[LEVELS:]
        elif action == UPDATE:
            if depth < len(lv):
                lv[depth] = (px, sz)
            else:
                lv.append((px, sz))
        elif action == REMOVE and depth < len(lv):
            del lv[depth]

    def levels(self, side: int) -> list[tuple[int, int]]:
        lv = self.sides[side]
        if self.mode == "price":
            pick = heapq.nlargest if side == BID else heapq.nsmallest
            return [(p, lv[p]) for p in pick(LEVELS, lv)]
        if self.mode == "set":
            return [x for x in lv if x is not None]
        return lv

    def best(self, side: int) -> int | None:
        lv = self.levels(side) if self.mode != "insert" else self.sides[side]
        return lv[0][0] if lv else None


def book_ok(bids, asks) -> bool:
    if not bids or not asks or bids[0][0] >= asks[0][0]:
        return False
    return (all(bids[i][0] > bids[i + 1][0] for i in range(len(bids) - 1))
            and all(asks[i][0] < asks[i + 1][0] for i in range(len(asks) - 1)))


def pick_depth_mode(f: CsvFile, sample_events: int = 300_000) -> tuple[str, dict[str, float]]:
    """Replay the first events under each reading; fewest crossed / out-of-order books wins."""
    builders = {m: BookBuilder(m) for m in MODES}
    bad = dict.fromkeys(MODES, 0)
    snaps = seen = 0
    prev = None
    for b in read_batches(f, block_mb=16):
        for t, typ, px, sz, dep, act in zip(b["ts"].tolist(), b["type"].astype(int).tolist(),
                                            np.rint(b["price"] / TICK).astype(np.int64).tolist(),
                                            b["volume"].tolist(), b["depth"].astype(int).tolist(),
                                            b["action"].astype(int).tolist()):
            if typ not in (BID, ASK) or dep < 1:
                continue
            if prev is not None and t != prev:
                snaps += 1
                for m, bk in builders.items():
                    bad[m] += not book_ok(bk.levels(BID), bk.levels(ASK))
            prev = t
            for bk in builders.values():
                bk.apply(typ, act, dep - 1, px, int(sz))
            seen += 1
            if seen >= sample_events:
                break
        if seen >= sample_events:
            break
    if not snaps:
        return "insert", {}
    rates = {m: round(bad[m] / snaps, 4) for m in MODES}
    return min(MODES, key=lambda m: (rates[m], MODES.index(m))), rates


# ---------------------------------------------------------------- write

@dataclass
class DayStats:
    rows: int = 0
    events: int = 0
    trades: int = 0
    buy_volume: int = 0
    sell_volume: int = 0
    aggressor_touch: int = 0
    aggressor_tick_rule: int = 0
    crossed_rows: int = 0
    padded_rows: int = 0
    out_of_order: int = 0
    first_ns: int = 0
    last_ns: int = 0
    files: list[str] = field(default_factory=list)


class DayWriter:
    """Streams one day's rows to Parquet; renames into place and writes the sidecar on close."""

    def __init__(self, out: Path, symbol: str, day: str, tag: str, meta: dict, stats: DayStats):
        self.out, self.symbol, self.day, self.tag, self.meta = out, symbol, day, tag, meta
        self.stem = f"CSV__{symbol}__mbp-10__rth__{day}__{tag}"
        self.tmp = out / f"{self.stem}.parquet.tmp"
        self.writer = pq.ParquetWriter(self.tmp, SCHEMA)
        self.stats = stats
        self._clear()

    def _clear(self) -> None:
        self.ts: list[int] = []
        self.bp: list[int] = []
        self.bs: list[int] = []
        self.ap: list[int] = []
        self.az: list[int] = []
        self.delta: list[int] = []

    def add(self, t: int, bids, asks, delta: int) -> None:
        st = self.stats
        if not st.rows:
            st.first_ns = t
        st.last_ns = t
        st.rows += 1
        if bids[0][0] >= asks[0][0]:
            st.crossed_rows += 1
        if len(bids) < LEVELS or len(asks) < LEVELS:
            st.padded_rows += 1
        self.ts.append(t)
        self.delta.append(delta)
        for lv, px, sz, step in ((bids, self.bp, self.bs, -1), (asks, self.ap, self.az, 1)):
            for p, s in lv:
                px.append(p)
                sz.append(s)
            last = lv[-1][0]
            for k in range(1, LEVELS - len(lv) + 1):  # pad missing levels one tick apart, size 0
                px.append(last + step * k)
                sz.append(0)
        if len(self.ts) >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        n = len(self.ts)
        if not n:
            return
        offs = pa.array(np.arange(0, (n + 1) * LEVELS, LEVELS, dtype=np.int32))

        def lst(vals, px: bool):
            a = np.asarray(vals, dtype=np.int64)
            return pa.ListArray.from_arrays(offs, pa.array(a * TICK if px else a))

        self.writer.write_table(pa.Table.from_arrays([
            pa.array(np.asarray(self.ts, dtype=np.int64), pa.timestamp("ns", tz="UTC")),
            pa.array(np.zeros(n, dtype=np.int64)),
            lst(self.bp, True), lst(self.bs, False), lst(self.ap, True), lst(self.az, False),
            pa.array(np.asarray(self.delta, dtype=np.int64)),
        ], schema=SCHEMA))
        self._clear()

    def close(self) -> dict:
        self.flush()
        self.writer.close()
        final = self.out / f"{self.stem}.parquet"
        for old in self.out.glob(f"CSV__*__mbp-10__rth__{self.day}__*.parquet"):  # one file per day
            old.unlink()
            old.with_suffix(".json").unlink(missing_ok=True)
        self.tmp.replace(final)
        st = self.stats
        side = {k: v for k, v in asdict(st).items() if k not in ("first_ns", "last_ns")}
        meta = {
            "dataset": "CSV", "symbol": self.symbol, "schema": "mbp-10", "day": self.day, "session": "rth",
            **self.meta, **side,
            "first_utc": str(np.datetime64(st.first_ns, "ns")) + "Z" if st.rows else None,
            "last_utc": str(np.datetime64(st.last_ns, "ns")) + "Z" if st.rows else None,
        }
        final.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return meta


# ---------------------------------------------------------------- convert

def convert_ticks(files: list[CsvFile], out: Path, symbol: str, session: tuple[int, int], *,
                  depth_mode: str = "auto", skip_days: set[str] = frozenset()) -> list[dict]:
    """L2 (or L1) tick files → day files. Files are read as one stream, oldest first."""
    if not files:
        return []
    kind = files[0].kind
    files = sorted(files, key=lambda f: f.first_ts or "")
    meta: dict = {"source": f"csv-{kind}", "levels": LEVELS if kind == "l2" else 1}
    if kind == "l2":
        if depth_mode == "auto":
            depth_mode, rates = pick_depth_mode(files[0])
            log(f"depth mode: {depth_mode}  (bad-book rate by reading: {rates})")
            meta["depth_mode_scores"] = rates
        meta["depth_mode"] = depth_mode
        book = BookBuilder(depth_mode)
    else:
        book = BookBuilder("price")  # L1: one level per side, replaced on every quote

    done: list[dict] = []
    w: DayWriter | None = None
    prev_t = prev_day = None
    prev_in = False
    delta = 0
    last_trade = None
    last_sign = 0
    stats: dict[int, DayStats] = {}  # per ET day number, session rows only
    sink = DayStats()                # events outside the session window

    def emit():
        nonlocal w, delta
        if prev_in and prev_day is not None:
            day = _day_str(prev_day)
            if day not in skip_days:
                if w is None or w.day != day:
                    if w is not None:
                        done.append(w.close())
                        log(f"  wrote {done[-1]['day']}: {done[-1]['rows']:,} rows")
                    w = DayWriter(out, symbol, day, kind, meta, stats.setdefault(prev_day, DayStats()))
                bids, asks = book.levels(BID), book.levels(ASK)
                if bids and asks:
                    if fname not in w.stats.files:
                        w.stats.files.append(fname)
                    w.add(prev_t, bids, asks, delta)
        delta = 0

    for f in files:
        fname = f.name
        t0, n_ev = time.time(), 0
        log(f"reading {f.name} ({f.size / 1e6:,.0f} MB, {kind.upper()})")
        for b in read_batches(f):
            days, in_sess = et_split(b["ts"], session)
            cols = (b["ts"].tolist(), days.tolist(), in_sess.tolist(), b["type"].astype(int).tolist(),
                    np.rint(b["price"] / TICK).astype(np.int64).tolist(),
                    np.rint(b["volume"]).astype(np.int64).tolist())
            if kind == "l2":
                cols += (b["depth"].astype(int).tolist(), b["action"].astype(int).tolist())
            else:
                cols += ([1] * len(b["ts"]), [UPDATE] * len(b["ts"]))
            for t, dn, ins, typ, px, sz, dep, act in zip(*cols):
                if typ > LAST:
                    continue
                ds = (stats.get(dn) or stats.setdefault(dn, DayStats())) if ins else sink
                if prev_t is not None:
                    if t < prev_t:
                        t = prev_t
                        ds.out_of_order += 1
                    if t != prev_t:
                        emit()
                        if t - prev_t >= GAP_NS:
                            book.reset()
                            last_trade, last_sign = None, 0
                prev_t, prev_day, prev_in = t, dn, ins
                ds.events += 1
                if typ == LAST:
                    if sz <= 0:
                        continue
                    bb, ba = book.best(BID), book.best(ASK)
                    if ba is not None and px >= ba:
                        sign = 1
                        ds.aggressor_touch += 1
                    elif bb is not None and px <= bb:
                        sign = -1
                        ds.aggressor_touch += 1
                    else:
                        sign = (1 if px > last_trade else -1 if px < last_trade else last_sign) \
                            if last_trade is not None else 0
                        ds.aggressor_tick_rule += 1
                    last_trade, last_sign = px, sign or last_sign
                    delta += sign * sz
                    ds.trades += 1
                    if sign > 0:
                        ds.buy_volume += sz
                    elif sign < 0:
                        ds.sell_volume += sz
                elif kind == "l2":
                    if dep >= 1:
                        book.apply(typ, act, dep - 1, px, sz)
                else:  # L1 quote replaces the touch
                    book.sides[typ].clear()
                    if sz > 0:
                        book.sides[typ][px] = sz
                n_ev += 1
            el = time.time() - t0
            log(f"  {f.name}: {n_ev:,} events · {n_ev / max(el, 1e-9):,.0f}/s · "
                f"at {str(np.datetime64(int(b['ts'][-1]) // 10**9, 's'))}Z")
    if prev_t is not None:
        emit()
    if w is not None:
        done.append(w.close())
        log(f"  wrote {done[-1]['day']}: {done[-1]['rows']:,} rows")
    return done


def convert_candles(files: list[CsvFile], out: Path, symbol: str, session: tuple[int, int]) -> list[str]:
    """Candles → bars/{symbol}__{day}.json (trade bars; bid bars if a day has no trade bars)."""
    if not files:
        return []
    parts = []
    for f in files:
        log(f"reading {f.name} ({f.size / 1e6:,.1f} MB, candles)")
        for b in read_batches(f):
            parts.append(pd.DataFrame(b))
    df = pd.concat(parts, ignore_index=True)
    days, ins = et_split(df["ts"].to_numpy(), session)
    df["day"] = days
    df = df[ins]
    (out / "bars").mkdir(parents=True, exist_ok=True)
    written = []
    for dn, g in df.groupby("day"):
        typ = LAST if (g["type"] == LAST).any() else BID
        g = g[g["type"] == typ].sort_values("ts").drop_duplicates("ts", keep="last")
        day = _day_str(dn)
        (out / "bars" / f"{symbol}__{day}.json").write_text(json.dumps({
            "day": day, "symbol": symbol, "type": {0: "bid", 1: "ask", 2: "last"}[typ],
            "t": (g["ts"] // 10**9).astype(int).tolist(),
            **{k: g[c].round(2).tolist() for k, c in
               (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"))},
            "v": g["volume"].round().astype(int).tolist(),
        }, separators=(",", ":")))
        written.append(day)
    log(f"  bars for {len(written)} days")
    return written


def _hhmm_ns(s: str) -> int:
    h, m = s.split(":")[:2]
    return (int(h) * 60 + int(m)) * 60 * 10**9


def ingest(folder: Path, out: Path = CSV_DIR, symbol: str | None = None, depth_mode: str = "auto",
           session: str = "08:30-17:00") -> dict:
    files = scan(folder)
    if not files:
        raise SystemExit(f"no .csv/.txt/.gz files under {folder}")
    symbol = symbol or guess_symbol(files)
    win = tuple(_hhmm_ns(x) for x in session.split("-"))
    out.mkdir(parents=True, exist_ok=True)
    by = {k: [f for f in files if f.kind == k] for k in ("l2", "l1", "candles")}
    log(f"symbol {symbol} · {len(by['l2'])} L2, {len(by['l1'])} L1, {len(by['candles'])} candle files → {out}")
    l2 = convert_ticks(by["l2"], out, symbol, win, depth_mode=depth_mode)
    l1 = convert_ticks(by["l1"], out, symbol, win, skip_days={d["day"] for d in l2})
    bars = convert_candles(by["candles"], out, symbol, win)
    summary = {"symbol": symbol, "out": str(out), "l2_days": [d["day"] for d in l2],
               "l1_days": [d["day"] for d in l1], "bar_days": bars,
               "skipped": [asdict(f) for f in files if f.kind == "unknown"]}
    log(f"done: {len(l2)} L2 days, {len(l1)} L1-only days, {len(bars)} bar days")
    for d in l2 + l1:
        bad = d["crossed_rows"] / max(d["rows"], 1)
        flag = "  ← check --depth-mode" if bad > 0.01 else ""
        log(f"  {d['day']}  {d['rows']:>10,} rows  {d['trades']:>8,} trades  "
            f"crossed {bad:.2%}  padded {d['padded_rows'] / max(d['rows'], 1):.2%}{flag}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CSV tick exports into backtest day files.")
    ap.add_argument("folder", type=Path, help=r"folder (or one file) with the CSVs, e.g. C:\Transfer")
    ap.add_argument("--out", type=Path, default=CSV_DIR)
    ap.add_argument("--symbol", default=None, help="e.g. MNQ or MNQZ6 (default: from file names, else MNQ)")
    ap.add_argument("--depth-mode", default="auto", choices=("auto", *MODES))
    ap.add_argument("--session", default="08:30-17:00", help="ET window to keep")
    ap.add_argument("--scan", action="store_true", help="list files and exit")
    args = ap.parse_args()
    if args.scan:
        for f in scan(args.folder):
            log(f"{f.kind:>8}  {f.size / 1e6:>9,.1f} MB  {f.first_ts or '-':>20}  "
                f"delim={f.delimiter!r} header={f.header}  {f.name}  {f.note}")
        return
    ingest(args.folder, args.out, args.symbol, args.depth_mode, args.session)


if __name__ == "__main__":
    sys.exit(main())
