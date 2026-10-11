"""Read MNQ text files: 1-minute-style candles, L1 ticks and L2 ticks.

Files are delimited text with no fixed name, so each file is classified by its
first data row (see `sniff`):

    candles  7 fields, 14-digit ts   ts, type, open, high, low, close, volume
    l1       5 fields, 20-digit ts   ts, level, type, price, volume
    l2       7 fields, 20-digit ts   ts, level, type, price, volume, depth, action

Timestamps are UTC `YYYYMMDDhhmmss[ffffff]`. A 20-digit value overflows int64,
so it is read as text and split into seconds and microseconds.
All loaders return int64 nanosecond `ts` columns, sorted ascending (stable).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

# Type codes (shared by all three files)
BID, ASK, LAST = 0, 1, 2
# L2 actions
ADD, UPDATE, REMOVE = 0, 1, 2

COLUMNS = {
    "candles": ["ts", "type", "open", "high", "low", "close", "volume"],
    "l1": ["ts", "level", "type", "price", "volume"],
    "l2": ["ts", "level", "type", "price", "volume", "depth", "action"],
}
SEPARATORS = [";", ",", "\t", "|", " "]
SUFFIXES = {".txt", ".csv", ".dat", ".tsv", ".gz", ""}


@dataclass(frozen=True)
class DataFile:
    path: Path
    kind: str          # candles | l1 | l2
    sep: str
    header: bool
    first_ns: int
    last_ns: int


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "r", newline="")


def _last_line(path: Path) -> str:
    if path.suffix == ".gz":  # no cheap seek; stream it
        line = ""
        with _open_text(path) as f:
            for line in f:
                pass
        return line.strip()
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        back = min(size, 4096)
        while True:
            f.seek(size - back)
            lines = [l for l in f.read(back).splitlines() if l.strip()]
            if len(lines) >= 2 or back == size:
                return lines[-1].decode().strip() if lines else ""
            back = min(size, back * 4)


def civil_to_ns(stamp: np.ndarray) -> np.ndarray:
    """int64 YYYYMMDDhhmmss → int64 ns since epoch (UTC). Vectorised days-from-civil."""
    s = stamp.astype(np.int64)
    sec, s = s % 100, s // 100
    mi, s = s % 100, s // 100
    hh, s = s % 100, s // 100
    d, s = s % 100, s // 100
    m, y = s % 100, s // 100
    y = y - (m <= 2)
    era = np.floor_divide(y, 400)
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return ((days * 86400 + hh * 3600 + mi * 60 + sec) * 1_000_000_000).astype(np.int64)


def parse_ts(text: str) -> int:
    """One timestamp string (14 or 20 digits) → ns."""
    ns = int(civil_to_ns(np.array([int(text[:14])]))[0])
    return ns + (int(text[14:20]) * 1000 if len(text) >= 20 else 0)


def sniff(path: Path) -> DataFile | None:
    """Classify a file by its first data row. None if it isn't one of ours."""
    try:
        with _open_text(path) as f:
            first = ""
            header = False
            for raw in f:
                if not raw.strip():
                    continue
                if not raw.strip()[0].isdigit():  # header row
                    header = True
                    continue
                first = raw.strip()
                break
    except (OSError, UnicodeDecodeError):
        return None
    if not first:
        return None
    for sep in SEPARATORS:
        parts = [p for p in first.split(sep)] if sep != " " else first.split()
        if len(parts) < 5:
            continue
        ts = parts[0]
        if not ts.isdigit():
            continue
        kind = {(14, 7): "candles", (20, 5): "l1", (20, 7): "l2"}.get((len(ts), len(parts)))
        if kind is None:
            continue
        last = _last_line(path)
        last_ts = (last.split(sep) if sep != " " else last.split())[0]
        last_ns = parse_ts(last_ts) if last_ts.isdigit() else parse_ts(ts)
        return DataFile(path, kind, sep, header, parse_ts(ts), last_ns)
    return None


def discover(root: Path) -> list[DataFile]:
    """Every candle / L1 / L2 file under `root` (a file or a folder), sorted by start time."""
    root = Path(root)
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    found = []
    for p in paths:
        if p.suffix.lower() not in SUFFIXES:
            continue
        df = sniff(p)
        if df is not None:
            found.append(df)
    return sorted(found, key=lambda d: (d.kind, d.first_ns))


def _read(df: DataFile) -> pa.Table:
    cols = COLUMNS[df.kind]
    ts_type = pa.int64() if df.kind == "candles" else pa.string()
    types = {"ts": ts_type, "type": pa.int8(), "level": pa.int8(), "depth": pa.int8(),
             "action": pa.int8(), "volume": pa.float64()}
    types.update({c: pa.float64() for c in ("open", "high", "low", "close", "price")})
    read = pacsv.ReadOptions(column_names=cols, skip_rows=1 if df.header else 0,
                             block_size=64 << 20)
    parse = pacsv.ParseOptions(delimiter=df.sep if df.sep != " " else " ",
                               ignore_empty_lines=True)
    conv = pacsv.ConvertOptions(column_types={c: types[c] for c in cols})
    return pacsv.read_csv(df.path, read_options=read, parse_options=parse, convert_options=conv)


def _ts_ns(col: pa.ChunkedArray, kind: str) -> np.ndarray:
    if kind == "candles":
        return civil_to_ns(col.to_numpy())
    col = pc.utf8_trim_whitespace(col)
    sec = pc.cast(pc.utf8_slice_codeunits(col, 0, 14), pa.int64()).to_numpy()
    us = pc.cast(pc.utf8_slice_codeunits(col, 14, 20), pa.int64()).to_numpy()
    return civil_to_ns(sec) + us * 1000


def load(files: list[DataFile], start_ns: int | None = None, end_ns: int | None = None) -> pd.DataFrame:
    """Concatenate files of one kind into one frame, optionally clipped to [start, end)."""
    if not files:
        return pd.DataFrame()
    kinds = {f.kind for f in files}
    assert len(kinds) == 1, f"mixed kinds: {kinds}"
    kind = kinds.pop()
    frames = []
    covered = None  # last ts already loaded; overlapping exports are clipped, not deduped
    for f in sorted(files, key=lambda f: f.first_ns):
        if start_ns is not None and f.last_ns < start_ns:
            continue
        if end_ns is not None and f.first_ns >= end_ns:
            continue
        t = _read(f)
        out = {"ts": _ts_ns(t.column("ts"), kind)}
        for c in COLUMNS[kind][1:]:
            out[c] = t.column(c).to_numpy()
        frame = pd.DataFrame(out)
        if covered is not None:
            frame = frame[frame.ts > covered]
        if len(frame):
            covered = int(frame.ts.max()) if covered is None else max(covered, int(frame.ts.max()))
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=COLUMNS[kind])
    data = pd.concat(frames, ignore_index=True)
    if start_ns is not None:
        data = data[data.ts >= start_ns]
    if end_ns is not None:
        data = data[data.ts < end_ns]
    # keep the file's row order inside one timestamp: it is the event order
    data = data.sort_values("ts", kind="stable").reset_index(drop=True)
    return data
