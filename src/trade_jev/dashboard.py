"""Backtest tab (viz/backtest.html): import CSVs, browse days, start runs, compare results.

Served by `trade_jev.view`. Imports and runs are child processes (`trade_jev.ingest`,
`trade_jev.run`) whose output the page polls, so the server stays responsive and a run started
here is exactly the CLI run.

  GET  /api/bt/env                   defaults, API key present, data folders
  GET  /api/bt/days                  every loadable day with its sidecar stats
  GET  /api/bt/day?day=<d>           1-min mid OHLC + aggressor volume (and CSV candles if imported)
  GET  /api/bt/results               every run in runs/index.jsonl that is on disk (baselines too)
  GET  /api/bt/result?id=<run>       summary, per-day equity (downsampled) and trades per policy
  GET  /api/bt/jobs                  imports / runs started from the page
  GET  /api/bt/job?id=<j>&since=<n>  one job's status and log lines from n
  POST /api/bt/scan    {path}                        what's in a CSV folder
  POST /api/bt/ingest  {path, symbol?, depth_mode?}  start an import
  POST /api/bt/run     {days, policies, settings…}   start a backtest
  POST /api/bt/cancel  {id}                          stop a job
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, fields
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from trade_jev import ROOT
from trade_jev.data import CSV_DIR, DATA_DIR, LIVE_DIR, TICK, list_days, point_value
from trade_jev.harness import Config
from trade_jev.ingest import guess_symbol, scan
from trade_jev.settings import DEFAULT

RUNS = ROOT / "runs"
POLICIES = ("jev", "imbalance", "random", "hold")
_TIME = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RUN_ID = re.compile(r"^[\w.-]+$")


# ---------------------------------------------------------------- jobs

class Job:
    def __init__(self, kind: str, title: str, argv: list[str], run_id: str | None = None):
        self.id = f"{kind}-{int(time.time() * 1000)}"
        self.kind, self.title, self.argv, self.run_id = kind, title, argv, run_id
        self.started = time.time()
        self.ended: float | None = None
        self.lines: deque[str] = deque(maxlen=4000)
        self.dropped = 0
        self.status = "running"
        self.returncode: int | None = None
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        self.proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace", bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            if len(self.lines) == self.lines.maxlen:
                self.dropped += 1
            self.lines.append(line.rstrip("\n"))
        self.returncode = self.proc.wait()
        self.ended = time.time()
        if self.status == "running":
            self.status = "done" if self.returncode == 0 else "failed"

    def cancel(self) -> None:
        if self.status == "running":
            self.status = "cancelled"
            self.proc.terminate()

    def to_dict(self, since: int = 0) -> dict:
        lines = list(self.lines)
        start = max(0, since - self.dropped)
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "run_id": self.run_id, "started": self.started, "ended": self.ended,
                "returncode": self.returncode, "next": self.dropped + len(lines),
                "lines": lines[start:]}


JOBS: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _start(kind: str, title: str, argv: list[str], run_id: str | None = None) -> dict:
    with _jobs_lock:
        busy = [j for j in JOBS.values() if j.kind == kind and j.status == "running"]
        if busy:
            raise ValueError(f"a {kind} is already running ({busy[0].title}); wait or cancel it")
        j = Job(kind, title, argv, run_id)
        JOBS[j.id] = j
    return j.to_dict()


def jobs() -> list[dict]:
    return [{k: v for k, v in j.to_dict().items() if k != "lines"} for j in
            sorted(JOBS.values(), key=lambda j: -j.started)]


def job(job_id: str, since: int = 0) -> dict:
    return JOBS[job_id].to_dict(since)


def cancel(body: dict) -> dict:
    j = JOBS[str(body.get("id"))]
    j.cancel()
    return j.to_dict()


# ---------------------------------------------------------------- env + data

def env() -> dict:
    gate = {"min_conf": DEFAULT.min_conf, "agree": DEFAULT.agree, "min_hold_s": DEFAULT.min_hold_s}
    return {
        "typesafe_key": bool(os.environ.get("TYPESAFE_API_KEY")),
        "csv_default": os.environ.get("TRADE_JEV_CSV", r"C:\Transfer"),
        "folders": {"csv": str(CSV_DIR), "databento": str(DATA_DIR), "live": str(LIVE_DIR)},
        "config": asdict(Config()), "gate": gate,
        "commission": {"MNQ": 0.62, "NQ": Config().commission},
        "avg_tokens": _avg_tokens(),
    }


def _avg_tokens() -> float:
    """Average Jev input tokens per call from past runs (for the cost estimate)."""
    for line in reversed(_index_lines()):
        for name in line.get("policies", {}):
            if name.startswith("jev"):
                s = RUNS / line["run_id"] / "summary.json"
                if s.exists():
                    t = json.loads(s.read_text()).get(name, {}).get("all", {}).get("avg_tokens")
                    if t:
                        return float(t)
    return 1200.0


def days() -> list[dict]:
    out = []
    for day, (sym, path) in sorted(list_days().items()):
        meta = path.with_suffix(".json")
        m = json.loads(meta.read_text()) if meta.exists() else {}
        rows = m.get("rows") or 0
        out.append({
            "day": day, "symbol": sym, "file": path.name, "folder": str(path.parent),
            "source": m.get("source") or ("databento" if path.name.startswith("GLBX") else "?"),
            "synthetic": bool(m.get("synthetic")),
            "levels": m.get("levels", 10), "rows": rows, "trades": m.get("trades"),
            "buy_volume": m.get("buy_volume"), "sell_volume": m.get("sell_volume"),
            "crossed": round(m["crossed_rows"] / rows, 4) if rows and "crossed_rows" in m else None,
            "padded": round(m["padded_rows"] / rows, 4) if rows and "padded_rows" in m else None,
            "depth_mode": m.get("depth_mode"), "point_value": point_value(sym),
            "bars": _bars_path(sym, day) is not None,
        })
    return out


def _bars_path(symbol: str, day: str) -> Path | None:
    p = CSV_DIR / "bars" / f"{symbol}__{day}.json"
    return p if p.exists() else None


def day_overview(day: str) -> dict:
    sym, path = list_days()[day]
    return _overview(day, sym, str(path), path.stat().st_mtime)


@lru_cache(maxsize=16)
def _overview(day: str, sym: str, path: str, _mtime: float) -> dict:
    """Per-minute mid OHLC (points) and buy / sell aggressor volume, read one row group at a time."""
    pf = pq.ParquetFile(path)
    mins, mids, deltas = [], [], []
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["ts_event", "bid_px", "ask_px", "trade_delta"])
        ts = t.column("ts_event").combine_chunks().cast("int64").to_numpy()
        bid = pc.list_element(t.column("bid_px"), 0).to_numpy()
        ask = pc.list_element(t.column("ask_px"), 0).to_numpy()
        mins.append(ts // 60_000_000_000)
        mids.append((bid + ask) / 2)
        deltas.append(t.column("trade_delta").combine_chunks().to_numpy())
    m, mid, d = np.concatenate(mins), np.concatenate(mids), np.concatenate(deltas)
    if not len(m):
        return {"day": day, "symbol": sym, "t": []}
    keys, first = np.unique(m, return_index=True)
    last = np.r_[first[1:], len(m)] - 1
    out = {
        "day": day, "symbol": sym, "tick": TICK,
        "t": (keys * 60).tolist(),
        "o": mid[first].tolist(), "c": mid[last].tolist(),
        "h": np.maximum.reduceat(mid, first).tolist(), "l": np.minimum.reduceat(mid, first).tolist(),
        "buy": np.add.reduceat(np.where(d > 0, d, 0), first).tolist(),
        "sell": np.add.reduceat(np.where(d < 0, -d, 0), first).tolist(),
    }
    if bp := _bars_path(sym, day):
        out["bars"] = json.loads(bp.read_text())
    return out


# ---------------------------------------------------------------- results

def _index_lines() -> list[dict]:
    idx = RUNS / "index.jsonl"
    if not idx.exists():
        return []
    return [json.loads(x) for x in idx.read_text().splitlines() if x.strip()]


def results() -> list[dict]:
    """Runs on disk, newest first; a re-run with the same id shows its latest record."""
    by_id: dict[str, dict] = {}
    for rec in _index_lines():
        if (RUNS / rec["run_id"] / "config.json").exists():
            by_id[rec["run_id"]] = rec
    out = []
    for rec in by_id.values():
        out.append({
            "run_id": rec["run_id"], "finished_at": rec.get("finished_at"), "days": rec["days"],
            "policies": {k: {"net_pnl": v["net_pnl"], "trades": v["trades"]} for k, v in rec["policies"].items()},
            "jev_cost_usd": rec.get("jev_cost_usd", 0), "config": rec.get("config"),
            "has_jev": any(k.startswith("jev") for k in rec["policies"]),
        })
    return sorted(out, key=lambda r: r.get("finished_at") or "", reverse=True)


def result(run_id: str, points_per_day: int = 600, max_trades: int = 5000) -> dict:
    if not _RUN_ID.match(run_id):
        raise KeyError(run_id)
    rd = RUNS / run_id
    cfg = json.loads((rd / "config.json").read_text())
    summary = json.loads((rd / "summary.json").read_text()) if (rd / "summary.json").exists() else {}
    res = json.loads((rd / "results.json").read_text()) if (rd / "results.json").exists() else {}
    equity = []
    for f in sorted((rd / "equity").glob("*.json")):
        e = json.loads(f.read_text())
        n = len(e.get("t_ns", []))
        if not n:
            continue
        idx = np.unique(np.r_[np.linspace(0, n - 1, min(n, points_per_day)).astype(int), n - 1])
        equity.append({
            "day": e["day"], "symbol": e.get("symbol"),
            "t": [e["t_ns"][i] // 10**9 for i in idx],
            "mid": [round(e["mid"][i], 2) for i in idx],
            "policies": {p: [v["equity"][i] for i in idx] for p, v in e["policies"].items()},
        })
    trades = {}
    for name in cfg.get("policies", []):
        tf = rd / name / "trades.csv"
        if tf.exists():
            with tf.open() as fh:
                rows = list(csv.DictReader(fh))
            trades[name] = [{k: (v if k in ("day", "reason") else float(v)) for k, v in r.items()}
                            for r in rows[-max_trades:]]
    return {"run_id": run_id, "config": cfg, "summary": summary, "record": res,
            "equity": equity, "trades": trades}


# ---------------------------------------------------------------- start

def scan_folder(body: dict) -> dict:
    path = Path(str(body.get("path", "")).strip().strip('"')).expanduser()
    if not path.exists():
        raise ValueError(f"not found: {path}")
    files = scan(path)
    return {"path": str(path), "symbol": guess_symbol(files), "files": [asdict(f) for f in files]}


def start_ingest(body: dict) -> dict:
    path = Path(str(body.get("path", "")).strip().strip('"')).expanduser()
    if not path.exists():
        raise ValueError(f"not found: {path}")
    argv = [sys.executable, "-u", "-m", "trade_jev.ingest", str(path)]
    if sym := re.sub(r"\W", "", str(body.get("symbol") or "")):
        argv += ["--symbol", sym]
    mode = str(body.get("depth_mode") or "auto")
    if mode not in ("auto", "insert", "price", "set"):
        raise ValueError(f"bad depth_mode {mode!r}")
    argv += ["--depth-mode", mode]
    return _start("ingest", f"import {path.name or path}", argv)


def start_run(body: dict) -> dict:
    avail = list_days()
    sel = [d for d in body.get("days", []) if isinstance(d, str) and _DAY.match(d)]
    missing = [d for d in sel if d not in avail]
    if not sel or missing:
        raise ValueError(f"pick days that have data (missing: {missing})" if missing else "pick at least one day")
    pols = [p for p in body.get("policies", []) if p in POLICIES]
    if not pols:
        raise ValueError("pick at least one policy")
    s = body.get("settings", {})
    cfg_defaults = asdict(Config())

    def num(key, default, cast=float, lo=None, hi=None):
        v = cast(s.get(key, default))
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ValueError(f"{key} must be in [{lo}, {hi}]")
        return v

    argv = [sys.executable, "-u", "-m", "trade_jev.run", "--days", ",".join(sorted(set(sel))),
            "--policies", ",".join(pols)]
    argv += ["--min-conf", str(num("min_conf", DEFAULT.min_conf, float, 0, 1)),
             "--agree", str(num("agree", DEFAULT.agree, int, 1, 50)),
             "--min-hold", str(num("min_hold_s", DEFAULT.min_hold_s, float, 0, 86400))]
    for f in fields(Config):
        v = s.get(f.name, cfg_defaults[f.name])
        if isinstance(f.default, str):
            if not _TIME.match(str(v)):
                raise ValueError(f"{f.name} must be HH:MM:SS")
            v = str(v) if len(str(v)) == 8 else f"{v}:00"
        else:
            v = type(f.default)(v)
            if v < 0:
                raise ValueError(f"{f.name} must be ≥ 0")
        argv += [f"--{f.name.replace('_', '-')}", str(v)]
    if float(s.get("cadence_s", 15)) < 1:
        raise ValueError("cadence_s must be ≥ 1")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-bt"
    argv += ["--run-id", run_id]
    title = f"{', '.join(pols)} on {len(sel)} day{'s' if len(sel) > 1 else ''}"
    return _start("run", title, argv, run_id)
