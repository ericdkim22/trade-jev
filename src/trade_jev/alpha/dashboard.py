"""JSON for the Alpha tab of the viewer (viz/alpha.html), read from runs/alpha/<run>/.

  GET /api/alpha/runs                       every trained run, newest first
  GET /api/alpha/run?id=<run>               summary, per-day stats, thresholds, deciles,
                                            importance, trades per threshold
  GET /api/alpha/day?id=<run>&day=<d>&q=<q> one OOS day: mid, prediction, cut, trades
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from trade_jev.alpha.model import deciles

ROOT = Path(__file__).resolve().parents[3]
ALPHA = ROOT / "runs" / "alpha"
MAX_POINTS = 2400


def _clean(obj):
    """NaN / inf → None so the payload is valid JSON."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return _clean(obj.item())
    return obj


def run_dir(run_id: str) -> Path | None:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        return None
    d = ALPHA / run_id
    return d if (d / "summary.json").exists() else None


def list_runs() -> list[dict]:
    out = []
    for s in ALPHA.glob("*/summary.json"):
        try:
            meta = json.loads(s.read_text())
        except (OSError, ValueError):
            continue
        out.append({"id": s.parent.name, "mtime": s.stat().st_mtime, **meta.get("summary", {})})
    return _clean(sorted(out, key=lambda r: r["mtime"], reverse=True))


@lru_cache(maxsize=4)
def _predictions(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def _preds(d: Path) -> pd.DataFrame:
    p = d / "predictions.parquet"
    return _predictions(str(p), p.stat().st_mtime)


def _trades(d: Path, q: float) -> pd.DataFrame:
    p = d / f"trades_q{q}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["ts", "day", "dir", "ticks", "pnl"])


def run_payload(run_id: str) -> dict | None:
    d = run_dir(run_id)
    if d is None:
        return None
    meta = json.loads((d / "summary.json").read_text())
    horizon = meta["summary"]["horizon_s"]
    pred = _preds(d)
    bt = pd.read_csv(d / "backtest.csv") if (d / "backtest.csv").exists() else pd.DataFrame()
    imp = pd.read_csv(d / "importance.csv", index_col=0).iloc[:, 0] if (d / "importance.csv").exists() else pd.Series(dtype=float)
    trades = {}
    for q in bt.get("q", []):
        t = _trades(d, q)
        trades[str(q)] = {"ts": (t.ts // 1_000_000).tolist(), "day": t.day.tolist(),
                          "dir": t.dir.tolist(), "ticks": t.ticks.round(2).tolist(), "pnl": t.pnl.round(2).tolist()}
    return _clean({
        "id": run_id, "summary": meta["summary"], "config": meta.get("config", {}),
        "cuts": meta.get("cuts"), "horizon_s": horizon,
        "folds": pd.read_csv(d / "folds.csv").to_dict("records") if (d / "folds.csv").exists() else [],
        "backtest": bt.to_dict("records"),
        "deciles": deciles(pred, horizon).reset_index().to_dict("records"),
        "importance": [{"feature": k, "gain": float(v)} for k, v in imp.head(25).items()],
        "days": sorted(pred.day.unique().tolist()),
        "trades": trades,
    })


def day_payload(run_id: str, day: str, q: float) -> dict | None:
    d = run_dir(run_id)
    if d is None:
        return None
    pred = _preds(d)
    p = pred[pred.day == day]
    if p.empty:
        return None
    step = max(1, math.ceil(len(p) / MAX_POINTS))
    # keep each bucket's strongest prediction so spikes survive downsampling
    idx = p.pred.abs().groupby(np.arange(len(p)) // step).idxmax().to_numpy()
    s = p.loc[idx]
    cut_col = f"cut_{q}"
    t = _trades(d, q)
    t = t[t.day == day]
    return _clean({
        "day": day, "q": q,
        "ts": (s.ts // 1_000_000).tolist(),
        "mid": ((s.px_in_bid + s.px_in_ask) / 2).round(3).tolist(),
        "pred": s.pred.round(4).tolist(),
        "cut": float(p[cut_col].iloc[0]) if cut_col in p else None,
        "trades": {"ts": (t.ts // 1_000_000).tolist(), "dir": t.dir.tolist(), "entry": t.entry.tolist(),
                   "exit": t.exit.tolist(), "ticks": t.ticks.round(2).tolist(), "pnl": t.pnl.round(2).tolist()},
    })
