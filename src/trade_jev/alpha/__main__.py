"""MNQ alpha signal: build features, walk-forward train/evaluate, predict.

    python -m trade_jev.alpha scan    --data C:\\Transfer
    python -m trade_jev.alpha build   --data C:\\Transfer --out runs/alpha/features.parquet
    python -m trade_jev.alpha train   --features runs/alpha/features.parquet --horizon 60
    python -m trade_jev.alpha predict --model runs/alpha/<id>/model --data <folder> --out signals.csv
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from trade_jev.alpha import io
from trade_jev.alpha.features import NS, Spec, build
from trade_jev.alpha.model import (TrainConfig, backtest, deciles, fit_final, load_model,
                                   single_feature_ic, walk_forward)

ROOT = Path(__file__).resolve().parents[3]


def _fmt_ns(ns: int) -> str:
    return pd.Timestamp(ns, unit="ns", tz="UTC").strftime("%Y-%m-%d %H:%M:%S")


def cmd_scan(a) -> None:
    files = io.discover(Path(a.data))
    by = defaultdict(list)
    for f in files:
        by[f.kind].append(f)
    if not files:
        print(f"no candle / L1 / L2 files recognised under {a.data}")
    for kind, fs in by.items():
        print(f"{kind}: {len(fs)} file(s), {_fmt_ns(fs[0].first_ns)} → {_fmt_ns(max(f.last_ns for f in fs))} UTC")
        for f in fs[:50]:
            print(f"   {f.path.name}  sep={f.sep!r}  {_fmt_ns(f.first_ns)} → {_fmt_ns(f.last_ns)}")


def _chunks(l1_files: list[io.DataFile], hours: float) -> list[tuple[int, int]]:
    """Time windows to build in, so a huge L1/L2 export never has to be in memory at once."""
    start = min(f.first_ns for f in l1_files)
    end = max(f.last_ns for f in l1_files) + 1
    step = int(hours * 3600 * NS)
    # align to 22:00 UTC: inside the daily CME halt (21:00–22:00 UTC in summer, 22:00–23:00 in winter)
    first = (start - 22 * 3600 * NS) // (24 * 3600 * NS) * (24 * 3600 * NS) + 22 * 3600 * NS
    return [(s, s + step) for s in range(first, end, step) if s + step > start]


def cmd_build(a) -> None:
    files = io.discover(Path(a.data))
    l1f = [f for f in files if f.kind == "l1"]
    l2f = [f for f in files if f.kind == "l2"]
    cf = [f for f in files if f.kind == "candles"]
    if not l1f:
        raise SystemExit(f"no L1 tick files under {a.data} (run `scan` to see what was found)")
    spec = Spec(step_s=a.step, horizons_s=tuple(a.horizons), latency_ms=a.latency_ms,
                session=a.session, l2_mode=a.l2_mode)
    candles = io.load(cf) if cf else None
    print(f"L1 {len(l1f)} file(s), L2 {len(l2f)} file(s), candles {len(cf)} file(s)"
          + (f" ({len(candles):,} bars)" if candles is not None else ""))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames, infos = [], []
    # warm-up so lookbacks/book are filled at the start of each chunk; labels need look-ahead
    warm = int(max(spec.windows_s) * NS + 600 * NS)
    ahead = int((max(spec.horizons_s) + 60) * NS)
    for s, e in _chunks(l1f, a.chunk_hours):
        t0 = time.time()
        l1 = io.load(l1f, s - warm, e + ahead)
        if l1.empty:
            continue
        l2 = io.load(l2f, s - warm - int(3600 * NS), e) if l2f else None  # extra hour so the book is built
        c = candles[(candles.ts >= s - 3 * 86400 * NS) & (candles.ts < e)] if candles is not None else None
        df, info = build(l1, l2, c, spec)
        df = df[(df.ts >= s) & (df.ts < e)]
        info.update(window=f"{_fmt_ns(s)} → {_fmt_ns(e)}", rows=len(df), secs=round(time.time() - t0, 1))
        print(f"  {info}", flush=True)
        infos.append(info)
        if len(df):
            frames.append(df)
    if not frames:
        raise SystemExit("no feature rows built")
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"spec": spec.__dict__, "data": str(a.data), "chunks": infos, "rows": len(df),
         "days": sorted(df.day.unique().tolist())}, indent=2, default=str))
    print(f"wrote {out}: {len(df):,} rows × {df.shape[1]} cols, {df.day.nunique()} days")


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index() if df.index.name else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False))


def cmd_train(a) -> None:
    df = pd.read_parquet(a.features)
    if a.session == "rth":
        df = df[(df.min_from_rth_open >= 0) & (df.min_from_rth_open < 390)]
    cfg = TrainConfig(horizon_s=a.horizon, min_train_days=a.min_train_days,
                      max_train_days=a.max_train_days, stride=a.stride, commission=a.commission)
    run = Path(a.out) if a.out else ROOT / "runs" / "alpha" / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    print(f"{len(df):,} rows, {df.day.nunique()} days, horizon {cfg.horizon_s}s → {run}")
    pred, folds, imp = walk_forward(df, cfg)
    if pred.empty:
        raise SystemExit(f"not enough days for walk-forward (have {df.day.nunique()}, "
                         f"need > {cfg.min_train_days}); lower --min-train-days")
    y = pred[f"fwd_{cfg.horizon_s}s_t"]
    m = y.notna()
    bts = [backtest(pred, cfg.horizon_s, q, cfg.commission) for q in cfg.quantiles]
    bt_table = pd.DataFrame([b for b, _ in bts])
    dec = deciles(pred, cfg.horizon_s)
    base = single_feature_ic(df, cfg.horizon_s)
    summary = {
        "horizon_s": cfg.horizon_s, "oos_days": len(folds), "oos_rows": int(m.sum()),
        "rank_ic": round(float(pred.pred[m].rank().corr(y[m].rank())), 4),
        "ic": round(float(np.corrcoef(pred.pred[m], y[m])[0, 1]), 4),
        "rank_ic_mean_daily": round(float(folds.rank_ic.mean()), 4),
        "rank_ic_t_stat": round(float(folds.rank_ic.mean() / folds.rank_ic.std() * np.sqrt(len(folds))), 2)
        if len(folds) > 1 else None,
        "pos_ic_days": f"{int((folds.rank_ic > 0).sum())}/{len(folds)}",
        "sign_hit": round(float((np.sign(pred.pred[m]) == np.sign(y[m]))[y[m] != 0].mean()), 4),
    }
    pred.to_parquet(run / "predictions.parquet", index=False)
    folds.to_csv(run / "folds.csv", index=False)
    bt_table.to_csv(run / "backtest.csv", index=False)
    for (b, tr) in bts:
        if len(tr):
            tr.to_csv(run / f"trades_q{b['q']}.csv", index=False)
    imp.rename("gain_share").to_csv(run / "importance.csv")
    side = Path(a.features).with_suffix(".json")
    spec = json.loads(side.read_text())["spec"] if side.exists() else Spec().__dict__
    meta = fit_final(df, cfg, run / "model", spec) if not a.no_final else None
    (run / "summary.json").write_text(json.dumps(
        {"summary": summary, "config": cfg.__dict__, "features": a.features,
         "cuts": meta and meta["cuts"]}, indent=2, default=str))
    report = f"""# MNQ alpha: {cfg.horizon_s}s horizon

Walk-forward by trading day; every number below is out of sample.

## Signal quality
{_md_table(pd.DataFrame([summary]).T.rename(columns={0: 'value'}).rename_axis('metric'))}

## Per day
{_md_table(folds.round(4))}

## Forward move by prediction decile (ticks)
{_md_table(dec)}

## Backtest: 1 MNQ, cross spread in and out, ${cfg.commission}/side, non-overlapping
Trade when |pred| ≥ the q-quantile of |pred| on the held-out validation day.
{_md_table(bt_table)}

## Top features (mean gain share)
{_md_table(imp.head(25).round(4).rename('gain').rename_axis('feature').to_frame())}

## Baseline: single-feature rank IC (full sample)
{_md_table(base.round(4).rename('rank_ic').rename_axis('feature').to_frame())}
"""
    (run / "report.md").write_text(report)
    print("\n" + json.dumps(summary, indent=2))
    print("\n" + bt_table.to_string(index=False))
    print(f"\nreport: {run / 'report.md'}")


def cmd_predict(a) -> None:
    predict, meta = load_model(Path(a.model))
    files = io.discover(Path(a.data))
    l1 = io.load([f for f in files if f.kind == "l1"])
    l2f = [f for f in files if f.kind == "l2"]
    cf = [f for f in files if f.kind == "candles"]
    spec = Spec(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta["spec"].items()})
    df, _ = build(l1, io.load(l2f) if l2f else None, io.load(cf) if cf else None, spec)
    for c in meta["features"]:
        if c not in df:
            df[c] = np.nan
    df["pred_ticks"] = predict(df)
    cut = meta["cuts"][str(a.q)]
    df["signal"] = np.where(df.pred_ticks >= cut, 1, np.where(df.pred_ticks <= -cut, -1, 0))
    df["time_utc"] = pd.to_datetime(df.ts, unit="ns", utc=True)
    df[["time_utc", "mid", "pred_ticks", "signal"]].to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} rows, {int((df.signal != 0).sum()):,} non-zero signals (cut {cut:.3f} ticks)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m trade_jev.alpha", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="list recognised data files")
    s.add_argument("--data", required=True)
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("build", help="data files → feature/label parquet")
    s.add_argument("--data", required=True)
    s.add_argument("--out", default=str(ROOT / "runs" / "alpha" / "features.parquet"))
    s.add_argument("--step", type=float, default=1.0, help="grid seconds")
    s.add_argument("--horizons", type=int, nargs="+", default=[30, 60, 300])
    s.add_argument("--latency-ms", type=int, default=250)
    s.add_argument("--session", choices=["all", "rth"], default="all")
    s.add_argument("--l2-mode", choices=["auto", "insert", "slot"], default="auto")
    s.add_argument("--chunk-hours", type=float, default=24.0)
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("train", help="walk-forward train + OOS report + final model")
    s.add_argument("--features", default=str(ROOT / "runs" / "alpha" / "features.parquet"))
    s.add_argument("--out")
    s.add_argument("--horizon", type=int, default=60)
    s.add_argument("--session", choices=["all", "rth"], default="all")
    s.add_argument("--min-train-days", type=int, default=3)
    s.add_argument("--max-train-days", type=int, default=0)
    s.add_argument("--stride", type=int, default=5)
    s.add_argument("--commission", type=float, default=0.62)
    s.add_argument("--no-final", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("predict", help="score new data with a saved model")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="signals.csv")
    s.add_argument("--q", type=float, default=0.9)
    s.set_defaults(fn=cmd_predict)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
