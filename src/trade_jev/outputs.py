"""Write a run's files (runs/<id>/…) and its runs/index.jsonl line. Shared by run.py and live.py."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from trade_jev import ROOT
from trade_jev.harness import Config, DayResult, Trade, equity_curve
from trade_jev.metrics import summarize


def write_config(out: Path, cfg: Config, days: list[str], policy_names: list[str],
                 encoder: str, model: str, gate: dict, **extra) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"config": asdict(cfg), "days": days, "policies": policy_names,
         "encoder": encoder, "model": model, "gate": gate, **extra},
        indent=2))


def write_equity(out: Path, day, grid: np.ndarray, day_results: list[DayResult]) -> None:
    eq_dir = out / "equity"
    eq_dir.mkdir(exist_ok=True)
    eq = {"day": day.day, "symbol": day.symbol, "policies": {}}
    for r in day_results:
        curve = equity_curve(day, grid, r.trades)
        eq["t_ns"], eq["mid"] = curve["t_ns"], curve["mid"]
        eq["policies"][r.policy] = {"equity": curve["equity"], "position": curve["position"]}
    (eq_dir / f"{day.day}.json").write_text(json.dumps(eq))


def write_results(out: Path, run_id: str, results: dict[str, list[DayResult]], days: list[str],
                  cfg: Config, *, wall_s: float, snapshots: int, book_rows: int,
                  jev_policies: list, encoder: str, model: str, gate: dict,
                  **extra) -> tuple[dict, dict]:
    """decisions.jsonl + trades.csv per policy, summary.json, results.json, index line."""
    half = len(days) // 2
    splits = {"all": days} if len(days) < 4 else {"all": days, "dev": days[:half], "holdout": days[half:]}
    summary = {}
    for name, rs in results.items():
        pdir = out / name
        pdir.mkdir(exist_ok=True)
        with (pdir / "decisions.jsonl").open("w") as fh:
            for r in sorted(rs, key=lambda r: r.day):
                for dec in r.decisions:
                    fh.write(json.dumps(dec) + "\n")
        with (pdir / "trades.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(Trade)])
            w.writeheader()
            for r in sorted(rs, key=lambda r: r.day):
                w.writerows(asdict(t) for t in r.trades)
        summary[name] = {s: summarize([r for r in rs if r.day in ds]) for s, ds in splits.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    record = {
        "run_id": run_id,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_time_s": wall_s,
        "days": days,
        "n_days": len(days),
        "cadence_s": cfg.cadence_s,
        "snapshots_decided": snapshots,
        "book_updates_replayed": book_rows,
        "jev_api_calls": sum(p.api_calls for p in jev_policies),
        "jev_cache_hits": sum(p.cache_hits for p in jev_policies),
        "jev_cost_usd": sum(m["all"]["api_cost_usd"] for n, m in summary.items() if n.startswith("jev")),
        "policies": {
            n: {"net_pnl": m["all"]["net_pnl"], "trades": m["all"]["trades"],
                "win_rate": m["all"]["win_rate"], "max_drawdown": m["all"]["max_drawdown"],
                "actions": m["all"]["actions"],
                **({"holdout_net_pnl": m["holdout"]["net_pnl"]} if "holdout" in m else {})}
            for n, m in summary.items()
        },
        "config": asdict(cfg),
        "encoder": encoder if jev_policies else None,
        "model": model if jev_policies else None,
        "gate": gate if jev_policies else None,
        **extra,
    }
    (out / "results.json").write_text(json.dumps(record, indent=2))
    with (ROOT / "runs" / "index.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return summary, record


def print_results(out: Path, record: dict, summary: dict) -> None:
    print(f"\nrun → {out}")
    print(f"{record['n_days']} days · {record['snapshots_decided']:,} snapshots decided · "
          f"{record['book_updates_replayed']:,} book updates · {record['wall_time_s']}s · "
          f"Jev calls: {record['jev_api_calls']:,} (+{record['jev_cache_hits']:,} cached) · "
          f"${record['jev_cost_usd']:.4f}")
    hdr = f"{'policy':>16} {'split':>8} {'net $':>11} {'trades':>7} {'win%':>6} {'sharpe':>7} {'maxDD':>9} {'actions'}"
    print(hdr)
    for name, by in summary.items():
        for s, m in by.items():
            wr = f"{m['win_rate'] * 100:.0f}" if m["win_rate"] is not None else "-"
            print(f"{name:>16} {s:>8} {m['net_pnl']:>11,.2f} {m['trades']:>7} {wr:>6} "
                  f"{str(m['sharpe_daily_ann']):>7} {m['max_drawdown']:>9,.0f} {m['actions']}")
