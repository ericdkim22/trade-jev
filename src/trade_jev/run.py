"""Run: backtest days while calling Jev; stores answers + trades in runs/<id>/.

  uv run python -m trade_jev.run --days 2026-06-23                     # jev + baselines, one day
  uv run python -m trade_jev.run --all                                 # all 15 days
  uv run python -m trade_jev.run --days 2026-06-23 --policies hold,random,imbalance   # no API key needed
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import fields
from datetime import datetime

from trade_jev import ROOT
from trade_jev.data import decision_grid, list_days, load_day
from trade_jev.harness import Config, DayResult, run_day
from trade_jev.outputs import print_results, write_config, write_equity, write_results
from trade_jev.policies import BASELINES, Gated, JevPolicy, JsonlCache, RateLimiter
from trade_jev.settings import DEFAULT


def build_policies(names: list[str], args) -> tuple[list, object | None]:
    out, client = [], None
    for n in names:
        if n == "jev":
            from typesafe_sdk import AsyncTypeSafeClient
            client = AsyncTypeSafeClient()
            jev = JevPolicy(client, JsonlCache(ROOT / "cache" / "jev.jsonl"),
                            RateLimiter(args.rps), encoder=args.encoder, model=args.model, question=args.question)
            out.append(Gated(jev, args.min_conf, args.agree, args.min_hold)
                       if args.agree > 1 or args.min_conf > 0 or args.min_hold > 0 else jev)
        else:
            out.append(BASELINES[n]())
    return out, client


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="", help="comma-separated YYYY-MM-DD")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--policies", default="jev,hold,random,imbalance")
    ap.add_argument("--encoder", default="raw_l10")
    ap.add_argument("--question", default="scalp", help="Jev question variant (policies.QUESTIONS)")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--rps", type=float, default=15.0, help="Jev requests/sec (limit is 20)")
    ap.add_argument("--parallel-days", type=int, default=3, help="days held in memory at once")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--min-conf", type=float, default=DEFAULT.min_conf, help="Jev: ignore answers below this prob")
    ap.add_argument("--agree", type=int, default=DEFAULT.agree, help="Jev: consecutive agreeing answers to act")
    ap.add_argument("--min-hold", type=float, default=DEFAULT.min_hold_s, help="Jev: min seconds before reversing")
    for f in fields(Config):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()

    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    available = list_days()
    days = sorted(available) if args.all else [d for d in args.days.split(",") if d]
    if not days:
        ap.error("pass --days or --all")
    missing = [d for d in days if d not in available]
    if missing:
        ap.error(f"no data for {missing}; have {sorted(available)}")

    policies, client = build_policies(args.policies.split(","), args)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "runs" / run_id
    gate = {"min_conf": args.min_conf, "agree": args.agree, "min_hold_s": args.min_hold}
    write_config(out, cfg, days, [p.name for p in policies], args.encoder, args.model, gate)

    results: dict[str, list[DayResult]] = {p.name: [] for p in policies}
    book_rows_seen = snapshots = 0
    t_start = time.time()
    sem = asyncio.Semaphore(args.parallel_days)

    async def one_day(d: str) -> None:
        async with sem:
            t0 = time.time()
            grid = decision_grid(d, cfg.start_et, cfg.end_et, cfg.cadence_s)
            day = await asyncio.to_thread(load_day, d, grid)
            # early closes (e.g. Juneteenth) end before end_et — never decide on a stale book
            grid = grid[(grid >= day.ts[0]) & (grid <= day.ts[-1])]
            nonlocal book_rows_seen, snapshots
            snapshots += len(grid)
            book_rows_seen += int(day.row_at(grid[-1]) - day.row_at(grid[0]) + 1) if len(grid) else 0
            print(f"[{d}] loaded {len(day.ts):,} rows in {time.time() - t0:.0f}s; "
                  f"{len(grid)} decisions × {len(policies)} policies", flush=True)
            day_res = await asyncio.gather(*(run_day(day, grid, p, cfg) for p in policies))
            for p, r in zip(policies, day_res):
                results[p.name].append(r)
                print(f"[{d}] {p.name:>16}: {len(r.trades):4d} trades  ${r.pnl:>10,.2f}", flush=True)
            write_equity(out, day, grid, list(day_res))
            del day

    try:
        await asyncio.gather(*(one_day(d) for d in days))
    finally:
        if client is not None:
            await client.aclose()

    jev = [getattr(p, "inner", p) for p in policies]
    jev = [p for p in jev if isinstance(p, JevPolicy)]
    summary, record = write_results(
        out, run_id, results, days, cfg, wall_s=round(time.time() - t_start, 1),
        snapshots=snapshots, book_rows=book_rows_seen, jev_policies=jev,
        encoder=args.encoder, model=args.model, gate=gate)
    print_results(out, record, summary)

    if jev:
        print("\nview it: uv run python -m trade_jev.view")

if __name__ == "__main__":
    asyncio.run(main())
