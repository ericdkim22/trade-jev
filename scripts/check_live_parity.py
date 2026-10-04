"""Check that the live engine reproduces the backtest exactly.

  uv run python scripts/check_live_parity.py runs/<id> [runs/<id> ...]

For every day of the given runs, streams the recorded day file through the live engine
(trade_jev.live, file mode, stored Jev answers) and compares its trades with the Python replay
(trade_jev.replay.replay_day) under the same settings. Every trade must match.
"""

import asyncio
import dataclasses
import sys
from pathlib import Path

from trade_jev.data import list_days
from trade_jev.live import Engine, ParquetSource, StoredAnswers
from trade_jev.replay import Settings, load_market, load_runs, replay_day
from trade_jev.settings import DEFAULT

CASES = [DEFAULT, Settings(0.5, 2, 120, 400, 0), Settings(0.0, 1, 0, 20, 40)]


def key(t):
    return (t.side, t.entry_ns, t.entry_px, t.exit_ns, t.exit_px, t.reason, t.pnl)


async def live_trades(rd, s: Settings):
    cfg = dataclasses.replace(rd.config, stop_ticks=s.stop_ticks, target_ticks=s.target_ticks)
    symbol, path = list_days()[rd.day]
    eng = Engine(rd.day, symbol, cfg, s, StoredAnswers(rd.answers), baselines=[])
    async for b in ParquetSource(path, speed=0).batches():
        await eng.feed(b)
        if eng.closed:
            break
    await eng.finish()
    return eng.results[eng.jev_name].trades


async def main(run_dirs: list[Path]) -> int:
    bad = 0
    print(f"{'day':<11} {'settings':<52} {'replay':>7} {'live':>7}  match")
    for day_s, rd in sorted(load_runs(run_dirs).items()):
        day, grid = load_market(rd)
        for s in CASES:
            ref = (await replay_day(rd, day, grid, s)).trades
            got = await live_trades(rd, s)
            ok = list(map(key, ref)) == list(map(key, got))
            bad += not ok
            print(f"{day_s:<11} {s.label():<52} {len(ref):>7} {len(got):>7}  {'ok' if ok else 'MISMATCH'}", flush=True)
            if not ok:
                for a, b in zip(map(key, ref), map(key, got)):
                    if a != b:
                        print(f"  first diff: replay {a}\n              live   {b}")
                        break
        del day
    print("all trades match" if not bad else f"{bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main([Path(p) for p in sys.argv[1:]])))
