"""Side variants on stored Jev answers (no Jev calls): long-only, short-only, stricter shorts, flipped answers.

  TRADE_JEV_DATA=data/history uv run python scripts/replay_sides.py runs/hist-raw_l10 [runs/...]
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

from trade_jev.harness import run_day
from trade_jev.policies import Decision, Gated
from trade_jev.replay import StoredAnswers, load_market, load_runs
from trade_jev.settings import DEFAULT as S

FLIP = {"BUY": "SELL", "SELL": "BUY", "HOLD": "HOLD"}


class Side(StoredAnswers):
    """Stored answers, edited per variant before the usual gate."""
    def __init__(self, answers, variant):
        super().__init__(answers)
        self.variant = variant

    async def __call__(self, ctx):
        d = await super().__call__(ctx)
        a, p = d.action, dict(d.probs)
        if self.variant == "long only" and a == "SELL" or self.variant == "short only" and a == "BUY":
            return Decision("HOLD", {"HOLD": 1.0})
        if self.variant == "shorts need 0.85" and a == "SELL" and p.get("SELL", 0) < 0.85:
            return Decision("HOLD", {"HOLD": 1.0})
        if self.variant == "flipped":
            return Decision(FLIP[a], {FLIP[k]: v for k, v in p.items()})
        return d


VARIANTS = ["as is", "long only", "short only", "shorts need 0.85", "flipped"]


def main(run_dirs):
    for rd_path in run_dirs:
        runs = load_runs([Path(rd_path)])
        per = {v: [] for v in VARIANTS}
        for day in sorted(runs):
            rd = runs[day]
            market = load_market(rd)
            for v in VARIANTS:
                r = asyncio.run(run_day(*market, Gated(Side(rd.answers, v), S.min_conf, S.agree, S.min_hold_s), rd.config))
                per[v].append((r.pnl, len(r.trades)))
        print(f"\n{rd_path} ({len(runs)} days)")
        for v, xs in per.items():
            p = [x for x, _ in xs]; n = len(p); m = sum(p) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in p) / (n - 1))
            print(f"  {v:18} net ${sum(p):8.0f}  ${m:6.1f}/day  t {m / (sd / math.sqrt(n)) if sd else 0:5.2f}  "
                  f"win days {sum(x > 0 for x in p)}/{n}  trades {sum(t for _, t in xs)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
