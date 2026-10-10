"""Write a synthetic CSV export (L2 ticks, L1 ticks, 1-min candles) in the vendor format.

  uv run python scripts/make_csv_sample.py [out_dir] [--days 2026-06-22,2026-06-23]
  uv run python -m trade_jev.ingest data/csv-sample          # → data/csv/

Made-up MNQ prices from a seeded random walk; only the file format is real. The L2 file uses
`insert` depth semantics (add shifts deeper levels down, remove shifts them up), so ingest's
auto depth-mode check should pick `insert`.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trade_jev.data import ET, LEVELS, TICK

ROOT = Path(__file__).resolve().parents[1]


def stamp(t: datetime, micros: bool = True) -> str:
    return t.strftime("%Y%m%d%H%M%S%f" if micros else "%Y%m%d%H%M%S")


def px(ticks: int) -> str:
    return f"{ticks * TICK:.2f}"


def one_day(day: str, rng: random.Random, l2, l1, candles, start="09:00", end="16:30") -> int:
    t = datetime.fromisoformat(f"{day}T{start}").replace(tzinfo=ET).astimezone(timezone.utc)
    t_end = datetime.fromisoformat(f"{day}T{end}").replace(tzinfo=ET).astimezone(timezone.utc)
    bb = 4 * 21_000 + rng.randint(-400, 400)       # best bid in ticks (~21,000)
    bids = [[bb - k, rng.randint(1, 30)] for k in range(LEVELS)]
    asks = [[bb + 1 + k, rng.randint(1, 30)] for k in range(LEVELS)]
    drift = 0.0
    n = 0
    bar = None

    def ev(side, act, depth, p, s):
        nonlocal n
        l2.write(f"{stamp(t)};2;{side};{px(p)};{s};{depth};{act}\n")
        n += 1

    def touch(side):
        lv = bids if side == 0 else asks
        l1.write(f"{stamp(t)};1;{side};{px(lv[0][0])};{lv[0][1]}\n")

    for side in (0, 1):  # opening snapshot: adds at depth 1..10
        for d, (p, s) in enumerate(bids if side == 0 else asks, 1):
            ev(side, 0, d, p, s)
        touch(side)

    while t < t_end:
        t += timedelta(microseconds=rng.randint(2_000, 120_000))
        drift = 0.995 * drift + rng.gauss(0, 0.02)
        r = rng.random()
        if r < 0.15:  # trade; sometimes it clears the touch and the market moves a tick
            up = rng.random() < 0.5 + max(-0.3, min(0.3, drift))
            side, lv = (1, asks) if up else (0, bids)
            q = rng.randint(1, 6)
            l2.write(f"{stamp(t)};1;2;{px(lv[0][0])};{q};;\n")
            l1.write(f"{stamp(t)};1;2;{px(lv[0][0])};{q}\n")
            m = (stamp(t, False)[:12])
            p = lv[0][0] * TICK
            if bar and bar[0] == m:
                bar[2], bar[3], bar[4], bar[5] = max(bar[2], p), min(bar[3], p), p, bar[5] + q
            else:
                if bar:
                    candles.write(f"{bar[0]}00;2;{bar[1]:.2f};{bar[2]:.2f};{bar[3]:.2f};{bar[4]:.2f};{bar[5]}\n")
                bar = [m, p, p, p, p, q]
            if q < lv[0][1]:
                lv[0][1] -= q
                ev(side, 1, 1, *lv[0])
            else:  # level gone: the touch moves away, the other side steps in behind it
                gone = lv.pop(0)[0]
                ev(side, 2, 1, gone, 0)
                far = lv[-1][0] + (1 if side == 1 else -1)
                lv.append([far, rng.randint(1, 30)])
                ev(side, 0, LEVELS, *lv[-1])
                other = bids if side == 1 else asks
                ev(1 - side, 2, LEVELS, *other.pop())
                other.insert(0, [gone, rng.randint(1, 8)])
                ev(1 - side, 0, 1, *other[0])
                touch(0)
                touch(1)
        else:  # resting size changes somewhere in the top 10
            side = rng.randint(0, 1)
            lv = bids if side == 0 else asks
            d = min(LEVELS, int(rng.expovariate(0.35)) + 1)
            lv[d - 1][1] = max(1, lv[d - 1][1] + rng.randint(-5, 6))
            ev(side, 1, d, *lv[d - 1])
            if d == 1:
                touch(side)
    if bar:
        candles.write(f"{bar[0]}00;2;{bar[1]:.2f};{bar[2]:.2f};{bar[3]:.2f};{bar[4]:.2f};{bar[5]}\n")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", type=Path, default=ROOT / "data" / "csv-sample")
    ap.add_argument("--days", default="2026-06-22,2026-06-23")
    ap.add_argument("--symbol", default="MNQU6")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    sym = args.symbol
    with (args.out / f"{sym}_L2_ticks.csv").open("w") as l2, \
         (args.out / f"{sym}_L1_ticks.csv").open("w") as l1, \
         (args.out / f"{sym}_1min_candles.csv").open("w") as candles:
        for day in args.days.split(","):
            n = one_day(day, rng, l2, l1, candles)
            print(f"{day}: {n:,} L2 events")
    print(f"wrote synthetic CSVs → {args.out}")


if __name__ == "__main__":
    main()
