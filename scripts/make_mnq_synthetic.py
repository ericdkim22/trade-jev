"""Write synthetic MNQ candle / L1 / L2 files in the C:\\Transfer text formats.

A hidden "pressure" process drives trade signs, top-of-book sizes and the next
price moves, so a working pipeline should find a small positive out-of-sample IC.
L2 rows follow insert semantics (Add pushes rows down, Remove pulls them up).

    python scripts/make_mnq_synthetic.py --out /tmp/mnq --days 6 --hours 3
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

TICK = 0.25
LEVELS = 10


def stamp(t: datetime, micro: bool = True) -> str:
    return t.strftime("%Y%m%d%H%M%S%f" if micro else "%Y%m%d%H%M%S")


def one_day(day: date, hours: float, rng: np.random.Generator, out: Path, edge: float = 1.0) -> None:
    """edge scales how much the hidden pressure moves price (0 = pure noise, nothing to find)."""
    start = datetime(day.year, day.month, day.day, 13, 30, tzinfo=timezone.utc)
    n = int(hours * 3600 * 8)                       # ~8 events / second
    dt = rng.exponential(1 / 8, n)
    t_s = np.cumsum(dt)
    bid = int(21000 / TICK)                          # price in ticks
    x = 0.0
    bsz = [int(rng.integers(3, 20)) for _ in range(LEVELS)]
    asz = [int(rng.integers(3, 20)) for _ in range(LEVELS)]
    l1, l2 = [], []
    t0 = stamp(start)                                # opening snapshot: full book as Adds
    for d in range(LEVELS):
        l2.append(f"{t0};2;0;{(bid - d) * TICK:.2f};{bsz[d]};{d + 1};0")
        l2.append(f"{t0};2;1;{(bid + 1 + d) * TICK:.2f};{asz[d]};{d + 1};0")
    bars: dict[int, list] = {}
    cum_vol = 0
    for i in range(n):
        t = start + timedelta(seconds=float(t_s[i]))
        ts = stamp(t)
        x = 0.997 * x + 0.08 * rng.standard_normal()
        p_up = 1 / (1 + np.exp(-2.5 * edge * x))
        u = rng.random()
        minute = int(t_s[i] // 60)
        if u < 0.35:                                 # trade
            side = 1 if rng.random() < 1 / (1 + np.exp(-1.5 * x)) else -1
            px = (bid + 1) * TICK if side > 0 else bid * TICK
            vol = int(rng.geometric(0.5))
            cum_vol += vol
            l1.append(f"{ts};1;2;{px:.2f};{vol}")
            if rng.random() < 0.02:
                l1.append(f"{ts};1;3;0;{cum_vol}")   # DailyVolume: must be ignored
            b = bars.setdefault(minute, [px, px, px, px, 0, bid * TICK, (bid + 1) * TICK])
            b[1], b[2], b[3], b[4] = max(b[1], px), min(b[2], px), px, b[4] + vol
            continue
        if u < 0.40:                                 # price move
            if rng.random() < p_up:                  # bid up a tick: new best bid, best ask lifted
                bid += 1
                bsz.insert(0, int(rng.integers(1, 8))); bsz.pop()
                l2.append(f"{ts};2;0;{bid * TICK:.2f};{bsz[0]};1;0")
                asz.pop(0); asz.append(int(rng.integers(3, 20)))
                l2.append(f"{ts};2;1;{(bid + 1) * TICK:.2f};{asz[0]};1;2")
                l2.append(f"{ts};2;1;{(bid + LEVELS) * TICK:.2f};{asz[-1]};{LEVELS};0")
            else:
                bid -= 1
                asz.insert(0, int(rng.integers(1, 8))); asz.pop()
                l2.append(f"{ts};2;1;{(bid + 1) * TICK:.2f};{asz[0]};1;0")
                bsz.pop(0); bsz.append(int(rng.integers(3, 20)))
                l2.append(f"{ts};2;0;{bid * TICK:.2f};{bsz[0]};1;2")
                l2.append(f"{ts};2;0;{(bid - LEVELS + 1) * TICK:.2f};{bsz[-1]};{LEVELS};0")
        else:                                        # size change, tilted by pressure
            s = int(rng.integers(0, 4))
            if rng.random() < 0.5:
                bsz[s] = max(1, int(rng.poisson(10 * np.exp(0.6 * x))))
                l2.append(f"{ts};2;0;{(bid - s) * TICK:.2f};{bsz[s]};{s + 1};1")
            else:
                asz[s] = max(1, int(rng.poisson(10 * np.exp(-0.6 * x))))
                l2.append(f"{ts};2;1;{(bid + 1 + s) * TICK:.2f};{asz[s]};{s + 1};1")
        l1.append(f"{ts};1;0;{bid * TICK:.2f};{bsz[0]}")
        l1.append(f"{ts};1;1;{(bid + 1) * TICK:.2f};{asz[0]}")
        b = bars.setdefault(minute, [bid * TICK] * 4 + [0, bid * TICK, (bid + 1) * TICK])
        b[5], b[6] = bid * TICK, (bid + 1) * TICK
    name = day.strftime("%Y%m%d")
    (out / f"MNQ_{name}_L1.txt").write_text("\n".join(l1) + "\n")
    (out / f"MNQ_{name}_L2.txt").write_text("\n".join(l2) + "\n")
    rows = []
    for m in sorted(bars):
        o, h, lo, c, v, bb, ba = bars[m]
        ts = stamp(start + timedelta(minutes=m), micro=False)
        rows.append(f"{ts},0,{bb:.2f},{bb:.2f},{bb:.2f},{bb:.2f},0")
        rows.append(f"{ts},1,{ba:.2f},{ba:.2f},{ba:.2f},{ba:.2f},0")
        rows.append(f"{ts},2,{o:.2f},{h:.2f},{lo:.2f},{c:.2f},{v}")
    (out / f"MNQ_{name}_1min.csv").write_text("\n".join(rows) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--days", type=int, default=6)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--edge", type=float, default=1.0, help="0 = no signal (leakage check)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    d = date(2026, 6, 8)
    made = 0
    while made < a.days:
        if d.weekday() < 5:
            one_day(d, a.hours, rng, out, a.edge)
            made += 1
            print(d)
        d += timedelta(days=1)


if __name__ == "__main__":
    main()
