"""Does Jev's read of a headline predict Nasdaq-100 futures (MNQ)? Headlines from the bz-premarket project's database.

  uv run python scripts/news_nq.py ask  --bz-db <copy of bz.sqlite3> [--since 2026-09-24T18:00] [--until ...]
  uv run python scripts/news_nq.py eval --bz-db <copy> --bars-db <copy of news_entry_bars.sqlite3> [--split 2026-10-03]
  uv run python scripts/news_nq.py nightly    # the pre-registered fade test on new days (research/NEWS_FADE.md)

Use copies (sqlite3 backup API), never bz's live database. One Jev call per headline asks three narrow questions about
Nasdaq-100 futures (QUESTIONS, version QV); the state is the headline as known when it arrived, with no prices.
Answers are cached in cache/news_nq.jsonl (keyed by headline id + QV): change the questions only with a new QV.

eval: MNQ price at the minute the headline was captured (the earliest one could act), moves 5 / 15 / 30 / 60 minutes
later. Size: AUC of Jev's answers for "a top-quartile 30-minute move". Direction: the signed move after headlines Jev
rates big with a clear side, net of ~5 ticks round trip, threshold picked on days before --split and tested after.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "news_nq.jsonl"
QV = 1
SIZES = ["under 0.1%: little or no reaction", "0.1-0.3%: a modest reaction", "0.3-1%: a sharp move",
         "over 1%: a violent move"]
QUESTIONS = {
    "market_moving": "This headline is news about the economy, interest rates, government policy, geopolitics or a "
                     "company big enough to move the whole US stock market or the Nasdaq-100 index, not just one "
                     "company's own stock.",
    "nq_move_size": "If traders react to this headline, how far will Nasdaq-100 futures move in the next 30 minutes "
                    "because of it?",
    "nq_reaction": "Which way will Nasdaq-100 futures move in the next 30 minutes because of this headline?",
}
TICK, COST_TICKS = 0.25, 5  # MNQ: spread + commission round trip, roughly

# Pre-registered 2026-10-10 (research/LOG.md): fade Jev's side on headlines it rates market-moving with a clear side.
BZ_LIVE_DB = Path(r"C:\Users\eric\Documents\bzJEV2\bz-premarket\bz-premarket\server\data\bz.sqlite3")
FADE_START, FADE_MM, FADE_SIDE, FADE_HOLD_MIN, FADE_N = "2026-10-12", 0.9, 0.6, 15, 50
FADE_OUT = ROOT / "research" / "news_fade.json"
FADE_REPORT = ROOT / "research" / "NEWS_FADE.md"


def jev_questions():
    from typesafe_sdk import Choice, Noul, Score
    return {"market_moving": Noul(instructions=QUESTIONS["market_moving"]),
            "nq_move_size": Score(instructions=QUESTIONS["nq_move_size"], criteria=SIZES),
            "nq_reaction": Choice(instructions=QUESTIONS["nq_reaction"],
                                  criteria={"up": "Nasdaq-100 futures rise.", "down": "Nasdaq-100 futures fall.",
                                            "unclear": "No clear direction, or too little reaction to tell."})}


def session(t: datetime) -> str:
    m = t.hour * 60 + t.minute
    return ("weekend" if t.weekday() >= 5 else "pre-market" if 240 <= m < 570 else "regular session"
            if 570 <= m < 960 else "after hours" if 960 <= m < 1200 else "overnight")


def headlines(conn, since, until, sources):
    q = ("SELECT id, ts_et, captured_at, ticker, tickers, title, teaser, source FROM news WHERE ts_et >= ? AND ts_et < ? "
         f"AND source IN ({','.join('?' * len(sources))}) ORDER BY ts_et")
    return [dict(zip(("id", "ts_et", "captured_at", "ticker", "tickers", "title", "teaser", "source"), r))
            for r in conn.execute(q, (since, until, *sources))]


def state(conn, h) -> dict:
    t = datetime.fromisoformat(h["ts_et"][:19])
    earlier = conn.execute("SELECT ts_et, title FROM news WHERE source = ? AND ts_et >= ? AND ts_et < ? AND id != ? "
                           "ORDER BY ts_et DESC LIMIT 5",
                           (h["source"], (t - timedelta(minutes=15)).isoformat(), h["ts_et"], h["id"])).fetchall()
    tickers = json.loads(h["tickers"] or "[]")
    return {"time": f"{t:%a %H:%M} ET, {session(t)}", "source": h["source"], "headline": h["title"],
            "teaser": (h["teaser"] or "").strip()[:300] or None, "tagged_tickers": tickers[:5] or None,
            "earlier_headlines_last_15_min": [f"{r[0][11:16]} ET {r[1]}" for r in earlier] or None}


def load_cache() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("v") == QV:
                out[r["id"]] = r
    return out


async def ask(conn, hs) -> None:
    from typesafe_sdk import AsyncTypeSafeClient
    from trade_jev.policies import RateLimiter
    done = load_cache()
    todo = [h for h in hs if h["id"] not in done]
    print(f"{len(hs)} headlines, {len(todo)} to ask", flush=True)
    client, lim, qs = AsyncTypeSafeClient(), RateLimiter(15), jev_questions()
    CACHE.parent.mkdir(exist_ok=True)
    fh = CACHE.open("a", encoding="utf-8")
    tokens = errors = 0

    async def one(h):
        nonlocal tokens, errors
        await lim.wait()
        try:
            r = await asyncio.wait_for(client.system_one(state(conn, h), qs, model="jev-latest"), 20)
        except Exception as e:  # noqa: BLE001 - count and move on; re-running asks what is missing
            errors += 1
            return
        a = r.answers
        rec = {"id": h["id"], "v": QV, "model": r.model, "tokens": r.usage.input_tokens or 0,
               "market_moving": a["market_moving"].noul,
               "size": a["nq_move_size"].score, "size_p": {str(k): v for k, v in a["nq_move_size"].probabilities.items()},
               "reaction": dict(a["nq_reaction"].probabilities)}
        tokens += rec["tokens"]
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
    for i in range(0, len(todo), 200):
        await asyncio.gather(*(one(h) for h in todo[i:i + 200]))
        print(f"  {min(i + 200, len(todo))}/{len(todo)} · ${tokens * 0.042 / 1e6:.3f} · errors {errors}", flush=True)
    await client.aclose()


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, rs = 0.0, 0, 0.0
    while i < len(allv):  # average ranks over ties
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2
        rs += r * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return (rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(conn, bars, hs, split) -> None:
    rows = bars.execute("SELECT t, close FROM mnq_bars ORDER BY t").fetchall()
    ts, px = [r[0] for r in rows], [r[1] for r in rows]

    def at(t: datetime):  # last full-minute close known at t; None if the market was closed (no bar within 3 min)
        i = bisect_left(ts, t.strftime("%Y-%m-%dT%H:%M:00")) - 1
        if i < 0 or (t - datetime.fromisoformat(ts[i])).total_seconds() > 240:
            return None
        return px[i]

    cache = load_cache()
    data = []
    for h in hs:
        a = cache.get(h["id"])
        if not a:
            continue
        t0 = datetime.fromisoformat(max(h["captured_at"], h["ts_et"])[:19])
        p0 = at(t0)
        mv = {m: at(t0 + timedelta(minutes=m)) for m in (5, 15, 30, 60)}
        if p0 is None or None in mv.values():
            continue
        data.append({**a, "day": h["ts_et"][:10], "title": h["title"], "mv": {m: (p - p0) / TICK for m, p in mv.items()},
                     "big_p": a["size_p"].get("2", 0) + a["size_p"].get("3", 0)})
    print(f"{len(data)} headlines with Jev answers and MNQ bars ({data[0]['day']}..{data[-1]['day']})")
    for m in (5, 15, 30, 60):
        cut = sorted(abs(d["mv"][m]) for d in data)[int(len(data) * 0.75)]
        pos = [d for d in data if abs(d["mv"][m]) > cut]
        neg = [d for d in data if abs(d["mv"][m]) <= cut]
        line = " · ".join(f"{k} {auc([d[k] for d in pos], [d[k] for d in neg]):.2f}" for k in ("market_moving", "size", "big_p"))
        print(f"  size, +{m:2d} min (top quartile = |move| > {cut:.0f} ticks): AUC {line}")

    def signed(d, m=30):
        r = d["reaction"]
        side = 1 if r.get("up", 0) > r.get("down", 0) else -1
        return d["mv"][m] * side

    def trades(rows, thr):
        sel = [d for d in rows if d["big_p"] >= thr and max(d["reaction"].get("up", 0), d["reaction"].get("down", 0)) >= 0.6]
        return [signed(d) - COST_TICKS for d in sel]

    def show(label, xs):
        if len(xs) < 3:
            return print(f"  {label}: {len(xs)} trades")
        n, mu = len(xs), sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) or 1
        print(f"  {label}: {n} trades, right {sum(x > 0 for x in xs) / n:.0%}, {mu:+.1f} ticks net each, t {mu / (sd / math.sqrt(n)):+.2f}")

    print("\nDirection after headlines Jev rates big (P(move >= 0.3%) >= threshold) with a clear side, 30-min hold:")
    for thr in (0.0, 0.3, 0.5, 0.7):
        show(f"all days, threshold {thr}", trades(data, thr))
    before, after = [d for d in data if d["day"] < split], [d for d in data if d["day"] >= split]
    if before and after:
        best = max((0.0, 0.3, 0.5, 0.7), key=lambda thr: sum(trades(before, thr)) if trades(before, thr) else -1e9)
        print(f"\nWalk-forward: threshold {best} picked on days before {split}, tested on {split} and after:")
        show("  before (tuning)", trades(before, best))
        show("  after (test)", trades(after, best))


def fade_side(a: dict) -> int | None:
    """+1 long / -1 short against Jev's side, or None if the headline doesn't qualify."""
    r = a["reaction"]
    up, down = r.get("up", 0), r.get("down", 0)
    if a["market_moving"] < FADE_MM or max(up, down) < FADE_SIDE:
        return None
    return -1 if up > down else 1


def summary(xs: list[float]) -> dict:
    """Net ticks per trade (after COST_TICKS) -> totals and the verdict rule: after FADE_N, worked if net > 0, t >= 2."""
    n = len(xs)
    mu = sum(xs) / n if n else 0.0
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    t = mu / (sd / math.sqrt(n)) if sd else (math.inf if mu > 0 and n > 1 else 0.0)
    verdict = "testing" if n < FADE_N else "worked" if sum(xs) > 0 and t >= 2 else "didn't work"
    return {"n": n, "net_ticks": round(sum(xs), 1), "net_usd": round(sum(xs) * TICK * 2, 2), "per_trade": round(mu, 1),
            "right": round(sum(x > 0 for x in xs) / n, 3) if n else None, "t": round(t, 2), "verdict": verdict}


def nightly() -> None:
    import numpy as np
    from trade_jev.data import LIVE_DIR, et_to_ns, list_days, load_day
    copy = ROOT / "data" / "news" / "bz-copy.sqlite3"
    copy.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{BZ_LIVE_DB.as_posix()}?mode=ro", uri=True)  # read-only on bz's live database
    dst = sqlite3.connect(copy)
    src.backup(dst)
    src.close()
    now = datetime.now().strftime("%Y-%m-%dT%H:%M")
    hs = headlines(dst, f"{FADE_START}T00:00", now, ["BZ Wire"])
    asyncio.run(ask(dst, hs))
    cache, days = load_cache(), list_days(LIVE_DIR)
    trades, skipped, loaded = [], {"outside 09:30-15:45": 0, "no recording": 0}, {}
    for h in hs:
        a = cache.get(h["id"])
        side = fade_side(a) if a else None
        if side is None:
            continue
        t0 = datetime.fromisoformat(max(h["captured_at"], h["ts_et"])[:19])
        d = t0.date().isoformat()
        if not ("09:30" <= t0.strftime("%H:%M") <= "15:45"):
            skipped["outside 09:30-15:45"] += 1
            continue
        if d not in days:
            skipped["no recording"] += 1
            continue
        if d not in loaded:
            loaded[d] = load_day(d, np.array([et_to_ns(d, "09:30:00")]), data_dir=LIVE_DIR)
        day = loaded[d]
        ns0 = et_to_ns(d, t0.strftime("%H:%M:%S"))
        ns1 = ns0 + FADE_HOLD_MIN * 60 * 10**9
        r0, r1 = int(day.row_at(ns0)), int(day.row_at(ns1))
        if r0 < 0 or ns0 - int(day.ts[r0]) > 60e9 or ns1 - int(day.ts[r1]) > 60e9:  # a gap in the recording
            skipped["no recording"] += 1
            continue
        move = (float(day.mid(r1)) - float(day.mid(r0))) / TICK
        trades.append({"day": d, "time": t0.strftime("%H:%M:%S"), "side": side, "title": h["title"][:120],
                       "market_moving": a["market_moving"], "reaction": a["reaction"],
                       "net_ticks": round(move * side - COST_TICKS, 2)})
    s = summary([t["net_ticks"] for t in trades])
    FADE_OUT.write_text(json.dumps({"rule": {"start": FADE_START, "market_moving_min": FADE_MM, "side_min": FADE_SIDE,
                                             "hold_min": FADE_HOLD_MIN, "cost_ticks": COST_TICKS, "verdict_after": FADE_N},
                                    "summary": s, "skipped": skipped, "trades": trades}, indent=1), encoding="utf-8")
    lines = ["# News fade test (pre-registered 2026-10-10)", "",
             f"Rule: on BZ Wire headlines from {FADE_START} that Jev rates market-moving >= {FADE_MM} with a clear side "
             f"(>= {FADE_SIDE}), trade AGAINST Jev's side at capture, hold {FADE_HOLD_MIN} min, mid to mid minus "
             f"{COST_TICKS} ticks. Verdict after {FADE_N} trades: worked if net > 0 and t >= 2.", "",
             f"**{s['verdict']}**: {s['n']}/{FADE_N} trades, net {s['net_ticks']:+} ticks (${s['net_usd']:+,.2f} on MNQ), "
             f"{s['per_trade']:+} per trade, right {s['right'] if s['right'] is not None else '-'}, t {s['t']:+}. "
             f"Skipped: {skipped}.", "",
             "| day | time | fade | net ticks | headline |", "|---|---|---|---|---|"]
    lines += [f"| {t['day']} | {t['time']} | {'long' if t['side'] > 0 else 'short'} | {t['net_ticks']:+} | {t['title']} |"
              for t in trades]
    FADE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"news fade: {s} / skipped {skipped} -> {FADE_REPORT}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ask", "eval", "nightly"))
    ap.add_argument("--bz-db")
    ap.add_argument("--bars-db")
    ap.add_argument("--since", default="2026-09-24T18:00")
    ap.add_argument("--until", default="2026-10-08T16:00")
    ap.add_argument("--sources", default="BZ Wire")
    ap.add_argument("--split", default="2026-10-03")
    a = ap.parse_args()
    if a.cmd == "nightly":
        return nightly()
    conn = sqlite3.connect(a.bz_db)
    hs = headlines(conn, a.since, a.until, a.sources.split(","))
    if a.cmd == "ask":
        asyncio.run(ask(conn, hs))
    else:
        evaluate(conn, sqlite3.connect(a.bars_db), hs, a.split)


if __name__ == "__main__":
    main()
