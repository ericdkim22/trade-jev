"""Automated variant research: a registry of Jev ideas, each judged only on days recorded after it was registered.

An idea is a Jev input (an encoder in `encode.ENCODERS`) plus filter / exit settings. Its fingerprint (`key`) is what
"already tried" means: a duplicate is refused with the earlier verdict, and a retired idea is never evaluated again.

  python -m trade_jev.research nightly      # what the scheduled task runs after the close
  python -m trade_jev.research add --encoder raw_l10 --stop-ticks 100 --target-ticks 50 --note "tighter exits"
  python -m trade_jev.research list

`nightly`:
 1. backtests every recorded day not yet backtested, once per encoder an open idea needs (Jev calls, cached)
 2. replays every open idea on every new day (no Jev calls) and stores the day's P&L in research/ledger.json
 3. verdicts: after MIN_OOS_DAYS days recorded after an idea was added, "worked" if it made money on those days and
    beat the live strategy on the same days, else "didn't work" (retired)
 4. new ideas: every encoder not yet tried (default settings), and the best untried grid setting on all days so far
    (in-sample; it is then judged out-of-sample like any other idea), at most one a night
 5. writes research/REPORT.md
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date
from pathlib import Path

from trade_jev import ROOT
from trade_jev.data import LIVE_DIR, list_days
from trade_jev.encode import ENCODERS
from trade_jev.replay import load_market, load_runs, replay_day
from trade_jev.settings import DEFAULT, Settings

DIR = ROOT / "research"
LEDGER = DIR / "ledger.json"
COMMISSION = 0.62        # MNQ, $ per side
MIN_OOS_DAYS = 10        # days after an idea was added before it gets a verdict
MAX_OPEN_GRID = 10       # grid ideas being tested at once (each new one waits for a slot)
MIN_GRID_DAYS = 5        # recorded days before the grid search proposes anything
BASELINE = "live"        # id of the live strategy's idea
GRID = {                 # as scripts/replay_grid.py
    "min_conf": [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "agree": [1, 2, 3, 4],
    "min_hold_s": [0, 30, 60, 120, 300],
    "stop_ticks": [0, 100, 200, 400],
    "target_ticks": [0, 100, 200],
}


# ---------------------------------------------------------------- registry

def key(encoder: str, s: Settings) -> str:
    return f"{encoder}|conf={float(s.min_conf):g}|agree={int(s.agree)}|hold={float(s.min_hold_s):g}" \
           f"|stop={int(s.stop_ticks)}|target={int(s.target_ticks)}" + \
           (f"|maxhold={float(s.max_hold_s):g}" if s.max_hold_s else "") + \
           (f"|breakeven={int(s.breakeven_ticks)}" if s.breakeven_ticks else "")


def settings_of(idea: dict) -> Settings:
    return Settings(**idea["settings"])


def load() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"ideas": []}


def save(ledger: dict) -> None:
    DIR.mkdir(exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1))
    tmp.replace(LEDGER)


def find(ledger: dict, k: str) -> dict | None:
    return next((i for i in ledger["ideas"] if i["key"] == k), None)


def add(ledger: dict, encoder: str, s: Settings, source: str, note: str, added: str, idea_id: str | None = None) -> dict:
    """Register an idea. Raises ValueError if the same idea was already tried."""
    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder {encoder!r}; have {sorted(ENCODERS)}")
    k = key(encoder, s)
    if (old := find(ledger, k)) is not None:
        raise ValueError(f"already tried as {old['id']!r} (added {old['added']}): {old['status']}")
    idea = {"id": idea_id or f"i{len(ledger['ideas']) + 1:03d}", "key": k, "encoder": encoder,
            "settings": asdict(s), "source": source, "note": note, "added": added,
            "status": "testing", "days": {}}
    ledger["ideas"].append(idea)
    return idea


def seed(ledger: dict) -> None:
    if not ledger["ideas"]:
        add(ledger, "raw_l10", DEFAULT, "live", "the live strategy: raw 10-level book, FINDINGS.md settings",
            "2026-10-08", BASELINE)["status"] = "live"
        add(ledger, "features", DEFAULT, "manual", "labeled, pre-computed features instead of the raw ladder",
            "2026-10-09")


# ---------------------------------------------------------------- verdicts

def oos_days(idea: dict) -> list[str]:
    return sorted(d for d in idea["days"] if d > idea["added"])


def judge(idea: dict, base: dict) -> None:
    """'worked' / "didn't work" once MIN_OOS_DAYS out-of-sample days exist (both ideas need the days)."""
    if idea["status"] != "testing":
        return
    days = [d for d in oos_days(idea) if d in base["days"]]
    if len(days) < MIN_OOS_DAYS:
        return
    net = sum(idea["days"][d]["pnl"] for d in days)
    base_net = sum(base["days"][d]["pnl"] for d in days)
    idea["status"] = "worked" if net > 0 and net > base_net else "didn't work"
    idea["verdict"] = {"on": str(date.today()), "days": len(days), "net": round(net, 2), "live_net": round(base_net, 2)}


# ---------------------------------------------------------------- nightly steps

def backtest(encoder: str, day: str) -> Path | None:
    """Jev answers for one encoder on one recorded day (default settings), via trade_jev.run. Cached Jev calls are free."""
    out = ROOT / "runs" / f"research-{encoder}-{day}"
    if (out / "results.json").exists():
        return out
    cmd = [sys.executable, "-m", "trade_jev.run", "--days", day, "--policies", "jev", "--encoder", encoder,
           "--commission", str(COMMISSION), "--run-id", out.name]
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True, text=True)
    if r.returncode or not (out / "results.json").exists():
        print(f"  backtest {encoder} {day} failed:\n{r.stdout[-800:]}{r.stderr[-800:]}", flush=True)
        return None
    print(f"  backtest {encoder} {day}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'ok'}", flush=True)
    return out


def recorded_days() -> list[str]:
    """Days the live sessions recorded, minus a day still being recorded (its parts aren't merged yet): a day is
    evaluated once, so it must be complete."""
    from trade_jev.live import parts_dir
    return sorted(d for d, (_, path) in list_days(LIVE_DIR).items() if not parts_dir(path).exists())


def evaluate(ledger: dict, days: list[str]) -> None:
    """Replay every open idea on every day it hasn't seen yet."""
    open_ideas = [i for i in ledger["ideas"] if i["status"] in ("live", "testing", "worked")]
    for day in days:
        todo = [i for i in open_ideas if day not in i["days"]]
        if not todo:
            continue
        runs = {e: backtest(e, day) for e in sorted({i["encoder"] for i in todo})}
        market = None
        for idea in todo:
            if runs[idea["encoder"]] is None:
                continue
            rd = load_runs([runs[idea["encoder"]]])[day]
            if market is None:
                market = load_market(rd)
            r = asyncio.run(replay_day(rd, *market, settings_of(idea)))
            idea["days"][day] = {"pnl": round(r.pnl, 2), "trades": len(r.trades)}
        save(ledger)


def _grid_day(day: str) -> list[float]:
    rd = load_runs([ROOT / "runs" / f"research-raw_l10-{day}"])[day]
    market = load_market(rd)

    async def go():
        return [round((await replay_day(rd, *market, Settings(*c))).pnl, 2) for c in itertools.product(*GRID.values())]
    return asyncio.run(go())


def propose(ledger: dict, days: list[str], today: str) -> None:
    for enc in sorted(ENCODERS):  # new Jev inputs added to encode.py
        if not any(i["encoder"] == enc for i in ledger["ideas"]):
            add(ledger, enc, DEFAULT, "new encoder", f"encoder {enc!r} found in encode.py", today)
            print(f"  new idea: encoder {enc}", flush=True)
    base_days = [d for d in days if d in find(ledger, key("raw_l10", DEFAULT))["days"]]
    open_grid = sum(1 for i in ledger["ideas"] if i["source"] == "grid" and i["status"] == "testing")
    if len(base_days) < MIN_GRID_DAYS or open_grid >= MAX_OPEN_GRID:
        return
    with ProcessPoolExecutor(max_workers=min(4, len(base_days))) as ex:
        per_day = list(ex.map(_grid_day, base_days))
    combos = list(itertools.product(*GRID.values()))
    ranked = sorted(range(len(combos)), key=lambda i: (sum(p[i] > 0 for p in per_day), sum(p[i] for p in per_day)),
                    reverse=True)
    for i in ranked:
        s = Settings(*combos[i])
        if find(ledger, key("raw_l10", s)) is None:
            net, wins = sum(p[i] for p in per_day), sum(p[i] > 0 for p in per_day)
            add(ledger, "raw_l10", s, "grid",
                f"best untried grid setting in-sample: {wins}/{len(per_day)} winning days, net ${net:,.0f}", today)
            print(f"  new idea: grid {s.label()} (in-sample {wins}/{len(per_day)} days, ${net:,.0f})", flush=True)
            return


def report(ledger: dict) -> str:
    base = find(ledger, key("raw_l10", DEFAULT))

    def row(i: dict) -> str:
        oos = [d for d in oos_days(i) if d in base["days"]]
        net = sum(i["days"][d]["pnl"] for d in oos)
        live = sum(base["days"][d]["pnl"] for d in oos)
        all_net = sum(v["pnl"] for v in i["days"].values())
        trades = sum(v["trades"] for v in i["days"].values())
        return (f"| {i['id']} | {i['encoder']} | {settings_of(i).label()} | {i['source']} | {i['added']} | "
                f"{len(oos)}/{MIN_OOS_DAYS} | ${net:,.2f} | ${live:,.2f} | ${all_net:,.2f} ({trades}) | {i['note']} |")

    head = ("| id | input | settings | source | added | days after added | net on them | live on them "
            "| net all days (trades) | note |\n|---|---|---|---|---|---|---|---|---|---|")
    groups = [("Worked: candidates to put live", "worked"), ("Testing", "testing"),
              ("Didn't work (retired, never re-run)", "didn't work")]
    days = sorted(base["days"])
    out = [f"# Jev variant research\n\nUpdated {date.today()} · {len(days)} recorded days "
           f"({days[0] if days else '-'} to {days[-1] if days else '-'}) · verdict after {MIN_OOS_DAYS} days recorded "
           f"after an idea was added: worked = made money and beat the live strategy on those days.\n",
           f"## Live strategy\n\n{head}\n{row(base)}\n"]
    for title, status in groups:
        ideas = [i for i in ledger["ideas"] if i["status"] == status]
        out.append(f"## {title} ({len(ideas)})\n\n" + (f"{head}\n" + "\n".join(map(row, ideas)) if ideas else "None yet.") + "\n")
    return "\n".join(out)


def nightly() -> None:
    ledger = load()
    seed(ledger)
    days = recorded_days()
    print(f"research: {len(days)} recorded days, {len(ledger['ideas'])} ideas", flush=True)
    evaluate(ledger, days)
    base = find(ledger, key("raw_l10", DEFAULT))
    for i in ledger["ideas"]:
        judge(i, base)
    propose(ledger, days, str(date.today()))
    evaluate(ledger, days)  # new ideas get their in-sample history too (it doesn't count for the verdict)
    save(ledger)
    (DIR / "REPORT.md").write_text(report(ledger), encoding="utf-8")
    print(f"research: report → {DIR / 'REPORT.md'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Automated Jev variant research")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("nightly")
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("--encoder", default="raw_l10")
    for f, v in asdict(DEFAULT).items():
        a.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    a.add_argument("--note", required=True)
    args = ap.parse_args()
    if args.cmd == "nightly":
        return nightly()
    ledger = load()
    seed(ledger)
    if args.cmd == "add":
        s = Settings(args.min_conf, args.agree, args.min_hold_s, args.stop_ticks, args.target_ticks,
                     args.max_hold_s, args.breakeven_ticks)
        try:
            idea = add(ledger, args.encoder, s, "manual", args.note, str(date.today()))
        except ValueError as e:
            sys.exit(f"not added: {e}")
        save(ledger)
        print(f"added {idea['id']}: {idea['key']} (judged on days after {idea['added']})")
    else:
        for i in ledger["ideas"]:
            print(f"{i['id']:6} {i['status']:12} {i['key']:55} {i['note']}")


if __name__ == "__main__":
    main()
