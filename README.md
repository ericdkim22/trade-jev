# trade-jev

Tests Jev (TypeSafe) as a BUY / SELL / HOLD trader on NQ L10 order-book data from Databento (15 trading days, Jun 8–26 2026). **Findings: [`results/FINDINGS.md`](results/FINDINGS.md).** Published Jev answers and results are in [`results/`](results/); where the market data goes and how it was pulled is in [`data/DATA.md`](data/DATA.md).

**Concepts**
- **Run:** a backtest that calls Jev and stores its answers and trades in `runs/<id>/`. It's the only step that costs money.
- **Settings:** harness parameters: cutoff, agreeing answers, min hold, stop, target, commission.
- **Replay:** a run's stored answers re-scored under new settings. It needs no API calls.
- **View:** one local web app that replays any mix of runs, days and settings live.
- **Live:** the same harness on a live Databento feed: Jev signals, paper trades and a live page. It places no orders.

Flow: **run once → replay many times → view**. Then **live** for forward testing on new days.

## 0) Setup

Install dependencies, add your API key, and point at the data. Databento day files go in `data/databento/` (gitignored), or anywhere set by `TRADE_JEV_DATA`; see [`data/DATA.md`](data/DATA.md). No data yet? Use the synthetic sample.

```bash
uv sync
echo "TYPESAFE_API_KEY=..." >> .env
echo "TRADE_JEV_DATA=/path/to/databento/files" >> .env        # optional; default data/databento/
TRADE_JEV_DATA=data/sample uv run python -m trade_jev.run --days 2026-06-23 --policies hold,random,imbalance   # smoke test
```

**In git:** code, docs, `data/sample/` (synthetic), `results/` (published runs), `runs/index.jsonl`. **Local only:** Databento data (incl. `data/live/` recordings), `runs/<id>/` (includes the order books Jev saw), `cache/`, `.env`.

## 1) Run

Asks Jev every 15s and trades 1 contract next to the hold / random / imbalance baselines. It prints Jev calls and P&L, saves to `runs/<id>/`, and appends to `runs/index.jsonl`, which the viewer lists.

```bash
uv run python -m trade_jev.run --days 2026-06-23                                   # one day
uv run python -m trade_jev.run --days 2026-06-08,2026-06-09,2026-06-10             # several days
uv run python -m trade_jev.run --all                                               # all 15 days
uv run python -m trade_jev.run --days 2026-06-23 --policies hold,random,imbalance  # baselines only, no API
```

## 2) Run settings

Defaults are the registered setting from [`results/FINDINGS.md`](results/FINDINGS.md): cutoff 0.7, 4 agreeing answers, 30s min hold, 200-tick stop, 100-tick target (`0` = off; defined in `src/trade_jev/settings.py`). Settings change which positions Jev sees, so a new run calls Jev again.

```bash
uv run python -m trade_jev.run --days 2026-06-23 \
  --min-conf 0.6 --agree 2 --min-hold 60 --stop-ticks 400 --target-ticks 0 \
  --cadence-s 15 --latency-ms 250 --commission 2.5
```

## 3) Replay

Replays one or more runs under many settings at once to find settings that hold up across days. It's free but approximate, because Jev answered with the run's own positions.

```bash
uv run python scripts/replay_sweep.py runs/<id> [runs/<id> ...]         # quick per-day table
uv run python scripts/replay_grid.py runs/<id> [runs/<id> ...]          # 1,920 settings → runs/replays/grid-<ts>.csv
uv run python scripts/analyze_replays.py runs/replays/grid-<ts>.csv     # rank + overfitting checks
uv run python scripts/findings.py --tune runs/<a> --test runs/<b> --grid runs/replays/grid-<ts>.csv  # tune vs test report
```

## 4) View

A local web app that lists every run in `runs/index.jsonl`. Pick any mix of runs and days, replay them live with the sliders, save presets to compare, press play, and click any point to see what Jev saw.

```bash
uv run python -m trade_jev.view                  # → http://localhost:8765
uv run python -m trade_jev.view --port 9000 --no-open
```

Links keep the view: `?sel=<run>@<day>,<run>@<day>&minConf=0.6&agree=2&minHold=60&stop=400&target=0` plus `&play=1` (autoplay) or `&at=11:02` (jump to a time).

## 5) Publish

Copies a run's Jev answers, probabilities, trades and results to `results/<id>/` for sharing. Order books and prices are stripped, since they're licensed Databento data.

```bash
uv run python scripts/publish_run.py runs/<id> [runs/<id> ...]
```

## 6) Live

Streams NQ L10 from Databento, asks Jev on the backtest's clock (09:30:00 ET + every 15s), applies the registered filters, and keeps a paper position with the backtest's fills and costs. Hold, random and imbalance baselines run alongside. One local server at `http://localhost:8765` serves **Replay**, **Live** and **Docs**. The Docs tab ([`viz/docs.html`](viz/docs.html)) covers use cases, costs and next steps.

**IBKR feed (default in this fork).** Needs IB Gateway on 127.0.0.1:4002 and the CME Real-Time (NP,L2)
subscription. Read-only, client ID 20; it places no orders. Defaults to the MNQ front month ($2 / point); `--root NQ`
for NQ. Pass `--commission` for your contract (the default 2.50 is NQ's). IBKR has no intraday replay of depth, so a
late start skips decisions until it has 60s of book. IBKR trades carry no aggressor side: a trade at or above the
previous ask counts as a buy, at or below the bid as a sell.

```bash
uv run python -m trade_jev.live --commission 0.62                  # IBKR, MNQ front month → /live
uv run python -m trade_jev.live --root NQ                          # IBKR, NQ
```

Every weekday it runs by itself: the Windows scheduled task `trade-jev-live` starts `scripts/live_daily.cmd` at
09:15 ET (log in `logs/live.log`), retries 3 times 5 minutes apart if the Gateway isn't up, and is stopped after
14 hours. Each session writes `runs/live-<day>/` and records the feed (10-level book + trades) to `data/live/`, in 1-minute part files merged into the day file at the close; a crash loses at most the last minute, and parts left by a crash are merged when the next session starts. The dashboard is at
http://localhost:8765/live while it runs. On a market holiday it waits for data and exits at 16:00 ET.

**Jev variants** (`--variants features`, on in the daily run): extra Jev traders that see the same moment through a different input encoder, with the same filters, exits and fills, each with its own paper position and its own row in *Jev vs. benchmarks*. `features` gives Jev pre-computed, labeled numbers (book imbalance at 1 / 5 / 10 levels, price change in ticks, net aggressor volume) and one-line summaries instead of the raw ladder, since Jev reads words better than it does arithmetic. Its Jev calls run in parallel with the main one.

**Databento feed:**

```bash
echo "DATABENTO_API_KEY=..." >> .env
uv run python -m trade_jev.live --feed databento --root NQ         # start before 09:30 ET → /live; Ctrl-C to stop
uv run python -m trade_jev.live --no-jev                           # feed + baselines only
uv run python -m trade_jev.live --from-file 2026-06-23 --speed 60 --stored-answers runs/<id>   # rehearsal, no subscription
uv run python scripts/check_live_parity.py runs/<id> [runs/<id> ...]   # live engine == backtest, trade for trade
```

A session writes `runs/live-<day>/`, the same layout as a run plus `manual_fills.jsonl`, and records the feed to `data/live/` in the day-file format. It then opens in Replay and works with the replay scripts. The only built-in difference from the backtest is that the order goes out when Jev answers, not at the decision time; each decision records `jev_ms` and `send_ns`.

## 6b) Backtest and tune on your own MNQ recordings

Every live session records its day to `data/live/`, and `trade_jev.run` reads those files like the Databento ones,
so each recorded day can be backtested and replayed. Always pass `--commission 0.62` for MNQ (the default 2.50 is NQ's).

```bash
# 1. Backtest recorded days: Jev + benchmarks. Answers Jev already gave (same state) come from cache/ for free.
uv run python -m trade_jev.run --all --commission 0.62 --run-id bt-raw
# 2. Tune filters and exits for free (no Jev calls): per-day table, then the 1,920-setting grid and its overfitting checks
uv run python scripts/replay_sweep.py runs/bt-raw
uv run python scripts/replay_grid.py runs/bt-raw
uv run python scripts/analyze_replays.py runs/replays/grid-<ts>.csv
# 3. Try a new input for Jev (a new encoder in encode.py): costs Jev calls, about $0.04 per recorded day
uv run python -m trade_jev.run --all --commission 0.62 --policies jev --encoder features --run-id bt-features
# 4. Look: Replay tab (http://localhost:8765/ while a live session runs, otherwise `python -m trade_jev.view`)
```

Before trusting a setting found this way, split the days: tune on the older ones, check on the newer
(`scripts/findings.py --tune runs/<a> --test runs/<b> --grid ...`). The author's +$20.8k was picked on the days it was
scored on; a result only counts once it holds on days it wasn't tuned on.

## 6b-2) Five years of MarketTick history

`python -m trade_jev.markettick` turns MarketTick MNQ CSV days (`C:\Transfer\YYYYMMDD.csv`, 2020-06 to 2025-05) into
day files in `data/history/` (about 90 s and 40 MB a day). Level 1 of each book is the exact Level 1 quote; levels 2-10
are rebuilt from MarketTick's Level 2 rows by price and are approximate (99.1% within one level of their stated depth);
trades are signed against the quote. Holiday sessions (e.g. MLK Day, Presidents' Day) are converted but left out of
backtests.

```bash
uv run python -m trade_jev.markettick sample C:\Transfer --days 60       # weekdays spread evenly over the 5 years
TRADE_JEV_DATA=data/history uv run python -m trade_jev.run --days <list> --encoder features --commission 0.62
```

The 59-day sample's day list is in `research/history-sample-days.txt`.

## 6c) Automated variant research

`python -m trade_jev.research nightly` runs every weekday at 16:30 ET (scheduled task `trade-jev-research`, log
`logs/research.log`) and keeps `research/ledger.json` (every idea ever tried, with per-day P&L) and `research/REPORT.md`.

- An idea is a Jev input (an encoder in `encode.py`) plus filter / exit settings. Its fingerprint decides "already
  tried": `research add` refuses a duplicate and prints the earlier verdict; retired ideas are never re-run.
- Each idea is judged only on days recorded **after** it was added: after 10 such days it "worked" if it made money and
  beat the live strategy on them, else it is retired as "didn't work".
- New ideas come from the free grid search (the best untried setting on all days so far, one a night, at most 10 being
  tested) and from any new encoder added to `encode.py`. To add your own:
  `uv run python -m trade_jev.research add --encoder raw_l10 --stop-ticks 100 --target-ticks 50 --note "tighter exits"`.
- Promoting a "worked" idea to the live session is left to you.

## 7) Test

Unit tests, plus a check that the page's replays match the Python replays.

```bash
uv run pytest -q
uv run python scripts/check_viewer.py runs/<id>
uv run python scripts/check_live_parity.py runs/<id>
uv run python scripts/make_sample.py        # regenerate the synthetic sample
```
