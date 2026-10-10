# Research log

Every experiment, with its verdict, so nothing is re-run. Ideas that fit the registry (an input + settings) also live in
`research/ledger.json` / `REPORT.md`; this log also covers experiments the registry can't express.

## 2026-10-10: MarketTick history sample (57 days, 2020-06..2025-05, MNQ, $0.62/side)

Backtests `runs/hist-<input>`, Jev answers ~$13.75 in total. Verdict per the checks set before the results.

| Experiment | Net | $/day | t | Win days | Verdict |
|---|---|---|---|---|---|
| live strategy (raw book, 0.7 / 4 / 30 s / 200 / 100) | -$2,248 | -39.4 | -1.18 | 24/57 | didn't work: positive in 1 of 6 years |
| features input | -$3,737 | -65.6 | -1.58 | 25/57 | didn't work |
| topbook input | -$5,118 | -89.8 | -2.34 | 22/57 | didn't work |
| context input | -$3,936 | -69.0 | -1.91 | 23/57 | didn't work |
| live + 5-min time stop | -$2,714 | -47.6 | -2.17 | 23/57 | didn't work |
| live + break-even after +50 ticks | -$2,525 | -44.3 | -1.61 | 24/57 | didn't work |
| features + 5-min time stop | -$4,800 | -84.2 | -2.34 | 19/57 | didn't work |
| random (costs only) | -$19,935 | -349.7 | -8.07 | 8/57 | benchmark |
| imbalance rule | +$262 | 4.6 | 0.19 | 27/57 | benchmark |

Side variants on the stored answers (`scripts/replay_sides.py`, no Jev calls):

| Input | as is | long only | short only | shorts need 0.85 | flipped |
|---|---|---|---|---|---|
| raw book | -$2,248 | -$455 | -$1,849 | -$324 | -$1,920 |
| context | -$3,936 | +$194 (t 0.18) | -$2,988 | -$1,276 | +$12 |

Verdict: didn't work. Doing the opposite of Jev loses about as much as following it, so the answers carry no direction
signal at this horizon; restricting sides only cuts the number of trades.

Hindsight grid on the raw-book history answers (`scripts/replay_grid.py runs/hist-raw_l10`,
`runs/replays/grid-20261010-084454.csv`): 1,920 filter / exit settings, picked after seeing the 57 days.

| | Net | t |
|---|---|---|
| best setting (0.7 / 3 agreeing / 300 s hold / 400 stop / 200 target) | +$1,721 | 0.76 |
| settings with a positive net | 52 of 1,920 | |
| settings with t >= 2 | 0 | |
| median setting | -$7,617 | |
| live setting | -$2,248 (rank 336) | -1.18 |

Verdict: didn't work. Even the best of 1,920 settings chosen in hindsight is indistinguishable from luck (the best of
1,920 draws from a strategy with no edge looks like this), so Jev's answers to the current question carry no tradeable
signal here and tuning filters / exits is pointless. Not registered as an idea for that reason.

## 2026-10-10: question variants on the history sample (57 days)

New Jev questions (`policies.QUESTIONS`), ~$7 in Jev answers (`runs/hist-raw_l10-target`, `runs/hist-raw_l10-next15`).

| Experiment | Net | $/day | t | Win days | vs live strategy, day by day | Verdict |
|---|---|---|---|---|---|---|
| i008 `target`: Jev told the +25 / -50 point exits, asked which side reaches its target first | -$3,580 | -62.8 | -1.84 | 25/57 | -$23/day (t -0.87) | didn't work |
| i009 `next15`: 15-minute direction, 15-min time stop, +/-25 points | -$3,984 | -69.9 | -2.05 | 23/57 | -$31/day (t -1.06) | didn't work |

Side variants (no Jev calls): target flipped -$700, long only -$792; next15 flipped -$3,964, long only -$954. No variant
positive. Both questions keep the short bias (next15: 1,390 shorts vs 439 longs over 2020-2025's rising market).
On one test moment all three questions got nearly the same answer (SELL 81% / 75% / 81%): the wording barely moves Jev.

Overall verdict after inputs (4), exits (2), questions (2), side filters and the hindsight grid: Jev's answers about
MNQ's order book carry no tradeable direction signal on 2020-2025. Retry only with a different kind of input (text /
news, where Jev is strong), not another variation of the order-book state.

## 2026-10-10: Jev on headlines (bz-premarket news) vs MNQ

Data: copies of bz's `bz.sqlite3` (headlines, Jev calls) and `news_entry_bars.sqlite3` (MNQZ6 1-min bars,
2026-09-24..10-08). Entry at the minute a headline was captured (the earliest one could act). ~1 min bars, no fills.

1. bz's existing Jev calls on SPY/QQQ-tagged stories (275 with bars, 09-29..10-08): direction right 46-51%
   (t -0.1..-1.5 against); materiality medium+ moved MNQ more at +5 min (AUC 0.62, median 96 vs 54 ticks), fading by
   60 min. Following MNQ's first-minute reaction after medium+ stories: <= +12 ticks before ~5 ticks costs, t <= 0.4,
   no better than low-materiality stories. Verdict: didn't work.
2. `scripts/news_nq.py` (QV 1): one Jev call per BZ Wire headline, three questions about Nasdaq-100 futures
   (market_moving, nq_move_size, nq_reaction). 6,577 asked ($0.22), 5,966 with bars (09-25..10-08).
   - Size: AUC 0.48-0.54 for a top-quartile move at +5/15/30/60 min. Verdict: didn't work.
   - Direction, big + clear side, 30-min hold: 150 trades right 48%, -25 ticks net each (t -1.27); stricter cut-offs
     do worse (t -1.9). Verdict: didn't work.
   - Cut found while looking (not pre-registered, several cuts tried): headlines with market_moving >= 0.9 and a clear
     side (48): Jev's side right 48% / 33% / 40% at +5 / 15 / 30 min, t -0.6 / -2.6 / -2.4. MNQ went the other way.
     Status: hypothesis only. Pre-registered test from 2026-10-12: FADE Jev's side on headlines with
     market_moving >= 0.9 and max(up, down) >= 0.6, entered at capture, 15-min hold, net of 5 ticks; verdict after
     50 such headlines with MNQ prices: worked if net > 0 and t >= 2, else didn't work.
