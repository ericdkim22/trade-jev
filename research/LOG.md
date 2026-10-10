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
