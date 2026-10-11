# MNQ machine-learning alpha

`python -m trade_jev.alpha` turns MNQ candle, L1 and L2 text files into a short-horizon
price-direction signal. It predicts the mid-price move over the next *H* seconds (in
ticks), checks the prediction day by day out of sample, and backtests it with real
spread and commission.

Code: [`src/trade_jev/alpha/`](src/trade_jev/alpha/). Tests: [`tests/test_alpha.py`](tests/test_alpha.py).

## Quick start (Windows, data in `C:\Transfer`)

```powershell
uv sync --extra alpha
uv run python -m trade_jev.alpha scan  --data C:\Transfer                 # what was recognised
uv run python -m trade_jev.alpha build --data C:\Transfer                 # → runs\alpha\features.parquet
uv run python -m trade_jev.alpha train --horizon 60                       # → runs\alpha\<ts>\report.md + model\
uv run python -m trade_jev.alpha train --horizon 30  --session rth        # other horizons / RTH only
uv run python -m trade_jev.alpha predict --model runs\alpha\<ts>\model --data D:\new_day --out signals.csv
```

**Dashboard:** `uv run python -m trade_jev.view` → **Alpha** tab at http://localhost:8765/alpha. Pick a run, a threshold and a day to see:
- headline stats
- cumulative P&L
- rank IC per day
- the decile chart
- the threshold table
- the day's price with every trade and the prediction against its cut
- feature importance

Every run under `runs/alpha/` appears there. Links keep the view: `?run=<id>&q=0.9&day=<YYYY-MM-DD>`.

No data handy? Write synthetic files in the same formats:
`uv run python scripts/make_mnq_synthetic.py --out data/mnq_synth --days 7` (`--edge 0` makes data with nothing to find).

## Input files

Files can have any name, extension (`.txt .csv .dat .tsv .gz`), separator (`; , tab |`) and
an optional header row. Each file is classified by its first row:

| Kind | Fields | Timestamp |
|---|---|---|
| Candles | ts, type, open, high, low, close, volume | 14 digits `YYYYMMDDhhmmss` |
| L1 ticks | ts, level, type, price, volume | 20 digits (+ microseconds) |
| L2 ticks | ts, level, type, price, volume, depth, action | 20 digits |

Type: 0 bid, 1 ask, 2 last trade (3–9 daily stats are ignored). L2 action: 0 add, 1 update, 2 remove.
L1 is required; L2 and candles are optional and add their features when present.

## How it works

1. **Book.** L2 rows rebuild a 10-level book. Feeds use `depth` in one of two ways: *insert*
   (Add pushes rows down, Remove pulls them up) or *slot* (each row is overwritten in place).
   `--l2-mode auto` (default) tries both on the first events and keeps the one whose book is
   less often crossed or out of order. `build` prints `l2_bad_share` per chunk. It should be
   near 0. If it isn't, try the other mode and tell me what you see.
2. **Grid.** One row every `--step` seconds (default 1s) while quotes are live. A quote gap over
   30s, such as the daily CME halt, starts a new segment. Lookbacks and labels never cross a segment.
3. **Features** at time *t* use only data up to *t* (a test enforces this):
   - L1: spread, top-of-book imbalance, microprice offset; per 5/15/60/300s window: return,
     order-flow imbalance (Cont–Kukanov–Stoikov), signed trade flow, volume, trade and quote
     counts, realised vol, VWAP gap, change in imbalance.
   - L2: depth imbalance over 1/3/5/10 levels and distance-weighted, depth, liquidity-weighted
     skew, price impact to fill 10/30 lots per side, and changes over 5/15/60s.
   - Candles (Last bars; mid of bid/ask bars if there are no Last bars): returns over 1/5/15/60
     bars in ATR units, ATR, position in the 60-bar range, volume z-score, EMA gaps. A bar is
     treated as known only once it has closed (timestamp + bar length).
   - Time of day (ET) and weekday.
4. **Labels.** `fwd_{H}s_t` is the mid move from *t* to *t+H* in ticks, for H = 30/60/300s.
5. **Model.** LightGBM regression, walk-forward by CME trading day. Each test day is
   predicted by a model trained only on earlier days. The last training day is held out
   for early stopping and to set trade thresholds. Training labels that overlap the
   next period are purged.
6. **Backtest.** One MNQ contract, no overlapping positions. Trade when |prediction| ≥
   threshold (the q-quantile of |prediction| on the held-out day, for q = 0.5…0.98).
   Each trade buys at the ask or sells at the bid 250ms after the signal and exits by
   crossing back H seconds later. Costs: $0.50/tick and $0.62 commission per side
   (`--commission`).

## Reading `report.md`

- **rank IC**: rank correlation between prediction and the realised move, out of sample.
  At these horizons 0.02–0.05 that holds day after day (`rank_ic_t_stat` > 2, most days
  positive) is a real but small edge. Much higher usually means a bug or bad data.
- **Deciles**: realised move per prediction decile. It should rise steadily from 1 to 10.
- **Backtest**: `avg_ticks_gross` must clear about 1 tick of spread plus about 2.5 ticks of
  commission ($1.24 round turn) to make money. That's a high bar for 30–60s MNQ horizons,
  so check the 300s horizon and the higher q thresholds too.
- **Baseline**: rank IC of the best single features. If the model doesn't beat `l1_imb`
  or `ofi_60s` alone, it isn't adding anything.

Pipeline check on synthetic data: with a planted signal, out-of-sample rank IC is about 0.55
and the model finds the planted feature (L2 depth imbalance). With `--edge 0` (no signal),
rank IC is −0.004 (t = 0.02) and every threshold loses about the spread plus commission,
so the evaluation does not leak future data.

## Outputs (`runs/alpha/`, gitignored)

| File | Contents |
|---|---|
| `features.parquet` (+ `.json`) | grid features, labels, entry/exit prices; build settings and per-chunk stats |
| `<run>/report.md`, `summary.json` | the out-of-sample report |
| `<run>/predictions.parquet` | every out-of-sample prediction with its label |
| `<run>/folds.csv`, `backtest.csv`, `trades_q*.csv`, `importance.csv` | per-day stats, backtest table, trades, feature importance |
| `<run>/model/` | final model trained on all days (`model.txt`, `model.json` with features, settings and thresholds) for `predict` |

## Known limits

- The L2 book is rebuilt from the file's own events, so the first minutes of a file can be
  thin until the book fills. Use files that start with a snapshot, or let the warm-up drop them.
- Candle timestamps are assumed to be the bar *open*. If they are the bar close, features
  lag one extra bar. That is safe, just slightly stale.
- Fills assume your one contract trades at the quoted touch. There is no queue position or
  market impact.
- With fewer than about 10 days the per-day statistics are noisy. More days beats more features.
