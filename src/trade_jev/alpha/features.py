"""Turn L1 / L2 / candle data into a feature table on a regular time grid.

Every feature at grid time t uses only data with ts <= t (candles: only bars that
have closed by t). Labels look forward from t and are kept in `fwd_*` / `px_*`
columns so they can never be picked up as features.

Grid: every `step_s` seconds inside "segments" of continuous quoting (a gap longer
than `max_gap_s` starts a new segment, e.g. the daily CME halt). Lookbacks and
labels never reach across a segment boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trade_jev.alpha import book
from trade_jev.alpha.io import ASK, BID, LAST

TICK = 0.25
NS = 1_000_000_000
ET = "America/New_York"


@dataclass(frozen=True)
class Spec:
    step_s: float = 1.0                      # grid spacing
    windows_s: tuple[int, ...] = (5, 15, 60, 300)
    horizons_s: tuple[int, ...] = (30, 60, 300)  # labels built for each
    latency_ms: int = 250                    # entry delay for the backtest prices
    max_gap_s: float = 30.0                  # quote gap that splits segments
    max_stale_s: float = 5.0                 # drop grid rows whose last quote is older
    l2_mode: str = "auto"                    # insert | slot | auto (see book.py)
    session: str = "all"                     # all | rth (09:30–16:00 ET)
    candle_lookbacks: tuple[int, ...] = (1, 5, 15, 60)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- L1 streams

def quotes(l1: pd.DataFrame) -> pd.DataFrame:
    """Top of book after every L1 bid/ask row: ts, bid, bid_sz, ask, ask_sz, ofi."""
    q = l1[l1.type.isin([BID, ASK])]
    is_bid = (q.type == BID).to_numpy()
    out = pd.DataFrame({
        "ts": q.ts.to_numpy(),
        "bid": np.where(is_bid, q.price, np.nan), "bid_sz": np.where(is_bid, q.volume, np.nan),
        "ask": np.where(~is_bid, q.price, np.nan), "ask_sz": np.where(~is_bid, q.volume, np.nan),
    }).ffill().dropna()
    out = out[(out.ask > out.bid) & (out.bid_sz > 0) & (out.ask_sz > 0)].reset_index(drop=True)
    # Order-flow imbalance (Cont, Kukanov & Stoikov 2014), one value per quote change
    b, bs, a, as_ = (out[c].to_numpy() for c in ("bid", "bid_sz", "ask", "ask_sz"))
    pb, pbs, pa, pas = (np.r_[x[:1], x[:-1]] for x in (b, bs, a, as_))
    e = (np.where(b >= pb, bs, 0) - np.where(b <= pb, pbs, 0)
         - np.where(a <= pa, as_, 0) + np.where(a >= pa, pas, 0))
    out["ofi"] = e
    return out


def trades(l1: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Trades signed against the quote just before them (quote rule, tick rule for mid prints)."""
    t = l1[(l1.type == LAST) & (l1.volume > 0)][["ts", "price", "volume"]].reset_index(drop=True)
    if t.empty:
        return t.assign(sign=np.array([], dtype=float))
    j = np.searchsorted(q.ts.to_numpy(), t.ts.to_numpy(), side="left") - 1  # strictly before
    ok = j >= 0
    bid = np.where(ok, q.bid.to_numpy()[np.maximum(j, 0)], np.nan)
    ask = np.where(ok, q.ask.to_numpy()[np.maximum(j, 0)], np.nan)
    px = t.price.to_numpy()
    sign = np.where(px >= ask, 1.0, np.where(px <= bid, -1.0, np.nan))
    tick = np.sign(np.diff(px, prepend=px[0]))
    tick = pd.Series(np.where(tick == 0, np.nan, tick)).ffill().fillna(0).to_numpy()
    t["sign"] = np.where(np.isnan(sign), tick, sign)
    return t


# ---------------------------------------------------------------- grid helpers

def make_grid(q: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    ts = q.ts.to_numpy()
    step = int(spec.step_s * NS)
    gap = np.diff(ts) > spec.max_gap_s * NS
    starts = np.r_[0, np.flatnonzero(gap) + 1]
    ends = np.r_[np.flatnonzero(gap), len(ts) - 1]
    parts, segs = [], []
    for k, (s, e) in enumerate(zip(starts, ends)):
        g = np.arange(-(-ts[s] // step) * step, ts[e] + 1, step, dtype=np.int64)
        parts.append(g)
        segs.append(np.full(len(g), k, dtype=np.int64))
    grid = pd.DataFrame({"ts": np.concatenate(parts) if parts else np.array([], np.int64),
                         "seg": np.concatenate(segs) if segs else np.array([], np.int64)})
    return grid


def _asof(src_ts: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Index of the last src row with ts <= at (-1 if none)."""
    return np.searchsorted(src_ts, at, side="right") - 1


def _cum_at(src_ts: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Cumulative sum of `values` over rows with ts <= at."""
    c = np.r_[0.0, np.cumsum(values)]
    return c[_asof(src_ts, at) + 1]


def _lag(x: np.ndarray, k: int, seg: np.ndarray) -> np.ndarray:
    """x shifted k rows back (forward if k < 0); NaN when that crosses a segment."""
    out = np.full(len(x), np.nan)
    if k == 0:
        return x.astype(float)
    if k > 0:
        out[k:] = x[:-k]
        same = np.r_[np.zeros(k, bool), seg[k:] == seg[:-k]]
    else:
        out[:k] = x[-k:]
        same = np.r_[seg[:k] == seg[-k:], np.zeros(-k, bool)]
    out[~same] = np.nan
    return out


# ---------------------------------------------------------------- feature blocks

def l1_features(grid: pd.DataFrame, q: pd.DataFrame, t: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    at = grid.ts.to_numpy()
    seg = grid.seg.to_numpy()
    qts = q.ts.to_numpy()
    i = _asof(qts, at)
    i0 = np.maximum(i, 0)
    bid, ask = q.bid.to_numpy()[i0], q.ask.to_numpy()[i0]
    bsz, asz = q.bid_sz.to_numpy()[i0], q.ask_sz.to_numpy()[i0]
    mid = (bid + ask) / 2
    f = {
        "stale_s": (at - qts[i0]) / NS,
        "bid": bid, "ask": ask, "mid": mid,
        "spread_t": (ask - bid) / TICK,
        "l1_imb": (bsz - asz) / (bsz + asz),
        "micro_dev_t": ((bid * asz + ask * bsz) / (bsz + asz) - mid) / TICK,
        "l1_depth_log": np.log1p(bsz + asz),
    }
    f["stale_s"][i < 0] = np.inf
    cum_ofi = _cum_at(qts, q.ofi.to_numpy(), at)
    cum_nq = _cum_at(qts, np.ones(len(qts)), at)
    tts = t.ts.to_numpy()
    vol = t.volume.to_numpy() if len(t) else np.array([])
    cum_v = _cum_at(tts, vol, at)
    cum_sv = _cum_at(tts, vol * t.sign.to_numpy() if len(t) else vol, at)
    cum_n = _cum_at(tts, np.ones(len(tts)), at)
    cum_pv = _cum_at(tts, vol * t.price.to_numpy() if len(t) else vol, at)
    r1 = mid - _lag(mid, 1, seg)
    cum_r2 = np.nancumsum(np.nan_to_num(r1) ** 2)
    depth = (bsz + asz) / 2
    for w in spec.windows_s:
        k = max(1, int(round(w / spec.step_s)))
        dv = cum_v - _lag(cum_v, k, seg)
        dsv = cum_sv - _lag(cum_sv, k, seg)
        f[f"ret_{w}s_t"] = (mid - _lag(mid, k, seg)) / TICK
        f[f"ofi_{w}s"] = (cum_ofi - _lag(cum_ofi, k, seg)) / np.maximum(depth, 1)
        f[f"flow_{w}s"] = dsv / np.where(dv > 0, dv, np.nan)
        f[f"svol_{w}s"] = dsv
        f[f"vol_{w}s_log"] = np.log1p(np.maximum(dv, 0))
        f[f"ntrade_{w}s"] = cum_n - _lag(cum_n, k, seg)
        f[f"nquote_{w}s"] = cum_nq - _lag(cum_nq, k, seg)
        f[f"rv_{w}s_t"] = np.sqrt(np.maximum(cum_r2 - _lag(cum_r2, k, seg), 0)) / TICK
        vwap = (cum_pv - _lag(cum_pv, k, seg)) / np.where(dv > 0, dv, np.nan)
        f[f"vwap_dev_{w}s_t"] = (mid - vwap) / TICK
        f[f"imb_chg_{w}s"] = f["l1_imb"] - _lag(f["l1_imb"], k, seg)
    return pd.DataFrame(f)


def l2_features(at: np.ndarray, seg: np.ndarray, l2: pd.DataFrame, mid: np.ndarray,
                spec: Spec) -> tuple[pd.DataFrame, str, float]:
    px, sz, mode = book.snapshots(l2, at, spec.l2_mode)
    bad = book.check_book(px)
    bpx, apx = px[:, 0], px[:, 1]
    bsz, asz = sz[:, 0], sz[:, 1]
    f: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in (1, 3, 5, 10):
            b, a = bsz[:, :k].sum(1), asz[:, :k].sum(1)
            f[f"l2_imb{k}"] = (b - a) / (b + a)
        w = np.exp(-0.5 * np.arange(book.LEVELS))
        b, a = (bsz * w).sum(1), (asz * w).sum(1)
        f["l2_imb_w"] = (b - a) / (b + a)
        f["l2_depth_log"] = np.log1p(bsz.sum(1) + asz.sum(1))
        # liquidity-weighted distance of each side from mid, in ticks
        bd = np.nansum(bsz * (mid[:, None] - bpx), 1) / bsz.sum(1) / TICK
        ad = np.nansum(asz * (apx - mid[:, None]), 1) / asz.sum(1) / TICK
        f["l2_com_skew_t"] = ad - bd
        # how far you'd walk for 10 / 30 lots: book thinness per side
        for side, p, s, sgn in (("b", bpx, bsz, -1), ("a", apx, asz, 1)):
            cs = np.cumsum(s, 1)
            for lots in (10, 30):
                j = np.minimum((cs < lots).sum(1), book.LEVELS - 1)
                f[f"l2_walk{lots}_{side}_t"] = sgn * (p[np.arange(len(p)), j] - mid) / TICK
        f["l2_max_lvl_b"] = np.argmax(bsz, 1).astype(float)
        f["l2_max_lvl_a"] = np.argmax(asz, 1).astype(float)
        f["l2_gap_t"] = (apx[:, 0] - bpx[:, 0]) / TICK - 1  # 0 when the L2 touch is one tick wide
    empty = np.isnan(bpx[:, 0]) | np.isnan(apx[:, 0])
    for v in f.values():
        v[empty] = np.nan
    for w in spec.windows_s[:3]:
        k = max(1, int(round(w / spec.step_s)))
        for c in ("l2_imb5", "l2_imb_w"):
            f[f"{c}_chg_{w}s"] = f[c] - _lag(f[c], k, seg)
        f[f"l2_depth_chg_{w}s"] = f["l2_depth_log"] - _lag(f["l2_depth_log"], k, seg)
    return pd.DataFrame(f), mode, bad


def candle_features(at: np.ndarray, candles: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Bar features as of t, using only bars that have closed by t.

    The bar timestamp is assumed to be the bar *open*; a bar is usable at ts + bar length.
    (If your timestamps are bar close this is one bar too cautious, never leaky.)
    """
    c = candles[candles.type == LAST]
    if c.empty:  # fall back to bid/ask bars → mid bars
        b, a = candles[candles.type == BID], candles[candles.type == ASK]
        if b.empty or a.empty:
            return pd.DataFrame(index=range(len(at)))
        m = b.merge(a, on="ts", suffixes=("_b", "_a"))
        c = pd.DataFrame({"ts": m.ts, **{k: (m[f"{k}_b"] + m[f"{k}_a"]) / 2
                                          for k in ("open", "high", "low", "close")},
                          "volume": m.volume_b})
    c = c.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    bar = int(np.median(np.diff(c.ts.to_numpy()))) if len(c) > 1 else 60 * NS
    close, high, low, vol = (c[k].to_numpy() for k in ("close", "high", "low", "volume"))
    s = pd.Series(close)
    prev = np.r_[close[0], close[:-1]]
    tr = np.max([high - low, np.abs(high - prev), np.abs(low - prev)], axis=0)
    atr = pd.Series(tr).rolling(14, min_periods=5).mean().to_numpy()
    f = {"c_atr14_t": atr / TICK}
    for n in spec.candle_lookbacks:
        f[f"c_ret_{n}b_atr"] = (close - s.shift(n).to_numpy()) / atr
    hi60 = pd.Series(high).rolling(60, min_periods=10).max().to_numpy()
    lo60 = pd.Series(low).rolling(60, min_periods=10).min().to_numpy()
    f["c_pos60"] = (close - lo60) / np.where(hi60 > lo60, hi60 - lo60, np.nan)
    v = pd.Series(np.log1p(vol))
    f["c_vol_z60"] = ((v - v.rolling(60, min_periods=10).mean())
                      / v.rolling(60, min_periods=10).std()).to_numpy()
    for n in (20, 60):
        f[f"c_ema{n}_gap_atr"] = (close - s.ewm(span=n, adjust=False).mean().to_numpy()) / atr
    f["c_bar_age_s"] = np.zeros(len(c))
    feats = pd.DataFrame(f)
    avail = c.ts.to_numpy() + bar
    j = _asof(avail, at)
    out = feats.iloc[np.maximum(j, 0)].reset_index(drop=True)
    out["c_bar_age_s"] = (at - avail[np.maximum(j, 0)]) / NS
    out[j < 0] = np.nan
    stale = out["c_bar_age_s"].to_numpy() > 3 * bar / NS  # missing bars (gap) → unknown
    out[stale] = np.nan
    return out


def time_features(at: np.ndarray) -> pd.DataFrame:
    t = pd.to_datetime(at, utc=True).tz_convert(ET)
    mins = (t.hour * 60 + t.minute + t.second / 60).to_numpy()
    return pd.DataFrame({
        "tod_sin": np.sin(2 * np.pi * mins / 1440), "tod_cos": np.cos(2 * np.pi * mins / 1440),
        "min_from_rth_open": mins - 570, "dow": t.dayofweek.to_numpy().astype(float),
    })


def session_day(at: np.ndarray) -> np.ndarray:
    """CME trading date: the Globex session opening 18:00 ET belongs to the next day."""
    t = pd.to_datetime(at, utc=True).tz_convert(ET) + pd.Timedelta(hours=6)
    return t.strftime("%Y-%m-%d").to_numpy()


# ---------------------------------------------------------------- labels

def labels(grid: pd.DataFrame, q: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Forward mid return per horizon plus the prices the backtest trades at.

    px_in_*: bid/ask at t + latency (entry); px_out_*_{h}: bid/ask at t + latency + h (exit).
    A label is NaN if its window crosses a segment boundary or the quote there is stale.
    """
    at = grid.ts.to_numpy()
    qts = q.ts.to_numpy()
    bid, ask = q.bid.to_numpy(), q.ask.to_numpy()
    qseg = np.r_[0, np.cumsum(np.diff(qts) > spec.max_gap_s * NS)]
    seg_of_grid = qseg[np.maximum(_asof(qts, at), 0)]
    lat = int(spec.latency_ms * 1e6)

    def at_time(t):
        j = _asof(qts, t)
        ok = (j >= 0) & (qseg[np.maximum(j, 0)] == seg_of_grid) & (t - qts[np.maximum(j, 0)] <= spec.max_stale_s * NS)
        j = np.maximum(j, 0)
        return np.where(ok, bid[j], np.nan), np.where(ok, ask[j], np.nan)

    mid_now = (lambda b, a: (b + a) / 2)(*at_time(at))
    b_in, a_in = at_time(at + lat)
    out = {"px_in_bid": b_in, "px_in_ask": a_in}
    for h in spec.horizons_s:
        b, a = at_time(at + lat + h * NS)
        out[f"px_out_bid_{h}"], out[f"px_out_ask_{h}"] = b, a
        bh, ah = at_time(at + h * NS)
        out[f"fwd_{h}s_t"] = ((bh + ah) / 2 - mid_now) / TICK
    return pd.DataFrame(out)


# ---------------------------------------------------------------- assembly

def build(l1: pd.DataFrame, l2: pd.DataFrame | None, candles: pd.DataFrame | None,
          spec: Spec) -> tuple[pd.DataFrame, dict]:
    q = quotes(l1)
    if q.empty:
        return pd.DataFrame(), {"rows": 0}
    t = trades(l1, q)
    grid = make_grid(q, spec)
    parts = [grid, l1_features(grid, q, t, spec)]
    info: dict = {"quotes": len(q), "trades": len(t)}
    at, seg = grid.ts.to_numpy(), grid.seg.to_numpy()
    if l2 is not None and not l2.empty:
        f2, mode, bad = l2_features(at, seg, l2, parts[1].mid.to_numpy(), spec)
        parts.append(f2)
        info["l2_mode"] = mode
        info["l2_bad_share"] = round(bad, 4)  # crossed / unsorted books; high → wrong mode or bad data
    if candles is not None and not candles.empty:
        parts.append(candle_features(at, candles, spec))
    parts.append(time_features(at))
    parts.append(labels(grid, q, spec))
    df = pd.concat(parts, axis=1)
    df.insert(1, "day", session_day(at))
    df = df[df.stale_s <= spec.max_stale_s]
    if spec.session == "rth":
        m = df.min_from_rth_open
        df = df[(m >= 0) & (m < 390)]
    info["rows"] = len(df)
    return df.reset_index(drop=True), info


LABEL_PREFIXES = ("fwd_", "px_")
NON_FEATURES = {"ts", "day", "seg", "bid", "ask", "mid", "stale_s"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers, raw prices and labels."""
    return [c for c in df.columns if c not in NON_FEATURES and not c.startswith(LABEL_PREFIXES)]
