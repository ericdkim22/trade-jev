"""Walk-forward training and out-of-sample evaluation of the alpha model.

Target: forward mid move over `horizon_s`, in ticks (winsorised for training).
Model:  LightGBM regressor (sklearn HistGradientBoosting if LightGBM is missing).

Walk-forward by trading day: for each test day, train only on earlier days, with a
purge so no training label window ends after the test day starts. The last training
day is held out for early stopping and to set trade thresholds, so thresholds are
never picked on in-sample predictions.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade_jev.alpha.features import NS, TICK, feature_columns

warnings.filterwarnings("ignore", category=FutureWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")
try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

POINT_VALUE = 2.0  # MNQ: $2 / point → $0.50 / tick


@dataclass
class TrainConfig:
    horizon_s: int = 60
    min_train_days: int = 3
    max_train_days: int = 0            # 0 = expanding window
    stride: int = 5                    # train on every n-th grid row (labels overlap heavily)
    winsor: float = 0.01
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.98)  # |pred| cut-offs to trade
    commission: float = 0.62           # $ per contract per side (incl. exchange fees)
    seed: int = 7
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 400


def _fit(X, y, Xv, yv, cfg: TrainConfig):
    if lgb is not None:
        m = lgb.LGBMRegressor(
            n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
            random_state=cfg.seed, verbose=-1)
        m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=cfg.learning_rate,
                                      min_samples_leaf=cfg.min_child_samples,
                                      early_stopping=True, random_state=cfg.seed)
    m.fit(np.vstack([X, Xv]), np.r_[y, yv])
    return m


def importances(model, cols: list[str]) -> pd.Series:
    if hasattr(model, "booster_"):
        g = model.booster_.feature_importance("gain")
        return pd.Series(g / g.sum(), index=cols).sort_values(ascending=False)
    return pd.Series(np.nan, index=cols)


def _clip(y: np.ndarray, q: float) -> np.ndarray:
    lo, hi = np.nanquantile(y, [q, 1 - q])
    return np.clip(y, lo, hi)


def _ic(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 30 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 30 or np.std(a[m]) == 0:
        return np.nan
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()))


def walk_forward(df: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """OOS predictions for every day after the first `min_train_days`.

    Returns (predictions, per-fold stats, mean feature importance).
    """
    y_col = f"fwd_{cfg.horizon_s}s_t"
    cols = feature_columns(df)
    days = sorted(df.day.unique())
    preds, folds, imps = [], [], []
    h_ns = cfg.horizon_s * NS
    for k in range(cfg.min_train_days, len(days)):
        test_day = days[k]
        train_days = days[:k] if not cfg.max_train_days else days[max(0, k - cfg.max_train_days):k]
        test = df[df.day == test_day]
        test_start = test.ts.min()
        tr = df[df.day.isin(train_days) & df[y_col].notna() & (df.ts + h_ns < test_start)]
        fit_part, val_part = tr[tr.day != train_days[-1]], tr[tr.day == train_days[-1]]
        # purge: the early-stopping day must not share label windows with the fit days
        fit_part = fit_part[fit_part.ts + h_ns < val_part.ts.min()]
        fit_part = fit_part.iloc[:: cfg.stride]
        if len(fit_part) < 1000 or len(val_part) < 100:
            continue
        model = _fit(fit_part[cols], _clip(fit_part[y_col].to_numpy(), cfg.winsor),
                     val_part[cols], _clip(val_part[y_col].to_numpy(), cfg.winsor), cfg)
        val_pred = model.predict(val_part[cols])
        cuts = {q: float(np.quantile(np.abs(val_pred), q)) for q in cfg.quantiles}
        p = model.predict(test[cols])
        out = test[["ts", "day", y_col, "px_in_bid", "px_in_ask",
                    f"px_out_bid_{cfg.horizon_s}", f"px_out_ask_{cfg.horizon_s}"]].copy()
        out["pred"] = p
        for q, c in cuts.items():
            out[f"cut_{q}"] = c
        preds.append(out)
        imps.append(importances(model, cols))
        folds.append({"day": test_day, "train_days": len(train_days), "train_rows": len(fit_part),
                      "trees": getattr(model, "best_iteration_", None),
                      "ic": _ic(p, test[y_col].to_numpy()),
                      "rank_ic": _rank_ic(p, test[y_col].to_numpy()),
                      "val_rank_ic": _rank_ic(val_pred, val_part[y_col].to_numpy())})
        print(f"  {test_day}: rank IC {folds[-1]['rank_ic']:+.4f}  (train {len(train_days)} days, "
              f"{len(fit_part):,} rows, {folds[-1]['trees']} trees)", flush=True)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False) if imps else pd.Series(dtype=float)
    return (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()), pd.DataFrame(folds), imp


def backtest(pred: pd.DataFrame, horizon_s: int, q: float, commission: float) -> tuple[dict, pd.DataFrame]:
    """One MNQ contract, non-overlapping trades: when flat and |pred| >= cut, cross the
    spread at t + latency, exit by crossing back at t + latency + horizon."""
    cut = pred[f"cut_{q}"].to_numpy()
    ts, p = pred.ts.to_numpy(), pred.pred.to_numpy()
    bi, ai = pred.px_in_bid.to_numpy(), pred.px_in_ask.to_numpy()
    bo, ao = pred[f"px_out_bid_{horizon_s}"].to_numpy(), pred[f"px_out_ask_{horizon_s}"].to_numpy()
    days = pred.day.to_numpy()
    h = horizon_s * NS
    rows, free_at = [], -1
    for i in np.flatnonzero(np.abs(p) >= cut):
        if ts[i] < free_at or np.isnan(bi[i]) or np.isnan(bo[i]):
            continue
        d = 1 if p[i] > 0 else -1
        entry, exit_ = (ai[i], bo[i]) if d > 0 else (bi[i], ao[i])
        ticks = d * (exit_ - entry) / TICK
        rows.append((ts[i], days[i], d, entry, exit_, ticks, ticks * TICK * POINT_VALUE - 2 * commission))
        free_at = ts[i] + h
    tr = pd.DataFrame(rows, columns=["ts", "day", "dir", "entry", "exit", "ticks", "pnl"])
    if tr.empty:
        return {"q": q, "trades": 0}, tr
    daily = tr.groupby("day").pnl.sum().reindex(sorted(pred.day.unique()), fill_value=0.0)
    return {
        "q": q, "trades": len(tr), "trades_per_day": round(len(tr) / len(daily), 1),
        "hit": round(float((tr.ticks > 0).mean()), 4),
        "avg_ticks_gross": round(float(tr.ticks.mean()), 3),
        "avg_pnl": round(float(tr.pnl.mean()), 3),
        "total_pnl": round(float(tr.pnl.sum()), 2),
        "daily_sharpe_ann": round(float(daily.mean() / daily.std() * np.sqrt(252)), 2) if daily.std() > 0 else None,
        "pos_days": f"{int((daily > 0).sum())}/{len(daily)}",
    }, tr


def deciles(pred: pd.DataFrame, horizon_s: int) -> pd.DataFrame:
    y = pred[f"fwd_{horizon_s}s_t"]
    m = y.notna()
    b = pd.qcut(pred.pred[m].rank(method="first"), 10, labels=False) + 1
    return (pd.DataFrame({"decile": b, "pred": pred.pred[m], "fwd_t": y[m]})
            .groupby("decile").agg(n=("fwd_t", "size"), pred_mean=("pred", "mean"),
                                   fwd_mean_t=("fwd_t", "mean"), up_share=("fwd_t", lambda s: (s > 0).mean()))
            .round(4))


def single_feature_ic(df: pd.DataFrame, horizon_s: int, top: int = 15) -> pd.Series:
    """Rank IC of each raw feature with the target: a sanity baseline for the model."""
    y = df[f"fwd_{horizon_s}s_t"].to_numpy()
    sample = np.arange(0, len(df), max(1, len(df) // 200_000))
    ics = {c: _rank_ic(df[c].to_numpy()[sample], y[sample]) for c in feature_columns(df)}
    s = pd.Series(ics).dropna()
    return s.reindex(s.abs().sort_values(ascending=False).index)[:top]


def fit_final(df: pd.DataFrame, cfg: TrainConfig, out: Path, spec: dict) -> dict:
    """Train on every day (last day for early stopping) and save model + metadata for `predict`."""
    y_col = f"fwd_{cfg.horizon_s}s_t"
    cols = feature_columns(df)
    days = sorted(df.day.unique())
    d = df[df[y_col].notna()]
    val = d[d.day == days[-1]]
    fit = d[(d.day != days[-1]) & (d.ts + cfg.horizon_s * NS < val.ts.min())].iloc[:: cfg.stride]
    model = _fit(fit[cols], _clip(fit[y_col].to_numpy(), cfg.winsor),
                 val[cols], _clip(val[y_col].to_numpy(), cfg.winsor), cfg)
    vp = model.predict(val[cols])
    meta = {"features": cols, "horizon_s": cfg.horizon_s, "config": asdict(cfg), "spec": spec,
            "cuts": {str(q): float(np.quantile(np.abs(vp), q)) for q in cfg.quantiles},
            "train_days": days}
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(out / "model.txt"))
    else:
        import pickle
        (out / "model.pkl").write_bytes(pickle.dumps(model))
    (out / "model.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def load_model(path: Path):
    meta = json.loads((path / "model.json").read_text())
    if (path / "model.txt").exists():
        booster = lgb.Booster(model_file=str(path / "model.txt"))
        return (lambda X: booster.predict(X[meta["features"]])), meta
    import pickle
    m = pickle.loads((path / "model.pkl").read_bytes())
    return (lambda X: m.predict(X[meta["features"]])), meta
