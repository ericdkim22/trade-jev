import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trade_jev.alpha import book, io
from trade_jev.alpha.features import Spec, build, feature_columns

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("mnq")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_mnq_synthetic.py"),
                    "--out", str(out), "--days", "1", "--hours", "0.5"], check=True, capture_output=True)
    return out


def test_civil_to_ns_matches_pandas():
    stamps = np.array([19700101000000, 20260611133000, 20240229235959, 20261231000001])
    want = pd.to_datetime(stamps.astype(str), format="%Y%m%d%H%M%S", utc=True).as_unit("ns").asi8
    assert (io.civil_to_ns(stamps) == want).all()
    assert io.parse_ts("20260611133000123456") == want[1] + 123_456_000


def test_sniff_classifies_each_format(synth):
    kinds = {f.path.name.split("_")[-1]: (f.kind, f.sep) for f in io.discover(synth)}
    assert kinds == {"1min.csv": ("candles", ","), "L1.txt": ("l1", ";"), "L2.txt": ("l2", ";")}


def test_sniff_header_and_load(tmp_path):
    p = tmp_path / "ticks.csv"
    p.write_text("Time,Level,Type,Price,Volume\n20260611133000000001,1,0,21000.25,5\n"
                 "20260611133000000002,1,1,21000.50,7\n")
    f = io.sniff(p)
    assert f.kind == "l1" and f.header
    df = io.load([f])
    assert list(df.type) == [0, 1] and df.ts.iloc[1] - df.ts.iloc[0] == 1000


def _l2(rows):
    return pd.DataFrame(rows, columns=["ts", "level", "type", "price", "volume", "depth", "action"])


def test_book_insert_vs_slot():
    # three bid levels, then a new best bid is Added at depth 1, then depth 2 is Removed
    rows = [(1, 2, 0, 100.0, 1, 1, 0), (1, 2, 0, 99.75, 2, 2, 0), (1, 2, 0, 99.5, 3, 3, 0),
            (2, 2, 0, 100.25, 9, 1, 0), (3, 2, 0, 0.0, 0, 2, 2)]
    px, sz, _ = book.snapshots(_l2(rows), np.array([1, 2, 3]), "insert")
    assert list(px[1, 0, :4]) == [100.25, 100.0, 99.75, 99.5]
    assert list(px[2, 0, :3]) == [100.25, 99.75, 99.5]
    px, _, _ = book.snapshots(_l2(rows), np.array([2]), "slot")
    assert list(px[0, 0, :3]) == [100.25, 99.75, 99.5]  # slot: Add overwrote row 1


def test_book_auto_picks_insert(synth):
    l2 = io.load([f for f in io.discover(synth) if f.kind == "l2"])
    mode, scores = book.pick_mode(l2)
    assert mode == "insert" and scores["insert"] < scores["slot"]


def test_features_are_causal(synth):
    """Cutting the data at T must not change any feature at t <= T - 1s."""
    files = io.discover(synth)
    l1 = io.load([f for f in files if f.kind == "l1"])
    l2 = io.load([f for f in files if f.kind == "l2"])
    c = io.load([f for f in files if f.kind == "candles"])
    spec = Spec(horizons_s=(30,))
    full, _ = build(l1, l2, c, spec)
    cut = int(l1.ts.iloc[len(l1) // 2])
    part, _ = build(l1[l1.ts <= cut], l2[l2.ts <= cut], c[c.ts <= cut], spec)
    cols = feature_columns(full)
    a = full[full.ts <= cut - 10**9].set_index("ts")[cols]
    b = part.set_index("ts").loc[a.index, cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)
    assert not any(c.startswith(("fwd_", "px_")) for c in cols)


def test_labels_are_forward_mid_moves(synth):
    files = io.discover(synth)
    l1 = io.load([f for f in files if f.kind == "l1"])
    df, _ = build(l1, None, None, Spec(horizons_s=(30,)))
    later = df.set_index("ts").mid.reindex(df.ts + 30 * 10**9).to_numpy()
    ok = ~np.isnan(later) & df.fwd_30s_t.notna().to_numpy()
    assert ok.sum() > 1000
    np.testing.assert_allclose(df.fwd_30s_t[ok], (later[ok] - df.mid[ok]) / 0.25)
