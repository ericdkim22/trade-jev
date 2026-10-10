import numpy as np

from trade_jev import markettick as M
from trade_jev.data import TICK, et_to_ns, load_day

# 2025-05-21 is EDT: 09:30 ET = 13:30 UTC
ROWS = """20250521133000000000;1;0;100.00;5
20250521133000000001;1;1;100.50;7
20250521133000000002;2;0;100.25;9;1;0
20250521133000000003;2;0;99.75;4;2;0
20250521133000000004;2;0;99.50;6;3;0
20250521133000000005;2;1;100.75;3;2;0
20250521133000000006;2;1;101.00;2;3;0
20250521133000000007;1;2;100.50;3
20250521133000000008;1;2;100.00;1
20250521133000050000;1;1;100.25;2
20250521133001000000;1;2;100.25;4
20250521133001000001;2;0;99.75;0;2;2
20250521133002000000;1;0;100.00;6
bad;row
"""


def test_convert_builds_rows_from_level1_and_a_price_keyed_book(tmp_path):
    src = tmp_path / "20250521.csv"
    src.write_text(ROWS)
    st = M.convert(src, out_dir=tmp_path)
    assert st["symbol"] == "MNQM5" and st["trades"] == 3 and st["crossed"] == 0
    day = load_day("2025-05-21", np.array([et_to_ns("2025-05-21", "09:30:05")]), data_dir=tmp_path)
    # row 1 at 13:30:00.000001: quote 100.00 x5 / 100.50 x7; 100.25 on the bid book is stale (≥ the quote): dropped
    assert day.bid[0] == round(100.00 / TICK) and day.ask[0] == round(100.50 / TICK)
    # the 100.25 ask quote changes the top → a row; its trade delta sums the two trades before it: +3 (at ask) -1 (at bid)
    assert day.ask[1] == round(100.25 / TICK) and int(day.cum_delta[1]) == 2
    # the 100.25 trade at the new ask comes in 1 s later with an unchanged top: held until the 100 ms throttle allows
    assert int(day.cum_delta[-1]) == 6
    book = day.book(int(day.book_rows[0]))
    bid_px, bid_sz, ask_px, ask_sz = book
    assert bid_px[0] == round(100.00 / TICK) and bid_sz[0] == 6  # level 1 = the Level 1 quote (last update 6)
    assert list(bid_px[1:3]) == [round(99.50 / TICK), round(99.50 / TICK) - 1]  # 99.75 removed; padding below
    assert list(ask_px[:3]) == [round(100.25 / TICK), round(100.75 / TICK), round(101.00 / TICK)]


def test_holiday_files_are_skipped(tmp_path):
    src = tmp_path / "20250526.csv"  # Memorial Day: only overnight rows
    src.write_text("20250526020000000000;1;0;100.00;5\n20250526020000000001;1;1;100.25;5\n")
    st = M.convert(src, out_dir=tmp_path)
    assert st["skipped"] and not list(tmp_path.glob("*.parquet")) and not (tmp_path / "parts" / "x").exists()
