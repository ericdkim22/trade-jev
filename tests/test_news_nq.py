import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import news_nq as N  # noqa: E402


def test_fade_trades_against_a_clear_market_moving_call():
    assert N.fade_side({"market_moving": 0.95, "reaction": {"up": 0.8, "down": 0.1}}) == -1
    assert N.fade_side({"market_moving": 0.95, "reaction": {"up": 0.1, "down": 0.7}}) == 1
    assert N.fade_side({"market_moving": 0.85, "reaction": {"up": 0.9, "down": 0.0}}) is None  # not market-moving enough
    assert N.fade_side({"market_moving": 0.99, "reaction": {"up": 0.5, "down": 0.4}}) is None  # no clear side


def test_verdict_only_after_fifty_and_needs_t_of_two():
    assert N.summary([10.0] * 49 + [12.0])["verdict"] != "testing"
    assert N.summary([10.0] * 10)["verdict"] == "testing"
    assert N.summary([5.0, -4.0] * 25)["verdict"] == "didn't work"  # positive but t < 2
    s = N.summary([8.0, 12.0] * 25)
    assert s["verdict"] == "worked" and s["n"] == 50 and s["net_usd"] == 250.0  # 500 ticks × $0.50
