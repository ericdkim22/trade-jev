import pytest

from trade_jev import research as R
from trade_jev.settings import DEFAULT, Settings


def ledger_with(days_live: dict, days_idea: dict, added="2026-10-09"):
    led = {"ideas": []}
    R.seed(led)
    base = R.find(led, R.key("raw_l10", DEFAULT))
    base["days"] = {d: {"pnl": p, "trades": 1} for d, p in days_live.items()}
    idea = R.add(led, "raw_l10", Settings(0.7, 4, 30, 100, 50), "manual", "tighter exits", added)
    idea["days"] = {d: {"pnl": p, "trades": 1} for d, p in days_idea.items()}
    return led, base, idea


def test_duplicates_are_refused_with_the_earlier_verdict():
    led, _, idea = ledger_with({}, {})
    idea["status"] = "didn't work"
    with pytest.raises(ValueError, match="didn't work"):
        R.add(led, "raw_l10", Settings(0.7, 4, 30.0, 100, 50), "grid", "same idea, other spelling", "2026-11-01")
    with pytest.raises(ValueError, match="already tried"):
        R.add(led, "raw_l10", DEFAULT, "manual", "the live strategy again", "2026-11-01")
    with pytest.raises(ValueError, match="unknown encoder"):
        R.add(led, "nope", DEFAULT, "manual", "", "2026-11-01")
    assert R.add(led, "features", Settings(0.6), "manual", "", "2026-11-01")["status"] == "testing"


def days(n, start=10, pnl=10.0):
    return {f"2026-10-{start + i:02d}": pnl for i in range(n)}


def test_verdict_counts_only_days_after_the_idea_was_added():
    # 20 great in-sample days don't count; 9 days after it was added aren't enough yet
    led, base, idea = ledger_with(days(9, start=10),
                                  {**{f"2026-09-{d:02d}": 500.0 for d in range(1, 21)}, **days(9, start=10, pnl=-5)})
    R.judge(idea, base)
    assert idea["status"] == "testing"
    base["days"]["2026-10-19"] = {"pnl": 10.0, "trades": 1}
    idea["days"]["2026-10-19"] = {"pnl": -5.0, "trades": 1}
    R.judge(idea, base)
    assert idea["status"] == "didn't work" and idea["verdict"]["days"] == 10 and idea["verdict"]["net"] == -50


def test_worked_needs_profit_and_beating_live():
    led, base, idea = ledger_with(days(10, pnl=-20), days(10, pnl=5))
    R.judge(idea, base)
    assert idea["status"] == "worked" and idea["verdict"]["live_net"] == -200
    led, base, idea = ledger_with(days(10, pnl=30), days(10, pnl=5))
    R.judge(idea, base)
    assert idea["status"] == "didn't work"  # profitable, but worse than the live strategy


def test_report_lists_every_group():
    led, base, idea = ledger_with(days(3), days(3))
    text = R.report(led)
    assert "## Live strategy" in text and "## Testing (2)" in text and "Didn't work" in text and idea["id"] in text


def test_history_retires_only_clear_losers_with_enough_days():
    led, base, idea = ledger_with({}, {})
    hist = lambda n, pnl: {f"2021-01-{d:02d}" if d <= 28 else f"2021-02-{d - 28:02d}": {"pnl": pnl, "trades": 2} for d in range(1, n + 1)}
    base["history"], idea["history"] = hist(39, 5.0), hist(39, -5.0)
    R.judge_history(idea, base)
    assert idea["status"] == "testing"  # 39 days: not enough
    base["history"], idea["history"] = hist(40, 5.0), hist(40, -5.0)
    R.judge_history(idea, base)
    assert idea["status"] == "didn't work" and idea["verdict"]["basis"] == "history"
    led, base, idea = ledger_with({}, {})
    base["history"], idea["history"] = hist(40, -10.0), hist(40, -5.0)
    R.judge_history(idea, base)
    assert idea["status"] == "testing"  # lost money, but less than the live strategy: keep testing on live days


def test_question_is_part_of_the_fingerprint_and_the_backtest():
    led = {"ideas": []}
    R.seed(led)
    i = R.add(led, "raw_l10", DEFAULT, "manual", "rules-aware question", "2026-10-10", question="target")
    assert i["key"].startswith("raw_l10/target|") and R.key("raw_l10", DEFAULT) == R.find(led, R.key("raw_l10", DEFAULT))["key"]
    with pytest.raises(ValueError, match="unknown question"):
        R.add(led, "raw_l10", DEFAULT, "manual", "", "2026-10-10", question="nope")
    args = R._run_args(("raw_l10", "next15"))
    assert args[args.index("--max-hold-s") + 1] == "900" and args[args.index("--question") + 1] == "next15"
    assert R._label(("raw_l10", "scalp")) == "raw_l10" and R._label(("features", "next15")) == "features-next15"


def test_jev_policy_names_and_questions():
    from trade_jev.policies import QUESTIONS, JevPolicy
    p = JevPolicy(None, None, None, encoder="raw_l10", question="next15")
    assert p.name == "jev[raw_l10/next15]" and p.question is QUESTIONS["next15"]
    assert JevPolicy(None, None, None).name == "jev[raw_l10]"
    assert "15 minutes" in QUESTIONS["next15"].instructions and "+25 points" in QUESTIONS["target"].instructions
