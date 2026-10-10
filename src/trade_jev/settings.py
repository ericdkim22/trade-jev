"""Harness settings: the one place defaults live.

The default is the registered "most consistent" setting from the 15-day replay grid
(results/FINDINGS.md): positive on 12 of 15 days, worst day −$1,065.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    min_conf: float = 0.7    # ignore Jev answers below this probability
    agree: int = 4           # consecutive same-side answers needed to act
    min_hold_s: float = 30   # seconds before a position may reverse
    stop_ticks: int = 200    # 0 = off
    target_ticks: int = 100  # 0 = off
    max_hold_s: float = 0    # time stop: close at market after this many seconds; 0 = off
    breakeven_ticks: int = 0  # once this many ticks in profit, the stop moves to the entry price; 0 = off

    def label(self) -> str:
        extra = (f" max_hold={self.max_hold_s:g}s" if self.max_hold_s else "") + \
                (f" breakeven={self.breakeven_ticks}" if self.breakeven_ticks else "")
        return (f"conf≥{self.min_conf} agree={self.agree} hold={self.min_hold_s}s "
                f"stop={self.stop_ticks or 'off'} target={self.target_ticks or 'off'}{extra}")


DEFAULT = Settings()
