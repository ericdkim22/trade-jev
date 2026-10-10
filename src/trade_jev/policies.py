"""Policies: context → BUY / SELL / HOLD. Jev plus offline baselines."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from typesafe_sdk import AsyncTypeSafeClient, Choice

from trade_jev.encode import ENCODERS, Context
from trade_jev.settings import DEFAULT, Settings

ACTIONS = ("BUY", "SELL", "HOLD")


@dataclass
class Decision:
    action: str
    probs: dict[str, float] = field(default_factory=dict)
    state: dict | None = None
    tokens: int = 0
    cached: bool = False
    raw_action: str | None = None  # what the inner policy said before gating


# ---------------------------------------------------------------- Jev

ACTION_QUESTION = Choice(
    instructions=(
        "You are a short-term scalper trading one Nasdaq-100 E-mini futures contract. "
        "Using the current order book (`order_book`), recent mid prices, net aggressor volume "
        "(positive = buyers lifting offers, negative = sellers hitting bids) and our current "
        "`position`, choose the action most likely to be profitable over the next 1 to 5 minutes. "
        "Each trade pays the bid-ask spread plus commission, so only act when the book and "
        "order flow point clearly in one direction."
    ),
    criteria={
        "BUY": "Be long one contract: open a long if flat, reverse to long if short, "
               "or keep the long if already long. Price is likely to rise.",
        "SELL": "Be short one contract: open a short if flat, reverse to short if long, "
                "or keep the short if already short. Price is likely to fall.",
        "HOLD": "Make no change: stay flat if flat, keep the current position if in one. "
                "No clear edge, or the current position is still fine.",
    },
)


_SIDES = ACTION_QUESTION.criteria
_INPUTS = ("Using the market state (order book or features derived from it, recent prices, net aggressor volume: "
           "positive = buyers lifting offers, negative = sellers hitting bids) and our current `position`, ")

# Question variants (research dimension next to the input encoder). "scalp" is the original; its cache keys are unchanged.
QUESTIONS: dict[str, Choice] = {
    "scalp": ACTION_QUESTION,
    # the exits Jev is actually traded with, instead of a 1-5 minute horizon a 25-point target rarely fits in
    "target": Choice(
        instructions=("You trade one Micro / E-mini Nasdaq-100 futures contract. Every position closes automatically at "
                      f"+{DEFAULT.target_ticks / 4:g} points (target) or -{DEFAULT.stop_ticks / 4:g} points (stop) "
                      "from its entry. " + _INPUTS + "choose the side most likely to reach its target before its stop. "
                      "Each trade pays the bid-ask spread plus commission, so only act when one side is clearly more "
                      "likely to reach its target first."),
        criteria=dict(_SIDES)),
    # a longer horizon, traded with matching exits (QUESTION_SETTINGS)
    "next15": Choice(
        instructions=("You trade one Micro / E-mini Nasdaq-100 futures contract. A position is closed after 15 minutes, "
                      "or earlier at +25 points (target) or -25 points (stop). " + _INPUTS + "choose the side the price "
                      "is more likely to move over the next 15 minutes. Each trade pays the bid-ask spread plus "
                      "commission, so only act when one direction is clearly more likely."),
        criteria=dict(_SIDES)),
}
QUESTION_SETTINGS: dict[str, Settings] = {  # exits a question is meant to be traded with (default: DEFAULT)
    "next15": Settings(stop_ticks=100, target_ticks=100, max_hold_s=900),
}


class RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class JsonlCache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec
        self.fh = path.open("a")

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, rec: dict) -> None:
        rec = {"key": key, **rec}
        self.data[key] = rec
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()


class JevPolicy:
    name = "jev"

    def __init__(self, client: AsyncTypeSafeClient, cache: JsonlCache, limiter: RateLimiter,
                 encoder: str = "raw_l10", model: str = "jev-latest", question: str = "scalp"):
        self.client, self.cache, self.limiter = client, cache, limiter
        self.encode = ENCODERS[encoder]
        self.model = model
        self.question = QUESTIONS[question]
        self.name = f"jev[{encoder}]" if question == "scalp" else f"jev[{encoder}/{question}]"
        self.api_calls = 0
        self.cache_hits = 0

    async def __call__(self, ctx: Context) -> Decision:
        state = self.encode(ctx)
        q = {"type": "choice", "instructions": self.question.instructions,
             "criteria": dict(self.question.criteria)}
        key = hashlib.sha256(json.dumps([self.model, state, q], sort_keys=True).encode()).hexdigest()
        if (hit := self.cache.get(key)) is not None:
            self.cache_hits += 1
            return Decision(hit["choice"], hit["probs"], state, hit.get("tokens", 0), cached=True)

        await self.limiter.wait()
        self.api_calls += 1
        res = await self.client.system_one(state, {"action": self.question}, model=self.model)
        ans = res.choices["action"]
        tokens = res.usage.input_tokens or 0
        self.cache.put(key, {"choice": ans.choice, "probs": dict(ans.probabilities),
                             "tokens": tokens, "model": res.model})
        return Decision(ans.choice, dict(ans.probabilities), state, tokens)


class Gated:
    """Act only when the last `agree` answers name the same side, each with p >= min_conf,
    and any open position has been held >= min_hold_s. Otherwise HOLD."""

    def __init__(self, inner, min_conf: float = DEFAULT.min_conf, agree: int = DEFAULT.agree,
                 min_hold_s: float = DEFAULT.min_hold_s):
        self.inner, self.min_conf, self.agree = inner, min_conf, agree
        self.min_hold_ns = int(min_hold_s * 1e9)
        self.name = inner.name
        self.hist: dict[str, list[str | None]] = {}  # per day

    async def __call__(self, ctx: Context) -> Decision:
        d = await self.inner(ctx)
        hist = self.hist.setdefault(ctx.day.day, [])
        hist.append(d.action if d.probs.get(d.action, 0) >= self.min_conf else None)
        last = hist[-self.agree:]
        side = last[0] if len(last) == self.agree and len(set(last)) == 1 else None
        p = ctx.position
        if side not in ("BUY", "SELL") or (p.side and ctx.t_ns - p.entry_ns < self.min_hold_ns):
            side = "HOLD"
        return Decision(side, d.probs, d.state, d.tokens, d.cached, raw_action=d.action)


# ---------------------------------------------------------------- baselines

class HoldPolicy:
    """Never trades. PnL should be exactly 0."""
    name = "hold"

    async def __call__(self, ctx: Context) -> Decision:
        return Decision("HOLD")


class RandomPolicy:
    """Deterministic per (day, t). Trades ~20% of decisions — a no-skill benchmark that pays costs."""
    name = "random"

    def __init__(self, p_trade: float = 0.2):
        self.p_trade = p_trade

    async def __call__(self, ctx: Context) -> Decision:
        rng = random.Random(f"{ctx.day.day}:{ctx.t_ns}")
        if rng.random() >= self.p_trade:
            return Decision("HOLD")
        return Decision(rng.choice(["BUY", "SELL"]))


class ImbalancePolicy:
    """Classic rule: distance-weighted L10 depth imbalance agrees with 60s aggressor flow."""
    name = "imbalance"

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    async def __call__(self, ctx: Context) -> Decision:
        _, bid_sz, _, ask_sz = ctx.book
        w = 1.0 / np.arange(1, bid_sz.size + 1)
        b, a = float((bid_sz * w).sum()), float((ask_sz * w).sum())
        imb = (b - a) / (b + a) if b + a else 0.0
        flow = ctx.delta_since(60)
        if imb > self.threshold and flow > 0:
            return Decision("BUY")
        if imb < -self.threshold and flow < 0:
            return Decision("SELL")
        return Decision("HOLD")


BASELINES = {"hold": HoldPolicy, "random": RandomPolicy, "imbalance": ImbalancePolicy}
