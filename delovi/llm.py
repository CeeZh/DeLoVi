"""OpenAI Responses API with a disk cache, used for QA, grading and memory building.

Completions are cached by (model, reasoning effort, messages), so rebuilding a memory or re-answering an
identical context costs nothing. `FakeLLM` answers offline for dry runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, fields
from pathlib import Path

from delovi.config import CACHE_DIR, PRICES


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached: bool = False  # served from the disk cache

    @property
    def cost_usd(self) -> float:
        base = next((v for k, v in PRICES.items() if self.model.startswith(k)), (0.0, 0.0, 0.0))
        return ((self.input_tokens - self.cached_input_tokens) * base[0] + self.cached_input_tokens * base[1]
                + self.output_tokens * base[2]) / 1e6


def _key(model: str, messages: list[dict], reasoning: str | None) -> str:
    return hashlib.sha1(json.dumps([model, reasoning, messages], ensure_ascii=False).encode()).hexdigest()


class LLM:
    def __init__(self, client=None, concurrency: int = 16, cache_dir: Path = CACHE_DIR / "llm", retries: int = 8):
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
        self.client, self.retries = client, retries
        self.sem = asyncio.Semaphore(concurrency)
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    async def complete(self, model: str, messages: list[dict], reasoning: str | None = None) -> Completion:
        """`reasoning`: an effort ("none", "low", "medium", "high"), or None for the model's default."""
        path = self.cache_dir / f"{_key(model, messages, reasoning)}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return Completion(**{f.name: data[f.name] for f in fields(Completion) if f.name in data}, cached=True)
        kwargs = {"reasoning": {"effort": reasoning}} if reasoning else {}
        async with self.sem:
            for attempt in range(self.retries):
                try:
                    resp = await self.client.responses.create(model=model, input=messages, **kwargs)
                    break
                except Exception:  # noqa: BLE001 -- rate limits, timeouts, 5xx
                    if attempt == self.retries - 1:
                        raise
                    await asyncio.sleep(min(60.0, 2 ** attempt * 2) * (0.5 + random.random()))
        u = resp.usage
        c = Completion(resp.output_text, resp.model, u.input_tokens,
                       getattr(u.input_tokens_details, "cached_tokens", 0) or 0, u.output_tokens,
                       getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0)
        record = {k: v for k, v in c.__dict__.items() if k != "cached"}
        record["request"] = {"model": model, "reasoning": reasoning, "messages": messages}
        tmp = path.with_suffix(f".{os.getpid()}.{id(c)}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        os.replace(tmp, path)
        return c


class BuildBudgetError(RuntimeError):
    """Building the memories of one evaluation reached the cost cap."""


class BuildBudget:
    """Cost cap on the memory building of one evaluation, in nominal USD (as if nothing were cached)."""

    def __init__(self, cap_usd: float | None):
        self.cap, self.used = cap_usd, 0.0

    def check(self) -> None:
        if self.cap is not None and self.used >= self.cap:
            raise BuildBudgetError(f"memory building reached the ${self.cap:.2f} cap (nominal ${self.used:.2f})")


class Tracked:
    """Wraps an LLM for one memory build: counts calls, tokens, nominal cost and real spend."""

    def __init__(self, llm, budget: BuildBudget | None = None):
        self.llm, self.budget = llm, budget
        self.calls = self.cached_calls = self.input_tokens = self.output_tokens = 0
        self.cost_usd = self.spent_usd = 0.0
        self.models: dict[str, int] = {}

    async def complete(self, model: str, messages: list[dict], reasoning: str | None = None) -> Completion:
        if self.budget is not None:
            self.budget.check()
        c = await self.llm.complete(model, messages, reasoning)
        self.calls += 1
        self.cached_calls += c.cached
        self.input_tokens += c.input_tokens
        self.output_tokens += c.output_tokens
        self.cost_usd += c.cost_usd
        self.spent_usd += 0.0 if c.cached else c.cost_usd
        if self.budget is not None:
            self.budget.used += c.cost_usd
        self.models[c.model] = self.models.get(c.model, 0) + 1
        return c

    def stats(self) -> dict:
        return {"calls": self.calls, "cached_calls": self.cached_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 4),
                "spent_usd": round(self.spent_usd, 4), "models": dict(self.models)}


class FakeLLM:
    """Offline stand-in for dry runs: a random letter, a fixed answer or a fake summary."""

    async def complete(self, model: str, messages: list[dict], reasoning: str | None = None) -> Completion:
        user = messages[-1]["content"]
        rng = random.Random(user[-200:])
        qa = "Confidence:" in messages[0]["content"]
        if "Choices:" in user:
            letters = [m[1] for m in re.findall(r"\((([A-J]))\)", user)]
            text = rng.choice(letters or ["A"])
            text = f"Answer: {text}\nConfidence: {rng.random():.2f}" if qa else text
        elif qa:
            text = f"Answer: fake answer\nConfidence: {rng.random():.2f}"
        elif "Final Score" in messages[0]["content"]:
            text = f"Analysis: fake.\n\nFinal Score:\n{rng.randint(0, 5)}"
        else:
            text = "fake answer"
        return Completion(text, "fake", len(json.dumps(messages)) // 4, 0, 20, 0)
