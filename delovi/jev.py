"""Requests to the decision model (Jev). Policies reach Jev only through a `Jev` handle, one per question.

- `choice(state, instructions, options)`: a ranking request. Jev reads the candidates jointly and returns a
  selection probability per candidate.
- `score(state, questions)`: up to JEV_SCORE_MAX_QUESTIONS score questions in one request; for each, Jev returns
  a probability per level and the expected level.

Requests are sized with a token estimate before they are sent. If Jev rejects a request as too long, the
request budget shrinks by 10% and `TooLong` is raised so the caller can split the request again. Every
response is appended to the question's replay file, so an interrupted run resumes without paying again.
An optional spend cap (`SearchBudget`) is shared by all questions of an evaluation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from delovi import guard
from delovi.config import (JEV_CHOICE_TOKENS, JEV_MAX_CHOICES, JEV_MODEL, JEV_PRICE_PER_MTOK, JEV_RPM,
                           JEV_SCORE_MAX_QUESTIONS, JEV_SCORE_TOKENS, JEV_TPS)

MODEL = JEV_MODEL
PRICE_PER_MTOK = JEV_PRICE_PER_MTOK
BUDGET = JEV_CHOICE_TOKENS
MAX_CHOICES = JEV_MAX_CHOICES
SCORE_MAX_QUESTIONS = JEV_SCORE_MAX_QUESTIONS
SCORE_BUDGET = JEV_SCORE_TOKENS
SHRINK = 0.9

# Token estimate fitted on earlier runs: fixed overhead, tokens per ASCII and non-ASCII character, and per option.
FIXED_TOKENS, TOKENS_PER_CHAR, NON_ASCII_TOKENS, OPTION_TOKENS = 206, 0.27, 1.21, 37
OVERFLOW_HINTS = ("max_tokens", "token", "context", "too long", "too large", "exceed")

log = logging.getLogger("jev")


class TooLong(Exception):
    """Jev rejected the request as too long."""


class SearchBudgetError(RuntimeError):
    """The evaluation's Jev requests reached the spend cap."""


class SearchBudget:
    """Jev spend cap of one evaluation, in nominal USD (replayed responses count as if paid)."""

    def __init__(self, cap_usd: float | None, used_tokens: int = 0):
        self.cap, self.used_tokens, self.exceeded = cap_usd, used_tokens, False

    @property
    def used_usd(self) -> float:
        return self.used_tokens / 1e6 * PRICE_PER_MTOK

    def check(self) -> None:
        if self.cap is not None and self.used_usd >= self.cap:
            self.exceeded = True
            raise SearchBudgetError(f"Jev search reached the ${self.cap:.2f} cap (nominal ${self.used_usd:.2f})")


@dataclass
class Reply:
    probs: dict  # choice: label -> probability; score: label -> {level, probs, confidence}
    input_tokens: int
    model: str


def text_tokens(s: str) -> float:
    k = sum(ord(ch) > 127 for ch in s)
    return TOKENS_PER_CHAR * (len(s) - k) + NON_ASCII_TOKENS * k


def option_tokens(label: str, option: dict) -> float:
    return OPTION_TOKENS + text_tokens(json.dumps({label: option}, ensure_ascii=False))


def estimate(state: dict, instructions: str, options: dict[str, dict]) -> float:
    return (FIXED_TOKENS + text_tokens(json.dumps(state, ensure_ascii=False)) + text_tokens(instructions)
            + sum(option_tokens(k, v) for k, v in options.items()))


def score_estimate(state: dict, questions: dict[str, dict]) -> float:
    return FIXED_TOKENS + text_tokens(json.dumps(state, ensure_ascii=False)) + sum(
        option_tokens(k, v) for k, v in questions.items())


class Pacer:
    """Spaces request starts so requests per minute and estimated tokens per second stay under the limits."""

    def __init__(self, rpm: float = JEV_RPM * 0.9, tps: float = JEV_TPS * 0.8):
        self.rpm, self.tps = rpm, tps
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self, tokens: float) -> None:
        async with self._lock:
            now = time.time()
            start = max(now, self._next)
            self._next = start + max(60.0 / self.rpm, tokens / self.tps)
        await asyncio.sleep(max(0.0, start - time.time()))


class LiveBackend:
    """Shared by every question of a run: one client, one pacer, one concurrency limit."""

    def __init__(self, client, concurrency: int = 20, pacer: Pacer | None = None, model: str = MODEL):
        self.client, self.model = client, model
        self.sem = asyncio.Semaphore(concurrency)
        self.pacer = pacer or Pacer()

    async def _send(self, state: dict, questions: dict, est: float):
        from typesafe_sdk import TypeSafeAPIError

        async with self.sem:
            await self.pacer.wait(est)
            try:
                return await self.client.system_one(state, questions, model=self.model)
            except TypeSafeAPIError as e:
                text = str(e).lower()
                if e.status == 413 or (e.status in (400, 422) and any(h in text for h in OVERFLOW_HINTS)):
                    raise TooLong(str(e)) from e
                raise

    async def ask(self, state: dict, question: dict, est: float) -> Reply:
        resp = await self._send(state, {"q": question}, est)
        ans = resp.choices["q"]
        return Reply({k: ans.probabilities.get(k, 0.0) for k in question["criteria"]}, resp.usage.input_tokens or 0,
                     resp.model)

    async def score(self, state: dict, questions: dict[str, dict], est: float) -> Reply:
        resp = await self._send(state, {k: {"type": "score", **q} for k, q in questions.items()}, est)
        out = {}
        for k, q in questions.items():
            a = resp.answers[k]
            probs = [a.probabilities.get(i, 0.0) for i in range(len(q["criteria"]))]
            out[k] = {"level": a.score, "probs": probs, "confidence": a.confidence}
        return Reply(out, resp.usage.input_tokens or 0, resp.model)


class FakeBackend:
    """Offline stand-in for dry runs: random probabilities; token counts from the estimate."""

    def __init__(self, seed: str = "fake"):
        self.seed = seed

    async def ask(self, state: dict, question: dict, est: float) -> Reply:
        labels = list(question["criteria"])
        rng = random.Random(f"{self.seed}:{len(labels)}:{labels[0]}:{labels[-1]}:{state.get('question', '')[:40]}")
        raw = [rng.choice([0, 0, 0, 1, 1, 5, 20, 20]) for _ in labels]
        if sum(raw) == 0:
            raw[0] = 1
        probs = [round(x / sum(raw), 2) for x in raw]
        return Reply(dict(zip(labels, probs)), round(est), "fake")

    async def score(self, state: dict, questions: dict[str, dict], est: float) -> Reply:
        out = {}
        for k, q in questions.items():
            rng = random.Random(f"{self.seed}:{k}:{state.get('question', '')[:40]}")
            raw = [rng.random() ** 3 for _ in q["criteria"]]
            probs = [x / sum(raw) for x in raw]
            out[k] = {"level": sum(i * p for i, p in enumerate(probs)), "probs": probs, "confidence": max(probs)}
        return Reply(out, round(est), "fake")


class Jev:
    """A policy's handle for one question. `budget` / `score_budget`: estimated tokens per request of each kind."""

    def __init__(self, backend, cache_path: Path | None = None, qid: str = "", search_budget: SearchBudget | None = None):
        self.backend, self.cache_path, self.qid = backend, cache_path, qid
        self.search_budget = search_budget
        self.budget = BUDGET
        self.score_budget = SCORE_BUDGET
        self.requests: list[dict] = []  # {index, kind, tag, options, est_tokens, input_tokens}
        self.input_tokens = 0
        self.sent_requests = 0  # requests sent by this process (not replayed)
        self.overflows = 0
        self._next_index = 0  # taken when a request starts, so concurrent requests get distinct indexes
        self._cache: dict[int, dict] = {}
        if cache_path and cache_path.exists():
            rows = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
            self._cache = {r.get("index", i): r for i, r in enumerate(rows)}
        self._fh = cache_path.open("a") if cache_path else None

    def close(self) -> None:
        if self._fh:
            self._fh.close()

    def fits(self, state: dict, instructions: str, options: dict[str, dict]) -> bool:
        return len(options) <= MAX_CHOICES and estimate(state, instructions, options) <= self.budget

    def groups(self, state: dict, instructions: str, options: list[tuple[str, dict]]) -> list[list[str]]:
        """Labels of `options`, in order, as the fewest contiguous groups that each fit one request, cut near
        equal token shares. An option too long on its own gets a group of its own."""
        fixed = estimate(state, instructions, {})
        costs = [option_tokens(k, v) for k, v in options]
        room = self.budget - fixed
        total = sum(costs)
        k = max(1, -(-len(options) // MAX_CHOICES), -(-int(total) // max(1, int(room))))
        while True:
            groups, cum = [[]], 0.0
            for i, c in enumerate(costs):
                if groups[-1] and len(groups) < k and cum + c / 2 > total * len(groups) / k:
                    groups.append([])
                groups[-1].append(i)
                cum += c
            if k >= len(options) or all(len(g) == 1 or (sum(costs[i] for i in g) <= room and len(g) <= MAX_CHOICES)
                                        for g in groups):
                return [[options[i][0] for i in g] for g in groups]
            k += 1

    def score_fits(self, state: dict, questions: dict[str, dict]) -> bool:
        return len(questions) <= SCORE_MAX_QUESTIONS and score_estimate(state, questions) <= self.score_budget

    def score_groups(self, state: dict, questions: list[tuple[str, dict]]) -> list[list[str]]:
        """Labels of `questions`, in order, packed greedily into requests of at most SCORE_MAX_QUESTIONS
        questions and `score_budget` estimated tokens."""
        fixed = score_estimate(state, {})
        out, cur, tok = [], [], fixed
        for k, q in questions:
            t = option_tokens(k, q)
            if cur and (len(cur) >= SCORE_MAX_QUESTIONS or tok + t > self.score_budget):
                out.append(cur)
                cur, tok = [], fixed
            cur.append(k)
            tok += t
        if cur:
            out.append(cur)
        return out

    async def choice(self, state: dict, instructions: str, options: dict[str, dict], tag: str = "") -> dict[str, float]:
        """One ranking request over `options` (label -> any JSON); a probability per label."""
        question = {"type": "choice", "instructions": instructions, "criteria": options}
        est = estimate(state, instructions, options)
        return await self._request("choice", list(options), est, tag, lambda: self.backend.ask(state, question, est))

    async def score(self, state: dict, questions: dict[str, dict], tag: str = "") -> dict[str, dict]:
        """One score request: `questions` maps a label to {"instructions": str or dict, "criteria": [level 0
        text, level 1 text, ...]}. Returns per label {"level": expected level, "probs": [...], "confidence"}."""
        questions = {k: {"instructions": q["instructions"], "criteria": list(q["criteria"])} for k, q in questions.items()}
        est = score_estimate(state, questions)
        return await self._request("score", list(questions), est, tag, lambda: self.backend.score(state, questions, est))

    async def _request(self, kind: str, labels: list[str], est: float, tag: str, send):
        """Replay the cached response at this request index, or check the spend cap and send."""
        index = self._next_index
        self._next_index += 1
        rec = {"index": index, "kind": kind, "tag": tag, "options": len(labels), "est_tokens": round(est)}
        hit = self._cache.get(index)
        if hit is not None and (hit["labels"] != labels or hit.get("kind", "choice") != kind):
            log.warning("%s: cached response %d does not match the request; dropping the rest", self.qid, index)
            self._cache, hit = {i: r for i, r in self._cache.items() if i < index}, None
        if hit is not None and hit.get("too_long"):
            self.requests.append({**rec, "input_tokens": 0, "too_long": True})
            self._shrink(kind)
            raise TooLong("replayed")
        if self.search_budget is not None:
            self.search_budget.check()
        if hit is not None:
            reply = Reply(dict(zip(labels, hit["probs"] if kind == "choice" else hit["answers"])), hit["input_tokens"],
                          hit["model"])
        else:
            try:
                reply = await send()
            except TooLong:
                self._shrink(kind)
                log.warning("%s: request %d (%s %s) rejected as too long at ~%.0f tokens", self.qid, index, kind, tag, est)
                self._write({"index": index, "kind": kind, "labels": labels, "too_long": True})
                self.requests.append({**rec, "input_tokens": 0, "too_long": True})
                raise
            self.sent_requests += 1
            guard.ATTRIBUTED["jev"] += 1
            answers = {"probs": [reply.probs[k] for k in labels]} if kind == "choice" else {
                "answers": [reply.probs[k] for k in labels]}
            self._write({"index": index, "kind": kind, "labels": labels, **answers,
                         "input_tokens": reply.input_tokens, "model": reply.model})
        self.requests.append({**rec, "input_tokens": reply.input_tokens})
        self.input_tokens += reply.input_tokens
        if self.search_budget is not None:
            self.search_budget.used_tokens += reply.input_tokens
        return reply.probs

    def _shrink(self, kind: str) -> None:
        self.overflows += 1
        if kind == "choice":
            self.budget *= SHRINK
        else:
            self.score_budget *= SHRINK

    def _write(self, record: dict) -> None:
        if self._fh:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
