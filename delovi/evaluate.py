"""Evaluate a policy (a folder with builder.py and retriever.py) on a question split.

    python -m delovi.evaluate --policy delovi_e --split egolifeqa --run-name delovi_e_egolifeqa
    python -m delovi.evaluate --policy delovi_e --split dev --run-name check --limit 1 --dry-run   # offline

Questions are grouped by memory and sorted by query time. Each memory is streamed once through the memory
policy, clip by clip; when the stream reaches a question's time, the tree is checkpointed and the question is
answered on a snapshot of the tree at that moment, while the stream goes on. Building stops after the last
question's time. Built trees are cached by the hash of builder.py; LLM completions and Jev responses are cached
too, so an interrupted run resumes where it stopped.

Output in runs/<run>/: config.json ("complete": true once every question is done), policy/ (a copy of the
policy), episodes/<qid>.json (prediction, score, tokens, every QA call and search step), summary.json, run.log.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib
import json
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from delovi import analyze, grade, guard, qa
from delovi import jev as jev_mod
from delovi.config import BUILD_CONCURRENCY, CACHE_DIR, CONCURRENCY, PARALLEL, ROOT, RUNS_DIR
from delovi.data import load_memory, load_questions
from delovi.jev import FakeBackend, Jev, LiveBackend, SearchBudget
from delovi.llm import LLM, BuildBudget, FakeLLM, Tracked
from delovi.tree import Tree, validate_tree

log = logging.getLogger("delovi")
POLICY_FILES = ("builder.py", "retriever.py")


def file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def load_policy(folder: Path):
    """Import builder.py and retriever.py of the policy folder as a package."""
    sys.path.insert(0, str(folder.parent))
    return (importlib.import_module(f"{folder.name}.builder"), importlib.import_module(f"{folder.name}.retriever"))


class MemoryStream:
    """One memory evaluated in streaming order. Each question is answered on a snapshot of the tree taken when
    the stream reaches its time (saved as <memory>/at_<time>.json, or full.json for questions without a time).
    The whole tree and its snapshots are cached per builder hash; when every needed snapshot exists the memory
    is not streamed again, otherwise it is rebuilt for the union of times (LLM calls come from the cache)."""

    def __init__(self, builder, memory_id: str, questions: list, llm, cache_dir: Path, budget: BuildBudget | None):
        self.builder, self.memory_id, self.llm, self.cache_dir, self.budget = builder, memory_id, llm, cache_dir, budget
        self.spent = 0.0  # USD this run paid to build this memory
        self.questions = sorted(questions, key=lambda q: (q.query_time is None, q.query_time or 0.0))
        self.path = cache_dir / (memory_id.replace("/", "__") + ".json")
        self.build: dict = {}
        self.snapshots: dict[float | None, Tree] = {}

    def snapshot_path(self, query_time: float | None) -> Path:
        name = "full" if query_time is None else f"at_{query_time!r}"
        return self.path.with_suffix("") / f"{name}.json"

    def take(self, tree: Tree, query_time: float | None) -> Tree:
        if query_time not in self.snapshots:
            snap = self.snapshots[query_time] = tree.snapshot(query_time)
            snap.save(self.snapshot_path(query_time))
        return self.snapshots[query_time]

    def cached(self, times: set) -> bool:
        build = self.path.with_suffix(".build.json")
        if not build.exists() or not all(self.snapshot_path(t).exists() for t in times):
            return False
        self.build = json.loads(build.read_text())
        self.snapshots = {t: Tree.load(self.snapshot_path(t)) for t in times}
        return True

    async def run(self, answer) -> None:
        """Await `answer(question, tree)` for every question while the stream goes on. A file lock per memory
        and builder lets processes that share the cache build each memory once."""
        lock = open(self.path.with_suffix(".lock"), "w")
        try:
            await asyncio.to_thread(fcntl.flock, lock, fcntl.LOCK_EX)
            tasks = await self._build(answer)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        await asyncio.gather(*tasks)

    async def _build(self, answer) -> list:
        if self.cached({q.query_time for q in self.questions}):
            return [asyncio.ensure_future(answer(q, self.snapshots[q.query_time])) for q in self.questions]
        memory = load_memory(self.memory_id)
        need = sorted({q.query_time for q in self.questions if q.query_time is not None})
        need_full = any(q.query_time is None for q in self.questions)
        old = Tree.load(self.path) if self.path.exists() else None  # keep the checkpoints of an earlier build
        cutoffs = sorted(set(need) | (set(old.checkpoints) if old else set()))
        t0, llm = time.monotonic(), Tracked(self.llm, self.budget)
        b = self.builder.MemoryBuilder(memory, llm)
        tasks: list = []
        try:
            return await self._stream(b, b.tree, memory, cutoffs, need_full, llm, t0, tasks, answer)
        except BaseException:
            for t in tasks:  # answers on a failed build do not count
                t.cancel()
            raise

    def _check_data(self, tree: Tree, memory) -> None:
        if tree.description != memory.description or tree.timeline != memory.timeline:
            raise ValueError(f"{self.memory_id}: the memory policy changed the recording's description or timeline")

    async def _stream(self, b, tree: Tree, memory, cutoffs: list, need_full: bool, llm, t0: float, tasks: list,
                      answer) -> list:
        qi, fed = 0, 0
        self._check_data(tree, memory)
        sync = getattr(b, "sync", None)
        for clip in memory.clips:
            if cutoffs and clip.end >= cutoffs[0]:
                if sync is not None:
                    await sync()  # a question arrives: the memory policy makes the nodes it has decided on
                self._check_data(tree, memory)
            while cutoffs and clip.end >= cutoffs[0]:
                tree.checkpoint(cutoffs.pop(0))
            while qi < len(self.questions) and self.questions[qi].query_time in tree.checkpoints:
                q = self.questions[qi]
                tasks.append(asyncio.ensure_future(answer(q, self.take(tree, q.query_time))))
                qi += 1
            if not cutoffs and not need_full:
                break  # nothing after the last question's time is needed
            await b.add(clip)
            fed += 1
        if fed == len(memory.clips):
            await b.finish()
            tree.finished = True
        for cutoff in cutoffs:
            tree.checkpoint(cutoff)
        for q in self.questions[qi:]:  # questions after the last clip see the tree before the root is added
            if q.query_time is not None:
                self.take(tree, q.query_time)
        tree.clips_fed = fed
        if tree.root is None:
            top = tree.orphans()
            if len(top) == 1 and not tree[top[0]].is_leaf:
                tree.set_root(top[0])
            else:
                tree.set_root(tree.make_node(top, ""))
        validate_tree(tree, memory.clips[:fed])
        self._check_data(tree, memory)
        self.spent = llm.spent_usd
        tree.save(self.path)
        self.build = {"memory_id": self.memory_id, "clips_fed": fed, "clips": len(memory.clips), "finished": tree.finished,
                      "checkpoints": len(tree.checkpoints), "seconds": round(time.monotonic() - t0, 1),
                      "tree": tree.stats(), **llm.stats()}
        self.path.with_suffix(".build.json").write_text(json.dumps(self.build, indent=2))
        log.info("built %s: %d/%d clips, %d nodes, %.0fs, %d LLM calls (%d cached), $%.3f", self.memory_id, fed,
                 len(memory.clips), len(tree), self.build["seconds"], llm.calls, llm.cached_calls, llm.cost_usd)
        tasks.extend(asyncio.ensure_future(answer(q, self.take(tree, q.query_time))) for q in self.questions[qi:])
        return tasks


def overlaps(node, spans: list[list[float]]) -> bool:
    return any(node.start <= b and a <= node.end for a, b in spans)


async def run_question(q, tree: Tree, stream: MemoryStream, retriever, backend, llm, args, out_dir: Path,
                       search_budget: SearchBudget | None = None) -> dict | None:
    final = out_dir / f"{q.qid.replace('/', '__')}.json"
    cache = final.with_suffix(".jev.jsonl")
    t0 = time.monotonic()
    try:
        jev = Jev(backend, None if args.dry_run else cache, q.qid, search_budget)
        checks: list[dict] = []

        async def ask(nodes, turn: int | None = None) -> SimpleNamespace:
            """The fixed QA step on any nodes of the question's tree; returns answer, confidence and response."""
            nodes = [int(n) for n in nodes]
            out = await llm.complete(qa.QA_MODEL, qa.messages(q, tree, nodes), qa.QA_REASONING)
            c = {"turn": turn, "nodes": nodes, "recorded": len(nodes), "prediction": qa.parse(q, out.text),
                 "confidence": qa.parse_confidence(out.text), "response": out.text, "model": out.model,
                 "input_tokens": out.input_tokens, "output_tokens": out.output_tokens,
                 "reasoning_tokens": out.reasoning_tokens, "cost_usd": out.cost_usd, "cached": out.cached}
            checks.append(c)
            return SimpleNamespace(answer=c["prediction"], confidence=c["confidence"], response=out.text)

        try:
            res = await retriever.retrieve(q, q.query_time, tree, jev, ask)
        finally:
            jev.close()
        if not checks:  # a policy that never asked: one QA call on everything it recorded
            await ask(res.recorded, turn=len(res.steps))
        last = checks[-1]
        pred = last["prediction"]
        score, detail = await grade.grade(q, pred, llm)
    except Exception as e:  # noqa: BLE001 -- one failed question must not stop the run
        log.exception("%s failed: %r; rerun to resume", q.qid, e)
        return None

    view = tree.view(q.query_time)
    by_tag, tokens_by_tag = Counter(), Counter()
    for r in jev.requests:
        by_tag[r["tag"]] += 1
        tokens_by_tag[r["tag"]] += r["input_tokens"]
    recorded = []
    for s in res.steps:
        n = tree[s.nid]
        recorded.append({"turn": s.turn, "nid": s.nid, "span": tree.span_label(n), "duration": tree.duration_label(n),
                         "seconds": n.duration, "depth": tree.depth(s.nid), "is_leaf": n.is_leaf, "chars": len(n.text),
                         "list_id": s.list_id, "n_candidates": len(s.candidates), "prob": s.probs.get(s.nid),
                         "children": s.children, "hits_evidence": bool(q.evidence) and overlaps(n, q.evidence)})
    hits = [r for r in recorded if r["hits_evidence"]]
    qa_in = sum(c["input_tokens"] for c in checks)
    qa_cost = sum(c["cost_usd"] for c in checks)
    jev_cost = jev.input_tokens / 1e6 * jev_mod.PRICE_PER_MTOK
    result = {
        "qid": q.qid, "dataset": q.dataset, "qtype": q.qtype, "memory_id": q.memory_id, "question": q.question,
        "options": q.options, "answer": q.answer, "query_time": q.query_time,
        "prediction": pred, "confidence": last["confidence"], "score": score, "response": last["response"],
        "turns": len(res.steps), "stopped": res.stop or "?",
        "checks": [{**{k: v for k, v in c.items() if k != "response"}, "response": c["response"][:500]} for c in checks],
        "search": {"requests": len(jev.requests), "by_tag": dict(by_tag), "input_tokens": jev.input_tokens,
                   "input_tokens_by_tag": dict(tokens_by_tag), "overflows": jev.overflows, "lists": res.lists,
                   "max_input_tokens": max((r["input_tokens"] for r in jev.requests), default=0),
                   "roots": len(view.roots())},
        "qa": {"model": last["model"], "calls": len(checks), "input_tokens": qa_in,
               "output_tokens": sum(c["output_tokens"] for c in checks),
               "reasoning_tokens": sum(c["reasoning_tokens"] for c in checks), "cost_usd": qa_cost,
               "final_input_tokens": last["input_tokens"],
               "context_chars": sum(len(tree[n].text) for n in last["nodes"])},
        "grading": detail,
        "total_input_tokens": jev.input_tokens + qa_in,
        "cost_usd": jev_cost + qa_cost + detail.get("cost_usd", 0.0),
        "spent_usd": 0.0 if args.dry_run else (  # what this run paid (cached calls are free)
            jev_cost + sum(c["cost_usd"] for c in checks if not c["cached"])
            + (0.0 if detail.get("cached") else detail.get("cost_usd", 0.0))),
        "evidence": {"spans": q.evidence, "any_hit": bool(hits), "leaf_hit": any(r["is_leaf"] for r in hits),
                     "first_hit_turn": hits[0]["turn"] if hits else None},
        "recorded": recorded,
        "steps": [{"turn": s.turn, "chosen": s.nid, "candidates": s.candidates,
                   "probs": {str(k): v for k, v in s.probs.items()}} for s in res.steps],
        "tree": view.stats(),
        "tree_snapshot": str(stream.snapshot_path(q.query_time).relative_to(ROOT)),
        "build": {k: v for k, v in stream.build.items() if k != "tree"},
        "requests": jev.requests,
        "seconds": round(time.monotonic() - t0, 1),
    }
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False))
    tmp.replace(final)
    cache.unlink(missing_ok=True)
    log.info("%s score %.2f | %d steps (%s), %d requests, %d QA calls, %s search + %s QA tokens, evidence %s | "
             "pred %r conf %s ans %r | $%.3f %.0fs", q.qid, score, len(res.steps), result["stopped"],
             len(jev.requests), len(checks), f"{jev.input_tokens:,}", f"{qa_in:,}", "hit" if hits else "miss",
             pred, last["confidence"], q.answer[:40], result["cost_usd"], result["seconds"])
    return result


def snapshot_policy(policy: Path, run_dir: Path) -> None:
    """Copy the policy into the run; a resumed run must have the same policy files."""
    snap = run_dir / "policy"
    if snap.exists():
        for f in POLICY_FILES:
            if (snap / f).read_bytes() != (policy / f).read_bytes():
                sys.exit(f"{run_dir.name} was evaluated with a different {f}; use a new --run-name")
        return
    snap.mkdir(parents=True)
    for f in (*POLICY_FILES, "__init__.py"):
        if (policy / f).exists():
            shutil.copy2(policy / f, snap / f)


async def run(args) -> None:
    policy = args.policy.resolve()
    run_dir = args.runs_dir / args.run_name
    out_dir = run_dir / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_policy(policy, run_dir)
    logging.getLogger().addHandler(logging.FileHandler(run_dir / "run.log"))
    problems = guard.check_policy(sorted(policy.glob("*.py")))
    if problems:
        sys.exit(f"PolicyCheckError: {'; '.join(problems)} ({guard.RULE})")
    try:  # violations of an earlier attempt stay
        prior = json.loads((run_dir / "config.json").read_text()).get("violations", [])
    except (OSError, json.JSONDecodeError):
        prior = []

    guard.install()
    if args.dry_run:
        backend, client = FakeBackend(), None
        llm = build_llm = FakeLLM()
    else:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        client = AsyncTypeSafeClient(model=jev_mod.MODEL, timeout=120.0,
                                     retry=RetryPolicy(max_retries=8, backoff_max=60.0, timeout=600.0))
        backend, llm = LiveBackend(client, args.concurrency), guard.Counted(LLM(concurrency=args.concurrency))
        build_llm = guard.Counted(LLM(concurrency=args.build_concurrency))
    guard.remove_keys()  # the clients have read them; the policy is imported after this
    builder, retriever = load_policy(policy)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["builder_hash"], config["retriever_hash"] = file_hash(policy / "builder.py"), file_hash(policy / "retriever.py")
    config["violations"] = prior
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    questions = load_questions(args.split, args.datasets, args.qids, args.limit)
    todo = [q for q in questions if not (out_dir / f"{q.qid.replace('/', '__')}.json").exists()]
    log.info("%d questions, %d done, %d to go; builder %s, retriever %s", len(questions), len(questions) - len(todo),
             len(todo), config["builder_hash"], config["retriever_hash"])

    cache_dir = CACHE_DIR / ("trees_dry" if args.dry_run else "trees") / config["builder_hash"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    qsem = asyncio.Semaphore(args.parallel)
    budget = BuildBudget(args.build_budget)
    done_tokens = 0  # Jev tokens of questions an earlier attempt finished count toward the cap
    for p in out_dir.glob("*.json"):
        try:
            done_tokens += json.loads(p.read_text())["search"]["input_tokens"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    search_budget = SearchBudget(args.search_budget, done_tokens)
    streams: list[MemoryStream] = []
    done, failed, t0 = {"n": 0, "cost": 0.0, "spent": 0.0}, [], time.monotonic()
    by_memory: dict[str, list] = {}
    for q in todo:
        by_memory.setdefault(q.memory_id, []).append(q)

    async def one_memory(memory_id, qs):
        stream = MemoryStream(builder, memory_id, qs, build_llm, cache_dir, budget)
        streams.append(stream)

        async def answer_one(q, tree):
            async with qsem:
                r = await run_question(q, tree, stream, retriever, backend, llm, args, out_dir, search_budget)
            if r is None:
                failed.append(q.qid)
                return
            done["n"] += 1
            done["cost"] += r["cost_usd"]
            done["spent"] += r["spent_usd"]
            el = time.monotonic() - t0
            log.info("[%d/%d] $%.2f so far, %.1f min elapsed, ETA %.1f min", done["n"], len(todo), done["cost"],
                     el / 60, el / done["n"] * (len(todo) - done["n"]) / 60)

        try:
            await stream.run(answer_one)
        except Exception as e:  # noqa: BLE001 -- a failed build must not stop the other memories
            log.exception("memory %s failed: %r; rerun to resume", memory_id, e)
            for q in qs:  # every answer on this memory is void
                (out_dir / f"{q.qid.replace('/', '__')}.json").unlink(missing_ok=True)
                failed.append(q.qid)

    try:
        await asyncio.gather(*(one_memory(m, qs) for m, qs in by_memory.items()))
    finally:
        if client is not None:
            await client.__aexit__(None, None, None)
    if failed:
        log.error("incomplete questions (rerun the same command to resume): %s", ", ".join(failed))
    violations = list(prior)
    if search_budget.exceeded:
        violations.append(f"SearchBudgetError: Jev search reached the ${args.search_budget:.2f} cap "
                          f"(nominal ${search_budget.used_usd:.2f})")
    if not args.dry_run and not failed:
        mismatch = guard.reconcile()
        if mismatch:
            violations.append(f"AccountingError: {mismatch}")
    for v in violations[len(prior):]:
        log.error("%s", v)
    analyze.summarize(run_dir)
    config["violations"] = violations
    config["search_nominal_usd"] = round(search_budget.used_usd, 4)
    config["complete"] = not failed and not violations
    config["build_spent_usd"] = round(sum(st.spent for st in streams), 4)
    config["spent_usd"] = round(done["spent"] + config["build_spent_usd"], 4)  # this invocation's real API spend
    config["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", type=Path, required=True, help="folder with builder.py and retriever.py")
    p.add_argument("--split", required=True, help="a file in data/questions/, e.g. dev, val, egolifeqa")
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--datasets", type=lambda s: s.split(","), help="comma-separated datasets to keep")
    p.add_argument("--qids", type=lambda s: s.split(","))
    p.add_argument("--limit", type=int, help="first N questions of each dataset")
    p.add_argument("--build-budget", type=float, help="cap in USD on the nominal cost of building the memories")
    p.add_argument("--search-budget", type=float, help="cap in USD on the nominal Jev cost of all questions")
    p.add_argument("--parent", help="run this policy was derived from (recorded)")
    p.add_argument("--note", default="", help="what changed in the policy (recorded)")
    p.add_argument("--parallel", type=int, default=PARALLEL, help="questions answered at once")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY, help="Jev requests and QA calls in flight")
    p.add_argument("--build-concurrency", type=int, default=BUILD_CONCURRENCY, help="memory-building LLM calls in flight")
    p.add_argument("--dry-run", action="store_true", help="fake Jev and fake LLM; no API calls")
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "typesafe_sdk", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()
    if args.summary_only:
        analyze.summarize(args.runs_dir / args.run_name)
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
