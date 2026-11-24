"""DeLoVi-S: Jev scores every clip caption before the question time, and the LLM answers from the top K.

    python -m delovi_s.run --split egolifeqa --run-name delovi_s_egolifeqa
    python -m delovi_s.run --split egolifeqa --run-name delovi_s_egolifeqa_gpt5 --qa-model gpt-5 \
        --scores-from delovi_s_egolifeqa          # another LLM on the same rankings (no new Jev requests)
    python -m delovi_s.run --split egolifeqa --run-name dry --limit 2 --dry-run   # offline

For each question, the candidates are the clips that ended before the question time (all clips when the
question has none). Jev scores them in requests of CLIPS_PER_REQUEST consecutive clips; a clip's score is the
expected level of a 4-level rubric (prompts.py). Clips are ranked by score, ties to the later clip, and the LLM
answers once from the top K. Output in runs/<run>/: scores/<qid>.json (the ranking), episodes/<qid>.json,
summary.json. Rankings and LLM completions are cached, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from delovi import grade
from delovi import jev as jev_mod
from delovi.config import CONCURRENCY, PARALLEL, QA_MODEL, RUNS_DIR
from delovi.data import load_memory, load_questions
from delovi.jev import FakeBackend, Jev, LiveBackend
from delovi.llm import LLM, FakeLLM
from delovi_s import prompts

# Hyper-parameters
TOP_K = 100  # clips the LLM reads
CLIPS_PER_REQUEST = 64  # clips per score request
QA_REASONING = None  # the QA model's default reasoning effort

log = logging.getLogger("delovi_s")


def key(qid: str) -> str:
    return qid.replace("/", "__")


async def rank(split: str, q, memory, backend, run_dir: Path) -> dict:
    """The question's ranking of its candidate clips: {"ranking": [[clip index, score], ...], tokens}."""
    path = run_dir / "scores" / f"{key(q.qid)}.json"
    if path.exists():
        return json.loads(path.read_text())
    cands = [c for c in memory.clips if q.query_time is None or c.end < q.query_time]
    groups = [cands[i:i + CLIPS_PER_REQUEST] for i in range(0, len(cands), CLIPS_PER_REQUEST)]
    jev = Jev(backend, path.with_suffix(".jev.jsonl"), q.qid)
    try:
        replies = await asyncio.gather(*(jev.score(*prompts.score_request(split, q, g), tag="score") for g in groups))
    finally:
        jev.close()
    level = {int(k[1:]): a["level"] for r in replies for k, a in r.items()}
    ranking = sorted(cands, key=lambda c: (-level[c.index], -c.index))
    out = {"qid": q.qid, "candidates": len(cands), "requests": len(jev.requests), "input_tokens": jev.input_tokens,
           "ranking": [[c.index, level[c.index]] for c in ranking]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    path.with_suffix(".jev.jsonl").unlink(missing_ok=True)
    return out


def hits(clip, q) -> bool:
    return any(clip.start <= e and clip.end >= s for s, e in q.evidence)


async def run_question(split: str, q, memory, backend, llm, args, run_dir: Path) -> dict:
    t0 = time.monotonic()
    ranked = await rank(split, q, memory, backend, args.runs_dir / (args.scores_from or args.run_name))
    top = [memory.clips[i] for i, _ in ranked["ranking"][:args.k]]
    out = await llm.complete(args.qa_model, prompts.qa_messages(split, q, memory, top), QA_REASONING)
    prediction = prompts.parse_answer(split, q, out.text)
    score, detail = await grade.grade(q, prediction, llm)
    jev_cost = ranked["input_tokens"] / 1e6 * jev_mod.PRICE_PER_MTOK
    ep = {"qid": q.qid, "dataset": q.dataset, "qtype": q.qtype, "memory_id": q.memory_id, "question": q.question,
          "options": q.options, "answer": q.answer, "prediction": prediction, "score": score, "response": out.text,
          "grading": detail, "k": args.k, "candidates": ranked["candidates"],
          "evidence_hit": any(hits(c, q) for c in top) if q.evidence else None,
          "search": {"requests": ranked["requests"], "input_tokens": ranked["input_tokens"], "cost_usd": jev_cost},
          "qa": {"model": out.model, "input_tokens": out.input_tokens, "output_tokens": out.output_tokens,
                 "reasoning_tokens": out.reasoning_tokens, "cost_usd": out.cost_usd, "cached": out.cached},
          "top": [c.index for c in top], "seconds": round(time.monotonic() - t0, 1)}
    (run_dir / "episodes" / f"{key(q.qid)}.json").write_text(json.dumps(ep, ensure_ascii=False, indent=1))
    log.info("%s: %s vs %s (%.2f), %d candidates, %s Jev tokens, %.0fs", q.qid, prediction, q.answer[:40], score,
             ranked["candidates"], f"{ranked['input_tokens']:,}", ep["seconds"])
    return ep


def summarize(run_dir: Path) -> dict:
    eps = [json.loads(p.read_text()) for p in sorted((run_dir / "episodes").glob("*.json"))]
    by: dict[str, list] = {}
    for e in eps:
        by.setdefault(e["dataset"], []).append(e)

    def stats(es):
        n = len(es)
        hit = [e["evidence_hit"] for e in es if e["evidence_hit"] is not None]
        return {"n": n, "accuracy": round(sum(e["score"] for e in es) / n, 4),
                "evidence_hit": round(sum(hit) / len(hit), 4) if hit else None,
                "candidates_per_q": round(sum(e["candidates"] for e in es) / n),
                "jev_tokens_per_q": round(sum(e["search"]["input_tokens"] for e in es) / n),
                "qa_input_tokens_per_q": round(sum(e["qa"]["input_tokens"] for e in es) / n),
                "jev_cost_usd": round(sum(e["search"]["cost_usd"] for e in es), 4),
                "qa_cost_usd": round(sum(e["qa"]["cost_usd"] for e in es), 4),
                "unparsed": sum(e["prediction"] is None for e in es)}

    s = {"run": run_dir.name, "n": len(eps), "by_dataset": {d: stats(es) for d, es in sorted(by.items())}}
    if eps:
        s["all"] = stats(eps)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
    return s


async def main(args) -> None:
    run_dir = args.runs_dir / args.run_name
    for sub in ("scores", "episodes"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(run_dir / "run.log"))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(jev_model=jev_mod.MODEL, clips_per_request=CLIPS_PER_REQUEST, qa_reasoning=QA_REASONING)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    qs = load_questions(args.split, qids=args.qids, limit=args.limit)
    todo = [q for q in qs if not (run_dir / "episodes" / f"{key(q.qid)}.json").exists()]
    log.info("%d questions, %d to go", len(qs), len(todo))
    if args.dry_run:
        backend, llm, client = FakeBackend(), FakeLLM(), None
    else:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        client = AsyncTypeSafeClient(model=jev_mod.MODEL, timeout=300.0,
                                     retry=RetryPolicy(max_retries=8, backoff_max=60.0, timeout=600.0))
        backend, llm = LiveBackend(client, args.concurrency), LLM(concurrency=args.concurrency)
    memories: dict = {}
    qsem = asyncio.Semaphore(args.parallel)
    failed = []

    async def one(q):
        async with qsem:
            if q.memory_id not in memories:
                memories[q.memory_id] = load_memory(q.memory_id)
            try:
                await run_question(args.split, q, memories[q.memory_id], backend, llm, args, run_dir)
            except Exception as e:  # noqa: BLE001 -- one failed question must not stop the run
                log.exception("%s failed: %r; rerun to resume", q.qid, e)
                failed.append(q.qid)

    try:
        await asyncio.gather(*(one(q) for q in todo))
    finally:
        if client is not None:
            await client.__aexit__(None, None, None)
    if failed:
        log.error("%d questions failed (rerun the same command to resume)", len(failed))
    print(json.dumps(summarize(run_dir), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=prompts.SPLITS)
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--k", type=int, default=TOP_K, help="clips the LLM reads")
    p.add_argument("--qa-model", default=QA_MODEL)
    p.add_argument("--scores-from", help="reuse the rankings of this run")
    p.add_argument("--qids", type=lambda s: s.split(","))
    p.add_argument("--limit", type=int, help="first N questions of each dataset")
    p.add_argument("--parallel", type=int, default=PARALLEL // 2, help="questions at once")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY * 3, help="Jev requests in flight")
    p.add_argument("--dry-run", action="store_true", help="fake Jev and fake LLM; no API calls")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "typesafe_sdk", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main(args))
