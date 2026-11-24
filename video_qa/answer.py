"""Answer a question about a video file with DeLoVi-S, the recursive hierarchical search, or DeLoVi-E.

    python -m video_qa.answer video.mp4 "What does the man put in the oven?"
    python -m video_qa.answer video.mp4 "What color is the car?" --options red blue green white --method delovi_s
    python -m video_qa.answer video.mp4 "What did I just pick up?" --time 00:41:30     # asked at 41:30

The first question on a video builds its memory (video_qa/memory.py), which later questions reuse. Without
--options, the question is open-ended. With --time, only the video before that time is used.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

from delovi import jev as jev_mod
from delovi import qa
from delovi.data import Question
from delovi.jev import FakeBackend, Jev, LiveBackend
from delovi.llm import LLM, FakeLLM, Tracked
from delovi.tree import Tree
from delovi_s import prompts
from delovi_s.run import CLIPS_PER_REQUEST, QA_REASONING, TOP_K
from video_qa.memory import CLIP_SECONDS, build_memory, clock

METHODS = ("delovi_e", "hierarchical_search", "delovi_s")


def seconds(text: str) -> float:
    """'01:02:03', '02:03' or '123' -> seconds."""
    sec = 0.0
    for part in text.split(":"):
        sec = sec * 60 + float(part)
    return sec


def spans(memory, indices: list[int]) -> str:
    """Clip indices as merged time ranges, e.g. '00:01:10-00:01:40, 00:12:00-00:12:10'."""
    out: list[list[float]] = []
    for i in sorted(set(indices)):
        c = memory.clips[i]
        if out and abs(out[-1][1] - c.start) < 1e-6:
            out[-1][1] = c.end
        else:
            out.append([c.start, c.end])
    return ", ".join(f"{clock(s)}-{clock(e)}" for s, e in out)


async def delovi_s(memory, q: Question, backend, llm) -> dict:
    """Jev scores every clip before the question time; the LLM answers from the top TOP_K."""
    cands = [c for c in memory.clips if q.query_time is None or c.end < q.query_time]
    jev = Jev(backend, None, q.qid)
    groups = [cands[i:i + CLIPS_PER_REQUEST] for i in range(0, len(cands), CLIPS_PER_REQUEST)]
    clip_seconds = memory.meta.get("clip_seconds", CLIP_SECONDS)
    replies = await asyncio.gather(*(jev.score(*prompts.score_request("video", q, g, clip_seconds)) for g in groups))
    level = {int(k[1:]): a["level"] for r in replies for k, a in r.items()}
    top = sorted(cands, key=lambda c: (-level[c.index], -c.index))[:TOP_K]
    out = await llm.complete(qa.QA_MODEL, prompts.qa_messages("video", q, memory, top), QA_REASONING)
    return {"answer": prompts.parse_answer("video", q, out.text), "confidence": None,
            "evidence": [c.index for c in top[:10]], "jev_tokens": jev.input_tokens, "llm_tokens": out.input_tokens,
            "llm_cost": out.cost_usd}


async def build_tree(builder, memory, llm, query_time: float | None) -> Tree:
    """Stream the clips through the memory policy; the tree as it stood at the question time."""
    b = builder.MemoryBuilder(memory, llm)
    tree = b.tree
    for clip in memory.clips:
        if query_time is not None and clip.end >= query_time:
            await b.sync()
            tree.checkpoint(query_time)
            return tree.snapshot(query_time)
        await b.add(clip)
    await b.finish()
    tree.finished = True
    if tree.root is None:
        top = tree.orphans()
        tree.set_root(top[0] if len(top) == 1 and not tree[top[0]].is_leaf else tree.make_node(top, ""))
    return tree.snapshot(None)


async def policy(name: str, memory, q: Question, backend, llm) -> dict:
    """A memory policy and a retrieval policy (hierarchical_search/ or delovi_e/)."""
    builder = importlib.import_module(f"{name}.builder")
    retriever = importlib.import_module(f"{name}.retriever")
    build_llm = Tracked(llm)
    tree = await build_tree(builder, memory, build_llm, q.query_time)
    jev = Jev(backend, None, q.qid)
    checks = []

    async def ask(nodes, turn=None):
        out = await llm.complete(qa.QA_MODEL, qa.messages(q, tree, list(nodes)), qa.QA_REASONING)
        checks.append(out)
        return SimpleNamespace(answer=qa.parse(q, out.text), confidence=qa.parse_confidence(out.text),
                               response=out.text)

    res = await retriever.retrieve(q, q.query_time, tree, jev, ask)
    if not checks:
        await ask(res.recorded)
    last = checks[-1]
    return {"answer": qa.parse(q, last.text), "confidence": qa.parse_confidence(last.text),
            "evidence": [n for n in res.recorded if tree[n].is_leaf], "jev_tokens": jev.input_tokens,
            "llm_tokens": sum(c.input_tokens for c in checks), "llm_cost": sum(c.cost_usd for c in checks),
            "build_cost": build_llm.cost_usd}


async def main(args) -> None:
    memory = await build_memory(args.video, args.clip_seconds, args.translate, args.description, args.dry_run)
    options = {chr(ord("A") + i): o for i, o in enumerate(args.options)} if args.options else None
    q = Question("video_qa", "video", memory.memory_id, args.question, options, "",
                 seconds(args.time) if args.time else None, [], "", {})
    if q.query_time is not None and q.query_time >= memory.clips[-1].end:
        q = Question(**{**q.__dict__, "query_time": None})  # asked after the whole video
    client = None
    if args.dry_run:
        backend, llm = FakeBackend(), FakeLLM()
    else:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        client = AsyncTypeSafeClient(model=jev_mod.MODEL, timeout=120.0,
                                     retry=RetryPolicy(max_retries=8, backoff_max=60.0, timeout=600.0))
        backend, llm = LiveBackend(client, 32), LLM(concurrency=32)
    try:
        r = await (delovi_s(memory, q, backend, llm) if args.method == "delovi_s"
                   else policy(args.method, memory, q, backend, llm))
    finally:
        if client is not None:
            await client.__aexit__(None, None, None)
    answer = r["answer"]
    if options and answer in options:
        answer = f"({answer}) {options[answer]}"
    cost = r["jev_tokens"] / 1e6 * jev_mod.PRICE_PER_MTOK + r["llm_cost"]
    print(f"\nQuestion: {q.question}")
    print(f"Answer ({args.method}): {answer}" + (f"  [confidence {r['confidence']}]" if r["confidence"] is not None else ""))
    print(f"Evidence: {spans(memory, r['evidence'])}")
    print(f"Input tokens: {r['jev_tokens']:,} Jev + {r['llm_tokens']:,} LLM; cost ${cost:.4f}"
          + (f" (+ ${r['build_cost']:.4f} nominal to build the memory tree, cached)" if "build_cost" in r else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("question")
    p.add_argument("--options", nargs="+", help="answer options; they are labeled A, B, C, ...")
    p.add_argument("--method", default="delovi_e", choices=METHODS)
    p.add_argument("--time", help="question time, e.g. 00:41:30 (default: after the whole video)")
    p.add_argument("--clip-seconds", type=int, default=CLIP_SECONDS)
    p.add_argument("--translate", action="store_true", help="transcribe the speech into English")
    p.add_argument("--description", help="what the video is, shown to the models (default: its length)")
    p.add_argument("--dry-run", action="store_true", help="fake models and fake captions; no API calls")
    asyncio.run(main(p.parse_args()))
