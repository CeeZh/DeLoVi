"""Re-answer a finished policy run with another LLM on the same final context (no new search).

    python scripts/requery.py --run runs/delovi_e_egolifeqa --model gpt-5
    python scripts/requery.py --run runs/delovi_e_egolifeqa --model gpt-6.1-sol

For every question, the nodes of the last QA call of the run are given to --model with the same QA prompt
(delovi/qa.py), at the model's default reasoning effort. The output is a new run, runs/<run>_<model>/, with the
same episode format; multiple-choice answers are scored by exact match, open-ended answers are left for
scripts/grade_mmlifelong.py. For DeLoVi-S, use `python -m delovi_s.run --qa-model <model> --scores-from <run>`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from delovi import qa  # noqa: E402
from delovi.config import ROOT  # noqa: E402
from delovi.data import load_questions  # noqa: E402
from delovi.llm import LLM  # noqa: E402
from delovi.tree import Tree  # noqa: E402


async def main(args) -> None:
    config = json.loads((args.run / "config.json").read_text())
    out_dir = args.run.parent / f"{args.run.name}_{args.model}"
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"split": config["split"], "requery_of": args.run.name,
                                                     "model": args.model}, indent=2))
    questions = {q.qid: q for q in load_questions(config["split"])}
    llm = LLM(concurrency=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)
    trees: dict[str, Tree] = {}

    async def one(path: Path) -> dict:
        out = out_dir / "episodes" / path.name
        if out.exists():
            return json.loads(out.read_text())
        e = json.loads(path.read_text())
        q = questions[e["qid"]]
        if e["tree_snapshot"] not in trees:
            trees[e["tree_snapshot"]] = Tree.load(ROOT / e["tree_snapshot"])
        nodes = e["checks"][-1]["nodes"]
        async with sem:
            c = await llm.complete(args.model, qa.messages(q, trees[e["tree_snapshot"]], nodes), None)
        pred = qa.parse(q, c.text)
        r = {"qid": q.qid, "dataset": q.dataset, "qtype": q.qtype, "prediction": pred, "answer": q.answer,
             "score": float(pred == q.answer) if q.multiple_choice else None, "response": c.text, "model": c.model,
             "input_tokens": c.input_tokens, "output_tokens": c.output_tokens, "reasoning_tokens": c.reasoning_tokens,
             "cost_usd": c.cost_usd, "base_prediction": e["prediction"], "base_score": e["score"]}
        out.write_text(json.dumps(r, ensure_ascii=False))
        return r

    rows = await asyncio.gather(*(one(p) for p in sorted((args.run / "episodes").glob("*.json"))))
    mc = [r for r in rows if r["score"] is not None]
    summary = {"run": out_dir.name, "split": config["split"], "model": args.model, "n": len(rows),
               "accuracy": sum(r["score"] for r in mc) / len(mc) if mc else None,
               "base_accuracy": sum(r["base_score"] for r in mc) / len(mc) if mc else None,
               "input_tokens_per_q": sum(r["input_tokens"] for r in rows) / len(rows) if rows else 0,
               "cost_usd": sum(r["cost_usd"] for r in rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="a finished run of delovi.evaluate")
    p.add_argument("--model", required=True, help="e.g. gpt-5 or gpt-6.1-sol")
    p.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main(p.parse_args()))
