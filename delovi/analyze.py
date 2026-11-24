"""Read a run's episodes: summary metrics and representative trajectories.

    python -m delovi.analyze summary runs/<run>
    python -m delovi.analyze trajectories runs/<run> [--per-group 2] [--seed 0] [--dataset egor1]

Accuracy is the mean per-question score (1/0 for multiple choice, judge score / 5 for open-ended answers).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from delovi.config import CACHE_DIR, ROOT
from delovi.tree import Tree


def episodes(run_dir: Path) -> list[dict]:
    return sorted((json.loads(p.read_text()) for p in (run_dir / "episodes").glob("*.json")), key=lambda r: r["qid"])


def _mean_of(rows: list[dict]):
    return sum(r["score"] for r in rows) / len(rows) if rows else None


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    mean = lambda f: sum(f(r) for r in rows) / n  # noqa: E731
    with_evidence = [r for r in rows if r["evidence"]["spans"]]
    confident = [r for r in rows if str(r.get("stopped", "")).startswith("confidence")]
    return {
        "n": n,
        "accuracy": mean(lambda r: r["score"]),
        "search_tokens_per_q": mean(lambda r: r["search"]["input_tokens"]),
        "qa_tokens_per_q": mean(lambda r: r["qa"]["input_tokens"]),
        "qa_final_tokens_per_q": mean(lambda r: r["qa"]["final_input_tokens"]),
        "qa_check_tokens_per_q": mean(lambda r: r["qa"]["input_tokens"] - r["qa"]["final_input_tokens"]),
        "total_tokens_per_q": mean(lambda r: r["total_input_tokens"]),
        "requests_per_q": mean(lambda r: r["search"]["requests"]),
        "turns_per_q": mean(lambda r: len(r["steps"])),
        "qa_calls_per_q": mean(lambda r: r["qa"]["calls"]),
        "stopped_confident": len(confident) / n,
        "accuracy_when_confident": _mean_of(confident),
        "evidence_hit": (sum(r["evidence"]["any_hit"] for r in with_evidence) / len(with_evidence)) if with_evidence else None,
        "evidence_leaf_hit": (sum(r["evidence"]["leaf_hit"] for r in with_evidence) / len(with_evidence)) if with_evidence else None,
        "accuracy_when_hit": _mean_of([r for r in with_evidence if r["evidence"]["any_hit"]]),
        "accuracy_when_miss": _mean_of([r for r in with_evidence if not r["evidence"]["any_hit"]]),
        "recorded_per_q": mean(lambda r: len(r["recorded"])),
        "leaves_recorded_per_q": mean(lambda r: sum(x["is_leaf"] for x in r["recorded"])),
        "overflows": sum(r["search"]["overflows"] for r in rows),
        "max_request_tokens": max(r["search"]["max_input_tokens"] for r in rows),
        "unparsed_predictions": sum(r["prediction"] is None for r in rows),
        "cost_usd": sum(r["cost_usd"] for r in rows),
        # nominal one-time cost of building the memories these questions used
        "build_cost_usd": sum({r["memory_id"]: (r.get("build") or {}).get("cost_usd") or 0.0 for r in rows}.values()),
        "accuracy_by_type": {t: round(sum(r["score"] for r in rows if r["qtype"] == t) / c, 3)
                             for t, c in sorted(Counter(r["qtype"] for r in rows).items())},
    }


def fill_build(run_dir: Path, rows: list[dict]) -> None:
    """Questions answered while their memory was still streaming have no build record yet; take it from the
    tree cache."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    if "builder_hash" not in cfg:
        return
    cache = CACHE_DIR / ("trees_dry" if cfg.get("dry_run") else "trees") / cfg["builder_hash"]
    records: dict[str, dict] = {}
    for r in rows:
        if (r.get("build") or {}).get("cost_usd") is not None:
            continue
        m = r["memory_id"]
        if m not in records:
            f = cache / (m.replace("/", "__") + ".build.json")
            records[m] = {k: v for k, v in json.loads(f.read_text()).items() if k != "tree"} if f.exists() else {}
        r["build"] = records[m]


def summarize(run_dir: Path) -> dict:
    rows = episodes(run_dir)
    if not rows:
        print("no finished questions")
        return {}
    fill_build(run_dir, rows)
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    s = {"run": run_dir.name, "all": _agg(rows), "by_dataset": {d: _agg(v) for d, v in sorted(by_ds.items())},
         "macro_accuracy": sum(_agg(v)["accuracy"] for v in by_ds.values()) / len(by_ds)}
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
    cols = ("n", "accuracy", "evidence_hit", "turns_per_q", "stopped_confident", "search_tokens_per_q",
            "qa_check_tokens_per_q", "qa_final_tokens_per_q", "total_tokens_per_q", "cost_usd", "build_cost_usd")
    print(f"{'':12}" + "".join(f"{c:>16}" for c in cols))
    for name, a in [*s["by_dataset"].items(), ("all", s["all"])]:
        cells = []
        for c in cols:
            v = a[c]
            cells.append(f"{v:>16}" if isinstance(v, int) else f"{'-':>16}" if v is None else
                         f"{v:>16,.0f}" if v > 100 else f"{v:>16.3f}")
        print(f"{name:12}" + "".join(cells))
    print(f"macro accuracy over datasets {s['macro_accuracy']:.3f}; overflows {s['all']['overflows']}, "
          f"largest request {s['all']['max_request_tokens']:,} tokens, unparsed {s['all']['unparsed_predictions']}")
    return s


def trajectories(run_dir: Path, per_group: int = 2, seed: int = 0, dataset: str | None = None, chars: int = 300) -> str:
    """Episodes grouped by (correct, evidence hit, cost tertile), `per_group` sampled from each with a fixed seed,
    each printed with every step's candidates and the chosen node's text."""
    rows = [r for r in episodes(run_dir) if dataset is None or r["dataset"] == dataset]
    if not rows:
        return "no episodes"
    costs = sorted(r["total_input_tokens"] for r in rows)
    cut = lambda c: "low" if c <= costs[len(costs) // 3] else "mid" if c <= costs[2 * len(costs) // 3] else "high"  # noqa: E731
    groups = defaultdict(list)
    for r in rows:
        hit = r["evidence"]["any_hit"] if r["evidence"]["spans"] else None
        groups[(r["score"] >= 0.5, hit, cut(r["total_input_tokens"]))].append(r)
    rng = random.Random(seed)
    trees: dict[str, Tree] = {}
    out = []
    for key in sorted(groups, key=str):
        picked = rng.sample(groups[key], min(per_group, len(groups[key])))
        out.append(f"\n{'=' * 100}\nGROUP correct={key[0]} evidence_hit={key[1]} cost={key[2]}  "
                   f"({len(groups[key])} episodes, showing {len(picked)})")
        for r in picked:
            tree_path = ROOT / r["tree_snapshot"]
            tree = trees.get(str(tree_path))
            if tree is None and tree_path.exists():
                tree = trees[str(tree_path)] = Tree.load(tree_path)
            out.append(f"\n--- {r['qid']} [{r['qtype']}] score {r['score']:.2f} | pred {r['prediction']!r} | answer {r['answer']!r}")
            out.append(f"Q: {r['question']}")
            if r["options"]:
                out.append("Options: " + " ".join(f"({k}) {v}" for k, v in sorted(r["options"].items())))
            out.append(f"evidence spans (s): {r['evidence']['spans'][:6]}; first hit turn {r['evidence']['first_hit_turn']}; "
                       f"tokens search {r['search']['input_tokens']:,} + QA {r['qa']['input_tokens']:,}; roots {r['search']['roots']}; "
                       f"stopped: {r.get('stopped', '?')}")
            checks = {(c["turn"] if c.get("turn") is not None else c["recorded"]): c for c in r.get("checks", [])}
            for st, rec in zip(r["steps"], r["recorded"]):
                cands = ", ".join(f"{'*' if c == st['chosen'] else ''}{c}:{st['probs'].get(str(c), '-')}" for c in st["candidates"][:12])
                more = f" (+{len(st['candidates']) - 12} more)" if len(st["candidates"]) > 12 else ""
                hit = " <EVIDENCE>" if rec["hits_evidence"] else ""
                text = tree[rec["nid"]].text if tree else ""
                out.append(f"  t{st['turn']:>2} heads[{cands}{more}] -> {rec['span']} ({rec['duration']}, depth {rec['depth']}, "
                           f"{rec['children']} children){hit}\n       {text[:chars]}{'…' if len(text) > chars else ''}")
                c = checks.get(st["turn"])
                if c:
                    out.append(f"  QA after t{st['turn']} ({c['recorded']} nodes): answer {c['prediction']!r}, "
                               f"confidence {c['confidence']}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("summary")
    a.add_argument("run_dir", type=Path)
    b = sub.add_parser("trajectories")
    b.add_argument("run_dir", type=Path)
    b.add_argument("--per-group", type=int, default=2)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--dataset")
    b.add_argument("--chars", type=int, default=300)
    args = p.parse_args()
    if args.cmd == "summary":
        summarize(args.run_dir)
    else:
        print(trajectories(args.run_dir, args.per_group, args.seed, args.dataset, args.chars))


if __name__ == "__main__":
    main()
