"""The evolution's state on disk: runs, methods (complete dev evaluations) and the Pareto frontier.

An experiment folder holds runs/<run>/ (one per evaluation), cache/ (LLM completions and memory trees shared by
its runs), frontier.json and LOG.md. A method is a complete run on the dev split; it is on the frontier when no
other method is at least as accurate and at most as costly with one of the two strictly better, and its dev
accuracy is at least the initial policy's minus INIT_FLOOR_MARGIN.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from evolution.config import CHECK_QA_WEIGHT, CONFIRM_SPLIT, FINAL_QA_WEIGHT, INIT_FLOOR_MARGIN, SELECT_SPLIT

EXP: Path = Path(".")  # the experiment folder, set by set_experiment()
RUNS: Path = EXP / "runs"


def set_experiment(path: Path) -> None:
    global EXP, RUNS
    EXP, RUNS = path.resolve(), path.resolve() / "runs"
    RUNS.mkdir(parents=True, exist_ok=True)
    (EXP / "cache").mkdir(exist_ok=True)


def weighted_tokens(a: dict) -> float:
    """The cost axis for one summary row."""
    return (a["search_tokens_per_q"] + CHECK_QA_WEIGHT * a["qa_check_tokens_per_q"]
            + FINAL_QA_WEIGHT * a["qa_final_tokens_per_q"])


def run_info(run: str) -> dict:
    """runs/<run>/iteration.json: what the loop started (kind: init, iteration or confirm) and what happened."""
    try:
        return json.loads((RUNS / run / "iteration.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_info(run: str, info: dict) -> None:
    (RUNS / run / "iteration.json").write_text(json.dumps(info, indent=2))


def complete(run: str) -> bool:
    try:
        return bool(json.loads((RUNS / run / "config.json").read_text()).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def unfinished() -> list[str]:
    """Runs the loop started whose evaluation is neither complete nor failed, oldest first."""
    out = [p.parent.name for p in RUNS.glob("*/iteration.json")
           if not complete(p.parent.name) and not run_info(p.parent.name).get("failed")]
    return sorted(out, key=lambda r: run_info(r).get("started", ""))


def failed_runs(parent: str | None = None) -> list[str]:
    out = [p.parent.name for p in RUNS.glob("*/iteration.json") if run_info(p.parent.name).get("failed")
           and (parent is None or run_info(p.parent.name).get("parent") == parent)]
    return sorted(out, key=lambda r: run_info(r).get("started", ""))


def methods() -> list[dict]:
    """Every complete evaluation with a summary, dev and val alike."""
    out = []
    for p in sorted(RUNS.glob("*/summary.json")):
        try:
            s = json.loads(p.read_text())
            c = json.loads((p.parent / "config.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not c.get("complete") or "all" not in s:
            continue
        a = s["all"]
        out.append({"run": p.parent.name, "split": c["split"], "parent": c.get("parent"), "note": c.get("note", ""),
                    "builder_hash": c["builder_hash"], "retriever_hash": c["retriever_hash"],
                    "n": a["n"], "accuracy": round(a["accuracy"], 4), "tokens_per_q": round(weighted_tokens(a)),
                    "search_tokens_per_q": round(a["search_tokens_per_q"]),
                    "qa_check_tokens_per_q": round(a["qa_check_tokens_per_q"]),
                    "qa_final_tokens_per_q": round(a["qa_final_tokens_per_q"]),
                    "cost_usd": round(a["cost_usd"], 3), "build_cost_usd": round(a.get("build_cost_usd", 0.0), 3),
                    "by_dataset": {d: round(v["accuracy"], 3) for d, v in s.get("by_dataset", {}).items()},
                    "full": not c.get("limit") and not c.get("datasets") and not c.get("qids"),
                    "finished_at": c.get("finished_at", "")})
    return out


def dominated(r: dict, others: list[dict]) -> bool:
    return any(o is not r and o["accuracy"] >= r["accuracy"] and o["tokens_per_q"] <= r["tokens_per_q"]
               and (o["accuracy"] > r["accuracy"] or o["tokens_per_q"] < r["tokens_per_q"]) for o in others)


def confirm_run(run: str, ms: list[dict]) -> str | None:
    """The complete val run of method `run` (runs/<run>_val, or any full val run with the same policy)."""
    me = next((m for m in ms if m["run"] == run), None)
    if me is None:
        return None
    cands = [m for m in ms if m["split"] == CONFIRM_SPLIT and m["full"] and m["builder_hash"] == me["builder_hash"]
             and m["retriever_hash"] == me["retriever_hash"]]
    named = [m for m in cands if m["run"] == f"{run}_{CONFIRM_SPLIT}"]
    return (named or cands or [{"run": None}])[0]["run"]


def update() -> dict:
    """Recompute the frontier from runs/ and write frontier.json."""
    ms = methods()
    sel = [m for m in ms if m["split"] == SELECT_SPLIT and m["full"]]
    children: dict[str, int] = {}
    for m in sel:
        if m["parent"]:
            children[m["parent"]] = children.get(m["parent"], 0) + 1
    init = next((m for m in sel if m["run"] == "init"), None)
    floor = round(init["accuracy"] - INIT_FLOOR_MARGIN, 6) if init else None
    for m in sel:
        m["children"] = children.get(m["run"], 0)
        c = confirm_run(m["run"], ms)
        cm = next((x for x in ms if x["run"] == c), None) if c else None
        m["val_run"] = c
        m["val_accuracy"], m["val_tokens_per_q"] = (cm["accuracy"], cm["tokens_per_q"]) if cm else (None, None)
        m["below_init"] = floor is not None and m["accuracy"] < floor
    kept = [m for m in sel if not m["below_init"]]
    for m in sel:
        m["frontier"] = not m["below_init"] and not dominated(m, kept)
    by_tokens = sorted(sel, key=lambda m: m["tokens_per_q"])
    f = {"experiment": EXP.name, "split": SELECT_SPLIT, "confirm_split": CONFIRM_SPLIT,
         "check_qa_weight": CHECK_QA_WEIGHT, "final_qa_weight": FINAL_QA_WEIGHT,
         "init_accuracy": init["accuracy"] if init else None, "init_floor_margin": INIT_FLOOR_MARGIN,
         "accuracy_floor": floor, "updated": dt.datetime.now().isoformat(timespec="seconds"),
         "frontier": [m["run"] for m in by_tokens if m["frontier"]],
         "confirmed": [m["run"] for m in by_tokens if m["val_run"]],
         "methods": by_tokens}
    (EXP / "frontier.json").write_text(json.dumps(f, indent=2))
    return f


def front_label(m: dict) -> str:
    return "yes" if m["frontier"] else "no (below the floor)" if m.get("below_init") else "no"


def table(f: dict) -> str:
    lines = [f"{'run':16}{'parent':16}{'dev acc':>8}{'tokens/q':>11}{'build$':>8}{'val acc':>9}{'kids':>5}  front  change"]
    for m in f["methods"]:
        va = f"{m['val_accuracy']:.3f}" if m["val_accuracy"] is not None else "-"
        mark = "*" if m["frontier"] else "<flr" if m.get("below_init") else ""
        lines.append(f"{m['run'][:15]:16}{(m['parent'] or '-')[:15]:16}{m['accuracy']:>8.3f}{m['tokens_per_q']:>11,}"
                     f"{m['build_cost_usd']:>8.2f}{va:>9}{m['children']:>5}  {mark:5}  {m['note'][:60]}")
    lines.append(f"tokens/q = search + {CHECK_QA_WEIGHT:g} x intermediate QA + {FINAL_QA_WEIGHT:g} x final QA input tokens"
                 + (f"; <flr: below the accuracy floor {f['accuracy_floor']:.3f}" if f.get("accuracy_floor") is not None else ""))
    if unfinished():
        lines.append(f"unfinished runs: {', '.join(unfinished())}")
    return "\n".join(lines)
