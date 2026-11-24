"""A small live check of the policy in this workspace, for the coding agent.

    python probe.py <qid> [<qid> ...]            # dev questions, with the real models
    python probe.py <qid> ... --dry-run          # the same with fake models, no cost

Building a memory up to a question's time is part of the probe, so questions early in a long recording or on a
short video are the cheap ones. All probes of one iteration share the budget in scratch/probe_ledger.json; the
script refuses once it is spent. Results go to scratch/probes/probe_<k>/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "scratch" / "probe_ledger.json"
OUT = HERE / "scratch" / "probes"
SPLIT = "dev"


def main() -> None:
    dry = "--dry-run" in sys.argv[1:]
    qids = [a for a in sys.argv[1:] if a != "--dry-run"]
    if not qids:
        sys.exit("usage: python probe.py <qid> [<qid> ...] [--dry-run]")
    known = {q["qid"] for q in json.loads((HERE / "data" / "questions" / f"{SPLIT}.json").read_text())}
    unknown = [q for q in qids if q not in known]
    if unknown:
        sys.exit(f"probe: not {SPLIT} questions: {', '.join(unknown)}")
    if not dry and not (os.environ.get("OPENAI_API_KEY") and os.environ.get("TYPESAFE_API_KEY")):
        sys.exit("probe: no API keys in the environment, so live checks are unavailable; use the dry run")
    book = json.loads(LEDGER.read_text())
    left = book["budget_usd"] - book["spent_usd"]
    if left <= 0.01:
        sys.exit(f"probe: the ${book['budget_usd']:.2f} budget of this iteration is spent")
    name = f"probe_{len(book['probes']) + 1}"
    run_dir = OUT / name
    cmd = [sys.executable, "-m", "delovi.evaluate", "--policy", "policy", "--run-name", name, "--runs-dir", str(OUT),
           "--split", SPLIT, "--qids", ",".join(qids), "--build-budget", f"{left:.4f}",
           "--search-budget", f"{left:.4f}", "--parallel", str(len(qids))] + (["--dry-run"] if dry else [])
    print(f"probe {name}: {len(qids)} questions, {'dry run' if dry else 'live'}; budget left ${left:.3f}", flush=True)
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.log", "w") as log:
        subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=False)
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        cfg = {}
    spent = cfg.get("spent_usd")
    if spent is None:  # killed or crashed before it could count: assume the worst
        spent = left
    book["spent_usd"] = round(book["spent_usd"] + spent, 4)
    book["probes"].append({"name": name, "spent_usd": spent, "seconds": round(time.monotonic() - t0)})
    LEDGER.write_text(json.dumps(book, indent=2))

    episodes = sorted((run_dir / "episodes").glob("*.json")) if run_dir.exists() else []
    for f in episodes:
        e = json.loads(f.read_text())
        checks = ", ".join(f"t{c.get('turn')}:{c['prediction']!r}@{c['confidence']}" for c in e["checks"])
        print(f"  {e['qid']:22} score {e['score']:.2f} (pred {e['prediction']!r}, answer {e['answer']!r}) | "
              f"{e['turns']} steps, stopped: {e['stopped']} | tokens: search {e['search']['input_tokens']:,} + "
              f"QA {e['qa']['input_tokens']:,} | QA checks {checks}")
    missing = len(qids) - len(episodes)
    if missing:
        errors = [ln.strip() for ln in (OUT / f"{name}.log").read_text(errors="replace").splitlines()
                  if "Error" in ln or "Exception" in ln][-3:]
        print(f"  {missing} question(s) without a result; last errors: {errors or 'none logged'}")
    print(f"spent ${spent:.3f}; budget left ${max(0.0, book['budget_usd'] - book['spent_usd']):.3f}; "
          f"details: {run_dir.relative_to(HERE)}/episodes/, log: {(OUT / (name + '.log')).relative_to(HERE)}")


if __name__ == "__main__":
    main()
