"""Grade the MM-Lifelong predictions of finished runs with the benchmark's official rubric grader (v1.1).

    python scripts/grade_mmlifelong.py --grader /path/to/MM-Lifelong/v1.1/grading/grade_predictions.py \
        runs/delovi_e_mml_day_test runs/delovi_e_mml_week_test runs/delovi_e_mml_month_val

Works on runs of delovi.evaluate, delovi_s.run and scripts/requery.py. The judge is GPT-5 (OPENAI_API_KEY). The
score is 100 x mean(0-5) / 5. Grades are cached in runs/<run>/official_grades.jsonl; the result is written to
runs/<run>/official_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from delovi.config import DATA_DIR, JUDGE_MODEL  # noqa: E402

CHUNKS = 8  # grader processes run in parallel


def grade_run(run_dir: Path, grader: Path) -> dict:
    split = json.loads((run_dir / "config.json").read_text())["split"]
    questions = {q["qid"]: q for q in json.loads((DATA_DIR / "questions" / f"{split}.json").read_text())}
    eps = [json.loads(p.read_text()) for p in sorted((run_dir / "episodes").glob("*.json"))]
    preds = [{"id": questions[e["qid"]]["extra"]["benchmark_id"], "model": run_dir.name,
              "prediction": e.get("prediction") or ""} for e in eps]
    out = run_dir / "official_grades.jsonl"
    done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    todo = [p for p in preds if p["id"] not in done]
    if todo:
        env = {**os.environ, "GRADER_API_KEY": os.environ["OPENAI_API_KEY"]}
        procs, parts = [], []
        for i in range(CHUNKS):
            chunk = todo[i::CHUNKS]
            if not chunk:
                continue
            fin, fout = run_dir / f"official.todo{i}.jsonl", run_dir / f"official.new{i}.jsonl"
            fout.unlink(missing_ok=True)
            fin.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in chunk))
            parts.append((fin, fout))
            procs.append(subprocess.Popen([sys.executable, str(grader), "--predictions", str(fin), "--output", str(fout),
                                           "--base-url", "https://api.openai.com/v1", "--model", JUDGE_MODEL], env=env))
        codes = [p.wait() for p in procs]
        with out.open("a") as fh:
            for fin, fout in parts:
                if fout.exists():
                    fh.write(fout.read_text())
                    fout.unlink()
                fin.unlink()
        if any(codes):
            print(f"{run_dir.name}: some grader processes failed ({codes}); rerun to finish")
    grades = {}
    for line in out.read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            grades[g["id"]] = g["score"]
    by_type = defaultdict(list)
    for e, p in zip(eps, preds):
        if p["id"] in grades:
            by_type[e["qtype"]].append(grades[p["id"]])
    scores = [grades[p["id"]] for p in preds if p["id"] in grades]
    return {"run": run_dir.name, "split": split, "n": len(preds), "graded": len(scores), "judge": JUDGE_MODEL,
            "score": round(100 * sum(scores) / (5 * len(scores)), 2) if scores else None,
            "by_type": {t: round(100 * sum(v) / (5 * len(v)), 1) for t, v in sorted(by_type.items())}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grader", type=Path, required=True, help="grade_predictions.py of the MM-Lifelong release")
    p.add_argument("runs", type=Path, nargs="+")
    args = p.parse_args()
    for run_dir in args.runs:
        res = grade_run(run_dir, args.grader)
        (run_dir / "official_summary.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res))


if __name__ == "__main__":
    main()
