"""Automated policy evolution over a Pareto frontier of accuracy and input tokens per question.

    python -m evolution.evolve --exp experiments/delovi_e iterate [--n 30]   # start, or resume after a stop
    python -m evolution.evolve --exp experiments/delovi_e frontier           # print the frontier table
    python -m evolution.evolve --exp experiments/delovi_e brief <run>        # print the prompt for parent <run>
    python -m evolution.evolve --exp experiments/delovi_e select-brief [run] # print the selection prompt

If no method exists yet, the initial policy (hierarchical_search/) is evaluated on dev as run `init`. Each
iteration then:
1. takes the parent chosen at the end of the previous iteration;
2. runs the coding agent in a private workspace with the parent's policy; it edits policy/ and ends its reply
   with a `CHANGE:` line;
3. evaluates the new policy on dev as runs/itNNN, updates frontier.json and LOG.md;
4. if the new method is on the frontier, evaluates it on val as runs/itNNN_val (the agents never see val);
5. asks a second agent call to choose the next parent from the frontier (a `NEXT:` line).
Ctrl-C stops the agent and the evaluation at once; `iterate` again resumes where the loop stopped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from delovi.config import DATA_DIR, ROOT
from evolution import briefs
from evolution import frontier as fr
from evolution.config import (AGENT_CMD, AGENT_EFFORT, AGENT_MODEL, BUILD_BUDGET_USD, CONFIRM_SPLIT, INIT_POLICY,
                              ITERATIONS, PARALLEL, PROBE_BUDGET_USD, SEARCH_BUDGET_USD, SELECT_SPLIT)


class AgentFailed(RuntimeError):
    """The agent command failed (e.g. a usage limit); the loop stops and `iterate` redoes the step later."""


def say(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


# -- workspaces --

def workspace(name: str, policy_from: Path, splits: list[str]) -> Path:
    """A private copy of the code for one run: the agent edits its policy/, the evaluation runs there. Only the
    questions and memories of `splits` are visible; runs/ shows the experiment's runs except the val runs when
    val is not among `splits`; cache/ is the experiment's."""
    w = fr.EXP / "work" / name
    shutil.rmtree(w, ignore_errors=True)
    w.mkdir(parents=True)
    shutil.copytree(ROOT / "delovi", w / "delovi", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "evolution" / "AGENT_README.md", w / "README.md")
    shutil.copy2(ROOT / "evolution" / "probe.py", w / "probe.py")
    (w / "policy").mkdir()
    for f in ("builder.py", "retriever.py", "__init__.py"):
        if (policy_from / f).exists():
            shutil.copy2(policy_from / f, w / "policy" / f)
    os.symlink(fr.EXP / "cache", w / "cache")
    (w / "runs").mkdir()
    for p in fr.RUNS.iterdir():
        if CONFIRM_SPLIT in splits or not p.name.endswith(f"_{CONFIRM_SPLIT}"):
            os.symlink(p, w / "runs" / p.name)
    (w / "data" / "questions").mkdir(parents=True)
    for s in splits:
        q_file = DATA_DIR / "questions" / f"{s}.json"
        os.symlink(q_file, w / "data" / "questions" / q_file.name)
        for mid in sorted({q["memory_id"] for q in json.loads(q_file.read_text())}):
            link = w / "data" / "memories" / mid
            if not link.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(DATA_DIR / "memories" / mid, link)
    (w / "scratch").mkdir()
    (w / "scratch" / "probe_ledger.json").write_text(json.dumps({"budget_usd": PROBE_BUDGET_USD, "spent_usd": 0.0,
                                                                 "probes": []}, indent=2))
    return w


def call(cmd, cwd: Path, shell: bool = False, input: str | None = None, stdout=subprocess.PIPE,
         stderr=subprocess.PIPE) -> tuple[int, str, str]:
    """A child process in its own process group; an interrupt stops the whole group."""
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.Popen(cmd, cwd=cwd, shell=shell, text=True, start_new_session=True, stdout=stdout, stderr=stderr,
                            stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL, env=env)
    try:
        out, err = proc.communicate(input)
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        raise
    return proc.returncode, out or "", err or ""


def run_agent(cmd: str, prompt: str, log_path: Path, cwd: Path, tag: str) -> str:
    """Run the agent with `prompt` on stdin; returns the text after the last `<tag>:` line of its reply."""
    code, out, err = call(cmd, cwd, shell=True, input=prompt)
    log_path.write_text(f"$ {cmd}\n\n--- stdin ---\n{prompt}\n\n--- stdout ---\n{out}\n\n--- stderr ---\n{err}\n")
    if code != 0:
        tail = " / ".join((err or out).strip().splitlines()[-2:])[:300]
        raise AgentFailed(f"agent failed (exit {code}): {tail or 'no output'}; see {log_path}")
    lines = [ln for ln in out.splitlines() if ln.strip().startswith(f"{tag}:")]
    return lines[-1].split(":", 1)[1].strip() if lines else ""


# -- evaluations --

def start_run(run: str, info: dict) -> None:
    (fr.RUNS / run).mkdir(parents=True, exist_ok=True)
    fr.write_info(run, {**info, "started": dt.datetime.now().isoformat(timespec="seconds")})


def failure_reason(run: str) -> str:
    done = len(list((fr.RUNS / run / "episodes").glob("*.json"))) if (fr.RUNS / run / "episodes").exists() else 0
    err = ""
    for log in ("eval.log", "run.log"):
        try:
            lines = [ln.strip() for ln in (fr.RUNS / run / log).read_text(errors="replace").splitlines()]
        except OSError:
            continue
        hits = [ln for ln in lines if "Error" in ln or "error" in ln or "Exception" in ln]
        err = hits[-1] if hits else err
    return f"{done} questions done; {err[:240] or 'no error message'}"


def evaluate(run: str, split: str, parent: str | None, note: str, w: Path, dry_run: bool) -> bool:
    """Evaluate w/policy as `run` on `split` (a second attempt finishes questions an API error left); True when
    every question finished."""
    cmd = [sys.executable, "-m", "delovi.evaluate", "--policy", "policy", "--run-name", run, "--split", split,
           "--runs-dir", str(fr.RUNS), "--note", note, "--parallel", str(PARALLEL),
           "--build-budget", str(BUILD_BUDGET_USD), "--search-budget", str(SEARCH_BUDGET_USD)]
    cmd += ["--parent", parent] if parent else []
    cmd += ["--dry-run"] if dry_run else []
    for attempt in (1, 2):
        say(f"{run}: evaluating on {split}" + (" (second attempt)" if attempt == 2 else ""))
        with open(fr.RUNS / run / "eval.log", "a") as out:
            code, _, _ = call(cmd, w, stdout=out, stderr=subprocess.STDOUT)
        if fr.complete(run):
            return True
        say(f"{run}: evaluation incomplete (exit {code}; {failure_reason(run)})")
    return False


def log_row(run: str, parent: str | None, note: str, f: dict, result: str | None = None) -> None:
    """One row of LOG.md: an evaluated method, a val run, or a run without a result (skipped / failed)."""
    log = fr.EXP / "LOG.md"
    text = log.read_text() if log.exists() else ("# log\n\n| run | parent | builder | retriever | change | split | acc "
                                                 "| acc by source | tokens/q | build $ | result |\n"
                                                 "|---|---|---|---|---|---|---|---|---|---|---|\n")
    if f"| {run} |" in text:
        return
    m = next((x for x in fr.methods() if x["run"] == run), None)
    fm = next((x for x in f["methods"] if x["run"] == run), None)
    if result or m is None:
        row = f"| {run} | {parent or '-'} | - | - | {briefs._cell(note)} | - | - | - | - | - | {briefs._cell(result or 'failed')} |\n"
    else:
        verdict = (("frontier" if fm["frontier"] else "failure: below floor" if fm.get("below_init") else "dominated")
                   if fm else f"{m['split']} of {fr.run_info(run).get('method', '?')}")
        by_src = ", ".join(f"{d} {v:.2f}".replace(" 0.", " .") for d, v in sorted(m["by_dataset"].items()))
        row = (f"| {run} | {parent or '-'} | {m['builder_hash']} | {m['retriever_hash']} | {briefs._cell(note)} | "
               f"{m['split']} | {m['accuracy']:.3f} | {by_src} | {m['tokens_per_q']:,} | {m['build_cost_usd']:.2f} | {verdict} |\n")
    log.write_text(text + row)


def evaluate_method(run: str, parent: str | None, note: str, w: Path, dry_run: bool) -> None:
    """Evaluate on dev, update the frontier and LOG.md, and confirm on val if the method is on the frontier."""
    if not evaluate(run, SELECT_SPLIT, parent, note, w, dry_run):
        info = fr.run_info(run)
        info["failed"] = failure_reason(run)
        fr.write_info(run, info)
        if not (fr.RUNS / run / "policy").exists():
            shutil.copytree(w / "policy", fr.RUNS / run / "policy", ignore=shutil.ignore_patterns("__pycache__"))
        log_row(run, parent, note, fr.update(), result=f"failed: {info['failed']}")
        say(f"{run}: FAILED ({info['failed']}); not a method. Logs: runs/{run}/eval.log")
        shutil.rmtree(w, ignore_errors=True)
        return
    f = fr.update()
    log_row(run, parent, note, f)
    m = next(x for x in f["methods"] if x["run"] == run)
    verdict = "on the frontier" if m["frontier"] else "below the accuracy floor" if m["below_init"] else "dominated"
    say(f"{run}: {SELECT_SPLIT} accuracy {m['accuracy']:.3f}, {m['tokens_per_q']:,} tokens/q, {verdict}\n{fr.table(f)}")
    shutil.rmtree(w, ignore_errors=True)
    if m["frontier"]:
        confirm(run, dry_run)


def confirm(run: str, dry_run: bool) -> None:
    """Evaluate method `run` on val as runs/<run>_val (resumed if unfinished)."""
    if fr.confirm_run(run, fr.methods()):
        return
    name = f"{run}_{CONFIRM_SPLIT}"
    me = next(m for m in fr.methods() if m["run"] == run)
    if not (fr.RUNS / name / "iteration.json").exists():
        start_run(name, {"kind": "confirm", "method": run, "dry_run": dry_run})
    say(f"{run} is on the frontier: evaluating it on {CONFIRM_SPLIT} as {name}")
    w = workspace(name, fr.RUNS / run / "policy", [SELECT_SPLIT, CONFIRM_SPLIT])
    if not evaluate(name, CONFIRM_SPLIT, me["parent"], me["note"], w, dry_run):
        raise RuntimeError(f"{name} did not finish; run `iterate` again to resume it")
    shutil.rmtree(w, ignore_errors=True)
    f = fr.update()
    log_row(name, me["parent"], me["note"], f)
    m = next(x for x in f["methods"] if x["run"] == run)
    say(f"{name}: {CONFIRM_SPLIT} accuracy {m['val_accuracy']:.3f} ({SELECT_SPLIT}: {m['accuracy']:.3f})")


# -- iterations --

def seed(args) -> None:
    """The first method: the initial policy evaluated on dev as `init`."""
    say(f"no method yet: evaluating {INIT_POLICY.name}/ on {SELECT_SPLIT} as `init`")
    info = fr.run_info("init")
    info.pop("failed", None)  # a failed earlier attempt is retried
    start_run("init", {"kind": "init", "dry_run": args.dry_run, **info})
    src = fr.RUNS / "init" / "policy" if (fr.RUNS / "init" / "policy").exists() else INIT_POLICY
    w = workspace("init", src, [SELECT_SPLIT])
    evaluate_method("init", None, "initial policy", w, info.get("dry_run", args.dry_run))
    if not fr.complete("init"):
        raise SystemExit("the initial policy's evaluation failed; see runs/init/eval.log")


def agent_phase(run: str, parent: str, w: Path, args) -> str | None:
    """The agent edits w/policy; returns its note, or None when it changed nothing."""
    f = fr.update()
    before = {p.name: p.read_bytes() for p in (w / "policy").glob("*.py")}
    say(f"{run}: agent working on parent {parent} (workspace work/{run})")
    note = run_agent(args.agent_cmd, briefs.improve(parent, f, fr.run_info(run).get("selected_reason")),
                     fr.RUNS / run / "agent.log", w, "CHANGE") or "(no CHANGE line)"
    info = fr.run_info(run)
    info["note"] = note
    try:
        ledger = json.loads((w / "scratch" / "probe_ledger.json").read_text())
        info["probe_spent_usd"], info["probes"] = ledger["spent_usd"], len(ledger["probes"])
        if ledger["probes"]:
            shutil.copytree(w / "scratch" / "probes", fr.RUNS / run / "probes", dirs_exist_ok=True)
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    fr.write_info(run, info)
    say(f"{run}: CHANGE: {note}")
    after = {p.name: p.read_bytes() for p in (w / "policy").glob("*.py")}
    return None if before == after else note


def skip(run: str, parent: str, w: Path) -> None:
    say(f"{run}: the agent changed nothing; skipping the evaluation")
    log_row(run, parent, fr.run_info(run).get("note", ""), fr.update(), result="skipped: no change")
    (fr.RUNS / run / "config.json").write_text(json.dumps({"complete": True, "skipped": True}, indent=2))
    shutil.rmtree(w, ignore_errors=True)


def select_parent(run: str | None, args) -> str:
    """Choose the next parent after iteration `run` (the only frontier member, else the selection agent; weighted
    sampling when its reply names no frontier method) and record it in runs/<run>/iteration.json."""
    f = fr.update()
    front = [m for m in f["methods"] if m["frontier"]]
    if not front:
        raise SystemExit("the frontier is empty")
    if len(front) == 1:
        choice, reason = front[0]["run"], "the only method on the frontier"
    else:
        say(f"{run or 'start'}: selection agent choosing the next parent")
        w = workspace(f"select_{run or 'start'}", INIT_POLICY, [SELECT_SPLIT])
        log_path = fr.RUNS / run / "select.log" if run else fr.EXP / "select.log"
        try:
            reply = run_agent(args.agent_cmd, briefs.select(f, run), log_path, w, "NEXT")
        finally:
            shutil.rmtree(w, ignore_errors=True)
        words = reply.replace("`", " ").replace(",", " ").split()
        choice = words[0].strip(".;:") if words else ""
        reason = reply[reply.find(choice) + len(choice):].strip(" .:;-—") if choice else ""
        if choice not in {m["run"] for m in front}:
            choice = random.choices(front, [1.0 / (1 + m["children"]) for m in front])[0]["run"]
            reason = "weighted sampling: the selection agent named no frontier method"
    if run:
        info = fr.run_info(run)
        info.update({"next_parent": choice, "next_reason": reason, "next_used": False})
        fr.write_info(run, info)
    say(f"{run or 'start'}: next parent {choice}: {reason}")
    return choice


def pending_choice() -> tuple[str, str, str] | None:
    """The unused choice recorded by an earlier iteration (marked used now), if its method is still on the
    frontier: (parent, reason, the run that chose it)."""
    front = set(fr.update()["frontier"])
    done = [p.parent.name for p in fr.RUNS.glob("it[0-9][0-9][0-9]/iteration.json")]
    for r in sorted(done, key=lambda r: fr.run_info(r).get("started", ""), reverse=True):
        info = fr.run_info(r)
        if "next_parent" in info and not info.get("next_used"):
            info["next_used"] = True
            fr.write_info(r, info)
            return (info["next_parent"], info["next_reason"], r) if info["next_parent"] in front else None
    return None


def iteration(args) -> None:
    parent, reason, chosen_by = pending_choice() or (select_parent(None, args), "", None)
    n = 1 + max([0] + [int(p.name[2:5]) for p in fr.RUNS.glob("it[0-9][0-9][0-9]")])
    run = f"it{n:03d}"
    start_run(run, {"kind": "iteration", "parent": parent, "dry_run": args.dry_run, "selected_by": chosen_by,
                    "selected_reason": reason or None})
    say(f"=== {run}: parent {parent} ===")
    finish_iteration(run, args)


def finish_iteration(run: str, args) -> None:
    """The agent phase (unless the run already has a policy snapshot), the evaluation and the next selection."""
    info = fr.run_info(run)
    parent = info["parent"]
    if (fr.RUNS / run / "policy").exists():  # the evaluation had started: continue it
        w = workspace(run, fr.RUNS / run / "policy", [SELECT_SPLIT])
        note = info.get("note", "")
    else:
        w = workspace(run, fr.RUNS / parent / "policy", [SELECT_SPLIT])
        note = agent_phase(run, parent, w, args)
        if note is None:
            skip(run, parent, w)
            select_parent(run, args)
            return
    evaluate_method(run, parent, note, w, info.get("dry_run", args.dry_run))
    select_parent(run, args)


def iterate(args) -> None:
    """Finish interrupted runs, seed the frontier if needed, then run iterations until --n are done."""
    for run in fr.unfinished():
        kind = fr.run_info(run).get("kind")
        say(f"resuming {run} ({kind})")
        if kind == "init":
            seed(args)
        elif kind == "confirm":
            confirm(fr.run_info(run)["method"], args.dry_run)
        else:
            finish_iteration(run, args)
    if not any(m["run"] == "init" for m in fr.methods()):
        seed(args)
    for m in fr.update()["methods"]:
        if m["frontier"] and not m["val_run"]:
            confirm(m["run"], args.dry_run)
    while sum(1 for p in fr.RUNS.glob("it[0-9][0-9][0-9]/iteration.json")) < args.n:
        iteration(args)
    say("done\n" + fr.table(fr.update()))


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", type=Path, required=True, help="the experiment folder")
    sub = p.add_subparsers(dest="cmd", required=True)
    it = sub.add_parser("iterate")
    it.add_argument("--n", type=int, default=ITERATIONS, help="iterations in total")
    it.add_argument("--agent-cmd", default=AGENT_CMD, help="the agent command; {model} and {effort} are filled in")
    it.add_argument("--dry-run", action="store_true", help="evaluate with fake Jev and fake LLM (tests the loop)")
    sub.add_parser("frontier")
    b = sub.add_parser("brief")
    b.add_argument("run")
    s = sub.add_parser("select-brief")
    s.add_argument("run", nargs="?")
    args = p.parse_args()
    fr.set_experiment(args.exp)
    if args.cmd == "frontier":
        print(fr.table(fr.update()))
    elif args.cmd == "brief":
        print(briefs.improve(args.run, fr.update()))
    elif args.cmd == "select-brief":
        print(briefs.select(fr.update(), args.run))
    else:
        args.agent_cmd = args.agent_cmd.format(model=AGENT_MODEL, effort=AGENT_EFFORT)
        signal.signal(signal.SIGTERM, _interrupt)
        try:
            iterate(args)
        except KeyboardInterrupt:
            say("interrupted; run `iterate` again to resume")
            sys.exit(130)
        except AgentFailed as e:
            say(f"{e}\nstopped; run `iterate` again to resume")
            sys.exit(1)


if __name__ == "__main__":
    main()
