"""The prompts of the two agent calls in each iteration: improving the parent policy, and choosing the next parent."""

from __future__ import annotations

import json

from delovi.config import DATA_DIR, JEV_PRICE_PER_MTOK
from evolution import frontier as fr
from evolution.config import (BUILD_BUDGET_USD, CHECK_QA_WEIGHT, FINAL_QA_WEIGHT, PROBE_BUDGET_USD,
                              SEARCH_BUDGET_USD, SELECT_SPLIT)

POLICY = "the memory policy and the retrieval policy"
FILES = "policy/builder.py (the memory policy) and policy/retriever.py (the retrieval policy)"


def _cell(text) -> str:
    return str(text if text not in (None, "") else "-").replace("|", "/").replace("\n", " ")


def objective(f: dict) -> str:
    floor, init = f.get("accuracy_floor"), f.get("init_accuracy")
    floor_note = ("" if floor is None or init is None else
                  f" The initial policy (`init`) scores {init:.3f}: the goal is methods more accurate than it. A"
                  f" method below {floor:.3f} (init's accuracy minus {f['init_floor_margin']:g}) is a failure"
                  f" and is never kept, however few tokens it uses; a method between {floor:.3f} and {init:.3f} can"
                  f" still be kept when it saves tokens.")
    return (f"- Accuracy: higher is better.{floor_note}\n"
            f"- Tokens: mean weighted input tokens per question = search tokens (every Jev request)\n"
            f"  + {CHECK_QA_WEIGHT:g} x QA input tokens of the intermediate QA calls (every QA call before the last, if any)\n"
            f"  + {FINAL_QA_WEIGHT:g} x QA input tokens of the final answer (the last QA call, whose answer is the prediction).\n"
            f"  Lower is better. Every tokens figure in this brief is weighted this way; summary.json and the episodes\n"
            f"  report the raw counts (search_tokens_per_q, qa_check_tokens_per_q for the intermediate calls,\n"
            f"  qa_final_tokens_per_q; search.input_tokens, qa.input_tokens, qa.final_input_tokens).\n"
            f"  Only the final QA call is required: its answer is the prediction. Intermediate QA calls are optional.\n"
            f"  The initial policy uses them as stop checks, to decide when to stop searching, but that is one design\n"
            f"  among many: a retriever may make no intermediate QA call at all (a fixed search budget, a stop decided\n"
            f"  from Jev's probabilities or from the memory's structure, one pass that gathers the evidence and asks\n"
            f"  once, etc.) or use them for something else. Do not assume the stopping rule must be kept or refined.")


def build_reference(f: dict) -> str:
    """The initial memory policy's build cost, overall and per recording, from its build records."""
    init = next((m for m in f["methods"] if m["run"] == "init"), None)
    if init is None:
        return ""
    text = (f"For reference, the initial builder (`init`) costs ${init['build_cost_usd']:.2f} at nominal prices to "
            f"build every {SELECT_SPLIT} memory")
    try:
        mids = sorted({q["memory_id"] for q in json.loads((DATA_DIR / "questions" / f"{SELECT_SPLIT}.json").read_text())})
        short, long = [], []
        for mid in mids:
            rec = json.loads((fr.EXP / "cache" / "trees" / init["builder_hash"] / f"{mid.replace('/', '__')}.build.json").read_text())
            meta = json.loads((DATA_DIR / "memories" / mid / "meta.json").read_text())
            hours = meta["n_clips"] * meta.get("clip_seconds", 30) / 3600
            (short if hours < 4 else long).append((hours, rec["cost_usd"]))
    except (OSError, KeyError, json.JSONDecodeError):
        return text + "."
    parts = []
    if short:
        parts.append(f"{len(short)} recording(s) under 4 hours cost ${sum(c for _, c in short) / len(short):.3f} each on"
                     f" average (at most ${max(c for _, c in short):.3f})")
    if long:
        hs, cs = [h for h, _ in long], [c for _, c in long]
        parts.append(f"{len(long)} longer recording(s) ({min(hs):.0f}-{max(hs):.0f} hours of clips, built up to their last"
                     f" question) cost ${min(cs):.2f}-${max(cs):.2f} each")
    return text + (": " + "; ".join(parts) if parts else "") + "."


def build_section(f: dict) -> str:
    ref = build_reference(f)
    init = next((m for m in f["methods"] if m["run"] == "init"), None)
    times = (f", about {BUILD_BUDGET_USD / init['build_cost_usd']:.0f}x the initial builder's"
             if init and init.get("build_cost_usd") else "")
    return (f"Memory building is not part of the tokens axis: its cost is reported separately and only capped. "
            f"{ref + ' ' if ref else ''}The cap is ${BUILD_BUDGET_USD:.0f} at nominal prices per evaluation{times}. A "
            f"builder may spend more than the initial one (more LLM calls, extra indexes or levels, longer summaries "
            f"where they pay off) when that buys accuracy or saves search and QA tokens. Built trees are cached by the "
            f"builder file's hash and LLM calls by their content, so a new builder is built once; do not avoid a "
            f"change to the memory policy because it needs a rebuild.")


def improve(run: str, f: dict, why: str | None = None) -> str:
    """The prompt of the agent that improves parent `run`; `why`: the reason the selection agent gave."""
    by_run = {x["run"]: x for x in f["methods"]}
    m = by_run[run]
    in_order = sorted(f["methods"], key=lambda x: (x["finished_at"], x["run"]))
    n_total = len(json.loads((DATA_DIR / "questions" / f"{SELECT_SPLIT}.json").read_text()))
    why_section = f"## Why this parent was chosen\n{why}\n\n" if why else ""
    origin = (f"The parent was made from `{m['parent']}` by this change: {m['note'] or '(no note)'}" if m["parent"]
              else "The parent is the initial policy.")
    children = [x for x in in_order if x["parent"] == run]
    failed = fr.failed_runs(run)
    if children or failed:
        rows = "\n".join(
            [f"| {x['run']} | {_cell(x['note'])} | {x['accuracy']:.3f} | {x['tokens_per_q']:,} | "
             f"{x['accuracy'] - m['accuracy']:+.3f} acc, {(x['tokens_per_q'] / m['tokens_per_q'] - 1) * 100:+.0f}% tokens | "
             f"{fr.front_label(x)} |" for x in children]
            + [f"| {r} | {_cell(fr.run_info(r).get('note'))} | - | - | evaluation failed: {_cell(fr.run_info(r)['failed'])} | no |"
               for r in failed])
        tried = (f"{len(children) + len(failed)} earlier attempt(s) changed this parent:\n"
                 f"| run | change | {SELECT_SPLIT} acc | tokens/q | vs parent | on frontier |\n|---|---|---|---|---|---|\n{rows}\n"
                 f"For each attempt, runs/<run>/agent.log holds that agent's analysis and reasoning, and\n"
                 f"`diff -r runs/{run}/policy runs/<run>/policy` shows its exact code change.")
    else:
        tried = "No earlier attempts on this parent: you are the first."
    all_rows = "\n".join(f"| {x['run']} | {x['parent'] or '-'} | {fr.front_label(x)} | {x['accuracy']:.3f} | "
                         f"{x['tokens_per_q']:,} | {_cell(x['note'])} |" for x in in_order)

    return f"""You are one iteration of an evolution loop that improves {POLICY} for long-video question answering.

## Your task
Study how the parent method performs and what has already been tried on it, then make deliberate changes
to {POLICY} that you expect to move the Pareto frontier. The loop evaluates your changes after you finish.

## Files
You may edit: {FILES}.
Everything else is fixed: delovi/ (memory tree, search interface, LLM calls, QA model, QA prompts, confidence,
grading, evaluation), probe.py and the question sets. The QA model and its reasoning effort are fixed.
Start by reading the files you may edit, and README.md, which states the contract your code must keep and the
APIs it can use.

## Objective
Each method is scored on two axes, on the {SELECT_SPLIT} split ({n_total} questions):
{objective(f)}
A method is kept when no other method is at least as good on both axes and better on one (the Pareto
frontier).
{build_section(f)}
Jev search is capped too: all questions of one evaluation together may spend at most ${SEARCH_BUDGET_USD:.0f} at nominal prices (${JEV_PRICE_PER_MTOK} per 1M input tokens, about
{SEARCH_BUDGET_USD / JEV_PRICE_PER_MTOK:,.0f}M tokens); past either cap the evaluation fails.

## The parent: {run}
{SELECT_SPLIT.capitalize()}: accuracy {m['accuracy']:.3f}, {m['tokens_per_q']:,} tokens per question (search {m['search_tokens_per_q']:,} + {CHECK_QA_WEIGHT:g} x intermediate QA {m['qa_check_tokens_per_q']:,} + {FINAL_QA_WEIGHT:g} x final answer {m['qa_final_tokens_per_q']:,}).
Its code is in policy/ now (also saved in runs/{run}/policy/). Its {SELECT_SPLIT} logs:
- runs/{run}/summary.json: metrics overall and per dataset.
- runs/{run}/episodes/<qid>.json: one file per question, with the prediction, score, confidence, every
  QA check, every search turn (candidates, probabilities, chosen node), the recorded nodes, the evidence spans
  and whether the search reached them, and token counts.
- cache/trees/{m['builder_hash']}/: the memory trees. An episode's `tree_snapshot` names the tree that question saw.
- `python -m delovi.analyze summary runs/{run}` and
  `python -m delovi.analyze trajectories runs/{run} --per-group 2` print a summary and representative questions.
Every other method's run is under runs/<run>/ in the same form.

{why_section}## Already tried on this parent
{origin}
{tried}

## All methods so far, in the order they were evaluated
| run | parent | on frontier | {SELECT_SPLIT} acc | tokens/q | changes from parent |
|---|---|---|---|---|---|
{all_rows}

## How to work
1. Read the files you may edit, and README.md.
2. Analyze the parent's {SELECT_SPLIT} logs thoroughly. Find where accuracy is lost: the evidence was never reached; it
   was reached but the answer was wrong; or the search stopped confidently on a wrong answer, etc. Find where
   tokens go: search versus QA, which tree levels, and how long the node texts are, etc.
3. Analyze every earlier attempt on this parent: which limitation it targeted, what it changed, and how the
   results moved, etc. Decide what worked, what did not, and why. Do not repeat an attempt that was dominated.
4. Choose changes that address the limitations you found and build on that evidence. Be ready to say why you
   expect them to improve accuracy, tokens, or both.
5. Make the edits, then check them with the offline dry run below.

## Rules
- Do NOT overfit constants to the dev questions. Measuring the logs and replaying them to estimate a change is
  welcome, but do not pick a value just because it scores best on the logged questions: give each constant a
  reason in its comment, in terms of the recordings (video durations, what the question asks, memory structure, how the search behaves, etc.).
- Prefer a structural change (e.g., modifying the memory structure, altering the search algorithm) when your analysis points to a structural limitation or minor changes can not make any significant improvement in accuracy or efficiency. Tune when the mechanism is new (e.g. the initial policy), or when earlier attempts show the mechanism is right and only its settings are off.
- Prioritize policies that improve accuracy, unless a policy can achieve substantial token savings while maintaining high accuracy.
- No dataset-specific changes. At test time the dataset is unknown, so do not branch on dataset or memory
  names or any other benchmark label. Base decisions on properties of the video itself:
  caption text, durations, the timeline, and the recording's description, etc. You can call LLMs to analyze these properties in the policy, but not the benchmark labels.
- You may modify the memory policy, the retrieval policy, or both. Both policies are equally important, so consider improvements to each without systematically favoring one over the other. Prioritize modifications based on their potential to improve overall performance, regardless of which policy they target.
- Videos differ, and past experience shows that the memory structure and the retrieval strategy that work
  best differ with them. Adapting the memory structure or the retrieval strategy to the features of the video
  (its length, its number of streams or days, caption density, speech, how repetitive its content is, etc.) can
  be effective. Decide from the video itself, never from a benchmark label.
- Every question counts. Do not trade accuracy on one kind of question for overall gains or token savings. Questions on the longest recordings cost the most tokens, so a change that gives up accuracy there buys a large drop in mean tokens for a small drop in mean accuracy. Do not make that trade. Also, questions that need several moments of the video (the order of events, how something changed, how often something happened, comparisons) are hurt most by stopping early or recording fewer clips, so keep enough evidence for them.
- Reducing QA tokens and intermediate tokens may appear to be an effective way to significantly lower total token consumption. However, these tokens are often critical for achieving high accuracy. Aggressively reducing them during evolution can lead to diminishing returns: once they become too limited, it may be difficult to improve accuracy in subsequent iterations, even with better search or retrieval strategies. Therefore, avoid systematically favoring token reductions at the expense of accuracy improvement potential. You may still optimize QA and intermediate token usage, but do so cautiously, ensuring that sufficient information and reasoning capacity are preserved for high-quality answers and future improvements.
- Do not run large-scale live evaluation; the loop evaluates your policy on {SELECT_SPLIT} after you finish. To check your code
  offline, run:
    python -m delovi.evaluate --policy policy --dry-run --run-name check --runs-dir scratch --split {SELECT_SPLIT} --limit 1
  This uses a fake search and a fake LLM. It checks that the policy runs and the trees are valid; its
  trees go to cache/trees_dry/. You are only allowed to use real search and real LLMs for small scale checks (see below).
- Live checks must be small. If the dry run cannot answer a question about your change (does the LLM
  follow the new prompt, does the search stop, is the build fast enough, etc.), refer to `python probe.py <qid> [<qid> ...]`: your policy on {SELECT_SPLIT} questions you choose (from data/questions/{SELECT_SPLIT}.json), with the real models, printing each question's checks, tokens and cost. Choose a few questions that exercise your change; questions on long recordings cost more to build and search (a new builder rebuilds the recording up to the question's time: about 10-15 minutes and $0.35 for a week). A probe has no time limit.
  All probes of this iteration share ${PROBE_BUDGET_USD:.2f}; the script refuses once it is spent. Make no other paid API calls:
  run live code only through `python probe.py`, never delovi.evaluate without --dry-run, and never load API keys yourself.
  If a probe is slow or fails, say so in your reply and do without it; do not work around it.
- Run every command in the foreground and wait for it to finish (a probe returns when it is done). Do not start
  background commands, monitors or wait loops: you run non-interactively, and a background task that never ends
  keeps this iteration from finishing even after your reply.
- Edit only {FILES}. Write scratch files only under scratch/.
- Work only inside this folder: do not read, list or search anything outside it (no `../` paths, no absolute
  paths to other folders). Everything you need is here.
- Policy code reaches Jev only through the `jev` handle (ranking and score requests, see README.md),
  the QA model only through `ask`, and other LLMs only through the builder's `llm`. It may not create API
  clients, open network connections, read environment variables or API keys, start processes, or reach into
  the harness (e.g. `jev.backend`). Every evaluation checks the code before running it and counts every API
  response; a policy that breaks this rule fails its evaluation.

## Reply
End your reply with one line that starts with `CHANGE:`, followed by a few sentences on that same line: the
limitations you targeted, what you changed, and the effect you expect on accuracy and tokens.
"""


def select(f: dict, just_finished: str | None) -> str:
    """The prompt of the agent that chooses the next parent from the frontier."""
    ms = f["methods"]
    in_order = sorted(ms, key=lambda x: (x["finished_at"], x["run"]))
    ds = sorted({d for m in ms for d in m["by_dataset"]})
    by_ds = lambda m: " | ".join(f"{m['by_dataset'].get(d, float('nan')):.2f}" for d in ds)  # noqa: E731
    kids = {m["run"]: [x for x in ms if x["parent"] == m["run"]] for m in ms}
    failed = fr.failed_runs()

    if just_finished:
        info = fr.run_info(just_finished)
        m = next((x for x in ms if x["run"] == just_finished), None)
        if m:
            dom = [o["run"] for o in ms if o["frontier"] and o is not m and o["accuracy"] >= m["accuracy"]
                   and o["tokens_per_q"] <= m["tokens_per_q"]]
            verdict = ("on the frontier" if m["frontier"] else
                       "a failure: below the accuracy floor (init's accuracy minus the margin), not kept" if m.get("below_init") else
                       f"dominated by {', '.join(dom) or 'other methods'}")
            happened = (f"Iteration `{just_finished}` changed `{m['parent']}`: {_cell(m['note'])}\n"
                        f"Result on {SELECT_SPLIT}: accuracy {m['accuracy']:.3f} ({by_ds(m).replace(' | ', ' / ')} on "
                        f"{' / '.join(ds)}), {m['tokens_per_q']:,} tokens per question: {verdict}.")
        elif info.get("failed"):
            happened = (f"Iteration `{just_finished}` changed `{info.get('parent')}`: {_cell(info.get('note'))}\n"
                        f"Its evaluation failed ({_cell(info['failed'])}), so it is not a method.")
        else:
            happened = f"Iteration `{just_finished}` (parent `{info.get('parent')}`) changed nothing and was skipped."
    else:
        happened = "No iteration has finished since the last selection (the loop is starting or resuming)."

    front_rows = "\n".join(
        f"| {m['run']} | {m['parent'] or '-'} | {m['accuracy']:.3f} | {m['tokens_per_q']:,} | {by_ds(m)} | "
        f"{len(kids[m['run']])} | {sum(k['frontier'] for k in kids[m['run']])} | {_cell(m['note'])} |"
        for m in ms if m["frontier"])
    all_rows = "\n".join(
        f"| {x['run']} | {x['parent'] or '-'} | {fr.front_label(x)} | {x['accuracy']:.3f} | "
        f"{x['tokens_per_q']:,} | {by_ds(x)} | {_cell(x['note'])} |" for x in in_order)
    failed_rows = "\n".join(f"| {r} | {fr.run_info(r).get('parent') or '-'} | evaluation failed: "
                            f"{_cell(fr.run_info(r)['failed'])} | {_cell(fr.run_info(r).get('note'))} |" for r in failed)
    failed_part = (f"\nRuns whose evaluation failed (not methods):\n| run | parent | result | change |\n|---|---|---|---|\n"
                   f"{failed_rows}\n") if failed else ""
    ds_cols = " | ".join(ds)
    ds_sep = "---|" * len(ds)

    return f"""You choose which method the next iteration of an evolution loop will improve. The loop evolves {POLICY} for
long-video question answering: each iteration, an agent takes one method from the Pareto frontier (the parent),
analyzes its logs and edits its policy; the result is evaluated and joins the frontier if no method dominates it.
You edit nothing: you study the state of the evolution and name the next parent.

## Objective of the evolution
Each method is scored on two axes, on the {SELECT_SPLIT} split:
{objective(f)}
A method is kept when no other method is at least as good on both axes and better on one (the Pareto frontier).
The aim is a frontier that is as high and as far left as possible over its whole range, not one point.

## What just happened
{happened}

## The frontier now (cheapest first)
| run | parent | {SELECT_SPLIT} acc | tokens/q | {ds_cols} | attempts on it | of which on the frontier | change from parent |
|---|---|---|---|{ds_sep}---|---|---|
{front_rows}

## All methods so far, in the order they were evaluated
| run | parent | on frontier | {SELECT_SPLIT} acc | tokens/q | {ds_cols} | change from parent |
|---|---|---|---|---|{ds_sep}---|
{all_rows}
{failed_part}
## Where to look
- runs/<run>/summary.json: metrics overall and per dataset; runs/<run>/episodes/<qid>.json: one file per question
  (prediction, score, confidence, every QA check, every search turn, the evidence spans and whether the search
  reached them, token counts); `python -m delovi.analyze summary runs/<run>` and
  `python -m delovi.analyze trajectories runs/<run> --per-group 2` print a summary and representative questions.
- runs/<run>/agent.log: the brief that iteration's agent received and its full reply (its analysis, reasoning
  and CHANGE line); runs/<run>/policy/: its code; `diff -r runs/<a>/policy runs/<b>/policy` shows a change.
You may read anything under runs/, and nothing outside this folder (no `../` paths, no absolute
paths to other folders). Do not run evaluations or make paid API calls.

## How to choose
Prefer accuracy over efficiency. Choose the most promising method that can expand the pareto frontier. Think in a long-term manner. Some methods may be more promising in the short term but less promising in the long term. Prefer parents whose next change can be more generalizable.

## Reply
End your reply with one line that starts with `NEXT: <run>` (a method on the frontier), followed on that same
line by a few sentences: why this method, and which limitation the next iteration should look at first.
"""
