# Workspace

This folder holds one policy for long-video question answering and the fixed harness that evaluates it. A policy
is two programs: `policy/builder.py` (the memory policy) turns a video's clip captions into a memory tree, and
`policy/retriever.py` (the retrieval policy) searches the tree with the decision model Jev and asks the QA model.

```
policy/builder.py, policy/retriever.py   the policy (yours to edit)
delovi/                                  the fixed harness: tree.py, jev.py, llm.py, qa.py, grade.py, evaluate.py, analyze.py
probe.py                                 a small live check of the policy (see your instructions)
data/                                    questions/dev.json and the memories (one folder per recording)
runs/<run>/                              every evaluated policy: policy/, summary.json, episodes/<qid>.json, agent.log
cache/                                   LLM completions and built trees (cache/trees/<builder hash>/), shared by all runs
```

## Data

A recording (`data/memories/<dataset>/<name>/`) is a list of base clips of 10 or 30 seconds, each with a caption
and, when there is speech, what was said after `Speech:`. Its `description` says what the recording is and is shown
to Jev and the QA model in every request. Times are seconds on the recording's timeline, shown as
`DAY2 14:03:00`, `stream 14 01:02:03` or `01:02:03`. A question may have a question time (`query_time`): it may only
use what was recorded before then. Questions are multiple choice (one letter) or open-ended (a short phrase).

## The memory policy

`MemoryBuilder(memory, llm)` builds the tree online, while the video streams in:

```python
b = MemoryBuilder(memory, llm)
for clip in memory.clips:           # in time order
    if <a question's time comes before clip ends>:
        await b.sync()              # then the harness checkpoints: the question sees the tree as it is now
    await b.add(clip)
await b.finish()                    # end of the video (not called when building stops after the last question)
tree = b.tree                       # the harness adds a root over the top level if b.tree.root is None
```

The contract:
- `add(clip)` receives the base clips one at a time, in time order, and may use only what it has seen. It should
  only start work (LLM calls as tasks) and return; `sync()` makes the nodes decided so far. `sync` says that a
  question is asked now, never what it is. `finish()` closes whatever is open and makes every node.
- The tree (`delovi/tree.py`) is append-only: `add_leaf` and `make_node` append nodes that never change afterwards (a
  node's parent is set when the parent is made). A question at time t sees the nodes that existed when the
  harness checkpointed t, so a node made late is invisible to earlier questions.
- Every clip is exactly one leaf with the clip's span and text; parents span their children; internal nodes have
  text; one root; at every checkpoint the leaves are exactly the clips that ended before it. The tree's
  `description` and `timeline` stay the recording's. `validate_tree` checks all of this after every build; a tree
  that fails voids every answer on that recording.
- Everything else is free: the builder may keep any state and call the LLM for any purpose (boundaries, entities,
  routing texts at any granularity, more levels). Depth and branching need not be uniform.

Tree API: `add_leaf(clip_index, start, end, text) -> nid`; `make_node(children, text) -> nid` (children must be
parentless); `orphans()`; `set_root(nid)`; `tree[nid]` is a `Node(start, end, text, children, parent, is_leaf)`;
`span_label(node)`, `duration_label(node)`, `time_label(sec)` and `place(sec)` give readable times.

LLM calls: `await llm.complete(model, messages, reasoning)` returns a completion with `.text` (OpenAI Responses API,
cached by model, effort and messages, so an unchanged call is free on a rebuild). Use `gpt-6-luna` with reasoning
`"none"` ($0.10 / $0.50 per 1M input / output tokens) unless more effort demonstrably changes the tree. Calls
awaited inside `add` run one after another and stall the stream: start them as tasks and make the nodes in
`sync`, in a fixed order, so the tree does not depend on API latency. Build cost is recorded per tree and capped
per evaluation (see your instructions).

What a good tree does: the retriever shows Jev the question and the text of candidate nodes and asks which one
contains the moment that answers the question, and the QA model reads the nodes the search found. An internal
node's text is therefore read for routing and as context: enough to decide whether a moment lies inside its span,
with as few characters as possible.

## The retrieval policy

`result = await retrieve(question, query_time, tree, jev, ask)`. Only nodes that existed at `query_time` are
visible: `tree.view(query_time)` gives `roots()`, `children(nid)` and `visible(nid)`. The result must have
`recorded` (node ids, in the order found), `steps` (objects with `turn`, `nid`, `list_id`, `candidates`, `probs`,
`children`, logged per step), `lists` (an int), `checks` (what `ask` returned) and `stop` (why the search ended).

The QA step: `check = await ask(nodes, turn=None)` runs the fixed QA model (`delovi/qa.py`: model, reasoning effort
and prompt are fixed) on any nodes of the question's tree, shown in time order with their time spans; `turn` is only
logged. It returns `check.answer` (a letter, or a short phrase), `check.confidence` (in [0, 1], or None) and
`check.response`. The answer of the last `ask` is the prediction; if the policy never asks, the harness asks once
on every recorded node. Only input tokens count toward tokens per question.

Jev (`delovi/jev.py`), through the `jev` handle only:
- `await jev.choice(state, instructions, options, tag="")`: a ranking request over `options` (label -> any JSON);
  returns a probability per label. Requests must fit Jev's limit (about 32k tokens, 255 options):
  `jev.fits(...)` checks one, `jev.groups(...)` splits a list into contiguous groups that fit.
- `await jev.score(state, questions, tag="")`: up to 64 score questions in one request; `questions` maps a label
  to {"instructions": str or dict, "criteria": [level 0 text, level 1 text, ...]}; returns per label {"level": the
  expected level, "probs": a probability per level, "confidence"}. `jev.score_groups(state, [(label, question),
  ...])` packs questions into requests.
- Both raise `TooLong` when Jev rejects a request as too long (the request budget then shrinks: split and retry).
  Every response's input tokens count toward the question's tokens, and all questions of one evaluation share a
  spend cap (see your instructions).

## Rules the harness enforces

Policy code reaches Jev only through `jev`, the QA model only through `ask`, and other LLMs only through the
builder's `llm`. It may not create API clients, open network connections, read environment variables or API keys,
start processes, or reach into the harness. Every evaluation checks the code before running it and counts every
API response; a policy that breaks this rule fails its evaluation.

## Commands

```bash
python -m delovi.evaluate --policy policy --dry-run --run-name check --runs-dir scratch --split dev --limit 1
python -m delovi.analyze summary runs/<run>
python -m delovi.analyze trajectories runs/<run> --per-group 2
python probe.py <qid> [<qid> ...]
```
