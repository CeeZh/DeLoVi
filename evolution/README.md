# Automated policy evolution

A coding agent (Claude Code) rewrites the memory policy and the retrieval policy over several iterations. The
captioner, the decision model, the LLM and its prompt, and the evaluation stay fixed.

```bash
python -m evolution.evolve --exp experiments/delovi_e iterate          # 30 iterations; run again to resume
python -m evolution.evolve --exp experiments/delovi_e frontier         # the frontier table
python -m evolution.evolve --exp experiments/delovi_e brief init       # the prompt an agent gets for parent `init`
```

**The loop.** The initial policy (`hierarchical_search/`) is evaluated on the dev split as run `init`. Each
iteration then:
1. takes the parent chosen at the end of the previous iteration;
2. gives the agent a private workspace with the parent's policy and the prompt from `briefs.py`: the objective, the
   parent's results and logs, the earlier attempts on the same parent, all methods so far, and the rules. The agent
   edits `policy/builder.py` and `policy/retriever.py`, and ends its reply with a `CHANGE:` line;
3. evaluates the new policy on dev as `runs/itNNN` and updates `frontier.json` and `LOG.md`;
4. if the new policy is on the Pareto frontier, evaluates it on val as `runs/itNNN_val`;
5. asks a second agent call to choose the next parent from the frontier (a `NEXT:` line).

**Objective.** Accuracy, and input tokens per question: decision-model tokens + 2 x tokens of intermediate LLM
calls + 4 x tokens of the final LLM call. A policy is kept when no other policy is at least as accurate and at most
as costly with one of the two strictly better. A policy more than 0.03 below the initial policy's dev accuracy is
discarded.

**What the agents see.** A workspace (`experiments/<name>/work/<run>/`) holds a copy of `delovi/`, the policy,
`probe.py`, the dev questions and their memories, the runs (without the val runs), the shared cache, and
`AGENT_README.md` as `README.md`. The agents never see the val questions or the test benchmarks. With
`probe.py`, the agent can run its policy live on a few dev questions it chooses, up to $3 per iteration.

**Choosing DeLoVi-E.** After the last iteration, the policy on the frontier with the highest val accuracy
(`val acc` in `frontier.json`) is DeLoVi-E; its code is in `runs/<run>/policy/`.

All hyper-parameters are in [`config.py`](config.py):

| hyper-parameter | value |
|---|---|
| `AGENT_MODEL`, `AGENT_EFFORT` | `claude-opus-5-5`, high |
| `ITERATIONS` | 30 |
| `CHECK_QA_WEIGHT`, `FINAL_QA_WEIGHT` | 2, 4 |
| `INIT_FLOOR_MARGIN` | 0.03 |
| `BUILD_BUDGET_USD`, `SEARCH_BUDGET_USD` | $20, $20 per evaluation (nominal) |
| `PROBE_BUDGET_USD` | $3 per iteration |
| `PARALLEL` | 32 questions at once |

Output in `experiments/<name>/`: `runs/<run>/` (`policy/`, `summary.json`, `episodes/`, `iteration.json`,
`agent.log` with the prompt and the agent's reply, `select.log`), `frontier.json`, `LOG.md` and `cache/`.
Ctrl-C stops the agent and the evaluation; `iterate` again continues where the loop stopped.
`--dry-run` evaluates with a fake decision model and a fake LLM, to test the loop offline (the agent still runs;
`--agent-cmd` replaces it).
