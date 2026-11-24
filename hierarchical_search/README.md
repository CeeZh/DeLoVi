# Recursive hierarchical search

The hand-designed policy that the evolution starts from (Algorithm 1 of the supplementary material).

**Memory policy** (`builder.py`). The clips of the video are the leaves of a tree. Whenever a 3-minute window is
complete, an LLM summarizes its clips; the 3-minute summaries are summarized into 15-minute summaries, and those
into 1-hour summaries. The tree is built online, as the video is observed.

**Retrieval policy** (`retriever.py`). The decision model ranks the top-level nodes into a ranked list. At each
step, the top node of every ranked list competes in one ranking request; the winner is added to the evidence and
its children are ranked into a new list. So the search can go deeper into a node or move to another part of the
video. Every 5 steps, the LLM answers from the evidence and reports its confidence; the search stops once the
confidence is above 0.8, or after 50 steps.

```bash
python -m delovi.evaluate --policy hierarchical_search --split egolifeqa --run-name hier_egolifeqa
python -m delovi.evaluate --policy hierarchical_search --split dev --run-name dry --limit 1 --dry-run   # offline check
```

| hyper-parameter | value | file |
|---|---|---|
| `LEVELS` | 3 min / 15 min / 1 h windows; summaries of at most 450 / 600 / 1000 words | `builder.py` |
| `LLM_MODEL`, `LLM_REASONING` | `gpt-6-luna`, no reasoning (summaries) | `builder.py` |
| `MAX_TURNS` | 50 search steps | `retriever.py` |
| `QA_EVERY` | the LLM answers every 5 steps | `retriever.py` |
| `STOP_CONFIDENCE` | 0.8 | `retriever.py` |
