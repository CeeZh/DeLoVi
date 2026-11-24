# DeLoVi-S

The simplest DeLoVi method. For each question, the decision model (Jev) scores the caption of every clip that
ended before the question time with a 4-level rubric (a *score* request), and the LLM answers once from the 100
best clips.

```bash
python -m delovi_s.run --split egolifeqa --run-name delovi_s_egolifeqa
python -m delovi_s.run --split egolifeqa --run-name dry --limit 2 --dry-run     # offline check
```

Splits: `egolifeqa`, `egomemreason`, `mml_day_test`, `mml_week_test`, `mml_month_val`, `videomme_long`, `lvbench`.
MM-Lifelong predictions are then graded with `scripts/grade_mmlifelong.py`.

To answer with another LLM from the same rankings (no new Jev requests):

```bash
python -m delovi_s.run --split egolifeqa --run-name delovi_s_egolifeqa_gpt5 --qa-model gpt-5 --scores-from delovi_s_egolifeqa
```

| hyper-parameter | value | where |
|---|---|---|
| `TOP_K` | 100 | clips the LLM reads (`--k`) |
| `CLIPS_PER_REQUEST` | 64 | consecutive clips per score request |
| `QA_REASONING` | None | the LLM's default reasoning effort |
| score rubric, QA prompts | | `prompts.py` |

`prompts.py` holds the requests of each benchmark. Only the sentence that describes the clip changes between
benchmarks. The QA prompts follow WorldMM's QA step; the clips are listed in ranked order for the
multiple-choice benchmarks and in time order for MM-Lifelong. Clips are ranked by their expected level, with
ties going to the later clip.

Output in `runs/<run>/`: `scores/<qid>.json` (the ranking of every candidate clip), `episodes/<qid>.json`,
`summary.json`.
