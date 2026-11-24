# Searching Ultra-Long Videos with Decision Models

DeLoVi (**De**cision Models for **Lo**ng **Vi**deo) retrieves evidence from ultra-long videos with a pretrained
decision model. Like an LLM, a decision model reads the question together with the candidate evidence; unlike an
LLM, it outputs a relevance score directly, without generating text, so it is fast and inexpensive.

This repository contains:

| folder | what it is |
|---|---|
| [`delovi_s/`](delovi_s) | **DeLoVi-S**: the decision model scores the caption of every clip, and an LLM answers from the top 100. |
| [`hierarchical_search/`](hierarchical_search) | **Recursive hierarchical search**: the hand-designed policy that the evolution starts from. |
| [`delovi_e/`](delovi_e) | **DeLoVi-E**: the memory and retrieval policies found by automated policy evolution. |
| [`evolution/`](evolution) | **Automated policy evolution**: a coding agent rewrites the policies over a Pareto frontier of accuracy and cost. |
| [`video_qa/`](video_qa) | Answer a question about your own video file with any of the three methods. |
| [`delovi/`](delovi) | Shared code: data loading, the memory tree, the decision-model client, the LLM client, the QA step, grading and the evaluation harness. |
| [`scripts/`](scripts) | MM-Lifelong official grading, and re-answering saved contexts with another LLM. |

## Setup

```bash
pip install -r requirements.txt          # Python 3.10+
pip install -U huggingface_hub           # the `hf` command, to download the data
export OPENAI_API_KEY=...                # GPT-6 Luna (QA, summaries), GPT-5 (open-ended grading)
export TYPESAFE_API_KEY=...              # Jev, the decision model
```

The evolution also needs [Claude Code](https://docs.claude.com/en/docs/claude-code) (`claude` on the PATH), and
`video_qa/` needs `ffmpeg`.

Download the data from [Hugging Face](https://huggingface.co/datasets/ceezh/DeLoVi) into `data/`:

```bash
hf download ceezh/DeLoVi --repo-type dataset --local-dir data
```

All settings shared by the methods (models, prices, request limits) are in [`delovi/config.py`](delovi/config.py);
each method keeps its own hyper-parameters at the top of its files, and the evolution's are in
[`evolution/config.py`](evolution/config.py).

## Data

Every video is turned into a memory of short clips (10 s for VideoMME-L, LVBench, LongVideoBench, HourVideo,
VRBench and HD-EPIC; 30 s otherwise), each with a caption and, when there is speech, the transcript after `Speech:`.

```
data/questions/<split>.json                    a list of questions
data/memories/<dataset>/<name>/meta.json       what the recording is: description, timeline, number of clips
data/memories/<dataset>/<name>/clips.json      [{"start": seconds, "end": seconds, "text": caption}, ...] in time order
```

A question has `qid`, `dataset`, `memory_id`, `question`, `options` (letter -> text, or null for open-ended
questions), `answer`, `query_time` (seconds on the memory's timeline; null when asked after the whole video),
`evidence` (annotated spans, if any), `qtype` and `extra`.

| split | questions | use |
|---|---|---|
| `dev` | 500 | the evolution's development set (Ego-R1, MA-EgoQA, MM-Lifelong, LongVideoBench, HourVideo, VRBench, HD-EPIC) |
| `val` | 350 | the evolution's validation set, same sources |
| `egolifeqa` | 500 | EgoLifeQA |
| `egomemreason` | 500 | EgoMemReason |
| `mml_day_test`, `mml_week_test`, `mml_month_val` | 200, 200, 623 | MM-Lifelong |
| `videomme_long` | 900 | VideoMME-L |
| `lvbench` | 1549 | LVBench |

## Running

All commands run from the repository root. A benchmark run writes `runs/<run>/` (`summary.json`, and one
`episodes/<qid>.json` per question); add `--dry-run --limit 1` to check it offline with a fake decision model and
a fake LLM. API responses are cached, so an interrupted run resumes with the same command. The test splits are
`egolifeqa`, `egomemreason`, `mml_day_test`, `mml_week_test`, `mml_month_val`, `videomme_long` and `lvbench`.

### DeLoVi-S

Jev scores the caption of every clip before the question time, and GPT-6 Luna answers from the top 100 clips.

```bash
python -m delovi_s.run --split egolifeqa --run-name delovi_s_egolifeqa
python -m delovi_s.run --split egolifeqa --run-name delovi_s_egolifeqa_gpt5 --qa-model gpt-5 \
    --scores-from delovi_s_egolifeqa          # GPT-5 answers from the same rankings, no new Jev requests
```

It costs about $23 on EgoLifeQA and $135 on the MM-Lifelong month split, almost all of it for Jev. See
[`delovi_s/README.md`](delovi_s/README.md) for the hyper-parameters and the per-benchmark prompts.

### Recursive hierarchical search

The hand-designed policy that the evolution starts from: a tree of 3-minute, 15-minute and 1-hour summaries,
searched recursively with Jev. It runs through the shared harness (`delovi/evaluate.py`), which builds each
memory online and answers every question on the memory as it stood at the question time.

```bash
python -m delovi.evaluate --policy hierarchical_search --split egolifeqa --run-name hier_egolifeqa
```

See [`hierarchical_search/README.md`](hierarchical_search/README.md) for the algorithm and its hyper-parameters.

### DeLoVi-E

The evolved policy, run through the same harness.

```bash
python -m delovi.evaluate --policy delovi_e --split egolifeqa --run-name delovi_e_egolifeqa
python scripts/requery.py --run runs/delovi_e_egolifeqa --model gpt-5   # GPT-5 answers from the same final context
```

It costs about $2 on EgoLifeQA, including building the memory. `scripts/requery.py` works for the hierarchical
search too. See [`delovi_e/README.md`](delovi_e/README.md) for the memory tree, the retrieval algorithm and the
hyper-parameters.

### Automated policy evolution

Claude Code rewrites the memory and retrieval policies for 30 iterations, starting from the recursive
hierarchical search, and the policy on the dev frontier with the best val accuracy is DeLoVi-E.

```bash
python -m evolution.evolve --exp experiments/delovi_e iterate     # run again to resume after a stop
```

It costs about $75 of API calls for 30 iterations, plus the coding agent's usage. See
[`evolution/README.md`](evolution/README.md) for the loop, the objective and its hyper-parameters.

### Your own video

```bash
python -m video_qa.answer video.mp4 "What does the person squeeze onto the plates?" --method delovi_e
```

The first question on a video builds its memory (captions and Whisper speech, about $0.4 per hour of video),
which later questions reuse. See [`video_qa/README.md`](video_qa/README.md) for multiple-choice questions,
question times and the other options.

### MM-Lifelong grading

MM-Lifelong answers are graded with the benchmark's official rubric grader, with GPT-5 as the judge. It works on
the runs of every method:

```bash
python scripts/grade_mmlifelong.py --grader /path/to/MM-Lifelong/v1.1/grading/grade_predictions.py \
    runs/delovi_e_mml_day_test runs/delovi_e_mml_week_test runs/delovi_e_mml_month_val
```

## Results in the paper (GPT-6 Luna)

| method | EgoLifeQA | EgoMemReason | MM-Lifelong | VideoMME-L | LVBench | input tokens / question |
|---|---|---|---|---|---|---|
| DeLoVi-S | 72.6 | 36.4 | 36.9 | 74.6 | 66.8 | 1.32M |
| Recursive hierarchical search | 66.2 | 40.4 | 33.0 | 76.9 | 68.1 | 214.5K |
| DeLoVi-E | 69.6 | 37.6 | 34.5 | 77.2 | 68.1 | 38.6K |

MM-Lifelong is the official score averaged over its 1,023 questions; token counts are averaged over the five
benchmarks. The decision model and the LLM are not deterministic, so a rerun typically moves a benchmark's
accuracy by about one point.

## License

The code is released under the [Apache-2.0 license](LICENSE). Some prompts are adapted from WorldMM, EgoMemReason
and MM-Lifelong, as noted in the code. The data on Hugging Face is derived from the original benchmarks and follows
their licenses.
