# DeLoVi-E

The policy found by the automated policy evolution (`evolution/`), started from the recursive hierarchical
search. It is fixed and applied to every benchmark without further changes.

**Memory policy** (`builder.py`). A tree built online over the clips: 3-minute and 15-minute nodes hold free-text
summaries, an hour holds a timeline with one line per 3-minute window, and for videos that span several days or
streams, a day holds an index with one line per hour (plus a free-text summary of each hour, which the decision
model reads when it ranks the hours of a day). A single video of up to two hours gets one timeline of the whole
video, which the LLM reads for every question.

**Retrieval policy** (`retriever.py`, Algorithm 2 of the supplementary material).
1. The hierarchical search descends the tree to a 3-minute window. Its clips form the first evidence window, and
   the LLM answers with a confidence; above 0.8, the answer is returned.
2. Otherwise, BM25 shortlists the clips whose captions share words with the question and its options, and the
   decision model re-ranks them in one request.
3. Each re-ranked clip, with its neighbors within 90 s, forms a further evidence window. The LLM answers again
   after 2, 4 and 7 windows; the search stops when the confidence is above 0.8, when the answer repeats (from the
   4th window on), or after 7 windows.

Questions that need several moments (counts, order, first or last occurrences, changes; detected from their
wording) are not answered from the first window alone and also take the hierarchical search's next window.
Questions about one moment on recordings longer than 12 hours stop after the second window.

```bash
python -m delovi.evaluate --policy delovi_e --split egolifeqa --run-name delovi_e_egolifeqa
python -m delovi.evaluate --policy delovi_e --split dev --run-name dry --limit 1 --dry-run   # offline check
```

Hyper-parameters are at the top of the two files:

| hyper-parameter | value | meaning |
|---|---|---|
| `LEVELS` | 3 min / 15 min / 1 h | windows of the tree (and summary lengths) |
| `LINE_WORDS`, `TL_WORDS` | 60, 30 | words per line of a day index / an hour timeline |
| `VIDEO_WINDOWS` | 40 | a single video with at most 40 windows (2 h) gets a whole-video timeline |
| `MAX_WINDOWS` | 7 | most evidence windows |
| `CHECK_AT` | 1, 2, 4, 7 | windows after which the LLM answers (2, 4, 7 for several-moment questions) |
| `STOP_CONFIDENCE` | 0.8 | confidence that stops the search |
| `STABLE_FROM` | 4 | from this window on, a repeated answer also stops the search |
| `LEAF_CHARS`, `FIRST_LEAF_CHARS` | 4,000, 10,000 | caption characters per window (the first window is kept whole) |
| `POOL_TOKENS` | 12,000 | size of the BM25 shortlist that the decision model re-ranks |
| `LEX_HALF` | 90 s | half width of a keyword window |
| `LONG_HOURS`, `LONG_LAST_WINDOW` | 12, 2 | one-moment questions on longer recordings stop after window 2 |
| `MAX_TURNS` | 30 | most steps of the hierarchical search |
