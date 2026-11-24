"""Questions and memories: data/questions/<split>.json and data/memories/<dataset>/<name>/{meta,clips}.json."""

from __future__ import annotations

import json
from dataclasses import dataclass

from delovi.config import DATA_DIR


@dataclass(frozen=True)
class Clip:
    index: int
    start: float  # seconds on the memory's timeline
    end: float
    text: str  # caption, with the transcript after "Speech:" when there is one


@dataclass(frozen=True)
class Memory:
    memory_id: str
    dataset: str
    timeline: dict  # how times are shown, see Tree.place
    description: str  # what the recording is; shown to the decision model and the LLM
    subject: str | None
    clips: list[Clip]
    meta: dict


@dataclass(frozen=True)
class Question:
    qid: str
    dataset: str
    memory_id: str
    question: str
    options: dict[str, str] | None  # letter -> text, or None for an open-ended question
    answer: str  # the letter, or the reference answer
    query_time: float | None  # seconds on the memory's timeline; None: asked after the whole video
    evidence: list[list[float]]  # annotated [start, end] spans, if any
    qtype: str
    extra: dict

    @property
    def multiple_choice(self) -> bool:
        return self.options is not None


def load_questions(split: str, datasets: list[str] | None = None, qids: list[str] | None = None,
                   limit: int | None = None) -> list[Question]:
    """The questions of a split; `limit` keeps the first N of each dataset."""
    qs = [Question(**r) for r in json.loads((DATA_DIR / "questions" / f"{split}.json").read_text())]
    if datasets:
        qs = [q for q in qs if q.dataset in datasets]
    if qids:
        wanted = set(qids)
        qs = [q for q in qs if q.qid in wanted]
    if limit:
        out, seen = [], {}
        for q in qs:
            if seen.get(q.dataset, 0) < limit:
                out.append(q)
                seen[q.dataset] = seen.get(q.dataset, 0) + 1
        qs = out
    return qs


def load_memory(memory_id: str) -> Memory:
    d = DATA_DIR / "memories" / memory_id
    meta = json.loads((d / "meta.json").read_text())
    clips = [Clip(i, c["start"], c["end"], c["text"]) for i, c in enumerate(json.loads((d / "clips.json").read_text()))]
    return Memory(memory_id, meta["dataset"], meta["timeline"], meta["description"], meta.get("subject"), clips, meta)
