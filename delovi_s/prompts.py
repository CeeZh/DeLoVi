"""DeLoVi-S requests for each evaluation benchmark: the score request to Jev and the QA prompt.

The score request is the same for every benchmark except the sentence that describes the clip and the video.
The QA prompts follow WorldMM's QA step (EgoLifeQA, EgoMemReason, VideoMME-L, LVBench) or an open-ended variant
of it (MM-Lifelong). For multiple choice the top-K clips are listed in ranked order; for MM-Lifelong in time order.
The split "video" is any single video (video_qa/), with the VideoMME-L requests.
"""

from __future__ import annotations

import re

SCORE_CRITERIA = [
    "`memory` involves different people, objects, places, and activities from the ones "
    "`question` asks about.",
    "`memory` shares the setting, people, or general activity of `question`, but does not "
    "mention the specific object, action, or event that `question` asks about.",
    "`memory` mentions the specific object, action, or event that `question` asks about, "
    "but not the detail needed to answer `question`.",
    "`memory` shows the moment that `question` asks about, with the detail needed to "
    "answer `question`.",
]
SCORE_ASK = "How useful is `memory` as evidence for answering `question`?"
OPTIONS_NOTE = " `answer_options` lists its candidate answers."

MML_CONTEXT = {
    "mml_day_test": (
        "`memory` describes a 30-second segment of a 24-hour recorded livestream in which a player "
        "plays the game Black Myth: Wukong; it gives the visual content and, after `Speech:`, the "
        "in-game dialogue (Chinese). `question` asks about events in that playthrough (\"the player\")."
    ),
    "mml_month_val": (
        "`memory` describes a 30-second segment of one of 23 livestreams recorded over about 51 days by "
        "a streamer travelling around China; it gives the visual content and, after `Speech:`, what was "
        "said (mostly the streamer, in English). `question` asks about events across those streams "
        "(\"the streamer\")."
    ),
    "mml_week_test": (
        "`memory` describes a 30-second clip of first-person video from Jake's head-mounted camera, "
        "recorded over seven days with his housemates; it gives the visual content and, after "
        "`Speech:`, the dialogue with speaker names. `question` asks about events in those days "
        "(\"the first-person protagonist\" is Jake)."
    ),
}

# WorldMM's QA system prompts (EgoLife and general video).
QA_EGOLIFE = """You are an AI assistant that answers questions about egocentric video experiences using retrieved memory context. Your task is to answer multiple choice questions based on this accumulated context. Always choose the most relevant answer from the given choices based on the evidence provided.\\

# Guidelines
- Analyze all provided context carefully.
- Choose the answer that best matches the evidence.
- If evidence is unclear, make the most reasonable inference.

# Output Format
Provide your answer as a single letter (A, B, C, or D) based on the evidence."""

QA_VIDEO = """You are an AI assistant that answers questions about video using retrieved memory context. Your task is to answer multiple choice questions based on this accumulated context. Always choose the most relevant answer from the given choices based on the evidence provided.\\

# Guidelines
- Analyze all provided context carefully.
- Choose the answer that best matches the evidence.
- If evidence is unclear, make the most reasonable inference.

# Output Format
Provide your answer as a single letter (A, B, C, or D) based on the evidence."""

# EgoMemReason's version of the WorldMM prompt (up to ten options).
QA_EMR = """You are an AI assistant that answers questions about egocentric video experiences using retrieved memory context. Your task is to answer multiple choice questions based on this accumulated context. Always choose the most relevant answer from the given choices based on the evidence provided.
# Guidelines
- Analyze all provided context carefully.
- Choose the answer that best matches the evidence.
- If evidence is unclear, make the most reasonable inference.

# Output Format
Provide your answer as a single letter corresponding to your chosen option based on the evidence."""

MC_FINAL = "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."

QA_MML = {
    "mml_day_test": (
        "You are an AI assistant that answers questions about a 24-hour recorded gameplay livestream "
        "(Black Myth: Wukong) using retrieved memory context: 30-second captions of the stream, each with "
        "the in-game dialogue after `Speech:`. \"The player\" is the streamer playing the game."
    ),
    "mml_month_val": (
        "You are an AI assistant that answers questions about a streamer's livestreams, recorded over about "
        "51 days while travelling around China, using retrieved memory context: 30-second captions of the "
        "streams, each with what was said after `Speech:`. \"The streamer\" is the person being filmed."
    ),
    "mml_week_test": (
        "You are an AI assistant that answers questions about a week of egocentric video from Jake's "
        "head-mounted camera using retrieved memory context: 30-second captions of the video, each with the "
        "dialogue (with speaker names) after `Speech:`. \"The first-person protagonist\" is Jake."
    ),
}
MML_GUIDELINES = """

# Guidelines
- Analyze all provided context carefully; the clips are ordered by time and each starts with its time.
- Base the answer on the evidence; if evidence is unclear, make the most reasonable inference.
- Answer the question directly and concisely.

# Output Format
Give only the final answer: a short phrase, a number, a time, a name, or an order of items.
If the question lists numbered options, answer with the option number(s) only."""
MML_FINAL = "\nPlease provide only the final answer, as briefly as possible."

SPLITS = ("egolifeqa", "egomemreason", "videomme_long", "lvbench", "mml_day_test", "mml_week_test", "mml_month_val")
# "video" (any single video, see video_qa/) uses the VideoMME-L and LVBench requests, also for open-ended questions.
VIDEO = ("videomme_long", "lvbench", "video")
QA_VIDEO_OPEN = ("You are an AI assistant that answers questions about a video using retrieved memory context: captions "
                 "of segments of the video, each with what was said after `Speech:`.")


def subject(q) -> str:
    """The person who asks an EgoLife question (the camera wearer)."""
    return q.extra["identity"].split("_", 1)[1].title() if "identity" in q.extra else "Jake"


def letters_text(options: dict) -> str:
    """'A, B, C, or D' for the letters of `options`."""
    ls = sorted(options)
    return ls[0] if len(ls) == 1 else f"{', '.join(ls[:-1])}, or {ls[-1]}"


def score_request(split: str, q, clips, clip_seconds: int = 10) -> tuple[dict, dict]:
    """The state and the score questions of one request over `clips`."""
    if split.startswith("mml_"):
        state = {"question": q.question}
        task = f"{MML_CONTEXT[split]} {SCORE_ASK}"
    elif split in VIDEO:
        state = {"question": q.question, **({"answer_options": q.options} if q.options else {})}
        task = (f"`memory` describes a {clip_seconds}-second clip of a video. `question` asks about the video."
                f"{OPTIONS_NOTE if q.options else ''} {SCORE_ASK}")
    else:
        state = {"question": q.question, "answer_options": q.options}
        who = subject(q)
        task = (f"`memory` describes a 30-second clip of first-person video from {who}'s head-mounted camera. "
                f"\"I\" in `memory` and in `question` both refer to {who}. {who} asked `question` at a later "
                f"time.{OPTIONS_NOTE} {SCORE_ASK}")
    questions = {f"c{c.index}": {"instructions": {"memory": c.text, "task": task}, "criteria": SCORE_CRITERIA}
                 for c in clips}
    return state, questions


def _hms(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def time_label(split: str, memory, clip) -> str:
    if split in ("egolifeqa", "egomemreason"):
        day = lambda t: f"DAY{int(t // 86400)} {_hms(t % 86400)}"  # noqa: E731
        return f"{day(clip.start)} - {day(clip.end)}"
    if split in VIDEO:
        return f"{_hms(clip.start)} - {_hms(clip.end)}"
    if split == "mml_week_test":
        return f"Day {int(clip.start // 86400)}, {_hms(clip.start % 86400)}"
    if split == "mml_month_val":
        offsets = memory.timeline["offsets"]
        stream = max((n for n, o in offsets.items() if o <= clip.start + 1e-6), key=lambda n: offsets[n])
        return f"Stream {stream}, {_hms(clip.start - offsets[stream])} into the stream"
    return f"{_hms(clip.start)} into the stream"


def qa_messages(split: str, q, memory, clips) -> list[dict]:
    """`clips`: the top-K clips in ranked order. Open-ended questions list them in time order."""
    if not q.options:
        clips = sorted(clips, key=lambda c: c.index)
    context = "\n\n".join(f"[{time_label(split, memory, c)}]\n{c.text}" for c in clips)
    if not q.options:
        system = QA_MML[split] if split.startswith("mml_") else QA_VIDEO_OPEN
        user = f"Query: {q.question}\n\nContext:\n{context}{MML_FINAL}"
        return [{"role": "system", "content": system + MML_GUIDELINES}, {"role": "user", "content": user}]
    choices = " ".join(f"({k}) {v}" for k, v in sorted(q.options.items()))
    if split == "egomemreason":
        system = QA_EMR
        final = f"\nPlease provide only the final answer from the choices given (e.g., {', '.join(sorted(q.options))})."
    elif split == "egolifeqa":
        system, final = QA_EGOLIFE, MC_FINAL
    else:
        system = QA_VIDEO.replace("(A, B, C, or D)", f"({letters_text(q.options)})")
        final = MC_FINAL.replace("(e.g., A, B, C, or D)", f"(e.g., {letters_text(q.options)})")
    user = f"Query: {q.question}\nChoices: {choices}\n\nContext:\n{context}{final}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_answer(split: str, q, text: str) -> str | None:
    if not q.options:
        text = re.sub(r"^(final answer|answer)\s*[:：]\s*", "", text.strip(), flags=re.IGNORECASE)
        return text.strip().strip("*").strip()
    text = text.strip().upper()
    if split in ("egomemreason", "video"):  # a bare letter first, otherwise the first standalone valid letter
        m = re.fullmatch(r"\(?([A-J])\)?[.:]?", text)
        if m and m.group(1) in q.options:
            return m.group(1)
        return next((m.group(1) for m in re.finditer(r"\b([A-J])\b", text) if m.group(1) in q.options), None)
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None
