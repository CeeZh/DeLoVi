"""The fixed QA step of the policy harness: the LLM answers from the nodes a retrieval policy chose.

The context is every chosen node's text in time order, each headed by its time span and duration. The LLM
replies with two lines, `Answer: ...` and `Confidence: <0-1>`; retrieval policies use the confidence to decide
when to stop searching. Multiple-choice answers are one letter; open-ended answers are a short phrase.
"""

from __future__ import annotations

import re

from delovi.config import QA_MODEL, QA_REASONING  # noqa: F401
from delovi.tree import Tree

MC_SYSTEM = """You are an AI assistant that answers questions about a long video recording using retrieved memory context. The context holds captions of segments of the recording, found by a search of the recording's memory for the question and listed in time order; the caption of a long segment is a summary, and a short segment may lie inside a longer one that is also listed. Your task is to answer multiple choice questions based on this accumulated context. Always choose the most relevant answer from the given choices based on the evidence provided.

# Guidelines
- Analyze all provided context carefully.
- Choose the answer that best matches the evidence.
- If evidence is unclear, make the most reasonable inference.
- The search may not have found the relevant moment yet. Your confidence must reflect whether the context actually contains the evidence for your answer: high only when it does, low when you are inferring or guessing.

# Output Format
Exactly two lines:
Answer: <a single letter ({letters})>
Confidence: <a number between 0 and 1, the probability that your answer is correct given this context>"""
MC_FINAL = "\nReply with exactly two lines: `Answer: <one of {letters}>` and `Confidence: <0 to 1>`."

# Open-ended questions (MM-Lifelong): what the recording is, per dataset.
OPEN_SYSTEM = {
    "mmlifelong": (
        "You are an AI assistant that answers questions about a streamer's livestreams, recorded over about "
        "51 days while travelling around China, using retrieved memory context: captions of segments of the "
        "streams, each with what was said after `Speech:`. \"The streamer\" is the person being filmed."),
    "mmlday": (
        "You are an AI assistant that answers questions about a 24-hour recorded gameplay livestream "
        "(Black Myth: Wukong) using retrieved memory context: captions of segments of the stream, each with the "
        "in-game dialogue after `Speech:`. \"The player\" is the streamer playing the game."),
    "mmlweek": (
        "You are an AI assistant that answers questions about a week of egocentric video from Jake's head-mounted "
        "camera using retrieved memory context: captions of segments of the video, each with the dialogue (with "
        "speaker names) after `Speech:`. \"The first-person protagonist\" is Jake."),
    "video": (  # any single video (video_qa/)
        "You are an AI assistant that answers questions about a video using retrieved memory context: captions of "
        "segments of the video, each with what was said after `Speech:`."),
}
OPEN_GUIDELINES = """

# Guidelines
- Analyze all provided context carefully; the captions are ordered by time and each starts with its time span.
- Base the answer on the evidence; if evidence is unclear, make the most reasonable inference.
- Answer the question directly and concisely.
- The search may not have found the relevant moment yet. Your confidence must reflect whether the context actually contains the evidence for your answer: high only when it does, low when you are inferring or guessing.

# Output Format
Exactly two lines:
Answer: <only the final answer: a short phrase, a number, a time, a name, or an order of items; if the question lists numbered options, the option number(s) only>
Confidence: <a number between 0 and 1, the probability that your answer is correct given this context>"""
OPEN_FINAL = "\nReply with exactly two lines: `Answer: <as briefly as possible>` and `Confidence: <0 to 1>`."


def context_text(tree: Tree, nodes: list[int]) -> str:
    ordered = sorted((tree[n] for n in nodes), key=lambda n: (n.start, -n.end))
    return "\n\n".join(f"[{tree.span_label(n)}] ({tree.duration_label(n)})\n{n.text}" for n in ordered)


def messages(question, tree: Tree, nodes: list[int]) -> list[dict]:
    context = context_text(tree, nodes)
    if question.multiple_choice:
        letters = ", ".join(sorted(question.options))
        choices = " ".join(f"({k}) {v}" for k, v in sorted(question.options.items()))
        when = f"\nQuestion time: {tree.time_label(question.query_time)}" if question.query_time is not None else ""
        user = (f"Recording: {tree.description}\nQuery: {question.question}{when}\nChoices: {choices}\n\n"
                f"Context:\n{context}{MC_FINAL.format(letters=letters)}")
        return [{"role": "system", "content": MC_SYSTEM.format(letters=letters)}, {"role": "user", "content": user}]
    user = f"Query: {question.question}\n\nContext:\n{context}{OPEN_FINAL}"
    return [{"role": "system", "content": OPEN_SYSTEM[question.dataset] + OPEN_GUIDELINES},
            {"role": "user", "content": user}]


ANSWER_LINE = re.compile(r"^[\s*_#>-]*(?:final\s+)?answer[\s*_]*[:：]\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
CONFIDENCE = re.compile(r"confidence[\s*_]*[:：=]?[\s*_]*(\d+(?:\.\d+)?|\.\d+)\s*(%|/\s*\d+(?:\.\d+)?)?", re.IGNORECASE)


def parse(question, text: str) -> str | None:
    """The predicted letter (None if unparsable), or the short open-ended answer."""
    m = ANSWER_LINE.search(text)
    body = m.group(1) if m else re.sub(r"^.*confidence.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    if question.multiple_choice:
        letters = "".join(sorted(question.options))
        m = re.search(rf"\b([{letters}])\b", body.strip().upper())
        return m.group(1) if m else None
    return body.strip().strip("*").strip() or None


def parse_confidence(text: str) -> float | None:
    """The confidence in [0, 1] ("0.8", "80%", "8/10"; a bare number above 1 is a percentage), or None."""
    m = CONFIDENCE.search(text)
    if not m:
        return None
    v, unit = float(m.group(1)), (m.group(2) or "").replace(" ", "")
    if unit.startswith("/"):
        v = v / float(unit[1:]) if float(unit[1:]) else 0.0
    elif unit == "%" or v > 1:
        v /= 100
    return min(max(v, 0.0), 1.0)
