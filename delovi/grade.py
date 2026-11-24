"""Scoring: exact letter match for multiple choice; for open-ended answers, an LLM judge compares the prediction
with the reference answer (0-5, reported as score / 5). This is MM-Lifelong's rubric-free judge (eval_acc.py in
its release), used for the month_dev questions of the dev and val sets, which have no rubrics. The MM-Lifelong
test splits are graded afterwards with the official rubric grader (scripts/grade_mmlifelong.py)."""

from __future__ import annotations

import re

from delovi.config import JUDGE_MODEL

JUDGE_REASONING = None

JUDGE_SYSTEM = """As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct “groundtruth” answer, and the candidate answer will be provided to you.
The following is a comparison table of some proper nouns; matching any one of them is considered correct.

You must FIRST provide a brief analysis explaining the semantic similarity between the groundtruth
and the candidate answer.

THEN, on a new line, output the final score.

Scoring criteria (semantic similarity only):

- 0: No similarity.
  The candidate answer is completely irrelevant, contradictory, or does not address the question at all.

- 1: Very low similarity.
  The candidate answer mentions a related topic or keyword, but fails to answer the question
  and does not convey the main meaning of the groundtruth.

- 2: Low similarity.
  The candidate answer addresses the question in a limited way, capturing some minor aspects,
  but misses or misrepresents the core idea or key facts of the groundtruth.

- 3: Moderate similarity.
  The candidate answer captures the main idea of the groundtruth,
  but omits several important details or includes noticeable inaccuracies.

- 4: High similarity.
  The candidate answer correctly captures the main idea and most key details of the groundtruth,
  with only minor omissions, simplifications, or non-critical inaccuracies.

- 5: Complete similarity.
  The candidate answer is semantically equivalent to the groundtruth,
  covering all essential information with no meaningful omissions or errors.

Special Rules:

- Hallucination-sensitive questions:
Score 5 only if all required items are correct;
if any item is incorrect, missing, or hallucinated, score 0 (no partial credit).

- Time-duration questions:
Allow errors within the range defined by the question; answers outside the range should receive score 0.

Output format (strictly follow):
Analysis:
<your analysis>

Final Score:
<an integer from 0 to 5>
"""


def judge_messages(question, prediction: str) -> list[dict]:
    user = (f"Question: {question.question}\nGroundtruth answer: {question.answer}\n"
            f"Candidate answer: {prediction}\nYour response: ")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_score(text: str) -> int:
    m = re.search(r"Final Score:\s*([0-5])", text, re.IGNORECASE) or re.search(r"\b([0-5])\b", text)
    return int(m.group(1)) if m else -1


async def grade(question, prediction: str | None, llm) -> tuple[float, dict]:
    """(score in [0, 1], detail). Multiple choice: 1.0 if the letter matches. Open: judge score / 5."""
    if question.multiple_choice:
        return float(prediction == question.answer), {"kind": "exact"}
    out = await llm.complete(JUDGE_MODEL, judge_messages(question, prediction or ""), JUDGE_REASONING)
    raw = parse_score(out.text)
    return max(raw, 0) / 5, {"kind": "judge", "model": out.model, "raw_score": raw, "response": out.text,
                             "input_tokens": out.input_tokens, "output_tokens": out.output_tokens,
                             "cost_usd": out.cost_usd, "cached": out.cached}
