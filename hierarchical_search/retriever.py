"""Retrieval policy of the recursive hierarchical search (Algorithm 1 of the supplementary material).

    result = await retrieve(question, query_time, tree, jev, ask)

The search keeps a set of ranked lists, one per opened node: Jev ranks the node's children. At each step, the
top node of every list competes in one more ranking request; the winner is removed from its list and added to
the evidence, and its children are ranked into a new list. So the search can go deeper or move to another part
of the video. Every QA_EVERY steps, the LLM answers from the evidence (`ask`) and the search stops once its
confidence is above STOP_CONFIDENCE, or after MAX_TURNS steps. The last answer is the prediction.

Requests that do not fit Jev's limit are split into contiguous groups. Ties go to the longer, then the earlier
segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from delovi.jev import Jev, TooLong, estimate, option_tokens
from delovi.tree import Tree

# Hyper-parameters
MAX_TURNS = 50  # most search steps per question
QA_EVERY = 5  # ask the LLM every QA_EVERY steps (0: only once, at the end)
STOP_CONFIDENCE = 0.8  # stop as soon as the LLM's confidence is above this

INSTRUCTIONS = (
    "`question` asks about the recording described in `memory`.{options}{time} Each choice is the caption of one "
    "segment of the recording. Segments differ in length; the caption of a long segment is a summary, so the "
    "moment `question` asks about can lie inside a long segment even when its caption mentions it only briefly. "
    "Which segment contains the moment that answers `question`?"
)


@dataclass
class Step:
    turn: int
    nid: int  # the node recorded at this step
    list_id: int  # the ranked list it came from
    candidates: list[int]  # the list heads that competed
    probs: dict[int, float]  # Jev's probabilities over them ({} when there was one head)
    children: int  # visible children, ranked into new list(s)


@dataclass
class RetrievalResult:
    recorded: list[int] = field(default_factory=list)  # the evidence, in the order found
    steps: list[Step] = field(default_factory=list)
    lists: int = 0  # ranked lists opened
    checks: list = field(default_factory=list)  # what `ask` returned, in order (the last one is the prediction)
    stop: str = ""  # why the search ended


def build_state(question, query_time: float | None, tree: Tree) -> tuple[dict, str]:
    state = {"question": question.question, "memory": tree.description}
    if question.options:
        state["answer_options"] = question.options
    if query_time is not None:
        state["question_time"] = tree.time_label(query_time)
    text = INSTRUCTIONS.format(
        options=" `answer_options` lists its candidate answers." if question.options else "",
        time=" The question was asked at `question_time`; every segment below ended before then." if query_time is not None else "")
    return state, text


def label(nid: int) -> str:
    return f"s{nid}"


def nid_of(lbl: str) -> int:
    return int(lbl[1:])


def option(tree: Tree, nid: int) -> dict:
    node = tree[nid]
    return {"time": tree.span_label(node), "duration": tree.duration_label(node), "caption": node.text}


def order(tree: Tree, probs: dict[int, float]) -> list[int]:
    """Nodes by probability; ties to the longer segment, then the earlier one."""
    return sorted(probs, key=lambda n: (-probs[n], -tree[n].duration, tree[n].start))


async def rank(jev: Jev, state: dict, text: str, tree: Tree, nids: list[int]) -> list[list[int]]:
    """`nids` as one or more ranked lists: one ranking request per group that fits a request."""
    if len(nids) <= 1:
        return [list(nids)] if nids else []
    while True:
        groups = jev.groups(state, text, [(label(n), option(tree, n)) for n in nids])
        try:
            out = []
            for g in groups:
                if len(g) == 1:
                    out.append([nid_of(g[0])])
                    continue
                probs = await jev.choice(state, text, {l: option(tree, nid_of(l)) for l in g}, tag="rank")
                out.append(order(tree, {nid_of(l): p for l, p in probs.items()}))
            return out
        except TooLong:
            continue  # the budget shrank; regroup


def cut_to_share(jev: Jev, state: dict, text: str, tree: Tree, nids: list[int]) -> dict[str, dict]:
    """Options whose captions are cut to an equal share of one request; used only when no two of them fit one
    request together."""
    share = (jev.budget - estimate(state, text, {})) / len(nids)
    out = {}
    for n in nids:
        opt = option(tree, n)
        while option_tokens(label(n), opt) > share and opt["caption"]:
            keep = int(len(opt["caption"]) * share / option_tokens(label(n), opt) * 0.95)
            opt = {**opt, "caption": opt["caption"][:keep]}
        out[label(n)] = opt
    return out


async def top1(jev: Jev, state: dict, text: str, tree: Tree, nids: list[int]) -> tuple[int, dict[int, float]]:
    """The most likely of `nids` and the probabilities of the request that decided it."""
    if len(nids) == 1:
        return nids[0], {}
    while True:
        groups = jev.groups(state, text, [(label(n), option(tree, n)) for n in nids])
        try:
            if len(groups) == 1 or len(groups) == len(nids):  # one request, or no two fit together
                options = ({label(n): option(tree, n) for n in nids} if len(groups) == 1
                           else cut_to_share(jev, state, text, tree, nids))
                probs = await jev.choice(state, text, options, tag="compare")
                probs = {nid_of(l): p for l, p in probs.items()}
                return order(tree, probs)[0], probs
            winners = []
            for g in groups:
                w, _ = await top1(jev, state, text, tree, [nid_of(l) for l in g])
                winners.append(w)
            return await top1(jev, state, text, tree, winners)
        except TooLong:
            continue


async def retrieve(question, query_time: float | None, tree: Tree, jev: Jev, ask, max_turns: int = MAX_TURNS,
                   qa_every: int = QA_EVERY, stop_confidence: float = STOP_CONFIDENCE) -> RetrievalResult:
    """`ask(nodes, turn)` is the fixed QA step; an unparsable confidence (None) counts as not confident."""
    view = tree.view(query_time)
    state, text = build_state(question, query_time, tree)
    lists = await rank(jev, state, text, tree, view.roots())
    result = RetrievalResult(lists=len(lists))
    asked_at = -1  # number of recorded nodes the last check saw
    for turn in range(1, max_turns + 1):
        heads = [(lst[0], i) for i, lst in enumerate(lists) if lst]
        if not heads:
            result.stop = "every list empty"
            break
        nid, probs = await top1(jev, state, text, tree, [h for h, _ in heads])
        list_id = next(i for h, i in heads if h == nid)
        lists[list_id].pop(0)
        result.recorded.append(nid)
        kids = view.children(nid)
        result.steps.append(Step(turn, nid, list_id, [h for h, _ in heads], probs, len(kids)))
        if qa_every and turn % qa_every == 0:
            check = await ask(result.recorded, turn=turn)
            result.checks.append(check)
            asked_at = len(result.recorded)
            if check.confidence is not None and check.confidence > stop_confidence:
                result.stop = f"confidence {check.confidence:.2f} > {stop_confidence} after turn {turn}"
                break
        if kids and turn < max_turns:  # a list opened after the last step would never be compared
            new = await rank(jev, state, text, tree, kids)
            lists.extend(new)
            result.lists += len(new)
    else:
        result.stop = f"turns ({max_turns})"
    if asked_at != len(result.recorded):  # the prediction sees every recorded node
        result.checks.append(await ask(result.recorded, turn=len(result.steps)))
    return result
