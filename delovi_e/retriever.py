"""Retrieval policy of DeLoVi-E (Algorithm 2 of the supplementary material).

    result = await retrieve(question, query_time, tree, jev, ask)

The evidence is gathered as windows of consecutive clips from two sources:
- Tree windows: the hierarchical search of hierarchical_search/retriever.py (ranked lists compared through their
  heads). When the selected node is a bottom window, its clips are added together as one window. A list of
  15-minute nodes is ranked directly as their 3-minute windows when these fit one request (flat windows), and a
  head that lost to a head still at the top of its list does not compete again.
- Keyword windows: BM25 matches the clip captions against the question and its options, Jev re-ranks the best
  clips in one request, and each re-ranked clip forms a window with its neighbors (LEX_HALF seconds either side).

The first window comes from the tree and the LLM answers from it (`ask`). If the answer is not confident, the
keyword channel opens and gives the later windows. Questions that need several moments (counts, order,
first/last occurrences, changes) alternate tree and keyword windows up to window TREE_WINDOWS_UNTIL and are not
answered from the first window alone. The LLM answers after the windows in CHECK_AT, and the search stops when
the confidence is above STOP_CONFIDENCE, when the answer repeats from window STABLE_FROM on, or after MAX_WINDOWS
windows. Questions about one moment on recordings with more than LONG_HOURS of video end after window
LONG_LAST_WINDOW. An open-ended answer saying that the context lacks the information never stops the search.
"""

from __future__ import annotations

import math
import re
import weakref
from dataclasses import dataclass, field

from delovi.jev import Jev, TooLong, estimate, option_tokens
from delovi.tree import Tree

# Hyper-parameters
MAX_TURNS = 30  # most tree search steps per question
MAX_WINDOWS = 7  # most evidence windows per question
TREE_WINDOWS_UNTIL = 4  # several-moment questions: tree and keyword windows alternate up to this window
TREE_WINDOWS_UNTIL_ONE = 1  # one-moment questions: the tree gives the first window only
CHECK_AT = (1, 2, 4, 7)  # the LLM answers after these windows
CHECK_AT_SEVERAL = (2, 4, 7)  # ... for questions that need several moments
SEVERAL_STOP_FROM = 4  # several-moment questions stop on confidence only from this window on
LONG_HOURS = 12  # recordings with more video than this (hours) ...
LONG_LAST_WINDOW = 2  # ... end one-moment questions after this window
STOP_CONFIDENCE = 0.8  # stop as soon as the LLM's confidence is above this
STABLE_FROM = 4  # from this window on, an answer equal to the previous check's also stops the search
LEAF_CHARS = 4000  # most caption characters of one window; the best-ranked clips are kept
FIRST_LEAF_CHARS = 10_000  # the same for the first tree window, which is kept whole in practice
POOL_TOKENS = 12_000  # estimated tokens of the BM25 clips that Jev re-ranks
LEX_HALF = 90  # a keyword window: the clips within this many seconds of the matched clip
BM25_K1, BM25_B = 1.2, 0.75  # BM25 term-frequency saturation and length normalization
FLAT_TOKENS = 12_000  # a list of 15-minute nodes is ranked as their 3-minute windows if those fit this

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
    list_id: int  # the ranked list it came from (-1: a clip of a tree window, -2: keyword window, -3: path node)
    candidates: list[int]  # the list heads that competed
    probs: dict[int, float]  # Jev's probabilities over them ({} when there was one head)
    children: int  # visible children


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


# An open-ended answer saying that the context does not contain the information.
ABSTAIN = re.compile(
    r"\b(not|never|isn['’]t|aren['’]t|wasn['’]t) (explicitly |clearly )?(specified|stated|mentioned|described|given|"
    r"provided|identified|identifiable|shown in|indicated|determinable|recorded|named|available|visible)|"
    r"(doesn['’]t|does not|didn['’]t|did not|don['’]t|do not) (explicitly |clearly )?(say|state|specify|mention|describe|identify|"
    r"show|indicate|include|contain|name|provide)|"
    r"\b(cannot|can['’]t|unable to) (be )?(determine|tell|identify|answer)|"
    r"\b(no|insufficient|not enough) (information|evidence|mention|details?)\b|^(unknown|unclear|n/?a)\.?$",
    re.IGNORECASE)


def abstains(question, answer) -> bool:
    """An open-ended answer all of whose clauses say the context lacks the information (or talk about it)."""
    if question.multiple_choice or not answer:
        return False
    clauses = [c for c in re.split(r";|\.\s|\s[—–-]\s", str(answer)) if c.strip(" .")]
    return bool(clauses) and bool(ABSTAIN.search(clauses[0])) and all(
        ABSTAIN.search(c) or re.search(r"\b(context|clips?|captions?)\b", c, re.IGNORECASE) for c in clauses)


# Words of a question whose answer depends on several moments: which occurrence, an earlier state, counts and
# frequencies, order, change and comparison ("in order to" states a purpose, not an order).
SEVERAL_MOMENTS = re.compile(
    r"\b(last time|the last|latest|most recent(ly)?|previous(ly)?|earlier|first time|the first|ever|again|"
    r"how many|how often|how much time|times|each time|every time|in total|total|more (often|frequently)|"
    r"most (often|frequently)|frequen\w*|sequence|(?<!in )order|in turn|chang\w*|compar\w*|differen\w*)\b|"
    r"\bbefore\s*\?*\s*$",
    re.IGNORECASE)


def several_moments(question) -> bool:
    return bool(SEVERAL_MOMENTS.search(question.question.strip()))


def same_answer(question, a, b) -> bool:
    if a is None or b is None or abstains(question, a):
        return False
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()
    return norm(a) == norm(b)


def label(nid: int) -> str:
    return f"s{nid}"


def nid_of(lbl: str) -> int:
    return int(lbl[1:])


# Day / stream node text: index lines, then after ROUTING_MARK one routing summary per child (see builder.py).
ROUTING_MARK = "\n\n<<ROUTING>>\n"
ROUTING_SEP = "\n<<NEXT>>\n"
_CAPTIONS: "weakref.WeakKeyDictionary[Tree, dict[int, str]]" = weakref.WeakKeyDictionary()


def caption(tree: Tree, nid: int) -> str:
    """What Jev reads for a node: a day node's index lines without the routing summaries, an hour's routing
    summary when its day node holds one, else the node's text."""
    cache = _CAPTIONS.setdefault(tree, {})
    if nid not in cache:
        node = tree[nid]
        text = node.text
        if ROUTING_MARK in text:
            text = text.split(ROUTING_MARK, 1)[0]
        elif node.parent is not None and ROUTING_MARK in tree[node.parent].text:
            parent = tree[node.parent]
            routes = parent.text.split(ROUTING_MARK, 1)[1].split(ROUTING_SEP)
            if len(routes) == len(parent.children) and routes[parent.children.index(nid)].strip() not in ("", "-"):
                text = routes[parent.children.index(nid)].strip()
        cache[nid] = text
    return cache[nid]


def option(tree: Tree, nid: int) -> dict:
    node = tree[nid]
    return {"time": tree.span_label(node), "duration": tree.duration_label(node), "caption": caption(tree, nid)}


INDEX_LINE = re.compile(r"^\[[^\]\n]+\] ")  # a day / stream index: one line per hour, starting with its time span
TIMELINE = re.compile(r"^\d\d:\d\d-\d\d:\d\d ")  # a timeline: one line per 3-minute window, starting with its clock span


def is_index(tree: Tree, n: int) -> bool:
    return not tree[n].is_leaf and bool(INDEX_LINE.match(tree[n].text))


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


async def window(jev: Jev, state: dict, text: str, tree: Tree, leaves: list[int], cap: int = LEAF_CHARS) -> list[int]:
    """The clips of a selected bottom window: all of them when they fit `cap` characters, else the ones Jev ranks
    highest, while they fit."""
    if sum(len(tree[n].text) for n in leaves) <= cap:
        return sorted(leaves, key=lambda n: tree[n].start)
    lists = await rank(jev, state, text, tree, leaves)
    ranked = [lst[i] for i in range(max(map(len, lists))) for lst in lists if i < len(lst)]
    out, chars = [], 0
    for n in ranked:
        if out and chars + len(tree[n].text) > cap:
            break
        out.append(n)
        chars += len(tree[n].text)
    return sorted(out, key=lambda n: tree[n].start)


# -- keyword channel --

STOPWORDS = frozenset(
    "a an the of to in on at for and or but nor is are was were be been being am do does did done doing have has had "
    "having i me my mine myself you your yours he him his she her hers it its we us our ours they them their theirs "
    "this that these those what which who whom whose when where why how with by from as about into onto after before "
    "during while than then there here not no yes so if just can could would should will may might shall must up down "
    "out over under again once all any both each few more most other some such only own same too very also now "
    "following happened happen time times first last one two s t don".split())
_TERMS: dict[str, tuple[dict, int]] = {}  # caption -> (term counts, length)


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def terms(text: str) -> list[str]:
    """Lower-case words (without stopwords, crudely stemmed) and, for Chinese text, character pairs."""
    text = text.lower()
    out = [stem(w) for w in (x.strip("'") for x in re.findall(r"[a-z0-9']+", text)) if len(w) > 1 and w not in STOPWORDS]
    for seg in re.findall(r"[一-鿿]+", text):
        out += [seg[i:i + 2] for i in range(len(seg) - 1)] or [seg]
    return out


def caption_terms(text: str) -> tuple[dict, int]:
    if text not in _TERMS:
        ts = terms(text)
        counts: dict[str, int] = {}
        for t in ts:
            counts[t] = counts.get(t, 0) + 1
        _TERMS[text] = (counts, len(ts))
    return _TERMS[text]


def bm25(question, tree: Tree, leaves: list[int]) -> list[tuple[float, int]]:
    """(score, clip) for the visible clips that share a word with the question or its options, best first."""
    query = question.question + (" " + " ".join(map(str, question.options.values())) if question.options else "")
    qt = frozenset(terms(query))
    docs = [caption_terms(tree[n].text) for n in leaves]
    if not qt or not docs:
        return []
    avg = sum(d[1] for d in docs) / len(docs) or 1.0
    matched = [(i, qt & d[0].keys()) for i, d in enumerate(docs)]
    df: dict[str, int] = {}
    for _, m in matched:
        for t in m:
            df[t] = df.get(t, 0) + 1
    n_docs = len(docs)
    idf = {t: math.log(1 + (n_docs - k + 0.5) / (k + 0.5)) for t, k in df.items()}
    out = []
    for i, m in matched:
        if not m:
            continue
        counts, length = docs[i]
        norm = BM25_K1 * (1 - BM25_B + BM25_B * length / avg)
        out.append((sum(idf[t] * counts[t] * (BM25_K1 + 1) / (counts[t] + norm) for t in m), leaves[i]))
    out.sort(key=lambda x: (-x[0], tree[x[1]].start))
    return out


async def lexical_seeds(jev: Jev, state: dict, text: str, tree: Tree, question,
                        leaves: list[int]) -> tuple[list[int], dict[int, float]]:
    """The clips the keyword channel visits, in order: the best BM25 clips (POOL_TOKENS of them) re-ranked by
    Jev, with the probabilities of that request."""
    pool, tokens = [], 0.0
    for _, n in bm25(question, tree, leaves):
        t = option_tokens(label(n), option(tree, n))
        if pool and tokens + t > POOL_TOKENS:
            break
        pool.append(n)
        tokens += t
    if len(pool) <= 1:
        return pool, {}
    pool.sort(key=lambda n: tree[n].start)  # candidates in time order, like every other list
    while True:
        groups = jev.groups(state, text, [(label(n), option(tree, n)) for n in pool])
        try:
            probs, lists = {}, []
            for g in groups:
                if len(g) == 1:
                    lists.append([nid_of(g[0])])
                    continue
                p = await jev.choice(state, text, {l: option(tree, nid_of(l)) for l in g}, tag="lexical")
                p = {nid_of(l): v for l, v in p.items()}
                probs.update(p)
                lists.append(order(tree, p))
            break
        except TooLong:
            continue
    return [lst[i] for i in range(max(map(len, lists))) for lst in lists if i < len(lst)], probs


def around(tree: Tree, leaves: list[int], pos: dict[int, int], seed: int, taken: set[int]) -> list[int]:
    """A keyword window: `seed` and the clips within LEX_HALF seconds of it, nearest first while their captions
    fit LEAF_CHARS, without clips already recorded; [] when `seed` was recorded already."""
    if seed in taken:
        return []
    mid = lambda n: (tree[n].start + tree[n].end) / 2
    c, i = mid(seed), pos[seed]
    near = []
    for step in (-1, 1):
        j = i + step
        while 0 <= j < len(leaves) and abs(mid(leaves[j]) - c) <= LEX_HALF:
            near.append(leaves[j])
            j += step
    near.sort(key=lambda n: abs(mid(n) - c))
    out, chars = [seed], len(tree[seed].text)
    for n in near:
        if n in taken:
            continue
        if chars + len(tree[n].text) > LEAF_CHARS:
            break
        out.append(n)
        chars += len(tree[n].text)
    return sorted(out, key=lambda n: tree[n].start)


def flatten(tree: Tree, view, nids: list[int], via: dict[int, int]) -> list[int]:
    """Flat windows: replace every node of `nids` whose children are all bottom windows (or lone clips) by those
    children, when no node of `nids` lies higher and the children fit FLAT_TOKENS; else `nids` unchanged.
    `via` maps each such window to the node it replaced."""
    def bottom(n: int) -> bool:
        return not tree[n].is_leaf and all(tree[k].is_leaf for k in view.children(n))

    def mid(n: int) -> bool:
        return not tree[n].is_leaf and not bottom(n) and all(tree[k].is_leaf or bottom(k) for k in view.children(n))

    if not any(mid(n) for n in nids) or not all(tree[n].is_leaf or bottom(n) or mid(n) for n in nids):
        return nids
    out = [k for n in nids for k in (view.children(n) if mid(n) else [n])]
    if sum(option_tokens(label(n), option(tree, n)) for n in out) > FLAT_TOKENS:
        return nids
    for n in nids:
        if mid(n):
            for k in view.children(n):
                via[k] = n
    return sorted(out, key=lambda n: (tree[n].start, -tree[n].duration))


async def retrieve(question, query_time: float | None, tree: Tree, jev: Jev, ask, max_turns: int = MAX_TURNS,
                   max_windows: int = MAX_WINDOWS, stop_confidence: float = STOP_CONFIDENCE) -> RetrievalResult:
    """`ask(nodes, turn)` is the fixed QA step; an unparsable confidence (None) counts as not confident."""
    view = tree.view(query_time)
    state, text = build_state(question, query_time, tree)
    via: dict[int, int] = {}  # flat window -> the 15-minute node it was ranked in place of
    paths: set[int] = set()  # 15-minute nodes recorded only as the path to a flat window
    lists = await rank(jev, state, text, tree, flatten(tree, view, view.roots(), via))
    result = RetrievalResult(lists=len(lists))
    leaves = sorted((n.nid for n in tree.nodes if n.is_leaf and view.visible(n.nid)), key=lambda n: tree[n].start)
    pos = {n: i for i, n in enumerate(leaves)}
    seeds: list[int] | None = None  # the keyword channel's clips, once it is open
    seed_probs: dict[int, float] = {}
    taken: set[int] = set()  # clips recorded so far
    asked: list[int] | None = None  # the nodes the last check saw
    seen: list[list[int]] = []  # the nodes each check saw, in order
    windows, turn = 0, 0
    tree_windows = 0  # tree windows recorded so far
    lexical_next = False  # whose window comes next (the tree's first)
    beaten: dict[int, set[int]] = {}  # head -> the heads that beat it in a comparison
    several = several_moments(question)
    check_at = CHECK_AT_SEVERAL if several else CHECK_AT
    tree_until = TREE_WINDOWS_UNTIL if several else TREE_WINDOWS_UNTIL_ONE
    long_last = (not several and sum(tree[n].end - tree[n].start for n in leaves) > LONG_HOURS * 3600)

    def qa_nodes(nodes: list[int]) -> list[int]:
        """The nodes the LLM reads: `nodes` without the texts that repeat others in the context."""
        have = set(nodes)
        out = []
        for n in nodes:
            if not tree[n].is_leaf:
                kids = view.children(n)
                if kids and all(tree[k].is_leaf and k in have for k in kids):
                    continue  # a bottom window whose clips are all in the context
                if is_index(tree, n) and (ROUTING_MARK in tree[n].text or any(k in have for k in kids)):
                    continue  # a day index once one of its hours is in the context, or one holding routing text
                if "\n" in tree[n].text and any(
                        m != n and tree[m].start <= tree[n].start and tree[m].end >= tree[n].end
                        and len(tree[m].text) > len(tree[n].text) and tree[n].text in tree[m].text
                        for m in have if not tree[m].is_leaf):
                    continue  # an hour timeline whose lines the video timeline already holds
                if n in paths and any(m != n and tree[m].start <= tree[n].start and tree[m].end >= tree[n].end
                                      and TIMELINE.match(tree[m].text) for m in have if not tree[m].is_leaf):
                    continue  # the 15-minute summary a flat window was ranked in place of
            out.append(n)
        return out

    async def new_window(source: str) -> str | None:
        """A window was recorded: answer if due; returns the reason to stop, if any."""
        nonlocal windows, asked
        windows += 1
        if windows in check_at:
            asked = list(result.recorded)
            check = await ask(qa_nodes(asked), turn=turn)
            seen.append(asked)
            result.checks.append(check)
            real = not abstains(question, check.answer)
            if (real and check.confidence is not None and check.confidence > stop_confidence
                    and (not several or windows >= SEVERAL_STOP_FROM)):
                return f"confidence {check.confidence:.2f} > {stop_confidence} after window {windows} ({source}), turn {turn}"
            if (windows >= STABLE_FROM and len(result.checks) >= 2
                    and same_answer(question, check.answer, result.checks[-2].answer)):
                return f"answer stable after window {windows} ({source}), turn {turn}"
            if long_last and real and windows >= LONG_LAST_WINDOW:
                return f"one moment on a long recording: last window {windows} ({source}), turn {turn}"
        if windows >= max_windows:
            return f"windows ({max_windows})"
        return None

    while True:
        if windows >= tree_until:  # the later windows are the keyword channel's
            lexical_next = True
        if lexical_next and seeds is None:  # open the keyword channel
            seeds, seed_probs = await lexical_seeds(jev, state, text, tree, question, leaves)
            result.lists += bool(seeds)
        if lexical_next and seeds:
            got: list[int] = []
            while seeds and not got:
                seed = seeds.pop(0)
                got = around(tree, leaves, pos, seed, taken)
            if got:
                for leaf in got:
                    result.recorded.append(leaf)
                    taken.add(leaf)
                    result.steps.append(Step(turn, leaf, -2, [seed], {seed: seed_probs[seed]} if seed in seed_probs
                                             and leaf == seed else {}, 0))
                lexical_next = False
                stop = await new_window("lexical")
                if stop:
                    result.stop = stop
                    break
                continue
        heads = [(lst[0], i) for i, lst in enumerate(lists) if lst]
        if not heads or turn >= max_turns:
            if seeds or (seeds is None and windows):  # the tree has nothing left: the keyword channel goes on alone
                lexical_next = True
                continue
            result.stop = "every list empty" if not heads else f"turns ({max_turns})"
            break
        turn += 1
        present = {h for h, _ in heads}
        open_heads = [h for h, _ in heads if not (beaten.get(h, set()) & present)] or list(present)
        nid, probs = await top1(jev, state, text, tree, open_heads)
        ranked = order(tree, probs) if probs else [nid]
        for i, h in enumerate(ranked):
            beaten.setdefault(h, set()).update(ranked[:i])
        list_id = next(i for h, i in heads if h == nid)
        lists[list_id].pop(0)
        if nid in taken:  # a clip already recorded by a keyword window
            continue
        up = via.get(nid)
        if up is not None and up not in result.recorded:  # the node a flat window was ranked in place of
            result.recorded.append(up)
            paths.add(up)
            result.steps.append(Step(turn, up, -3, [], {}, 0))
        result.recorded.append(nid)
        kids = view.children(nid)
        result.steps.append(Step(turn, nid, list_id, list(open_heads), probs, len(kids)))
        if tree[nid].is_leaf:
            taken.add(nid)
        fresh = [k for k in kids if tree[k].is_leaf and k not in taken]
        inner = [k for k in kids if not tree[k].is_leaf]
        if fresh:
            cap = FIRST_LEAF_CHARS if not tree_windows else LEAF_CHARS
            tree_windows += 1
            for leaf in await window(jev, state, text, tree, fresh, cap):
                result.recorded.append(leaf)
                taken.add(leaf)
                result.steps.append(Step(turn, leaf, -1, list(fresh), {}, 0))
            lexical_next = True
            stop = await new_window("tree")
            if stop:
                result.stop = stop
                break
        if inner and turn < max_turns:  # a list opened after the last step would never be compared
            new = await rank(jev, state, text, tree, flatten(tree, view, inner, via))
            lists.extend(new)
            result.lists += len(new)
    final = list(result.recorded)
    if asked != final:  # the prediction sees the final context
        result.checks.append(await ask(qa_nodes(final), turn=len(result.steps)))
        seen.append(final)
    if abstains(question, result.checks[-1].answer):
        # the last answer is an abstention: ask again on the latest context that gave a real answer
        real = [i for i, c in enumerate(result.checks) if c.answer and not abstains(question, c.answer)]
        if real:
            result.checks.append(await ask(qa_nodes(seen[real[-1]]), turn=len(result.steps)))
            result.stop += "; re-asked on the last context with a real answer"
    return result
