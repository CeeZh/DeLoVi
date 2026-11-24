"""Memory policy of DeLoVi-E, obtained by automated policy evolution from the recursive hierarchical search.

The tree is built online over the base clips:
- 3-minute and 15-minute nodes: free-text event summaries (SUMMARY_PROMPT).
- Hour nodes: a timeline with one line per 3-minute window (TL_PROMPT), which keeps the order, frequency and
  simultaneous actions of events.
- Day (or stream) nodes, for videos that span several days or streams: an index with one line per hour
  (LINE_PROMPT). After the index lines, the node also stores a free-text summary of each hour (ROUTING_MARK).
  The decision model reads these summaries to rank the hours of a day, while the LLM reads the hour timelines.
- A single video of up to two hours gets one visible node over the whole video whose text is the timeline of
  all its 3-minute windows, so the LLM reads it for every question.

Streaming interface as in hierarchical_search/builder.py: add(clip), sync(), finish(), tree.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable

from delovi.data import Clip, Memory
from delovi.tree import Tree

# Hyper-parameters
LLM_MODEL = "gpt-6-luna"
LLM_REASONING = "none"
LEVELS = [(180, 450), (900, 600), (3600, 1000)]  # (window seconds, max words of its summary), bottom-up
TIMELINE_LEVEL = 2  # the hour level's text is a timeline (its word limit above is used for its routing summary)
LINE_WORDS = 60  # words per line of a day / stream index
TL_WORDS = 30  # words per timeline line, cut at a clause boundary
TL_ASK = 22  # words asked of the LLM per timeline line (it tends to overshoot the limit)
VIDEO_WINDOWS = 40  # a single video with at most this many 3-minute windows (two hours) gets a video timeline
ROUTING_MARK = "\n\n<<ROUTING>>\n"  # day node text: index lines | ROUTING_MARK | one summary per child
ROUTING_SEP = "\n<<NEXT>>\n"
# Event summary in the style of EgoLife's hierarchical captions, kept in the captions' point of view.
SUMMARY_PROMPT = """You summarize the captions of consecutive segments of one recording into an event summary. The summary is used to decide whether a specific moment lies inside this span of the recording, and as context for answering questions about it.

#### Requirements

1. **Structure the events clearly**
   - Merge related content into major events and arrange them in chronological order.
   - Group events by location, task, or theme. Each event has a clear start, progression, and key turning points.

2. **Retain key information**
   - The actions and decisions of the people involved, by name. Keep the captions' point of view: if they are written in the first person ("I"), "I" is the camera wearer, so keep "I"; otherwise refer to people by name or role (for example "the streamer").
   - Discussions, decisions, plans, and task arrangements that matter, with their outcome.
   - The purpose and method of key actions (for example "orders takeout on a phone", "writes the plan on a whiteboard").
   - The specific, searchable details: names, distinctive objects, places, on-screen text, titles, numbers and prices, and the topics of what was said.

3. **Be concise**
   - Keep the facts clear; leave out atmosphere, emotions, and abstract commentary.
   - Drop trivial chatter; keep the core topic and conclusion of each discussion.

4. **Strictly adhere to the facts**
   - Do not assume or add anything the captions do not say.
   - Keep the true chronological order of events.

#### Output format
One paragraph per major event: start with one sentence naming the event, then the key actions and details, then its outcome. At most {max_words} words in total. Do not mention the date, the time, or the word count."""


# One line of an hour's timeline, written from one 3-minute window's summary (or a lone clip's caption).
TL_PROMPT = """You write one line of a timeline of a recording: what happened in one segment of about three minutes. You are given the segment's caption or summary. The line is read to find when something happened, in what order, how often, and what else happened at the same time.

Say who did what, by name, in time order, separated by semicolons; when several people act at once, give each person's action ("Jake washes the pot while Tasha and Lucia discuss the cake"). Keep the place, distinctive objects, numbers and counts ("opens the oven twice", "pays 50 yuan"), on-screen text, games and titles, and the topic or outcome of what was said. Keep the point of view of the text: if it is written in the first person ("I"), keep "I"; otherwise refer to people by name or role. Facts only, nothing the text does not say. No introduction, no time stamps.

At most {max_words} words."""


# One line of a day / stream index, written from one hour summary (or a shorter segment's text).
LINE_PROMPT = """You write one line of an index of a long recording. You are given the caption or summary of one segment. The line is read to decide whether a specific moment that a question asks about lies in this segment.

List the segment's events in time order, separated by semicolons, each with its specific, searchable details: the people involved, by name; the place; distinctive objects; the activity; games, titles, on-screen text and numbers; the topics of what was said. Keep the point of view of the text: if it is written in the first person ("I"), keep "I"; otherwise refer to people by name or role. Facts only, nothing the text does not say. No full sentences, no introduction, no time stamps.

At most {max_words} words."""


class MemoryBuilder:
    """Fixed windows as in the hierarchical search, plus a top level (the unit) that groups the hours of one day
    or stream into an index node. When the first clip of a new day or stream arrives, every open window of the
    previous one is closed, so a question early on the next day already sees the finished day as one node."""

    def __init__(self, memory: Memory, llm):
        self.tree = Tree(memory.timeline, memory.description)
        self.llm = llm
        self.top = len(LEVELS)  # the unit level
        self.open: list[list[Item]] = [[] for _ in range(self.top + 1)]  # the open window at each level
        self.pending: list[Item] = []  # decided but not yet made, in the order their windows closed
        self.unit: str | None = None  # the day / stream of the last clip
        self.units = 0  # unit nodes decided so far
        self.multi = memory.timeline.get("type") in ("days", "videos")  # a plain timeline is one video

    def window(self, level: int, item: "Item") -> tuple:
        if level == self.top:
            return (self.tree.place(item.start)[0],)
        return (self.tree.place(item.start)[0], item.start // LEVELS[level][0])

    async def add(self, clip: Clip) -> None:
        unit = self.tree.place(clip.start)[0]
        if self.unit is not None and unit != self.unit:
            self.close_all()  # the previous day / stream is over
        self.unit = unit
        nid = self.tree.add_leaf(clip.index, clip.start, clip.end, clip.text)
        self.push(0, Item.made(self.tree, nid))

    def close_all(self) -> None:
        for level in range(self.top + 1):
            if self.open[level]:
                self.close(level)

    def push(self, level: int, item: "Item") -> None:
        """An item arrives at `level`: close the open window there first if `item` starts a new one."""
        if level > self.top:
            return  # above the top level: stays under the root
        if level == self.top and self.multi and item.line is None:  # its index line, written while the unit goes on
            item.line = asyncio.ensure_future(index_line(self.tree, item, self.llm))
        group = self.open[level]
        if group and self.window(level, item) != self.window(level, group[-1]):
            self.close(level)
        self.open[level].append(item)

    def close(self, level: int) -> None:
        group, self.open[level] = self.open[level], []
        if len(group) == 1:  # nothing to summarize: the child itself moves up
            self.push(level + 1, group[0])
            return
        if level == self.top:
            task = index(self.tree, group)
            self.units += 1
        elif level == TIMELINE_LEVEL:
            task = timeline(self.tree, group, self.llm)
        else:
            task = summarize(self.tree, group, SUMMARY_PROMPT, self.llm, max_words=LEVELS[level][1])
        item = Item(group, asyncio.ensure_future(task))
        item.level = level
        if level == TIMELINE_LEVEL and self.multi:  # the hour's free-text summary, stored in its day node
            item.routing = asyncio.ensure_future(
                summarize(self.tree, group, SUMMARY_PROMPT, self.llm, max_words=LEVELS[level][1]))
        self.pending.append(item)
        self.push(level + 1, item)

    async def sync(self) -> None:
        """Make every pending node, in the order the windows closed (children before parents)."""
        pending, self.pending = self.pending, []
        for item in pending:
            text = (await item.task).strip()
            item.nid = self.tree.make_node([c.nid for c in item.children], text)

    async def finish(self) -> None:
        for level in range(self.top):
            if self.open[level]:
                self.close(level)
        if self.units:  # the last day / stream of a multi-day recording
            self.close(self.top)
            await self.sync()
            return
        # a single video: its top items stay as they are, under a video timeline when it is short enough
        tops = self.open[self.top]
        video = None
        if len(tops) > 1 and sum(len(windows(t)) for t in tops) <= VIDEO_WINDOWS:
            video = Item(tops, asyncio.ensure_future(timeline(self.tree, tops, self.llm)))
            video.level = self.top
            self.pending.append(video)
        await self.sync()
        if video is not None:
            self.tree.set_root(self.tree.make_node([video.nid], ""))
        elif len(tops) == 1 and tops[0].level == TIMELINE_LEVEL and len(windows(tops[0])) <= VIDEO_WINDOWS:
            self.tree.set_root(self.tree.make_node([tops[0].nid], ""))  # the hour timeline is the whole video


class Item:
    """A node to be: an existing node (`nid` set, e.g. a leaf) or a decided one whose text is still being
    written (`task`); `nid` is set when the node is made."""

    def __init__(self, children: list["Item"], task: Awaitable[str] | None = None, nid: int | None = None,
                 start: float | None = None, end: float | None = None):
        self.children, self.task, self.nid = children, task, nid
        self.line: Awaitable[str] | None = None  # its line of the day / stream index (unit-level items)
        self.routing: Awaitable[str] | None = None  # multi-day hours: the summary the search ranks it by
        self.level = -1  # the level whose window made it (-1: a leaf)
        self.start = start if start is not None else min(c.start for c in children)
        self.end = end if end is not None else max(c.end for c in children)

    @classmethod
    def made(cls, tree: Tree, nid: int) -> "Item":
        n = tree[nid]
        return cls([], None, nid, n.start, n.end)

    @property
    def duration(self) -> float:
        return self.end - self.start

    async def text(self, tree: Tree) -> str:
        return tree[self.nid].text if self.task is None else (await self.task).strip()

    async def route_text(self, tree: Tree) -> str:
        """The text the search ranks this item by: its routing summary if it has one, else its text."""
        return (await self.routing).strip() if self.routing is not None else await self.text(tree)


async def summarize(tree: Tree, children: list[Item], prompt: str, llm, max_words: int = 150,
                    model: str = LLM_MODEL, reasoning: str | None = LLM_REASONING) -> str:
    """One LLM call: the children's texts with their time spans -> the parent's text."""
    texts = await asyncio.gather(*(c.text(tree) for c in children))
    body = "\n\n".join(f"[{tree.span_label(c)}] ({tree.duration_label(c)})\n{t}" for c, t in zip(children, texts))
    messages = [{"role": "system", "content": prompt.format(max_words=max_words)},
                {"role": "user", "content": f"Recording: {tree.description}\n\nCaptions:\n{body}"}]
    out = await llm.complete(model, messages, reasoning)
    return out.text.strip()


async def index_line(tree: Tree, item: Item, llm, max_words: int = LINE_WORDS) -> str:
    """One line of a day / stream index, written from the item's routing text (a leaf's caption is used as is)."""
    text = await item.route_text(tree)
    if item.task is not None or not tree[item.nid].is_leaf:
        messages = [{"role": "system", "content": LINE_PROMPT.format(max_words=max_words)},
                    {"role": "user", "content": f"Recording: {tree.description}\n\nSegment "
                                                f"[{tree.span_label(item)}] ({tree.duration_label(item)}):\n{text}"}]
        text = (await llm.complete(LLM_MODEL, messages, LLM_REASONING)).text
    words = " ".join(text.split()).split(" ")
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def windows(item: Item) -> list[Item]:
    """The bottom-level items under `item`: 3-minute nodes, or clips that were alone in their window."""
    if item.level <= 0:
        return [item]
    return [w for c in item.children for w in windows(c)]


def clock_span(tree: Tree, item: Item) -> str:
    """'14:03-14:06': the item's span as clock times within its day / stream (or video)."""
    hm = lambda sec: f"{int(sec) // 3600:02d}:{int(sec) % 3600 // 60:02d}"
    return f"{hm(tree.place(item.start)[1])}-{hm(tree.place(item.end)[1])}"


async def timeline_line(tree: Tree, item: Item, llm, max_words: int = TL_WORDS) -> str:
    """One timeline line for a bottom-level item, at most `max_words` words, ending at a clause boundary (';' or
    '.') when one falls in the second half (a lone clip's caption is cut the same way)."""
    text = await item.text(tree)
    if item.level >= 0:
        messages = [{"role": "system", "content": TL_PROMPT.format(max_words=TL_ASK)},
                    {"role": "user", "content": f"Recording: {tree.description}\n\nSegment "
                                                f"[{tree.span_label(item)}] ({tree.duration_label(item)}):\n{text}"}]
        text = (await llm.complete(LLM_MODEL, messages, LLM_REASONING)).text
    words = " ".join(text.split()).split(" ")
    if len(words) <= max_words:
        return " ".join(words)
    words = words[:max_words]
    ends = [i for i, w in enumerate(words) if w.endswith((";", "."))]
    if ends and ends[-1] >= max_words // 2:
        return " ".join(words[:ends[-1] + 1]).rstrip(";.") + "."
    return " ".join(words) + " ..."


async def timeline(tree: Tree, children: list[Item], llm) -> str:
    """The text of an hour node (or a video timeline): one line per 3-minute window, headed by its clock span."""
    items = [w for c in children for w in windows(c)]
    lines = await asyncio.gather(*(timeline_line(tree, w, llm) for w in items))
    return "\n".join(f"{clock_span(tree, w)} {line.strip()}" for w, line in zip(items, lines))


async def index(tree: Tree, children: list[Item]) -> str:
    """The text of a day / stream node: its children's index lines, each headed by the child's time span, then
    after ROUTING_MARK one routing summary per child ("-" for a child that has none)."""
    lines = await asyncio.gather(*(c.line for c in children))
    text = "\n".join(f"[{tree.span_label(c)}] {line.strip()}" for c, line in zip(children, lines))
    routes = [(await c.routing).strip() if c.routing is not None else "" for c in children]
    if any(routes):
        clean = lambda r: r.replace(ROUTING_MARK.strip(), "").replace(ROUTING_SEP.strip(), "").strip() or "-"
        text += ROUTING_MARK + ROUTING_SEP.join(clean(r) for r in routes)
    return text
