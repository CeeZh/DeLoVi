"""Memory policy of the recursive hierarchical search: a fixed-window hierarchy built online.

The base clips are the leaves. Whenever a 3-minute window of the video is complete, an LLM summarizes its clips;
the 3-minute summaries are summarized into 15-minute summaries, and those into 1-hour summaries. A window with a
single child is not wrapped, so short videos stay shallow. Windows also break where the day or stream changes.

Streaming interface (see delovi/evaluate.py):

    b = MemoryBuilder(memory, llm)
    await b.add(clip)       # every base clip, in time order; starts LLM calls as tasks and returns
    await b.sync()          # before a question is answered: make the nodes decided so far
    await b.finish()        # end of the video: close every open window
    tree = b.tree
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


class MemoryBuilder:
    """Level k groups the items of level k-1 (leaves for k=0) whose start falls in the same LEVELS[k] window. A
    closed window becomes an Item at once (its summary runs as a task) and moves up; its node is made at the next
    sync, in the order the windows closed, so the tree does not depend on API latency."""

    def __init__(self, memory: Memory, llm):
        self.tree = Tree(memory.timeline, memory.description)
        self.llm = llm
        self.open: list[list[Item]] = [[] for _ in LEVELS]  # the open window at each level
        self.pending: list[Item] = []  # decided but not yet made, in the order their windows closed

    def window(self, level: int, item: "Item") -> tuple:
        return (self.tree.place(item.start)[0], item.start // LEVELS[level][0])

    async def add(self, clip: Clip) -> None:
        nid = self.tree.add_leaf(clip.index, clip.start, clip.end, clip.text)
        self.push(0, Item.made(self.tree, nid))

    def push(self, level: int, item: "Item") -> None:
        """An item arrives at `level`: close the open window there first if `item` starts a new one."""
        if level >= len(LEVELS):
            return  # above the top level: stays under the root
        group = self.open[level]
        if group and self.window(level, item) != self.window(level, group[-1]):
            self.close(level)
        self.open[level].append(item)

    def close(self, level: int) -> None:
        group, self.open[level] = self.open[level], []
        if len(group) == 1:  # nothing to summarize: the child itself moves up
            self.push(level + 1, group[0])
            return
        task = summarize(self.tree, group, SUMMARY_PROMPT, self.llm, max_words=LEVELS[level][1])
        item = Item(group, asyncio.ensure_future(task))
        self.pending.append(item)
        self.push(level + 1, item)

    async def sync(self) -> None:
        """Make every pending node, in the order the windows closed (children before parents)."""
        pending, self.pending = self.pending, []
        for item in pending:
            text = (await item.task).strip()
            item.nid = self.tree.make_node([c.nid for c in item.children], text)

    async def finish(self) -> None:
        for level in range(len(LEVELS)):
            if self.open[level]:
                self.close(level)
        await self.sync()


class Item:
    """A node to be: an existing node (`nid` set, e.g. a leaf) or a decided one whose text is still being
    written (`task`); `nid` is set when the node is made."""

    def __init__(self, children: list["Item"], task: Awaitable[str] | None = None, nid: int | None = None,
                 start: float | None = None, end: float | None = None):
        self.children, self.task, self.nid = children, task, nid
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


async def summarize(tree: Tree, children: list[Item], prompt: str, llm, max_words: int = 150,
                    model: str = LLM_MODEL, reasoning: str | None = LLM_REASONING) -> str:
    """One LLM call: the children's texts with their time spans -> the parent's text."""
    texts = await asyncio.gather(*(c.text(tree) for c in children))
    body = "\n\n".join(f"[{tree.span_label(c)}] ({tree.duration_label(c)})\n{t}" for c, t in zip(children, texts))
    messages = [{"role": "system", "content": prompt.format(max_words=max_words)},
                {"role": "user", "content": f"Recording: {tree.description}\n\nCaptions:\n{body}"}]
    out = await llm.complete(model, messages, reasoning)
    return out.text.strip()
