"""The memory tree: base clips as leaves, internal nodes with text, one root.

A memory policy builds the tree online with `add_leaf` and `make_node`. Nodes are append-only and never change
once made, so the tree a question asked at time t sees is exactly the first n nodes, where n is recorded by
`checkpoint(t)` while the video streams in. `validate_tree` checks this contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Node:
    nid: int
    start: float  # seconds on the memory's timeline
    end: float
    text: str  # the clip's caption (leaf) or a summary (internal node); the root's text is never shown
    parent: int | None = None  # set once, when the parent is made
    children: tuple[int, ...] = ()
    clip: int | None = None  # index of the base clip, for a leaf

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_leaf(self) -> bool:
        return self.clip is not None


class Tree:
    def __init__(self, timeline: dict, description: str = ""):
        self.timeline = timeline
        self.description = description
        self.nodes: list[Node] = []
        self.root: int | None = None
        self.checkpoints: dict[float, int] = {}  # question time -> number of nodes that existed then
        self.clips_fed: int | None = None  # base clips the memory policy has received
        self.finished = False  # the whole video is in the tree

    # -- construction --

    def add_leaf(self, clip_index: int, start: float, end: float, text: str) -> int:
        self.nodes.append(Node(len(self.nodes), start, end, text, clip=clip_index))
        return self.nodes[-1].nid

    def make_node(self, children: list[int], text: str) -> int:
        """A parent over `children` (which must not have a parent yet), spanning them."""
        if not children:
            raise ValueError("a node needs at least one child")
        children = tuple(sorted(children, key=lambda c: (self.nodes[c].start, self.nodes[c].end)))
        node = Node(len(self.nodes), min(self.nodes[c].start for c in children),
                    max(self.nodes[c].end for c in children), text, children=children)
        for c in children:
            if self.nodes[c].parent is not None:
                raise ValueError(f"node {c} already has a parent")
        for c in children:
            object.__setattr__(self.nodes[c], "parent", node.nid)
        self.nodes.append(node)
        return node.nid

    def set_root(self, nid: int) -> None:
        self.root = nid

    def orphans(self) -> list[int]:
        """Nodes without a parent, in time order (the top level built so far)."""
        return sorted((n.nid for n in self.nodes if n.parent is None and n.nid != self.root),
                      key=lambda s: (self.nodes[s].start, -self.nodes[s].duration))

    def checkpoint(self, cutoff: float) -> None:
        self.checkpoints[cutoff] = len(self.nodes)

    # -- access --

    def __getitem__(self, nid: int) -> Node:
        return self.nodes[nid]

    def __len__(self) -> int:
        return len(self.nodes)

    def depth(self, nid: int) -> int:
        d = 0
        while self.nodes[nid].parent is not None:
            nid, d = self.nodes[nid].parent, d + 1
        return d

    def view(self, cutoff: float | None) -> "View":
        """The tree as a question at `cutoff` sees it (None: the whole tree)."""
        if cutoff is None:
            return View(self, None)
        if cutoff not in self.checkpoints:
            raise KeyError(f"no checkpoint at {cutoff}")
        return View(self, self.checkpoints[cutoff])

    def snapshot(self, cutoff: float | None) -> "Tree":
        """A standalone copy of the tree as a question at `cutoff` sees it (None: the whole tree)."""
        limit = len(self.nodes) if cutoff is None else self.checkpoints[cutoff]
        snap = Tree(self.timeline, self.description)
        snap.nodes = [replace(n, parent=n.parent if n.parent is not None and n.parent < limit else None)
                      for n in self.nodes[:limit]]
        snap.root = self.root if self.root is not None and self.root < limit else None
        snap.checkpoints = {} if cutoff is None else {cutoff: limit}
        snap.clips_fed = sum(n.is_leaf for n in snap.nodes)
        snap.finished = cutoff is None and self.finished
        return snap

    # -- time labels --

    def clock(self, sec: float) -> str:
        sec = max(0, int(round(sec)))
        return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"

    def place(self, sec: float) -> tuple[str, float]:
        """(prefix such as 'DAY2' or 'stream 14', seconds within it)."""
        t = self.timeline
        if t["type"] == "days":
            return f"DAY{int(sec // 86400)}", sec % 86400
        if t["type"] == "videos":
            name, off = max(((n, o) for n, o in t["offsets"].items() if o <= sec + 1e-6), key=lambda x: x[1],
                            default=(next(iter(t["offsets"])), 0.0))
            return f"{t.get('unit', 'video')} {name}", sec - off
        return "", sec

    def time_label(self, sec: float) -> str:
        """'DAY3 19:59:50', 'stream 14 01:02:03' or '01:02:03'."""
        p, s = self.place(sec)
        return f"{p} {self.clock(s)}".strip()

    def span_label(self, node: Node) -> str:
        """'DAY2 14:03:00 to 14:03:30'; the end has its own prefix when it differs from the start's."""
        p0, s0 = self.place(node.start)
        p1, s1 = self.place(node.end)
        start = f"{p0} {self.clock(s0)}".strip()
        end = self.clock(s1) if p1 == p0 else f"{p1} {self.clock(s1)}".strip()
        return f"{start} to {end}"

    @staticmethod
    def duration_label(node: Node) -> str:
        sec = node.duration
        if sec < 90:
            return f"{round(sec)} s"
        if sec < 90 * 60:
            return f"{round(sec / 60)} min"
        return f"{sec / 3600:.1f} h"

    # -- persistence --

    def to_json(self) -> dict:
        return {"timeline": self.timeline, "description": self.description, "root": self.root,
                "clips_fed": self.clips_fed, "finished": self.finished,
                "checkpoints": [[c, n] for c, n in sorted(self.checkpoints.items())],
                "nodes": [{"nid": n.nid, "start": n.start, "end": n.end, "text": n.text, "parent": n.parent,
                           "children": list(n.children), "clip": n.clip} for n in self.nodes]}

    @classmethod
    def from_json(cls, d: dict) -> "Tree":
        tree = cls(d["timeline"], d.get("description", ""))
        tree.nodes = [Node(**{**n, "children": tuple(n["children"])}) for n in d["nodes"]]
        tree.root = d["root"]
        tree.checkpoints = {float(c): int(n) for c, n in d.get("checkpoints", [])}
        tree.clips_fed, tree.finished = d.get("clips_fed"), bool(d.get("finished", True))
        return tree

    def save(self, path: Path) -> None:
        """Readable JSON: the tree's fields one per line, then one node per line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_json()
        nodes = d.pop("nodes")
        lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in d.items()]
        lines += [' "nodes": ['] + [f"  {json.dumps(n, ensure_ascii=False)}{',' if i < len(nodes) - 1 else ''}"
                                     for i, n in enumerate(nodes)] + [" ]", "}"]
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes("\n".join(lines).encode())
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Tree":
        return cls.from_json(json.loads(path.read_bytes()))

    def stats(self) -> dict:
        internal = [n for n in self.nodes if not n.is_leaf and n.nid != self.root]
        depths = [self.depth(n.nid) for n in self.nodes if n.is_leaf]
        return {"nodes": len(self.nodes), "leaves": sum(n.is_leaf for n in self.nodes), "internal": len(internal),
                "leaf_depth_min": min(depths, default=0), "leaf_depth_max": max(depths, default=0),
                "root_children": len(self.nodes[self.root].children) if self.root is not None else 0,
                "mean_internal_chars": sum(len(n.text) for n in internal) / len(internal) if internal else 0,
                "mean_branching": (sum(len(n.children) for n in internal) / len(internal)) if internal else 0}


class View:
    """The tree as one question sees it: the first `limit` nodes (None: all). A visible node whose parent is
    hidden (or is the root) is a root of the view, so finished parts of an unfinished segment stay reachable."""

    def __init__(self, tree: Tree, limit: int | None):
        self.tree, self.limit = tree, limit

    def visible(self, nid: int) -> bool:
        return self.limit is None or nid < self.limit

    def roots(self) -> list[int]:
        out = [n.nid for n in self.tree.nodes if n.nid != self.tree.root and self.visible(n.nid)
               and (n.parent is None or n.parent == self.tree.root or not self.visible(n.parent))]
        return sorted(out, key=lambda s: (self.tree[s].start, -self.tree[s].duration))

    def children(self, nid: int) -> list[int]:
        return [c for c in self.tree[nid].children if self.visible(c)]

    def stats(self) -> dict:
        nodes = self.tree.nodes[:self.limit] if self.limit is not None else self.tree.nodes
        leaves = [n for n in nodes if n.is_leaf]
        internal = [n for n in nodes if not n.is_leaf and n.nid != self.tree.root]
        depths = [1 + sum(1 for _ in self._ancestors(n.nid)) for n in leaves]
        return {"nodes": len(nodes), "leaves": len(leaves), "internal": len(internal), "roots": len(self.roots()),
                "leaf_depth_min": min(depths, default=0), "leaf_depth_max": max(depths, default=0),
                "mean_internal_chars": sum(len(n.text) for n in internal) / len(internal) if internal else 0}

    def _ancestors(self, nid: int):
        p = self.tree[nid].parent
        while p is not None and self.visible(p) and p != self.tree.root:
            yield p
            p = self.tree[p].parent


def validate_tree(tree: Tree, clips: list) -> None:
    """Raise ValueError unless: one root; every clip is exactly one leaf with the clip's span and text; links
    agree; every node spans its children and was made after them; internal nodes have text; every node is
    reachable from the root; at every checkpoint the leaves are exactly the clips that ended before it."""
    problems = []
    if tree.root is None or not 0 <= tree.root < len(tree):
        raise ValueError("tree has no root")
    if tree[tree.root].parent is not None:
        problems.append("root has a parent")
    for n in tree.nodes:
        if n.nid != tree.root and n.parent is None:
            problems.append(f"node {n.nid} has no parent")
        if n.parent is not None and n.nid not in tree[n.parent].children:
            problems.append(f"node {n.nid} missing from its parent's children")
        if not n.end > n.start:
            problems.append(f"node {n.nid} has an empty or negative span")
        if not n.is_leaf and n.nid != tree.root and not n.text.strip():
            problems.append(f"internal node {n.nid} has no text")
        if n.is_leaf and n.children:
            problems.append(f"leaf {n.nid} has children")
        for c in n.children:
            if c >= n.nid:
                problems.append(f"child {c} of {n.nid} was made after its parent")
            if tree[c].parent != n.nid:
                problems.append(f"child {c} of {n.nid} points to another parent")
            if tree[c].start < n.start - 1e-6 or tree[c].end > n.end + 1e-6:
                problems.append(f"child {c} lies outside node {n.nid}")
        if sorted(n.children, key=lambda c: tree[c].start) != list(n.children):
            problems.append(f"children of {n.nid} are not in time order")
    seen, stack = set(), [tree.root]
    while stack:
        nid = stack.pop()
        if nid in seen:
            problems.append(f"node {nid} reached twice (cycle or shared child)")
            break
        seen.add(nid)
        stack.extend(tree[nid].children)
    if len(seen) != len(tree):
        problems.append(f"{len(tree) - len(seen)} nodes unreachable from the root")
    leaves = [n for n in tree.nodes if n.is_leaf]
    got = [n.clip for n in leaves]
    if len(set(got)) != len(got):
        problems.append("a clip appears as more than one leaf")
    if set(got) != set(range(len(clips))):
        problems.append(f"leaves cover {len(set(got))} clips, expected {len(clips)}")
    for n in leaves:
        if 0 <= n.clip < len(clips):
            c = clips[n.clip]
            if abs(n.start - c.start) > 1e-6 or abs(n.end - c.end) > 1e-6 or n.text != c.text:
                problems.append(f"leaf {n.nid} does not match clip {n.clip}")
                break
    for cutoff, limit in tree.checkpoints.items():
        expected = {i for i, c in enumerate(clips) if c.end < cutoff}
        seen_leaves = {n.clip for n in tree.nodes[:limit] if n.is_leaf}
        if seen_leaves != expected:
            problems.append(f"checkpoint {cutoff}: {len(seen_leaves)} leaves, expected {len(expected)}")
            break
    if problems:
        raise ValueError("invalid tree: " + "; ".join(problems[:8]) + (" ..." if len(problems) > 8 else ""))
