"""Keeps a policy's spending visible during evolution: a check of its code, API keys out of reach, and a count of
every API response.

1. `check_policy(files)` parses the policy source before it is imported and reports forbidden imports, names and
   attributes (own network or API clients, the environment and API keys, processes, eval/exec, the harness's
   internals). Policies reach Jev only through their `jev` handle, the QA LLM through `ask`, and other LLMs
   through the memory policy's `llm`.
2. `remove_keys()` deletes the API keys from the environment once the harness has made its clients.
3. `install()` counts every successful HTTP response of the process; after a live evaluation, `reconcile()`
   compares the count with the calls the harness accounted for.

These in-process checks stop accidental and obvious bypasses, not code written to defeat them.
"""

from __future__ import annotations

import ast
import asyncio
import os
from collections import Counter
from pathlib import Path

RESPONSES: Counter = Counter()  # host -> successful responses this process received
ATTRIBUTED: Counter = Counter()  # "llm" / "jev" -> API calls the harness accounted for

KEY_ENVS = ("OPENAI_API_KEY", "TYPESAFE_API_KEY")

FORBIDDEN_MODULES = {
    "openai", "typesafe_sdk", "httpx", "httpx2", "requests", "aiohttp", "urllib", "urllib3", "http", "socket", "ssl",
    "websockets", "subprocess", "multiprocessing", "ctypes", "gc", "inspect", "importlib", "builtins", "sys",
    "signal", "pty", "shlex", "asyncio.subprocess", "__main__", "probe", "delovi.evaluate", "delovi.guard",
}
FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "globals", "locals", "vars", "breakpoint", "__builtins__", "__loader__",
    "__spec__",
}
FORBIDDEN_ATTRS = {
    "backend", "client", "_client", "api_key", "_config", "environ", "environb", "getenv", "getenvb", "putenv",
    "unsetenv", "system", "popen", "fork", "forkpty", "kill", "execv", "execve", "execvp", "execvpe", "execl",
    "execle", "execlp", "execlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "posix_spawn",
    "__dict__", "__globals__", "__subclasses__", "__bases__", "__mro__", "__code__", "__closure__", "__builtins__",
    "f_globals", "f_locals", "f_back", "f_builtins", "gi_frame", "cr_frame", "ag_frame", "tb_frame",
}
RULE = ("policy code may not make its own API or network calls, read the environment or API keys, start processes, "
        "or reach into the harness (use the `jev` handle, `ask`, and the memory policy's `llm`)")


def check_policy(files: list[Path]) -> list[str]:
    """Every forbidden import, name or attribute in `files`, as `file:line: what`."""
    problems = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(), str(path))
        except SyntaxError as e:
            problems.append(f"{path.name}:{e.lineno}: syntax error: {e.msg}")
            continue
        for node in ast.walk(tree):
            what = None
            if isinstance(node, ast.Import):
                bad = [a.name for a in node.names if _forbidden_module(a.name)]
                what = f"import {', '.join(bad)}" if bad else None
            elif isinstance(node, ast.ImportFrom) and node.module and _forbidden_module(node.module):
                what = f"from {node.module} import ..."
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                what = f"use of {node.id}"
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                what = f"attribute .{node.attr}"
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in ("getattr", "setattr", "hasattr", "delattr") and len(node.args) > 1
                  and isinstance(node.args[1], ast.Constant) and node.args[1].value in FORBIDDEN_ATTRS):
                what = f"{node.func.id}(..., {node.args[1].value!r})"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in KEY_ENVS:
                what = f"string {node.value!r}"
            if what:
                problems.append(f"{path.name}:{node.lineno}: {what}")
    return problems


def _forbidden_module(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in FORBIDDEN_MODULES)


def remove_keys() -> None:
    for k in KEY_ENVS:
        os.environ.pop(k, None)


def install() -> None:
    """Count every successful response at the HTTP transport of both SDKs (idempotent)."""
    for mod_name in ("httpx", "httpx2"):
        try:
            mod = __import__(mod_name)
        except ImportError:
            continue
        for cls in (mod.Client, mod.AsyncClient):
            if not getattr(cls.send, "_counted", False):
                _wrap(cls)


def _wrap(cls) -> None:
    original = cls.send
    if asyncio.iscoroutinefunction(original):
        async def send(self, request, *a, **kw):
            response = await original(self, request, *a, **kw)
            _count(request, response)
            return response
    else:
        def send(self, request, *a, **kw):
            response = original(self, request, *a, **kw)
            _count(request, response)
            return response
    send._counted = True
    cls.send = send


def _count(request, response) -> None:
    if 200 <= response.status_code < 300:
        RESPONSES[request.url.host] += 1


class Counted:
    """Same `complete` as the wrapped LLM; counts the calls not served from the disk cache."""

    def __init__(self, llm):
        self._llm = llm

    async def complete(self, model: str, messages: list[dict], reasoning: str | None = None):
        c = await self._llm.complete(model, messages, reasoning)
        if not c.cached:
            ATTRIBUTED["llm"] += 1
        return c


def reconcile() -> str | None:
    """None when every successful API response was accounted for, else a description of the difference."""
    sent, attributed = sum(RESPONSES.values()), ATTRIBUTED["llm"] + ATTRIBUTED["jev"]
    if sent == attributed:
        return None
    return (f"{sent} successful API responses at the transport ({dict(RESPONSES)}) but {attributed} accounted for "
            f"({ATTRIBUTED['llm']} LLM calls, {ATTRIBUTED['jev']} Jev requests): some call bypassed the accounting")
