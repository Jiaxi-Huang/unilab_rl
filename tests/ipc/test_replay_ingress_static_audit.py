"""Static audit for replay ingress graph-candidate publication ordering."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "uni_rl" / "ipc" / "replay_buffer.py"

FORBIDDEN_ALLOCATION_CALLS = frozenset({"empty", "zeros", "ones", "cat", "stack"})
FORBIDDEN_HOST_CALLS = frozenset({"cpu", "tolist", "numpy", "item", "nonzero"})
FORBIDDEN_STREAM_CALLS = frozenset({"synchronize", "current_stream"})


@dataclass(frozen=True)
class Violation:
    method: str
    line: int
    kind: str
    expression: str


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    result = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(result) == 1, f"expected one {name!r} method"
    return result[0]


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def test_replay_ingress_publication_path_has_no_unclassified_host_detours() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    method = _method(tree, "_add_to_ingress")
    violations: list[Violation] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        called = _call_name(node)
        expression = _unparse(node)
        if called in FORBIDDEN_STREAM_CALLS:
            violations.append(Violation("_add_to_ingress", node.lineno, "stream-sync", expression))
        if called in FORBIDDEN_HOST_CALLS:
            violations.append(Violation("_add_to_ingress", node.lineno, "host-detour", expression))
        if called in FORBIDDEN_ALLOCATION_CALLS and not any(
            keyword.arg == "out" for keyword in node.keywords
        ):
            violations.append(
                Violation("_add_to_ingress", node.lineno, "capture-time-allocation", expression)
            )
    assert not violations, "replay ingress static audit is nonzero:\n" + "\n".join(
        f"{item.method}:{item.line} {item.kind}: {item.expression}" for item in violations
    )
