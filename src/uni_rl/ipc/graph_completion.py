"""Neutral graph-completion contract consumed by replay ingress."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

_LAST_SEQUENCE = 0
_IMMEDIATE_SEQUENCE = itertools.count()


@runtime_checkable
class GraphCompletionTokenProtocol(Protocol):
    """Structural producer receipt consumed without backend dependencies.

    A producer creates the receipt only after enqueueing all graph outputs and
    Torch publications on an ordered device stream. Replay ingress gates its
    first read with :meth:`wait_before_read`; it must not synchronize the
    producer stream on the publication hot path.
    """

    @property
    def contract_version(self) -> int: ...

    @property
    def operation(self) -> str: ...

    @property
    def sequence(self) -> int: ...

    @property
    def stream_id(self) -> int: ...

    def wait_before_read(self, stream: Any | None = None) -> None:
        """Make the completed producer prefix visible before this read."""
        ...


def wait_graph_completion(
    completion: GraphCompletionTokenProtocol | None,
    *,
    stream: Any | None = None,
) -> GraphCompletionTokenProtocol | None:
    """Gate a consumer read on an optional public completion receipt."""
    if completion is None:
        return None
    if not isinstance(completion, GraphCompletionTokenProtocol):
        raise TypeError("graph completion must implement the public wait_before_read protocol")
    if completion.contract_version != 1:
        raise ValueError(f"unsupported graph completion contract {completion.contract_version}")
    if completion.operation not in ("step", "selected_reset", "mask_reset"):
        raise ValueError(f"unsupported graph completion operation {completion.operation!r}")
    if isinstance(completion.sequence, bool) or not isinstance(completion.sequence, int):
        raise TypeError("graph completion sequence must be an integer")
    if isinstance(completion.stream_id, bool) or not isinstance(completion.stream_id, int):
        raise TypeError("graph completion stream_id must be an integer")
    global _LAST_SEQUENCE
    if completion.sequence < _LAST_SEQUENCE:
        raise ValueError(
            f"stale graph completion sequence {completion.sequence}; "
            f"last published sequence is {_LAST_SEQUENCE}"
        )
    completion.wait_before_read(stream)
    _LAST_SEQUENCE = completion.sequence
    return completion


@dataclass
class ImmediateGraphCompletion:
    """Already-complete receipt for eager, host, and null execution paths."""

    contract_version: int = 1
    operation: str = "step"
    sequence: int = field(default_factory=lambda: next(_IMMEDIATE_SEQUENCE))
    stream_id: int = 0

    def wait_before_read(self, stream: Any | None = None) -> None:
        del stream
