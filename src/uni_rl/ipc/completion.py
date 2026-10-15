"""Neutral ordered-completion contract consumed by replay ingress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

COMPLETION_CONTRACT_VERSION = 3
MAX_COMPLETION_PRODUCER_ID_BYTES = 256
COMPLETION_OPERATIONS = frozenset({"step"})


def _validate_receipt_metadata(
    *,
    contract_version: int,
    operation: str,
    sequence: int,
    producer_id: str,
) -> None:
    """Validate the neutral public receipt fields."""
    if not isinstance(contract_version, int) or isinstance(contract_version, bool):
        raise TypeError("completion contract_version must be an integer")
    if contract_version != COMPLETION_CONTRACT_VERSION:
        raise ValueError(f"unsupported completion contract {contract_version}")
    if operation not in COMPLETION_OPERATIONS:
        raise ValueError(f"unsupported completion operation {operation!r}")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise TypeError("completion sequence must be a non-negative integer")
    if not isinstance(producer_id, str) or not producer_id.strip():
        raise TypeError("completion producer_id must be a non-empty string")
    if len(producer_id.encode("utf-8")) > MAX_COMPLETION_PRODUCER_ID_BYTES:
        raise ValueError("completion producer_id exceeds the IPC metadata limit")


@runtime_checkable
class CompletionTokenProtocol(Protocol):
    """Structural producer receipt consumed without backend dependencies."""

    @property
    def contract_version(self) -> int: ...

    @property
    def operation(self) -> str: ...

    @property
    def sequence(self) -> int: ...

    @property
    def producer_id(self) -> str: ...

    def wait_before_read(self, stream: Any | None = None) -> None:
        """Make the completed producer prefix visible before this read."""
        ...


def wait_completion(
    completion: CompletionTokenProtocol | None,
    *,
    stream: Any | None = None,
    operation: str = "step",
    producer_id: str | None = None,
    last_sequence: int = 0,
) -> CompletionTokenProtocol | None:
    """Gate a consumer read on an optional public completion receipt."""
    if completion is None:
        return None
    if not isinstance(completion, CompletionTokenProtocol):
        raise TypeError("completion must implement the public wait_before_read protocol")
    if completion.contract_version != COMPLETION_CONTRACT_VERSION:
        raise ValueError(f"unsupported completion contract {completion.contract_version}")
    if completion.operation not in COMPLETION_OPERATIONS:
        raise ValueError(f"unsupported completion operation {completion.operation!r}")
    if completion.operation != operation:
        raise ValueError(f"expected {operation!r} completion, got {completion.operation!r}")
    if producer_id is not None and completion.producer_id != producer_id:
        raise ValueError(
            f"completion producer {completion.producer_id!r} does not match {producer_id!r}"
        )
    if isinstance(completion.sequence, bool) or not isinstance(completion.sequence, int):
        raise TypeError("completion sequence must be an integer")
    if not isinstance(completion.producer_id, str) or not completion.producer_id.strip():
        raise TypeError("completion producer_id must be a non-empty string")
    if completion.sequence < last_sequence:
        raise ValueError(
            f"stale completion sequence {completion.sequence}; "
            f"last published sequence is {last_sequence}"
        )
    completion.wait_before_read(stream)
    return completion


@dataclass
class ImmediateCompletion:
    """Already-complete receipt for eager, host, and null execution paths."""

    contract_version: int = COMPLETION_CONTRACT_VERSION
    operation: str = "step"
    sequence: int = 0
    producer_id: str = "host-producer"

    def wait_before_read(self, stream: Any | None = None) -> None:
        del stream


@dataclass
class EventCompletion:
    """Replay-owned ingress publication receipt backed by a recorded CUDA event."""

    contract_version: int
    operation: str
    sequence: int
    producer_id: str
    event: Any

    def wait_before_read(self, stream: Any | None = None) -> None:
        if stream is None:
            self.event.synchronize()
        else:
            self.event.wait(stream)


def record_ingress_publication(
    completion: CompletionTokenProtocol,
    *,
    stream: Any | None,
) -> CompletionTokenProtocol:
    """Record a replay-owned publication receipt after ingress writes are enqueued."""
    if stream is None or getattr(stream.device, "type", None) != "cuda":
        return completion
    import torch

    _validate_receipt_metadata(
        contract_version=completion.contract_version,
        operation=completion.operation,
        sequence=completion.sequence,
        producer_id=completion.producer_id,
    )
    event = torch.cuda.Event(interprocess=True)
    event.record(stream)
    return EventCompletion(
        contract_version=completion.contract_version,
        operation=completion.operation,
        sequence=completion.sequence,
        producer_id=completion.producer_id,
        event=event,
    )


def publication_receipt(
    completion: CompletionTokenProtocol,
    *,
    sequence: int,
) -> CompletionTokenProtocol:
    """Normalize a waited producer receipt for ingress publication.

    Producer packages can implement the public structural protocol without
    inheriting from this package's concrete classes. Ingress has already waited
    on that protocol before writing, so preserve its public metadata in the
    local immediate receipt rather than coupling the contract to a backend type.
    """
    publication_sequence = int(sequence)
    _validate_receipt_metadata(
        contract_version=completion.contract_version,
        operation=completion.operation,
        sequence=completion.sequence,
        producer_id=completion.producer_id,
    )
    if isinstance(completion, EventCompletion):
        return EventCompletion(
            contract_version=completion.contract_version,
            operation=completion.operation,
            sequence=publication_sequence,
            producer_id=completion.producer_id,
            event=completion.event,
        )
    return ImmediateCompletion(
        contract_version=completion.contract_version,
        operation=completion.operation,
        sequence=publication_sequence,
        producer_id=completion.producer_id,
    )
