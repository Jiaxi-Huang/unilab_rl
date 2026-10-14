"""Neutral graph-completion contract consumed by replay ingress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

MAX_GRAPH_IPC_HANDLE_BYTES = 256
MAX_GRAPH_CAPTURE_ID_BYTES = 256
GRAPH_COMPLETION_OPERATIONS = frozenset(
    {"step", "selected_reset", "mask_reset", "collector_tick", "env_update"}
)


def _validate_public_metadata(
    *,
    contract_version: int,
    operation: str,
    sequence: int,
    stream_id: int,
    capture_id: str | None,
    ipc_handle: bytes,
) -> None:
    """Validate IPC receipt metadata without interpreting the event handle."""
    if not isinstance(contract_version, int) or isinstance(contract_version, bool):
        raise TypeError("graph completion contract_version must be an integer")
    if contract_version != 1:
        raise ValueError(f"unsupported graph completion contract {contract_version}")
    if operation not in GRAPH_COMPLETION_OPERATIONS:
        raise ValueError(f"unsupported graph completion operation {operation!r}")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise TypeError("graph completion sequence must be a non-negative integer")
    if not isinstance(stream_id, int) or isinstance(stream_id, bool) or stream_id < 0:
        raise TypeError("graph completion stream_id must be a non-negative integer")
    if capture_id is not None:
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise TypeError("graph completion capture_id must be a non-empty string or None")
        if len(capture_id.encode("utf-8")) > MAX_GRAPH_CAPTURE_ID_BYTES:
            raise ValueError("graph completion capture_id exceeds the IPC metadata limit")
    else:
        raise ValueError("an interprocess graph completion requires a capture_id")
    if not isinstance(ipc_handle, bytes) or not 0 < len(ipc_handle) <= MAX_GRAPH_IPC_HANDLE_BYTES:
        raise TypeError("graph completion ipc_handle must be non-empty bounded bytes")


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

    @property
    def capture_id(self) -> str | None: ...

    def wait_before_read(self, stream: Any | None = None) -> None:
        """Make the completed producer prefix visible before this read."""
        ...

    def ipc_handle(self) -> bytes: ...


def wait_graph_completion(
    completion: GraphCompletionTokenProtocol | None,
    *,
    stream: Any | None = None,
    operation: str = "step",
    capture_id: str | None = None,
    last_sequence: int = 0,
) -> GraphCompletionTokenProtocol | None:
    """Gate a consumer read on an optional public completion receipt."""
    if completion is None:
        return None
    if not isinstance(completion, GraphCompletionTokenProtocol):
        raise TypeError("graph completion must implement the public wait_before_read protocol")
    if completion.contract_version != 1:
        raise ValueError(f"unsupported graph completion contract {completion.contract_version}")
    if completion.operation not in GRAPH_COMPLETION_OPERATIONS:
        raise ValueError(f"unsupported graph completion operation {completion.operation!r}")
    if completion.operation != operation:
        raise ValueError(f"expected {operation!r} graph completion, got {completion.operation!r}")
    if capture_id is not None and completion.capture_id != capture_id:
        raise ValueError(
            f"graph completion capture {completion.capture_id!r} does not match {capture_id!r}"
        )
    if isinstance(completion.sequence, bool) or not isinstance(completion.sequence, int):
        raise TypeError("graph completion sequence must be an integer")
    if isinstance(completion.stream_id, bool) or not isinstance(completion.stream_id, int):
        raise TypeError("graph completion stream_id must be an integer")
    if completion.sequence < last_sequence:
        raise ValueError(
            f"stale graph completion sequence {completion.sequence}; "
            f"last published sequence is {last_sequence}"
        )
    completion.wait_before_read(stream)
    return completion


@dataclass
class ImmediateGraphCompletion:
    """Already-complete receipt for eager, host, and null execution paths."""

    contract_version: int = 1
    operation: str = "step"
    sequence: int = 0
    stream_id: int = 0
    capture_id: str | None = None

    def wait_before_read(self, stream: Any | None = None) -> None:
        del stream

    def ipc_handle(self) -> bytes:
        raise NotImplementedError("this graph completion receipt has no IPC transport")


@dataclass
class EventGraphCompletion:
    """Consumer-local receipt reconstructed from a public CUDA IPC event."""

    contract_version: int
    operation: str
    sequence: int
    stream_id: int
    capture_id: str | None
    event: Any

    def wait_before_read(self, stream: Any | None = None) -> None:
        if stream is None:
            self.event.synchronize()
        else:
            self.event.wait(stream)

    def ipc_handle(self) -> bytes:
        return bytes(self.event.ipc_handle())


def record_ingress_publication(
    completion: GraphCompletionTokenProtocol,
    *,
    stream: Any | None,
) -> GraphCompletionTokenProtocol:
    """Record a publication receipt after ingress writes are enqueued.

    The producer graph event orders backend outputs, while this event also
    includes the CUDA ingress-slot copies submitted after that graph event.
    It remains stream-local: neither endpoint synchronizes the producer stream.
    """
    if stream is None or getattr(stream.device, "type", None) != "cuda":
        return completion
    import torch

    _validate_public_metadata(
        contract_version=completion.contract_version,
        operation=completion.operation,
        sequence=completion.sequence,
        stream_id=completion.stream_id,
        capture_id=completion.capture_id,
        ipc_handle=b"metadata-only",
    )
    event = torch.cuda.Event(interprocess=True)
    event.record(stream)
    return EventGraphCompletion(
        contract_version=completion.contract_version,
        operation=completion.operation,
        sequence=completion.sequence,
        stream_id=completion.stream_id,
        capture_id=completion.capture_id,
        event=event,
    )


def collector_tick_receipt(
    completion: GraphCompletionTokenProtocol,
    *,
    sequence: int,
) -> GraphCompletionTokenProtocol:
    """Return a neutral receipt for one complete collector publication tick.

    The producer calls this only after every output and ingress-slot write for
    the tick is enqueued. The operation name is changed to ``collector_tick``
    so downstream waits cannot mistake a backend-only step receipt for complete
    transition publication. Event ownership and stream identity are unchanged.
    """
    contract_version = int(completion.contract_version)
    operation = "collector_tick"
    tick_sequence = int(sequence)
    stream_id = int(completion.stream_id)
    capture_id = completion.capture_id
    if isinstance(completion, EventGraphCompletion):
        return EventGraphCompletion(
            contract_version=contract_version,
            operation=operation,
            sequence=tick_sequence,
            stream_id=stream_id,
            capture_id=capture_id,
            event=completion.event,
        )
    if isinstance(completion, ImmediateGraphCompletion):
        return ImmediateGraphCompletion(
            contract_version=contract_version,
            operation=operation,
            sequence=tick_sequence,
            stream_id=stream_id,
            capture_id=capture_id,
        )
    raise TypeError("collector tick publication requires the producer's public concrete receipt")


def import_graph_completion(
    *,
    contract_version: int,
    operation: str,
    sequence: int,
    stream_id: int,
    capture_id: str | None,
    ipc_handle: bytes,
    device: Any,
) -> EventGraphCompletion:
    """Reconstruct a neutral receipt from its public CUDA IPC descriptor."""
    import torch

    _validate_public_metadata(
        contract_version=contract_version,
        operation=operation,
        sequence=sequence,
        stream_id=stream_id,
        capture_id=capture_id,
        ipc_handle=ipc_handle,
    )
    event = torch.cuda.Event.from_ipc_handle(device, ipc_handle)
    return EventGraphCompletion(
        contract_version=contract_version,
        operation=operation,
        sequence=sequence,
        stream_id=stream_id,
        capture_id=capture_id,
        event=event,
    )
