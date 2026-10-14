"""Neutral ordered-completion contract tests."""

from __future__ import annotations

import multiprocessing as mp

import pytest
import torch

from uni_rl.ipc.completion import (
    EventCompletion,
    ImmediateCompletion,
    import_shared_completion,
    publication_receipt,
    record_ingress_publication,
    wait_completion,
)


def test_immediate_completion_is_pass_through() -> None:
    completion = ImmediateCompletion()

    assert wait_completion(completion, stream=object()) is completion


def test_wait_completion_validates_public_metadata() -> None:
    bad_contract = ImmediateCompletion(contract_version=1)
    bad_operation = ImmediateCompletion(operation="learner-update")
    bad_sequence = ImmediateCompletion(sequence=True)
    bad_stream = ImmediateCompletion(stream_id=True)
    bad_producer = ImmediateCompletion(producer_id=" ")

    with pytest.raises(ValueError, match="contract"):
        wait_completion(bad_contract)
    with pytest.raises(ValueError, match="operation"):
        wait_completion(bad_operation)
    with pytest.raises(TypeError, match="sequence"):
        wait_completion(bad_sequence)
    with pytest.raises(TypeError, match="stream_id"):
        wait_completion(bad_stream)
    with pytest.raises(TypeError, match="producer_id"):
        wait_completion(bad_producer)


def test_wait_completion_rejects_non_protocol_producer() -> None:
    with pytest.raises(TypeError, match="wait_before_read"):
        wait_completion(object())  # type: ignore[arg-type]


def test_wait_completion_rejects_stale_sequences() -> None:
    current = ImmediateCompletion(sequence=10**9)
    stale = ImmediateCompletion(sequence=10**9 - 1)

    wait_completion(current, last_sequence=10**9 - 1)
    with pytest.raises(ValueError, match="stale"):
        wait_completion(stale, last_sequence=10**9)


def test_wait_completion_binds_operation_and_producer() -> None:
    completion = ImmediateCompletion(operation="step", producer_id="producer-a")

    with pytest.raises(ValueError, match="expected 'mask_reset'"):
        wait_completion(completion, operation="mask_reset")
    with pytest.raises(ValueError, match="producer"):
        wait_completion(completion, producer_id="producer-b")
    assert wait_completion(completion, producer_id="producer-a") is completion


def test_removed_whole_collector_operation_is_rejected() -> None:
    completion = ImmediateCompletion()
    completion.operation = "whole_collector"

    with pytest.raises(ValueError, match="unsupported completion operation"):
        wait_completion(completion)


def test_immediate_completion_has_no_undeclared_ipc_transport() -> None:
    with pytest.raises(NotImplementedError, match="IPC"):
        ImmediateCompletion().ipc_handle()


def test_ingress_publication_preserves_cpu_receipt() -> None:
    completion = ImmediateCompletion(producer_id="producer-cpu")

    assert record_ingress_publication(completion, stream=None) is completion


def test_publication_receipt_accepts_foreign_structural_producer() -> None:
    class ForeignImmediateCompletion:
        def __init__(self, **overrides: object) -> None:
            self.contract_version: int = 2
            self.operation: str = "mask_reset"
            self.sequence: int = 17
            self.stream_id: int = 5
            self.producer_id: str = "producer-foreign"
            for name, value in overrides.items():
                setattr(self, name, value)

        def wait_before_read(self, stream=None) -> None:
            del stream

        def ipc_handle(self) -> bytes:
            raise NotImplementedError

    token = ForeignImmediateCompletion()
    assert not isinstance(token, ImmediateCompletion)

    published = publication_receipt(token, sequence=19)

    assert isinstance(published, ImmediateCompletion)
    assert published is not None
    assert published.contract_version == 2
    assert published.operation == "mask_reset"
    assert published.sequence == 19
    assert published.stream_id == 5
    assert published.producer_id == "producer-foreign"


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ({"contract_version": 1}, "contract"),
        ({"operation": "whole_collector"}, "operation"),
        ({"sequence": True}, "sequence"),
        ({"stream_id": True}, "stream"),
        ({"producer_id": " "}, "producer"),
    ),
)
def test_publication_receipt_rejects_malformed_foreign_metadata(field: dict, expected: str) -> None:
    class ForeignImmediateCompletion:
        contract_version = field.get("contract_version", 2)
        operation = field.get("operation", "step")
        sequence = field.get("sequence", 1)
        stream_id = field.get("stream_id", 0)
        producer_id = field.get("producer_id", "producer-foreign")

        def wait_before_read(self, stream=None) -> None:
            del stream

        def ipc_handle(self) -> bytes:
            raise NotImplementedError

    with pytest.raises((TypeError, ValueError), match=expected):
        publication_receipt(ForeignImmediateCompletion(), sequence=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ingress_publication_records_interprocess_event() -> None:
    completion = ImmediateCompletion(
        operation="step",
        sequence=7,
        stream_id=11,
        producer_id="producer-cuda",
    )
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        published = record_ingress_publication(completion, stream=stream)

    assert isinstance(published, EventCompletion)
    assert published.operation == completion.operation
    assert published.sequence == completion.sequence
    assert published.stream_id == completion.stream_id
    assert published.producer_id == completion.producer_id
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        wait_completion(published, stream=consumer)
    consumer.synchronize()
    assert isinstance(published.ipc_handle(), bytes)


def _spawned_collector(queue, release) -> None:
    """Write a CUDA ingress slot, then publish an IPC event without syncing."""
    device = torch.device("cuda", index=torch.cuda.current_device())
    stream = torch.cuda.current_stream(device)
    slot = torch.full((4, 3), 17.0, dtype=torch.float32, device=device).share_memory_()
    completion = ImmediateCompletion(
        operation="step",
        sequence=9,
        stream_id=int(stream.cuda_stream),
        producer_id="producer-spawn",
    )
    published = record_ingress_publication(completion, stream=stream)
    queue.put(
        (
            slot,
            published.contract_version,
            published.operation,
            published.sequence,
            published.stream_id,
            published.producer_id,
            published.ipc_handle(),
            device,
        )
    )
    assert release.wait(timeout=30)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spawned_collector_completion_receipt_imports_and_orders_read() -> None:
    queue = mp.get_context("spawn").Queue()
    release = mp.get_context("spawn").Event()
    process = mp.get_context("spawn").Process(
        target=_spawned_collector,
        args=(queue, release),
    )
    process.start()
    try:
        payload = queue.get(timeout=30)
        slot, version, operation, sequence, stream_id, producer_id, handle, device = payload

        imported = import_shared_completion(
            contract_version=version,
            operation=operation,
            sequence=sequence,
            stream_id=stream_id,
            producer_id=producer_id,
            ipc_handle=handle,
            device=device,
        )
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            wait_completion(imported, stream=consumer)
            copied = slot.clone()
        consumer.synchronize()
        torch.testing.assert_close(copied.cpu(), torch.full((4, 3), 17.0))
    finally:
        release.set()
        process.join(timeout=30)
    assert process.exitcode == 0


@pytest.mark.parametrize(
    "field",
    (
        {"contract_version": 1},
        {"operation": "learner-update"},
        {"producer_id": " "},
        {"producer_id": "x" * 257},
        {"ipc_handle": b""},
        {"ipc_handle": b"x" * 257},
    ),
)
def test_import_shared_completion_rejects_malformed_metadata(field: dict) -> None:
    valid = {
        "contract_version": 2,
        "operation": "step",
        "sequence": 1,
        "stream_id": 2,
        "producer_id": "producer-a",
        "ipc_handle": b"x" * 64,
        "device": object(),
    }
    with pytest.raises((TypeError, ValueError)):
        import_shared_completion(**(valid | field))  # type: ignore[arg-type]
