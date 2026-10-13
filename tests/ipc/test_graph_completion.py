"""Neutral graph-completion contract tests."""

from __future__ import annotations

import multiprocessing as mp

import pytest
import torch

from uni_rl.ipc.graph_completion import (
    EventGraphCompletion,
    ImmediateGraphCompletion,
    import_graph_completion,
    record_ingress_publication,
    wait_graph_completion,
)


def test_immediate_completion_is_pass_through() -> None:
    completion = ImmediateGraphCompletion()

    assert wait_graph_completion(completion, stream=object()) is completion


def test_wait_graph_completion_validates_public_metadata() -> None:
    bad_contract = ImmediateGraphCompletion(contract_version=2)
    bad_operation = ImmediateGraphCompletion(operation="learner-update")
    bad_sequence = ImmediateGraphCompletion(sequence=True)
    bad_stream = ImmediateGraphCompletion(stream_id=True)

    with pytest.raises(ValueError, match="contract"):
        wait_graph_completion(bad_contract)
    with pytest.raises(ValueError, match="operation"):
        wait_graph_completion(bad_operation)
    with pytest.raises(TypeError, match="sequence"):
        wait_graph_completion(bad_sequence)
    with pytest.raises(TypeError, match="stream_id"):
        wait_graph_completion(bad_stream)


def test_wait_graph_completion_rejects_non_protocol_producer() -> None:
    with pytest.raises(TypeError, match="wait_before_read"):
        wait_graph_completion(object())  # type: ignore[arg-type]


def test_wait_graph_completion_rejects_stale_sequences() -> None:
    current = ImmediateGraphCompletion(sequence=10**9)
    stale = ImmediateGraphCompletion(sequence=10**9 - 1)

    wait_graph_completion(current, last_sequence=10**9 - 1)
    with pytest.raises(ValueError, match="stale"):
        wait_graph_completion(stale, last_sequence=10**9)


def test_wait_graph_completion_binds_operation_and_capture() -> None:
    completion = ImmediateGraphCompletion(operation="step", capture_id="capture-a")

    with pytest.raises(ValueError, match="expected 'mask_reset'"):
        wait_graph_completion(completion, operation="mask_reset")
    with pytest.raises(ValueError, match="capture"):
        wait_graph_completion(completion, capture_id="capture-b")
    assert wait_graph_completion(completion, capture_id="capture-a") is completion


def test_immediate_completion_has_no_undeclared_ipc_transport() -> None:
    with pytest.raises(NotImplementedError, match="IPC"):
        ImmediateGraphCompletion().ipc_handle()


def test_ingress_publication_preserves_cpu_receipt() -> None:
    completion = ImmediateGraphCompletion(capture_id="capture-cpu")

    assert record_ingress_publication(completion, stream=None) is completion


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ingress_publication_records_interprocess_event() -> None:
    completion = ImmediateGraphCompletion(
        operation="step",
        sequence=7,
        stream_id=11,
        capture_id="capture-cuda",
    )
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        published = record_ingress_publication(completion, stream=stream)

    assert isinstance(published, EventGraphCompletion)
    assert published.operation == completion.operation
    assert published.sequence == completion.sequence
    assert published.stream_id == completion.stream_id
    assert published.capture_id == completion.capture_id
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        wait_graph_completion(published, stream=consumer)
    consumer.synchronize()
    assert isinstance(published.ipc_handle(), bytes)


def _spawned_collector(queue, release) -> None:
    """Write a CUDA ingress slot, then publish an IPC event without syncing."""
    device = torch.device("cuda", index=torch.cuda.current_device())
    stream = torch.cuda.current_stream(device)
    slot = torch.full((4, 3), 17.0, dtype=torch.float32, device=device).share_memory_()
    completion = ImmediateGraphCompletion(
        operation="step",
        sequence=9,
        stream_id=int(stream.cuda_stream),
        capture_id="capture-spawn",
    )
    published = record_ingress_publication(completion, stream=stream)
    queue.put(
        (
            slot,
            published.contract_version,
            published.operation,
            published.sequence,
            published.stream_id,
            published.capture_id,
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
        slot, version, operation, sequence, stream_id, capture_id, handle, device = payload

        imported = import_graph_completion(
            contract_version=version,
            operation=operation,
            sequence=sequence,
            stream_id=stream_id,
            capture_id=capture_id,
            ipc_handle=handle,
            device=device,
        )
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            wait_graph_completion(imported, stream=consumer)
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
        {"contract_version": 2},
        {"operation": "learner-update"},
        {"capture_id": None},
        {"ipc_handle": b""},
        {"ipc_handle": b"x" * 257},
    ),
)
def test_import_graph_completion_rejects_malformed_metadata(field: dict) -> None:
    valid = {
        "contract_version": 1,
        "operation": "step",
        "sequence": 1,
        "stream_id": 2,
        "capture_id": "capture-a",
        "ipc_handle": b"x" * 64,
        "device": object(),
    }
    with pytest.raises((TypeError, ValueError)):
        import_graph_completion(**(valid | field))  # type: ignore[arg-type]
