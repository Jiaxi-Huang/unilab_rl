"""Neutral ordered-completion contract tests."""

from __future__ import annotations

import pytest
import torch

from uni_rl.ipc.completion import (
    EventCompletion,
    ImmediateCompletion,
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
    bad_producer = ImmediateCompletion(producer_id=" ")

    with pytest.raises(ValueError, match="contract"):
        wait_completion(bad_contract)
    with pytest.raises(ValueError, match="operation"):
        wait_completion(bad_operation)
    with pytest.raises(TypeError, match="sequence"):
        wait_completion(bad_sequence)
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


@pytest.mark.parametrize("operation", ("whole_collector", "selected_reset", "mask_reset"))
def test_removed_completion_operations_are_rejected(operation: str) -> None:
    completion = ImmediateCompletion()
    completion.operation = operation

    with pytest.raises(ValueError, match="unsupported completion operation"):
        wait_completion(completion)


def test_ingress_publication_preserves_cpu_receipt() -> None:
    completion = ImmediateCompletion(producer_id="producer-cpu")

    assert record_ingress_publication(completion, stream=None) is completion


def test_publication_receipt_accepts_foreign_structural_producer() -> None:
    class ForeignImmediateCompletion:
        def __init__(self, **overrides: object) -> None:
            self.contract_version: int = 3
            self.operation: str = "step"
            self.sequence: int = 17
            self.producer_id: str = "producer-foreign"
            for name, value in overrides.items():
                setattr(self, name, value)

        def wait_before_read(self, stream=None) -> None:
            del stream

    token = ForeignImmediateCompletion()
    assert not isinstance(token, ImmediateCompletion)

    published = publication_receipt(token, sequence=19)

    assert isinstance(published, ImmediateCompletion)
    assert published is not None
    assert published.contract_version == 3
    assert published.operation == "step"
    assert published.sequence == 19
    assert published.producer_id == "producer-foreign"


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ({"contract_version": 1}, "contract"),
        ({"operation": "whole_collector"}, "operation"),
        ({"operation": "selected_reset"}, "operation"),
        ({"operation": "mask_reset"}, "operation"),
        ({"sequence": True}, "sequence"),
        ({"producer_id": " "}, "producer"),
    ),
)
def test_publication_receipt_rejects_malformed_foreign_metadata(field: dict, expected: str) -> None:
    class ForeignImmediateCompletion:
        contract_version = field.get("contract_version", 3)
        operation = field.get("operation", "step")
        sequence = field.get("sequence", 1)
        producer_id = field.get("producer_id", "producer-foreign")

        def wait_before_read(self, stream=None) -> None:
            del stream

    with pytest.raises((TypeError, ValueError), match=expected):
        publication_receipt(ForeignImmediateCompletion(), sequence=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ingress_publication_records_interprocess_event() -> None:
    completion = ImmediateCompletion(
        operation="step",
        sequence=7,
        producer_id="producer-cuda",
    )
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        published = record_ingress_publication(completion, stream=stream)

    assert isinstance(published, EventCompletion)
    assert published.operation == completion.operation
    assert published.sequence == completion.sequence
    assert published.producer_id == completion.producer_id
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        wait_completion(published, stream=consumer)
    consumer.synchronize()
