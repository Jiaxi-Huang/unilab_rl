"""Neutral graph-completion contract tests."""

from __future__ import annotations

import pytest

import uni_rl.ipc.graph_completion as graph_completion
from uni_rl.ipc.graph_completion import (
    ImmediateGraphCompletion,
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
    monkey_sequence = object.__new__(ImmediateGraphCompletion)
    object.__setattr__(monkey_sequence, "sequence", 10**9)
    stale = ImmediateGraphCompletion(sequence=10**9 - 1)

    graph_completion._LAST_SEQUENCE = 10**9 - 1
    try:
        wait_graph_completion(monkey_sequence)
    finally:
        graph_completion._LAST_SEQUENCE = 0
    graph_completion._LAST_SEQUENCE = 10**9
    try:
        with pytest.raises(ValueError, match="stale"):
            wait_graph_completion(stale)
    finally:
        graph_completion._LAST_SEQUENCE = 0
