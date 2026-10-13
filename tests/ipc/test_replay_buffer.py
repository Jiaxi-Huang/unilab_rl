"""Tests for the bounded off-policy replay ingress."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time

import pytest
import torch

from uni_rl.ipc.graph_completion import ImmediateGraphCompletion
from uni_rl.ipc.replay_buffer import ReplayBuffer

_SPAWN_CTX = mp.get_context("spawn")
_OBS_DIM = 8
_ACTION_DIM = 3


def _make_buf(*, capacity: int = 128, slot_rows: int = 8, depth: int = 2) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=capacity,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=slot_rows,
        ingress_depth=depth,
    )


def _random_batch(n: int):
    return (
        torch.randn(n, _OBS_DIM),
        torch.randn(n, _ACTION_DIM),
        torch.randn(n),
        torch.randn(n, _OBS_DIM),
        torch.zeros(n),
        torch.zeros(n),
    )


def _terminal_contract_batch(n: int):
    obs, actions, rewards, next_obs, dones, truncated = _random_batch(n)
    terminal_mask = torch.tensor([index % 2 == 1 for index in range(n)], dtype=torch.bool)
    return (
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        terminal_mask,
        torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(n, _OBS_DIM).contiguous(),
        torch.zeros(n, 3),
        torch.ones(n, 3),
        torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(n, 3).contiguous(),
    )


def _take_and_commit(buf: ReplayBuffer) -> tuple[int, int, torch.Tensor]:
    ingress = buf.take_published_ingress()
    assert ingress is not None
    slot, start, count, packed = ingress
    buf.commit_ingress(slot=slot, start=start, count=count)
    return start, count, packed


def test_host_allocation_is_capacity_independent_and_commit_is_device_owned():
    small = _make_buf(capacity=32, slot_rows=4)
    large = _make_buf(capacity=3200, slot_rows=4)
    assert small.host_storage_bytes == large.host_storage_bytes
    assert not hasattr(small, "_storage")

    obs, act, rew, nobs, done, trunc = _random_batch(4)
    small.add(obs, act, rew, nobs, done, trunc)
    assert small.published_ptr == 4
    assert int(small.ptr[0]) == 0
    assert int(small.size[0]) == 0

    start, count, packed = _take_and_commit(small)
    assert (start, count) == (0, 4)
    torch.testing.assert_close(packed[:, small._obs_sl], obs)
    torch.testing.assert_close(packed[:, small._act_sl], act)
    assert int(small.ptr[0]) == 4
    assert int(small.size[0]) == 4
    diagnostics = small.ingress_diagnostics()
    assert diagnostics == {
        "ingress_depth": 2,
        "ingress_slot_rows": 4,
        "published_sequence": 1,
        "release_sequence": 1,
        "occupancy": 0,
        "high_water_occupancy": 1,
        "backpressure_waits": 0,
        "backpressure_wait_s": 0.0,
        "early_returns": 0,
        "dropped_batches": 0,
        "closed_returns": 0,
        "stop_returns": 0,
    }
    small.close()
    large.close()


def test_ingress_patches_terminal_rows_before_publication():
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=4,
        critic_dim=3,
    )
    terminal_mask = torch.tensor([False, True, False, True])
    terminal_obs = torch.full((4, _OBS_DIM), 17.0)
    terminal_critic = torch.full((4, 3), 23.0)
    obs, act, rew, nobs, done, trunc = _random_batch(4)

    buf.add(
        obs,
        act,
        rew,
        nobs,
        done,
        trunc,
        terminal_mask=terminal_mask,
        terminal_next_obs=terminal_obs,
        critic=torch.zeros(4, 3),
        next_critic=torch.ones(4, 3),
        terminal_next_critic=terminal_critic,
    )

    _, _, packed = _take_and_commit(buf)
    torch.testing.assert_close(packed[terminal_mask, buf._nobs_sl], terminal_obs[terminal_mask])
    torch.testing.assert_close(
        packed[terminal_mask, buf._ncritic_sl],
        terminal_critic[terminal_mask],
    )
    buf.close()


def test_add_batch_chunks_partial_final_rows_and_terminal_contracts():
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=3,
        critic_dim=3,
        ingress_depth=2,
    )
    batch = _terminal_contract_batch(5)
    (
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        terminal_mask,
        terminal_obs,
        critic,
        next_critic,
        terminal_critic,
    ) = batch
    expected_next_obs = next_obs.clone()
    expected_next_obs[terminal_mask] = terminal_obs[terminal_mask]
    expected_next_critic = next_critic.clone()
    expected_next_critic[terminal_mask] = terminal_critic[terminal_mask]

    buf.add_batch(*batch)
    assert buf.published_ptr == 5

    first_start, first_count, first = _take_and_commit(buf)
    second_start, second_count, second = _take_and_commit(buf)
    assert (first_start, first_count) == (0, 3)
    assert (second_start, second_count) == (3, 2)

    packed = torch.cat((first, second))
    torch.testing.assert_close(packed[:, buf._obs_sl], obs)
    torch.testing.assert_close(packed[:, buf._act_sl], actions)
    torch.testing.assert_close(packed[:, buf._rew_col], rewards)
    torch.testing.assert_close(packed[:, buf._nobs_sl], expected_next_obs)
    torch.testing.assert_close(packed[:, buf._done_col], dones)
    torch.testing.assert_close(packed[:, buf._trunc_col], truncated)
    torch.testing.assert_close(packed[:, buf._critic_sl], critic)
    torch.testing.assert_close(packed[:, buf._ncritic_sl], expected_next_critic)
    torch.testing.assert_close(
        packed[terminal_mask, buf._nobs_sl],
        terminal_obs[terminal_mask],
    )
    torch.testing.assert_close(
        packed[terminal_mask, buf._ncritic_sl],
        terminal_critic[terminal_mask],
    )

    diagnostics = buf.ingress_diagnostics()
    assert diagnostics["published_sequence"] == 2
    assert diagnostics["release_sequence"] == 2
    assert diagnostics["occupancy"] == 0
    assert diagnostics["high_water_occupancy"] == 2
    assert diagnostics["dropped_batches"] == 0
    assert int(buf.ptr[0]) == 5
    assert int(buf.size[0]) == 5
    buf.close()


def test_add_batch_uses_single_slot_when_batch_fits():
    buf = _make_buf(slot_rows=4)
    batch = _random_batch(3)

    buf.add_batch(*batch)

    assert buf.published_ptr == 3
    start, count, packed = _take_and_commit(buf)
    assert (start, count) == (0, 3)
    torch.testing.assert_close(packed[:, buf._obs_sl], batch[0])
    diagnostics = buf.ingress_diagnostics()
    assert diagnostics["published_sequence"] == 1
    assert diagnostics["release_sequence"] == 1
    buf.close()


def test_add_batch_backpressures_between_chunks_until_learner_commits():
    buf = _make_buf(capacity=16, slot_rows=3, depth=1)
    batch = _random_batch(6)
    add_started = threading.Event()
    add_finished = threading.Event()

    def add_chunked() -> None:
        add_started.set()
        buf.add_batch(*batch)
        add_finished.set()

    thread = threading.Thread(target=add_chunked)
    thread.start()
    assert add_started.wait(timeout=1.0)
    deadline = time.monotonic() + 2.0
    while buf.published_ptr < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert buf.published_ptr == 3
    # The producer's semaphore acquire has a 50 ms timeout. Hold the occupied
    # slot beyond that boundary so this exercises a completed backpressure
    # retry rather than the release/acquire race at the timeout edge.
    time.sleep(0.06)
    first_start, first_count, first = _take_and_commit(buf)
    assert (first_start, first_count) == (0, 3)
    torch.testing.assert_close(first[:, buf._obs_sl], batch[0][:3])
    assert add_finished.wait(timeout=2.0)
    thread.join(timeout=1.0)

    second_start, second_count, second = _take_and_commit(buf)
    assert (second_start, second_count) == (3, 3)
    torch.testing.assert_close(second[:, buf._obs_sl], batch[0][3:])
    diagnostics = buf.ingress_diagnostics()
    assert diagnostics["published_sequence"] == 2
    assert diagnostics["release_sequence"] == 2
    assert diagnostics["high_water_occupancy"] == 1
    assert diagnostics["backpressure_waits"] >= 1
    assert diagnostics["dropped_batches"] == 0
    buf.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ingress_publication_requires_completion_read_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_ingress_read(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("diagnostics must not read CUDA ingress tensors")

    protected_ingress_slots = []

    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cuda",
        ingress_slot_rows=4,
        ingress_device="cuda",
    )
    try:
        for slot in buf._ingress_slots:
            monkeypatch.setattr(slot, "cpu", fail_ingress_read)
            monkeypatch.setattr(slot, "item", fail_ingress_read)
            protected_ingress_slots.append(slot)
        batch = tuple(tensor.to("cuda") for tensor in _random_batch(4))
        with pytest.raises(RuntimeError, match="graph completion receipt"):
            buf.add(*batch)

        class StreamCompletion(ImmediateGraphCompletion):
            def wait_before_read(self, stream=None):
                assert isinstance(stream, torch.cuda.Stream)

        buf.add(
            *batch,
            completion=StreamCompletion(operation="step", capture_id="capture-ingress"),
        )

        assert protected_ingress_slots
        assert buf.ingress_diagnostics()["high_water_occupancy"] == 1
        assert buf.published_ptr == 4
        ingress = buf.take_published_ingress()
        assert ingress is not None
        slot, start, count, packed = ingress
        buf.commit_ingress(slot=slot, start=start, count=count)

        assert (start, count) == (0, 4)
        torch.testing.assert_close(packed[:, buf._obs_sl], batch[0])
        torch.testing.assert_close(packed[:, buf._act_sl], batch[1])
    finally:
        buf.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_add_batch_reuses_one_completion_gate_for_each_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waited_streams: list[object] = []

    class RecordingCompletion(ImmediateGraphCompletion):
        def wait_before_read(self, stream=None):
            waited_streams.append(stream)

    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cuda",
        ingress_slot_rows=2,
        ingress_depth=3,
        ingress_device="cuda",
    )
    batch = tuple(tensor.to("cuda") for tensor in _random_batch(5))
    del monkeypatch

    try:
        assert (
            buf.add_batch(
                *batch,
                completion=RecordingCompletion(capture_id="capture-ingress"),
            )
            is True
        )
        chunks = [_take_and_commit(buf) for _ in range(3)]
        assert [count for _, count, _ in chunks] == [2, 2, 1]
        assert all(isinstance(stream, torch.cuda.Stream) for stream in waited_streams)
        assert len(waited_streams) == 3
        packed_obs = torch.cat(
            (
                chunks[0][2][:, buf._obs_sl],
                chunks[1][2][:, buf._obs_sl],
                chunks[2][2][:, buf._obs_sl],
            )
        )
        torch.testing.assert_close(packed_obs, batch[0])
    finally:
        buf.close()


def test_ingress_stores_done_and_truncated_contract():
    buf = _make_buf(slot_rows=3)
    truncated = torch.tensor([0.0, 1.0, 0.0])
    dones = torch.tensor([1.0, 1.0, 0.0])
    buf.add(
        torch.zeros(3, _OBS_DIM),
        torch.zeros(3, _ACTION_DIM),
        torch.zeros(3),
        torch.zeros(3, _OBS_DIM),
        dones,
        truncated,
    )

    _, _, packed = _take_and_commit(buf)
    torch.testing.assert_close(packed[:, buf._done_col], dones)
    torch.testing.assert_close(packed[:, buf._trunc_col], truncated)
    buf.close()


def test_ingress_rejects_stale_completion_within_one_buffer() -> None:
    buf = _make_buf()

    assert buf.add(
        *_random_batch(4),
        completion=ImmediateGraphCompletion(sequence=2),
    )
    with pytest.raises(ValueError, match="stale graph completion"):
        buf.add(
            *_random_batch(4),
            completion=ImmediateGraphCompletion(sequence=1),
        )
    buf.close()


def test_ingress_writes_packed_columns_without_cat(monkeypatch: pytest.MonkeyPatch):
    buf = ReplayBuffer(
        capacity=8,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=4,
        critic_dim=5,
    )
    obs = torch.randn(4, _OBS_DIM)
    actions = torch.randn(4, _ACTION_DIM)
    critic = torch.randn(4, 5)

    def _fail_cat(*args, **kwargs):
        del args, kwargs
        raise AssertionError("ReplayBuffer.add must write packed columns directly")

    monkeypatch.setattr(torch, "cat", _fail_cat)
    buf.add(
        obs,
        actions,
        torch.randn(4),
        torch.randn(4, _OBS_DIM),
        torch.zeros(4),
        torch.zeros(4),
        critic=critic,
        next_critic=torch.randn(4, 5),
    )

    _, _, packed = _take_and_commit(buf)
    torch.testing.assert_close(packed[:, buf._obs_sl], obs)
    torch.testing.assert_close(packed[:, buf._act_sl], actions)
    torch.testing.assert_close(packed[:, buf._critic_sl], critic)
    buf.close()


def test_ingress_rejects_collection_chunk_larger_than_slot():
    buf = _make_buf(slot_rows=4)
    with pytest.raises(ValueError, match="slots hold 4"):
        buf.add(*_random_batch(5))
    buf.close()


def test_ingress_backpressures_until_committed_slot_is_released():
    buf = _make_buf(capacity=16, slot_rows=4, depth=1)
    buf.add(*_random_batch(4))
    assert buf.ingress_diagnostics()["occupancy"] == 1
    assert buf.ingress_diagnostics()["high_water_occupancy"] == 1
    add_started = threading.Event()
    add_finished = threading.Event()

    def add_second() -> None:
        add_started.set()
        buf.add(*_random_batch(4))
        add_finished.set()

    thread = threading.Thread(target=add_second)
    thread.start()
    assert add_started.wait(timeout=1.0)
    assert not add_finished.wait(timeout=0.05)
    # The producer's semaphore acquire has a 50 ms timeout. Hold the full slot
    # beyond that boundary so this test exercises a completed backpressure retry
    # rather than the release/acquire race at the timeout edge.
    time.sleep(0.02)
    _take_and_commit(buf)
    assert add_finished.wait(timeout=1.0)
    thread.join(timeout=1.0)
    _take_and_commit(buf)
    diagnostics = buf.ingress_diagnostics()
    assert diagnostics["published_sequence"] == 2
    assert diagnostics["release_sequence"] == 2
    assert diagnostics["occupancy"] == 0
    assert diagnostics["high_water_occupancy"] == 1
    assert diagnostics["backpressure_waits"] == 1
    assert diagnostics["backpressure_wait_s"] >= 0.04
    assert diagnostics["dropped_batches"] == 0
    buf.close()


def test_add_batch_stop_retains_only_published_prefix():
    stop_event = _SPAWN_CTX.Event()
    buf = _make_buf(capacity=16, slot_rows=2, depth=1)
    buf.attach_stop_event(stop_event)
    batch = _random_batch(4)
    add_started = threading.Event()
    add_finished = threading.Event()
    results: list[bool] = []

    def add_chunked() -> None:
        add_started.set()
        results.append(buf.add_batch(*batch))
        add_finished.set()

    thread = threading.Thread(target=add_chunked)
    thread.start()
    deadline = time.monotonic() + 2.0
    while buf.published_ptr < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert buf.published_ptr == 2
    stop_event.set()
    assert add_finished.wait(timeout=2.0)
    thread.join(timeout=1.0)

    assert results == [False]
    diagnostics = buf.ingress_diagnostics()
    assert diagnostics["published_sequence"] == 1
    assert diagnostics["stop_returns"] == 1
    assert diagnostics["early_returns"] == 1
    assert diagnostics["dropped_batches"] == 1
    buf.close()


def test_add_batch_close_stops_after_published_prefix():
    buf = _make_buf(capacity=16, slot_rows=2, depth=1)
    batch = _random_batch(4)
    add_started = threading.Event()
    add_finished = threading.Event()
    results: list[bool] = []

    def add_chunked() -> None:
        add_started.set()
        results.append(buf.add_batch(*batch))
        add_finished.set()

    thread = threading.Thread(target=add_chunked)
    thread.start()
    assert add_started.wait(timeout=1.0)
    deadline = time.monotonic() + 2.0
    while buf.published_ptr < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert buf.published_ptr == 2
    buf.close()
    assert add_finished.wait(timeout=2.0)
    thread.join(timeout=1.0)

    assert results == [False]
    diagnostics = buf.ingress_diagnostics()
    assert diagnostics["published_sequence"] == 1
    assert diagnostics["closed_returns"] == 1
    assert diagnostics["early_returns"] == 1
    assert diagnostics["dropped_batches"] == 1


def test_ingress_shutdown_and_abnormal_stop_count_dropped_batches():
    stop_event = _SPAWN_CTX.Event()
    stopping = _make_buf(capacity=16, slot_rows=4, depth=1)
    stopping.attach_stop_event(stop_event)
    stopping.add(*_random_batch(4))
    stop_event.set()
    stopping.add(*_random_batch(4))

    stopped = stopping.ingress_diagnostics()
    assert stopped["stop_returns"] == 1
    assert stopped["early_returns"] == 1
    assert stopped["dropped_batches"] == 1
    assert stopped["published_sequence"] == 1
    assert stopped["release_sequence"] == 0
    stopping.close()

    closed = _make_buf(capacity=16, slot_rows=4, depth=1)
    closed.add(*_random_batch(4))
    closed.close()
    closed.add(*_random_batch(4))
    diagnostics = closed.ingress_diagnostics()
    assert diagnostics["closed_returns"] == 1
    assert diagnostics["early_returns"] == 1
    assert diagnostics["dropped_batches"] == 1
    assert diagnostics["published_sequence"] == 1


def _collector_add(buf: ReplayBuffer, chunks: int) -> None:
    for _ in range(chunks):
        buf.add(*_random_batch(8))


def _collector_publish_cuda_completion(buf: ReplayBuffer) -> None:
    batch = tuple(tensor.to("cuda") for tensor in _random_batch(4))
    completion = ImmediateGraphCompletion(
        operation="step", sequence=1, capture_id="capture-cross-process"
    )
    assert buf.add(*batch, completion=completion) is True


def test_spawned_collector_publishes_bounded_chunks():
    buf = _make_buf(capacity=128, slot_rows=8)
    process = _SPAWN_CTX.Process(target=_collector_add, args=(buf, 4))
    process.start()

    committed = 0
    deadline = time.monotonic() + 15.0
    while committed < 32 and time.monotonic() < deadline:
        ingress = buf.take_published_ingress()
        if ingress is None:
            time.sleep(0.001)
            continue
        slot, start, count, _ = ingress
        buf.commit_ingress(slot=slot, start=start, count=count)
        committed += count

    process.join(timeout=15)
    assert process.exitcode == 0
    assert committed == 32
    assert int(buf.ptr[0]) == 32
    assert int(buf.size[0]) == 32
    diagnostics = buf.ingress_diagnostics()
    assert diagnostics["published_sequence"] == 4
    assert diagnostics["release_sequence"] == 4
    assert 1 <= diagnostics["high_water_occupancy"] <= 2
    assert diagnostics["dropped_batches"] == 0
    buf.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spawned_collector_transports_ingress_completion_to_learner_copy() -> None:
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cuda",
        ingress_slot_rows=4,
        ingress_device="cuda",
    )
    process = _SPAWN_CTX.Process(target=_collector_publish_cuda_completion, args=(buf,))
    process.start()
    try:
        deadline = time.monotonic() + 15.0
        while buf.published_ptr == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        imported = buf.import_shared_graph_completion()
        assert imported is not None
        assert imported.capture_id == "capture-cross-process"
        assert imported.sequence == 1
        assert buf.graph_capture_id == "capture-cross-process"
        ingress = buf.take_published_ingress()
        assert ingress is not None
        slot, start, count, packed = ingress
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            from uni_rl.ipc.graph_completion import wait_graph_completion

            wait_graph_completion(
                imported,
                stream=consumer,
                capture_id="capture-cross-process",
                last_sequence=0,
            )
            copied = packed.clone()
        consumer.synchronize()
        assert copied.shape == (4, buf.storage_width)
        buf.commit_ingress(slot=slot, start=start, count=count)
    finally:
        process.join(timeout=15)
        buf.close()
    assert process.exitcode == 0


def test_empty_terminal_contract_skips_host_bool_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _make_buf(capacity=16, slot_rows=4, depth=2)
    called = False

    def fail_any(mask):
        raise AssertionError("empty terminal contract host-converted terminal_mask.any()")

    monkeypatch.setattr(torch.Tensor, "any", fail_any)
    obs, actions, rewards, next_obs, dones, truncated = _random_batch(4)
    empty_terminal_obs = torch.full((4, _OBS_DIM), 17.0)

    try:
        buf.add(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            truncated,
            terminal_mask=torch.zeros(4, dtype=torch.bool),
            terminal_next_obs=empty_terminal_obs,
        )
    finally:
        monkeypatch.undo()

    _, _, packed = _take_and_commit(buf)
    torch.testing.assert_close(packed[:, buf._nobs_sl], next_obs)
    assert not called
    buf.close()


def test_terminal_contract_patch_is_fixed_width_masked_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=4,
        critic_dim=3,
    )

    def fail_host_call(*_args, **_kwargs):
        raise AssertionError("terminal publication must not convert masks to host")

    monkeypatch.setattr(torch.Tensor, "any", fail_host_call)
    monkeypatch.setattr(torch.Tensor, "item", fail_host_call)
    monkeypatch.setattr(torch.Tensor, "tolist", fail_host_call)
    monkeypatch.setattr(torch.Tensor, "cpu", fail_host_call)
    try:
        buf.add(
            *_terminal_contract_batch(4)[:6],
            terminal_mask=torch.tensor([False, True, False, True]),
            terminal_next_obs=torch.full((4, _OBS_DIM), 17.0),
            critic=torch.zeros(4, 3),
            next_critic=torch.zeros(4, 3),
            terminal_next_critic=torch.full((4, 3), 19.0),
        )
    finally:
        monkeypatch.undo()

    _, _, packed = _take_and_commit(buf)
    torch.testing.assert_close(
        packed[:, buf._nobs_sl],
        torch.stack(
            (
                packed[0, buf._nobs_sl],
                torch.full((_OBS_DIM,), 17.0),
                packed[2, buf._nobs_sl],
                torch.full((_OBS_DIM,), 17.0),
            )
        ),
    )
    torch.testing.assert_close(packed[:, buf._ncritic_sl][1], torch.full((3,), 19.0))
    torch.testing.assert_close(packed[:, buf._ncritic_sl][0], torch.zeros(3))
    buf.close()


def test_commit_ingress_rejects_out_of_order_publication() -> None:
    buf = _make_buf(capacity=32, slot_rows=4, depth=2)
    first = _terminal_contract_batch(4)
    second = _terminal_contract_batch(4)
    buf.add(*first)
    buf.add(*second)

    first_ingress = buf.take_published_ingress()
    second_ingress = buf.take_published_ingress()
    assert first_ingress is not None and second_ingress is not None
    second_slot, second_start, second_count, _ = second_ingress

    with pytest.raises(RuntimeError, match="publication order"):
        buf.commit_ingress(slot=second_slot, start=second_start, count=second_count)

    first_slot, first_start, first_count, _ = first_ingress
    buf.commit_ingress(slot=first_slot, start=first_start, count=first_count)
    buf.commit_ingress(slot=second_slot, start=second_start, count=second_count)
    assert int(buf.ptr[0]) == 8
    assert int(buf.size[0]) == 8
    buf.close()


def test_commit_ingress_rejects_noncontiguous_start() -> None:
    buf = _make_buf(capacity=32, slot_rows=4, depth=1)
    buf.add(*_random_batch(4))
    ingress = buf.take_published_ingress()
    assert ingress is not None
    slot, _start, count, _ = ingress

    with pytest.raises(RuntimeError, match="not contiguous"):
        buf.commit_ingress(slot=slot, start=4, count=count)

    buf.commit_ingress(slot=slot, start=0, count=count)
    assert int(buf.ptr[0]) == 4
    buf.close()
