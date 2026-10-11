"""Off-policy logger timing-window contracts."""

from __future__ import annotations

import pytest

from uni_rl.logging.offpolicy import OffPolicyLogger


def test_tail_env_steps_per_sec_uses_bounded_window() -> None:
    logger = OffPolicyLogger(algo_name="SAC", num_envs=4, max_iterations=150)
    for iteration in range(1, 151):
        logger.log_step(
            iteration=iteration,
            iteration_time=0.5,
            extra_info={"throughput_steps": 8},
        )

    assert logger.tail_window_iterations == 100
    assert logger._tail_iteration_times.maxlen == 100
    assert len(logger._tail_iteration_times) == 100
    assert logger._tail_throughput_env_steps == 100 * 8
    assert logger._get_tail_env_steps_per_sec() == pytest.approx(16.0)


def test_tail_window_reports_partial_window_and_evicts_oldest_steps() -> None:
    logger = OffPolicyLogger(algo_name="SAC", num_envs=4, max_iterations=150)

    logger.log_step(iteration=1, iteration_time=0.25, extra_info={"throughput_steps": 2})

    assert len(logger._tail_iteration_times) == 1
    assert logger._tail_throughput_env_steps == 2
    assert logger._get_tail_env_steps_per_sec() == pytest.approx(8.0)

    for iteration in range(2, 102):
        logger.log_step(
            iteration=iteration,
            iteration_time=0.5,
            extra_info={"throughput_steps": 8},
        )

    assert len(logger._tail_iteration_times) == 100
    assert logger._tail_throughput_env_steps == 794
    assert logger._get_tail_env_steps_per_sec() == pytest.approx(794 / 50.0)
