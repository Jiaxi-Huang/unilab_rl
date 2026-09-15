"""Shared collector-metrics drain for the async runners.

``APPORunner`` and ``OffPolicyRunner`` consume the same collector metrics
message protocol; this module owns the shared dispatch so each runner only
keeps a thin wrapper encoding its own semantics (collector-error propagation,
``buffer_size`` handling, trace events).
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Any


def drain_collector_metrics(
    queue: Any,
    reward_history: deque,
    reward_components: dict,
    logger: Any,
    trace_recorder: Any | None = None,
    *,
    runner_label: str,
    raise_on_collector_error: bool,
    require_buffer_size: bool,
    log_collector_reward: bool = True,
) -> None:
    """Drain all pending collector metrics messages and dispatch them to the logger.

    ``runner_label`` prefixes the stderr drain-error line.
    ``raise_on_collector_error`` controls whether a collector ``error`` message
    propagates as ``RuntimeError`` (off-policy) or is reported on stderr and
    swallowed (APPO). ``require_buffer_size`` controls whether ``log_collector``
    requires a ``buffer_size`` key (off-policy replay) or defaults it to 0
    (APPO uses shared memory, not a separate buffer).
    """
    while True:
        try:
            metrics = queue.get_nowait()
        except Exception:
            break
        if "error" in metrics:
            logger.log_status(f"[red]Collector ERROR: {metrics['error']}[/]")
            error = RuntimeError(f"Collector process failed: {metrics['error']}")
            if raise_on_collector_error:
                raise error
            print(f"[{runner_label}] metrics drain error: {error}", file=sys.stderr)
            break

        try:
            updated_reward = False
            if "runtime_manifest" in metrics:
                logger.update_runtime_manifest(metrics["runtime_manifest"])
            if "mean_ep_reward" in metrics:
                reward_history.append(metrics["mean_ep_reward"])
                updated_reward = True
            if "reward_components" in metrics:
                reward_components.clear()
                reward_components.update(metrics["reward_components"])
            if "mean_ep_length" in metrics:
                logger.update_ep_length(metrics["mean_ep_length"])
            if "collector_timing_ms" in metrics:
                logger.update_collector_timing(metrics["collector_timing_ms"])
            active_steps_per_sec = metrics.get("collector_active_steps_per_sec")
            if active_steps_per_sec is not None:
                logger.update_collector_active_steps_per_sec(float(active_steps_per_sec))
            if "timeout_rate" in metrics:
                logger.update_timeout_rate(float(metrics["timeout_rate"]))
            if "total_steps" in metrics and (not require_buffer_size or "buffer_size" in metrics):
                logger.log_collector(
                    metrics["total_steps"],
                    metrics.get("buffer_size", 0),
                    (
                        metrics.get("mean_ep_reward", 0.0)
                        if updated_reward and log_collector_reward
                        else 0.0
                    ),
                )
            if trace_recorder and "trace_events" in metrics:
                trace_recorder.extend(metrics["trace_events"])
        except Exception as exc:
            print(f"[{runner_label}] metrics drain error: {exc}", file=sys.stderr)
            break
