"""Device-resident collector metric aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class TensorMetricFlush:
    """One host transfer of completed episode and done-count metrics."""

    rewards: list[float]
    lengths: list[int]
    done_count: int
    timeout_count: int
    reward_component_sums: dict[str, float] | None = None


class TensorCollectorMetrics:
    """Accumulate tensor-collector metrics and flush them in one D2H transfer.

    A fixed ``interval * num_envs`` window removes per-step scalar ``.item()``,
    ``.tolist()``, and dynamic ``nonzero()`` host synchronization. Compaction is
    performed on device once per flush; only the compacted result reaches host.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        interval: int,
        device: str | torch.device,
        reward_component_names: Sequence[str] = (),
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("tensor collector metrics require a positive environment count")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("tensor collector metrics require a positive interval")
        self.num_envs = int(num_envs)
        self.interval = int(interval)
        self.device = torch.device(device)
        self.reward_component_names = tuple(reward_component_names)
        self.reward_component_count = len(self.reward_component_names)
        self.reward_component_sums = (
            torch.zeros(self.reward_component_count, dtype=torch.float64, device=self.device)
            if self.reward_component_count
            else None
        )
        self.current_rewards = torch.zeros(self.num_envs, device=self.device)
        self.current_lengths = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.finished_rewards = torch.zeros(
            (self.interval, self.num_envs), dtype=torch.float64, device=self.device
        )
        self.finished_lengths = torch.zeros(
            (self.interval, self.num_envs), dtype=torch.float64, device=self.device
        )
        self.finished_valid = torch.zeros(
            (self.interval, self.num_envs), dtype=torch.bool, device=self.device
        )
        self.done_count = torch.zeros((), dtype=torch.int64, device=self.device)
        self.timeout_count = torch.zeros((), dtype=torch.int64, device=self.device)
        self._cursor = 0
        self._updates = 0
        self._finalized = False

    @property
    def ready(self) -> bool:
        return self._updates >= self.interval

    def update(
        self,
        rewards: torch.Tensor,
        done: torch.Tensor,
        timeout: torch.Tensor,
        reward_components: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(rewards, torch.Tensor):
            raise TypeError("tensor metric rewards must be Torch tensors")
        if not rewards.is_floating_point():
            raise TypeError("tensor metric rewards must have a floating dtype")
        if not isinstance(done, torch.Tensor) or not isinstance(timeout, torch.Tensor):
            raise TypeError("tensor metric done and timeout values must be Torch tensors")
        if rewards.shape != (self.num_envs,) or done.shape != rewards.shape:
            raise ValueError("tensor metric values must have environment shape")
        if timeout.shape != done.shape:
            raise ValueError("tensor metric timeout values must have environment shape")
        if done.dtype != torch.bool or timeout.dtype != torch.bool:
            raise TypeError("tensor metric done and timeout values must be boolean tensors")
        if (
            rewards.device != self.device
            or done.device != self.device
            or timeout.device != self.device
        ):
            raise ValueError("tensor metric values must live on one device")
        if self.reward_component_count:
            if not isinstance(reward_components, torch.Tensor):
                raise TypeError("tensor metric reward components must be a Torch tensor")
            expected_components = (self.num_envs, self.reward_component_count)
            if reward_components.shape != expected_components:
                raise ValueError(
                    f"tensor metric reward components must have shape {expected_components}"
                )
            if reward_components.dtype != torch.float32 or not reward_components.is_contiguous():
                raise TypeError("tensor metric reward components must be contiguous float32")
            if reward_components.device != self.device:
                raise ValueError(f"tensor metric reward components must live on {self.device}")
        elif reward_components is not None:
            raise ValueError("tensor metric has no reward component names")
        if self._updates >= self.interval:
            raise RuntimeError("tensor metric window was not flushed before reuse")

        episode_rewards = self.current_rewards + rewards
        episode_lengths = self.current_lengths + 1
        row = self._cursor
        self.finished_rewards[row].copy_(episode_rewards)
        self.finished_lengths[row].copy_(episode_lengths)
        self.finished_valid[row].copy_(done)
        self.current_rewards = torch.where(done, torch.zeros_like(episode_rewards), episode_rewards)
        self.current_lengths = torch.where(done, torch.zeros_like(episode_lengths), episode_lengths)
        self.done_count += done.sum()
        self.timeout_count += timeout.sum()
        if self.reward_component_sums is not None and reward_components is not None:
            self.reward_component_sums += reward_components.sum(dim=0, dtype=torch.float64)
        self._cursor = (row + 1) % self.interval
        self._updates += 1

    def flush(self) -> TensorMetricFlush:
        if self._finalized:
            raise RuntimeError("tensor metric window was already finalized")
        if not self.ready:
            raise RuntimeError("tensor metric window is not ready")
        return self._flush()

    def final_flush(self) -> TensorMetricFlush:
        """Flush completed episodes once at collector shutdown.

        A partial shutdown flush uses the same device-side compaction and single
        packed host transfer as the periodic path.  Unfinished episode
        accumulators remain in ``current_rewards`` and ``current_lengths`` and
        are therefore never published as completed episodes.  Calling this
        method again is idempotent and performs no additional device transfer.
        """
        if self._finalized:
            return TensorMetricFlush([], [], 0, 0)
        self._finalized = True
        if self._updates == 0:
            return TensorMetricFlush([], [], 0, 0)
        return self._flush()

    def _flush(self) -> TensorMetricFlush:
        valid = self.finished_valid.reshape(-1)
        rewards = self.finished_rewards.reshape(-1)[valid]
        lengths = self.finished_lengths.reshape(-1)[valid]
        # Float64 exactly represents the bounded integer lengths and counts,
        # while rewards retain at least their accumulated float32 precision.
        # Concatenation keeps episode values and scalar counters to one D2H.
        host_values = torch.cat(
            (
                rewards,
                lengths,
                self.reward_component_sums
                if self.reward_component_sums is not None
                else torch.empty(0, dtype=torch.float64, device=self.device),
                self.done_count.reshape(1),
                self.timeout_count.reshape(1),
            )
        ).cpu()
        episode_count = int(rewards.numel())
        reward_values = host_values[:episode_count].tolist()
        component_tail = self.reward_component_count + 2
        length_values = (
            host_values[episode_count:-component_tail].tolist()
            if self.reward_component_count
            else host_values[episode_count:-2].tolist()
        )
        component_values = (
            host_values[-(self.reward_component_count + 2) : -2].tolist()
            if self.reward_component_count
            else []
        )
        result = TensorMetricFlush(
            rewards=[float(value) for value in reward_values],
            lengths=[int(value) for value in length_values],
            done_count=int(host_values[-2].item()),
            timeout_count=int(host_values[-1].item()),
            reward_component_sums=(
                {
                    name: float(value)
                    for name, value in zip(
                        self.reward_component_names, component_values, strict=True
                    )
                }
                if self.reward_component_sums is not None
                else None
            ),
        )
        self.finished_rewards.zero_()
        self.finished_lengths.zero_()
        self.finished_valid.zero_()
        self.done_count.zero_()
        self.timeout_count.zero_()
        if self.reward_component_sums is not None:
            self.reward_component_sums.zero_()
        self._cursor = 0
        self._updates = 0
        return result


__all__ = ["TensorCollectorMetrics", "TensorMetricFlush"]
