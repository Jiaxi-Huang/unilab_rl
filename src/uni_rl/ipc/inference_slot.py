"""Single-slot observation/action exchange for learner-owned inference."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
import torch

_SPAWN_CTX = mp.get_context("spawn")

_IDLE = 0
_OBS_READY = 1
_ACTION_READY = 2


class SharedInferenceSlot:
    """Fixed shared-memory slot with strict single-request ownership."""

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if min(num_envs, obs_dim, action_dim) <= 0:
            raise ValueError("SharedInferenceSlot dimensions must be positive")
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA inference slots requested on unavailable {target}")
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = target
        self.observations = torch.empty(
            (self.num_envs, self.obs_dim), dtype=torch.float32, device=target
        ).share_memory_()
        self.dones = torch.empty(self.num_envs, dtype=torch.float32, device=target).share_memory_()
        self.actions = torch.empty(
            (self.num_envs, self.action_dim), dtype=torch.float32, device=target
        ).share_memory_()
        self._state = _SPAWN_CTX.Value("i", _IDLE)
        self._request_tick = _SPAWN_CTX.Value("q", -1)
        self._response_tick = _SPAWN_CTX.Value("q", -1)
        self._policy_version = _SPAWN_CTX.Value("q", -1)
        self._lock = _SPAWN_CTX.Lock()

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.observations, self.dones, self.actions)
        )

    def publish_observation(
        self,
        *,
        tick_id: int,
        observations: np.ndarray | torch.Tensor,
        dones: np.ndarray | torch.Tensor,
    ) -> None:
        if isinstance(observations, torch.Tensor):
            if not isinstance(dones, torch.Tensor):
                raise TypeError("observations and dones must both be tensors")
            observation_tensor = observations.to(dtype=torch.float32)
            done_tensor = dones.to(dtype=torch.float32).reshape(-1)
            if observation_tensor.device != self.observations.device:
                raise ValueError("inference observations must live on the slot device")
        else:
            if isinstance(dones, torch.Tensor):
                raise TypeError("observations and dones must both be arrays")
            observation_tensor = torch.from_numpy(
                np.ascontiguousarray(observations, dtype=np.float32)
            ).to(self.observations.device)
            done_tensor = torch.from_numpy(
                np.ascontiguousarray(dones, dtype=np.float32).reshape(-1)
            ).to(self.dones.device)
        expected_obs_shape = (self.num_envs, self.obs_dim)
        if tuple(observation_tensor.shape) != expected_obs_shape:
            raise ValueError(
                f"Inference observation shape must be {expected_obs_shape}, "
                f"got {tuple(observation_tensor.shape)}"
            )
        if tuple(done_tensor.shape) != (self.num_envs,):
            raise ValueError(
                f"Inference dones shape must be {(self.num_envs,)}, got {tuple(done_tensor.shape)}"
            )
        with self._lock:
            if self._state.value != _IDLE:
                raise RuntimeError("Inference slot cannot be reused before action consumption")
            self.observations.copy_(observation_tensor)
            self.dones.copy_(done_tensor)
            if self.device.type == "cuda":
                # Keep conservative CPU-ready ordering, but wait only for the
                # producer stream that issued the copies. This is deliberately
                # not a device-wide barrier or a cross-process CUDA-event
                # protocol.
                torch.cuda.current_stream(self.device).synchronize()
            self._request_tick.value = int(tick_id)
            self._state.value = _OBS_READY

    def copy_observation_to(
        self,
        *,
        tick_id: int,
        observations: torch.Tensor,
        dones: torch.Tensor,
        non_blocking: bool = False,
    ) -> None:
        if tuple(observations.shape) != (self.num_envs, self.obs_dim):
            raise ValueError("Learner observation destination shape mismatch")
        if tuple(dones.shape) != (self.num_envs,):
            raise ValueError("Learner dones destination shape mismatch")
        with self._lock:
            self._require_state(_OBS_READY, tick_id, self._request_tick, "observation")
            observations.copy_(self.observations, non_blocking=non_blocking)
            dones.copy_(self.dones, non_blocking=non_blocking)

    def publish_action(
        self,
        *,
        tick_id: int,
        policy_version: int,
        actions: torch.Tensor,
        non_blocking: bool = False,
    ) -> None:
        if tuple(actions.shape) != (self.num_envs, self.action_dim):
            raise ValueError("Learner action shape mismatch")
        with self._lock:
            self._require_state(_OBS_READY, tick_id, self._request_tick, "observation")
            # A CPU slot returns NumPy actions that the tensor collector
            # restages into a persistent host tensor, while the learner may
            # still own its actor on CUDA. Preserve that implicit
            # synchronization boundary; CUDA IPC slots remain exact-device.
            if actions.device != self.actions.device and not (
                self.device.type == "cpu" and actions.device.type == "cuda"
            ):
                raise ValueError("inference actions must live on the slot device")
            self.actions.copy_(actions, non_blocking=non_blocking)
            if self.device.type == "cpu" and actions.device.type == "cuda":
                # CPU ready flags cannot order an asynchronous D2H copy. Wait
                # for the producer stream before publishing the ready state.
                torch.cuda.current_stream(actions.device).synchronize()
            if self.device.type == "cuda":
                # Retain the same conservative CPU-ready ordering for device-local
                # publication without waiting for unrelated learner streams.
                torch.cuda.current_stream(self.device).synchronize()
            self._response_tick.value = int(tick_id)
            self._policy_version.value = int(policy_version)
            self._state.value = _ACTION_READY

    def consume_action(self, *, tick_id: int) -> tuple[np.ndarray | torch.Tensor, int]:
        with self._lock:
            self._require_state(_ACTION_READY, tick_id, self._response_tick, "action")
            actions = (
                self.actions.clone() if self.device.type == "cuda" else self.actions.numpy().copy()
            )
            policy_version = int(self._policy_version.value)
            self._state.value = _IDLE
            return actions, policy_version

    def _require_state(self, state: int, tick_id: int, tick_value: Any, label: str) -> None:
        if self._state.value != state:
            raise RuntimeError(f"Inference {label} is not ready")
        actual_tick = int(tick_value.value)
        if actual_tick != int(tick_id):
            raise RuntimeError(
                f"Inference {label} tick mismatch: expected {tick_id}, got {actual_tick}"
            )

    def close(self) -> None:
        for name in ("observations", "dones", "actions"):
            if hasattr(self, name):
                delattr(self, name)

    cleanup = close
