"""THE tensor-native env contract consumed by uni_rl.

uni_rl never constructs environments itself and never imports an env
registry. Environments are injected by the caller (e.g. UniLab's training
scripts) as an :data:`EnvFactory`.

Contract summary (tensor-native, autoresetting vectorized env):

- ``step(actions)`` accepts a contiguous ``torch.float32`` action tensor on the
  environment's authoritative device. It returns a state whose ``obs`` is a
  mapping of ``(num_envs, dim)`` Torch tensors keyed by observation group,
  along with floating-point ``reward`` and boolean ``terminated`` /
  ``truncated`` tensors. ``final_observation`` is the optional mapping of
  pre-reset tensors for environments that terminated on that step.
- ``obs_groups_spec`` maps observation group name -> dim, e.g. ``{"obs": 48}``
  or ``{"obs": 48, "critic": 101}``. ``"obs"`` is the actor group;
  ``"critic"`` is the optional privileged critic group.
- ``reset(env_indices)`` accepts a one-dimensional Torch integer tensor on the
  environment device (or ``None``) and returns selected-row observations.
  ``init_state()`` performs cold-path initialization and returns the first
  state; ``state`` is ``None`` before it is called.
- Collectors keep observations, actions, rewards, termination flags, and reset
  indices tensor-native. NumPy is only a host serialization/diagnostic carrier,
  never a trainer boundary.
- Optional cold-path metadata remains available through
  :func:`get_algo_capabilities`; algorithm code must not probe private env
  attributes.
- Optional backend tensor-runtime diagnostics (for example whether the
  physics backend replays a CUDA graph) are collected through
  :func:`get_backend_runtime_diagnostics` on cold paths and published into
  the off-policy runtime manifest.

Because collectors run in ``multiprocessing`` spawn subprocesses, an
``EnvFactory`` must be picklable by reference — use a top-level function or
``functools.partial`` of one, never a closure or lambda.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class EnvStateProtocol(Protocol):
    """One vectorized tensor env step/reset result (structural contract)."""

    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, Any]
    final_observation: dict[str, torch.Tensor] | None


@runtime_checkable
class EnvPlayCapabilitiesProtocol(Protocol):
    """Playback capabilities read by collector diagnostics wiring."""

    supports_physics_state_playback: bool


@runtime_checkable
class EnvAlgoCapabilitiesProtocol(Protocol):
    """Optional env metadata for algorithm-side features (all fields optional).

    Every field defaults to ``None`` meaning "not provided"; consumers must
    fall back gracefully instead of requiring any field. Read via
    :func:`get_algo_capabilities` on cold paths only — never inside
    ``step``/``reset`` hot loops.
    """

    @property
    def action_low(self) -> torch.Tensor | None:
        """Per-dimension lower action bounds, shape ``(action_dim,)``."""
        ...

    @property
    def action_high(self) -> torch.Tensor | None:
        """Per-dimension upper action bounds, shape ``(action_dim,)``."""
        ...

    @property
    def joint_names(self) -> tuple[str, ...] | None:
        """Joint names in action order, for symmetry-augmentation maps."""
        ...

    @property
    def reward_component_names(self) -> tuple[str, ...] | None:
        """Reward component names for fixed-width tensor metric aggregation."""
        ...


@dataclass(frozen=True)
class EnvAlgoCapabilities:
    """Default capability carrier with every field unset."""

    action_low: torch.Tensor | None = None
    action_high: torch.Tensor | None = None
    joint_names: tuple[str, ...] | None = None
    reward_component_names: tuple[str, ...] | None = None


@runtime_checkable
class SupportsAlgoCapabilitiesProtocol(Protocol):
    """Optional provider protocol: envs expose ``algo_capabilities``."""

    @property
    def algo_capabilities(self) -> EnvAlgoCapabilitiesProtocol: ...


@runtime_checkable
class SupportsBackendDiagnosticsProtocol(Protocol):
    """Optional provider protocol: envs expose backend runtime diagnostics."""

    def get_backend_runtime_diagnostics(self) -> Mapping[str, Any]:
        """Return a plain-JSON snapshot of backend tensor-runtime diagnostics.

        Cold paths only. The payload carries at least ``available`` (bool);
        when available it also carries ``backend_type`` (str) and a
        ``diagnostics`` mapping of optimization name to
        ``{"requested": bool, "enabled": bool, "disable_reason": str | None}``.
        """
        ...


@runtime_checkable
class SupportsCollectorGraphModeProtocol(Protocol):
    """Optional provider protocol: envs negotiate collector graph execution.

    This is a cold-path owner contract. A collector may request a mode after
    env initialization and before its first control tick; it must never call
    the negotiation method inside ``step``/``reset``. ``disabled`` and
    ``eager`` are independent explicit modes: backend physics graph state, or
    a backend ``UNAVAILABLE`` graph lifecycle, must never be interpreted as
    collector mode selection.
    """

    def configure_collector_graph(
        self,
        mode: str,
        *,
        allow_eager_fallback: bool = False,
    ) -> None:
        """Negotiate ``disabled``, ``eager``, or ``graph`` execution.

        Unsupported graph negotiation fails closed by default. If the owner
        explicitly allows fallback, the returned manifest alone records the
        eager mode, reason, and fallback count.
        """
        ...

    def get_collector_graph_manifest(self) -> Mapping[str, Any]:
        """Return requested/enabled/state/reason/counter evidence."""
        ...


@dataclass(frozen=True)
class CollectorGraphNegotiation:
    """Result of the neutral collector mode negotiation."""

    mode: str = "disabled"
    manifest: Mapping[str, Any] | None = None


COLLECTOR_GRAPH_MODES = frozenset({"disabled", "eager", "graph"})


def negotiate_collector_graph(
    env: Any,
    mode: str,
    *,
    allow_eager_fallback: bool = False,
) -> CollectorGraphNegotiation:
    """Negotiate collector execution through the optional env protocol.

    All modes fail closed when an env does not implement the public protocol;
    there is no implicit graph/eager inference from backend state. The call is
    cold-path only and returns a JSON-safe manifest snapshot.
    """
    if not isinstance(mode, str):
        raise TypeError(f"collector graph mode must be a string, got {type(mode).__name__}")
    normalized = mode.strip().lower()
    if normalized not in COLLECTOR_GRAPH_MODES:
        choices = ", ".join(sorted(COLLECTOR_GRAPH_MODES))
        raise ValueError(f"collector graph mode must be one of {choices}; got {mode!r}")
    provider = env if isinstance(env, SupportsCollectorGraphModeProtocol) else None
    if provider is None:
        if normalized == "disabled":
            return CollectorGraphNegotiation(mode="disabled")
        raise NotImplementedError("env does not implement the public collector graph mode protocol")
    provider.configure_collector_graph(
        normalized,
        allow_eager_fallback=allow_eager_fallback,
    )
    manifest = provider.get_collector_graph_manifest()
    if not isinstance(manifest, Mapping):
        raise TypeError("collector graph manifest must be a mapping")
    negotiated = str(manifest.get("mode", normalized))
    if negotiated not in COLLECTOR_GRAPH_MODES:
        raise ValueError(f"env returned unsupported collector graph mode {negotiated!r}")
    return CollectorGraphNegotiation(mode=negotiated, manifest=manifest)


@runtime_checkable
class EnvProtocol(Protocol):
    """Vectorized tensor env consumed by uni_rl runners and collectors."""

    @property
    def num_envs(self) -> int: ...

    @property
    def device(self) -> torch.device:
        """Authoritative device for actions, reset indices, and state tensors."""
        ...

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """Observation group dims, e.g. ``{"obs": 48, "critic": 101}``."""
        ...

    @property
    def observation_space(self) -> Any:
        """Gym-style flat observation space (only ``.shape`` is read)."""
        ...

    @property
    def action_space(self) -> Any:
        """Gym-style action space (only ``.shape`` is read)."""
        ...

    @property
    def state(self) -> EnvStateProtocol | None:
        """Current state; ``None`` before :meth:`init_state` is called."""
        ...

    @property
    def cfg(self) -> Any:
        """Env config; runners read ``max_episode_seconds`` and ``ctrl_dt``."""
        ...

    @property
    def play_capabilities(self) -> EnvPlayCapabilitiesProtocol: ...

    def init_state(self) -> EnvStateProtocol: ...

    def step(self, actions: torch.Tensor) -> EnvStateProtocol: ...

    def reset(
        self, env_indices: torch.Tensor | None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]: ...

    def set_nan_guard(self, guard: Any) -> None: ...

    def close(self) -> None: ...


# Injected env constructor. Arguments are ``(num_envs, env_cfg_override)``; the
# override mapping is opaque to uni_rl and interpreted by the factory owner.
EnvFactory = Callable[[int, Mapping[str, Any] | None], EnvProtocol]


def get_algo_capabilities(env: Any) -> EnvAlgoCapabilitiesProtocol:
    """Return optional env capabilities, or an all-``None`` default.

    Cold paths only (runner init and dimension probes). Never call from
    ``step``/``reset`` hot loops and never probe private environment
    attributes.
    """
    if isinstance(env, SupportsAlgoCapabilitiesProtocol):
        return env.algo_capabilities
    return EnvAlgoCapabilities()


def get_backend_runtime_diagnostics(env: Any) -> Mapping[str, Any] | None:
    """Return the env's backend runtime diagnostics, or ``None``.

    Cold paths only (collector startup). Returns ``None`` when the env does
    not implement :class:`SupportsBackendDiagnosticsProtocol`; callers record
    an explicit unavailable marker instead of probing private attributes.
    """
    if isinstance(env, SupportsBackendDiagnosticsProtocol):
        return env.get_backend_runtime_diagnostics()
    return None


__all__ = [
    "EnvAlgoCapabilities",
    "EnvAlgoCapabilitiesProtocol",
    "EnvFactory",
    "EnvPlayCapabilitiesProtocol",
    "EnvProtocol",
    "EnvStateProtocol",
    "COLLECTOR_GRAPH_MODES",
    "CollectorGraphNegotiation",
    "SupportsAlgoCapabilitiesProtocol",
    "SupportsBackendDiagnosticsProtocol",
    "SupportsCollectorGraphModeProtocol",
    "get_algo_capabilities",
    "get_backend_runtime_diagnostics",
    "negotiate_collector_graph",
]
