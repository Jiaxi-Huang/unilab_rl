"""Safe import of released SONIC actor weights."""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch._weights_only_unpickler import _get_allowed_globals

from uni_rl.algos.sonic.network import SonicBackbone

_UPSTREAM_METADATA_GLOBALS = frozenset(
    {
        "accelerate.state.PartialState",
        "accelerate.utils.dataclasses.DistributedType",
        "transformers.trainer_pt_utils.AcceleratorConfig",
        "transformers.trainer_utils.HubStrategy",
        "transformers.trainer_utils.IntervalStrategy",
        "transformers.trainer_utils.SaveStrategy",
        "transformers.trainer_utils.SchedulerType",
        "transformers.training_args.OptimizerNames",
        "trl.trainer.ppo_config.PPOConfig",
        "trl.trainer.utils.OnlineTrainerState",
    }
)
_SELECTED_PREFIXES = ("encoders.g1.", "encoders.smpl.", "decoders.g1_dyn.", "decoders.g1_kin.")


class _CheckpointMetadata:
    def __new__(cls, *args: Any, **kwargs: Any) -> "_CheckpointMetadata":
        del args, kwargs
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _RestrictedCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        qualified_name = f"{module}.{name}"
        allowed = _get_allowed_globals()
        if qualified_name in allowed:
            return allowed[qualified_name]
        if qualified_name == "collections.deque":
            return deque
        if qualified_name in _UPSTREAM_METADATA_GLOBALS:
            return _CheckpointMetadata
        raise pickle.UnpicklingError(f"Unsupported checkpoint global: {qualified_name}")


class _RestrictedCheckpointPickle:
    __name__ = "unilab_sonic_restricted_pickle"
    Unpickler = _RestrictedCheckpointUnpickler


@dataclass(frozen=True)
class SonicCheckpointReport:
    loaded: tuple[str, ...]
    ignored: tuple[str, ...]


def load_sonic_checkpoint_file(checkpoint_path: str | Path) -> dict[str, Any]:
    """Safely load a released or UniLab SONIC checkpoint dictionary."""

    checkpoint = _load_checkpoint(Path(checkpoint_path))
    if not isinstance(checkpoint, dict):
        raise ValueError("SONIC checkpoint must contain a dictionary")
    return checkpoint


def classify_sonic_checkpoint(checkpoint: dict[str, Any]) -> str:
    """Return the unambiguous SONIC checkpoint format."""

    from uni_rl.algos.sonic.flashsac import (
        SONIC_FLASHSAC_CHECKPOINT_KIND,
        SONIC_FLASHSAC_CHECKPOINT_VERSION,
    )

    is_internal = checkpoint.get("checkpoint_kind") == SONIC_FLASHSAC_CHECKPOINT_KIND
    is_release = "policy_state_dict" in checkpoint or "actor_model_state_dict" in checkpoint
    if is_internal and is_release:
        raise ValueError("Ambiguous SONIC checkpoint contains release and UniLab markers")
    if is_internal:
        version = checkpoint.get("format_version")
        if version != SONIC_FLASHSAC_CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported UniLab SONIC checkpoint version: {version!r}")
        required = {
            "actor",
            "sonic_model_config",
            "actor_group_names",
            "actor_group_dims",
            "sonic_log_std_min",
            "sonic_log_std_max",
            "sonic_noise_zeta_mu",
            "sonic_noise_zeta_max",
            "sonic_action_scale",
            "sonic_std_conditioning",
            "sonic_distribution",
            "sonic_policy_head",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"UniLab SONIC checkpoint is missing fields: {missing}")
        if checkpoint["sonic_std_conditioning"] != "flashsac_actor_hidden":
            raise ValueError(
                "Unsupported UniLab SONIC std conditioning: "
                f"{checkpoint['sonic_std_conditioning']!r}"
            )
        if checkpoint["sonic_distribution"] != "native_flashsac_normal_tanh":
            raise ValueError(
                "Unsupported UniLab SONIC action distribution: "
                f"{checkpoint['sonic_distribution']!r}"
            )
        if checkpoint["sonic_policy_head"] != "shared_flashsac_normal_tanh_v3":
            raise ValueError(
                "Unsupported UniLab SONIC policy head: "
                f"{checkpoint['sonic_policy_head']!r}"
            )
        return "unilab"
    if is_release:
        return "sonic_release"
    raise ValueError("Checkpoint is neither SONIC release nor UniLab SONIC FlashSAC format")


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        expected = _UPSTREAM_METADATA_GLOBALS | {"collections.deque"}
        unexpected = unsafe - expected
        if unexpected:
            raise RuntimeError(
                f"Checkpoint contains unsupported globals: {sorted(unexpected)}"
            ) from None
        return torch.load(
            path,
            map_location="cpu",
            pickle_module=_RestrictedCheckpointPickle,
            weights_only=False,
        )


def _extract_actor_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("SONIC checkpoint must contain a dictionary")
    state = checkpoint.get(
        "actor_model_state_dict", checkpoint.get("policy_state_dict", checkpoint)
    )
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("SONIC checkpoint does not contain an actor state_dict")
    return state


def load_sonic_checkpoint(
    model: SonicBackbone,
    checkpoint_path: str | Path,
) -> SonicCheckpointReport:
    """Load the complete selected G1+SMPL backbone and reject partial matches."""

    source = _extract_actor_state(_load_checkpoint(Path(checkpoint_path)))
    normalized = {
        key.removeprefix("actor_module."): value
        for key, value in source.items()
        if isinstance(value, torch.Tensor)
    }
    target = model.state_dict()
    required = {key: value for key, value in target.items() if key.startswith(_SELECTED_PREFIXES)}
    missing = sorted(set(required) - set(normalized))
    mismatched = sorted(
        key
        for key, value in required.items()
        if key in normalized and tuple(normalized[key].shape) != tuple(value.shape)
    )
    if missing or mismatched:
        raise ValueError(
            "SONIC checkpoint is incompatible with the selected backbone "
            f"(missing={missing}, shape_mismatch={mismatched})"
        )
    compatible = {key: normalized[key] for key in required}
    model.load_state_dict(compatible, strict=False)
    loaded_source_keys = {
        f"actor_module.{key}" if f"actor_module.{key}" in source else key for key in compatible
    }
    ignored = tuple(sorted(set(source) - loaded_source_keys))
    return SonicCheckpointReport(loaded=tuple(sorted(compatible)), ignored=ignored)


__all__ = [
    "SonicCheckpointReport",
    "classify_sonic_checkpoint",
    "load_sonic_checkpoint",
    "load_sonic_checkpoint_file",
]
