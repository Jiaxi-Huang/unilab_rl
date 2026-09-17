"""Reference-mode actor BC: expert actions from packed next observations."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from uni_rl.algos.flash_sac.learner import FlashSACLearner
from uni_rl.algos.sonic.config import SonicAuxLossConfig, SonicModelConfig
from uni_rl.algos.sonic.flashsac import SonicFlashSACLearner

ACTION_DIM = 29
NUM_FUTURE_FRAMES = 2
ACTOR_OBS_DIM = 16
OBS_DIM = ACTOR_OBS_DIM + NUM_FUTURE_FRAMES * (64 + 84) + 2


def _model_config() -> SonicModelConfig:
    return SonicModelConfig(
        num_future_frames=NUM_FUTURE_FRAMES,
        actor_obs_dim=ACTOR_OBS_DIM,
        token_dim=8,
        g1_encoder_hidden_dims=(32,),
        smpl_encoder_hidden_dims=(32,),
        g1_motion_decoder_hidden_dims=(32,),
        g1_control_decoder_hidden_dims=(32,),
    )


def _make_learner(**kwargs: Any) -> SonicFlashSACLearner:
    defaults: dict[str, Any] = {
        "model_config": _model_config(),
        "auxiliary_config": SonicAuxLossConfig(),
        "action_low": torch.full((ACTION_DIM,), -1.0),
        "action_high": torch.full((ACTION_DIM,), 1.0),
        "actor_group_names": ("obs", "g1_reference", "smpl_reference", "encoder_index"),
        "actor_group_dims": (ACTOR_OBS_DIM, NUM_FUTURE_FRAMES * 64, NUM_FUTURE_FRAMES * 84, 2),
        "log_std_min": -10.0,
        "log_std_max": 2.0,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "critic_obs_dim": 24,
        "actor_hidden_dim": 16,
        "critic_hidden_dim": 16,
        "actor_num_blocks": 1,
        "critic_num_blocks": 1,
        "num_atoms": 5,
        "device": "cpu",
        "use_compile": False,
    }
    defaults.update(kwargs)
    return SonicFlashSACLearner(**defaults)


def _reference_learner(**kwargs: Any) -> SonicFlashSACLearner:
    torch.manual_seed(7)
    kwargs.setdefault("actor_bc_alpha", 2.5)
    kwargs.setdefault("actor_bc_target", "reference")
    kwargs.setdefault("bc_joint_default", torch.zeros(ACTION_DIM, dtype=torch.float32))
    kwargs.setdefault("bc_action_scale", torch.full((ACTION_DIM,), 0.5))
    return _make_learner(**kwargs)


def _batch(batch_size: int = 12) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    obs = torch.randn(batch_size, OBS_DIM)
    next_obs = torch.randn(batch_size, OBS_DIM)
    # The packed tail carries one-hot G1/SMPL encoder masks.
    masks = torch.zeros(batch_size, 2)
    masks[torch.arange(batch_size), torch.arange(batch_size) % 2] = 1.0
    obs[:, -2:] = masks
    next_obs[:, -2:] = masks
    return {
        "obs": obs,
        "next_obs": next_obs,
        "actions": torch.tanh(torch.randn(batch_size, ACTION_DIM)),
        "critic": torch.randn(batch_size, 24),
        "dones": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
        "rewards": torch.randn(batch_size),
        "next_critic": torch.randn(batch_size, 24),
    }


def _plant_reference_joints(next_obs: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    """Write reference joints into the first g1 command frame of packed rows."""
    result = next_obs.clone()
    result[:, ACTOR_OBS_DIM : ACTOR_OBS_DIM + ACTION_DIM] = joints
    return result


def test_reference_expert_actions_invert_the_action_contract() -> None:
    learner = _reference_learner()
    joints = torch.randn(5, ACTION_DIM) * 0.2
    next_obs = _plant_reference_joints(torch.randn(5, OBS_DIM), joints)
    expert = learner._reference_expert_actions(next_obs)
    expected = torch.clamp(joints / 0.5, -1.0, 1.0)
    assert torch.allclose(expert, expected, atol=1e-6)


def test_reference_bc_masks_terminal_rows() -> None:
    learner = _reference_learner()
    next_obs = torch.randn(4, OBS_DIM)
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0])
    _, keep = learner._reference_bc_batch_terms(next_obs, dones)
    assert torch.equal(keep, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_actor_loss_respects_bc_mask_and_alpha() -> None:
    learner = _reference_learner()
    batch_size = 6
    log_probs = torch.randn(batch_size)
    q_values = [torch.randn(batch_size, 1) for _ in range(2)]
    actions = torch.tanh(torch.randn(batch_size, ACTION_DIM))
    expert = torch.tanh(torch.randn(batch_size, ACTION_DIM))
    temp = torch.tensor(0.1)
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    loss_masked, _ = learner._actor_loss_tensors(log_probs, q_values, actions, expert, temp, mask)
    min_q = torch.minimum(q_values[0], q_values[1])
    sac = (temp * log_probs - min_q).mean()
    kept = ((actions[:3] - expert[:3]) ** 2).mean()
    expected = sac + 2.5 * min_q.abs().mean() * kept
    assert torch.allclose(loss_masked, expected, atol=1e-5)

    # An all-done mask disables the BC term entirely.
    loss_blocked, _ = learner._actor_loss_tensors(
        log_probs, q_values, actions, expert, temp, torch.zeros(batch_size)
    )
    assert torch.allclose(loss_blocked, sac, atol=1e-6)


def test_actor_bc_alpha_anneals_with_scheduler_steps() -> None:
    learner = _reference_learner(actor_bc_alpha_end=0.25, learning_rate_decay_steps=100)
    assert learner._effective_actor_bc_alpha() == pytest.approx(2.5)
    for _ in range(50):
        learner.actor_scheduler.step()
    assert learner._effective_actor_bc_alpha() == pytest.approx(1.375)
    for _ in range(100):
        learner.actor_scheduler.step()
    assert learner._effective_actor_bc_alpha() == pytest.approx(0.25)


def test_reference_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="bc_joint_default"):
        _make_learner(actor_bc_target="reference", actor_bc_alpha=1.0)
    with pytest.raises(ValueError, match="obs_normalization"):
        _reference_learner(obs_normalization=True)
    with pytest.raises(ValueError, match="reference"):
        FlashSACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=6,
            actor_bc_alpha=1.0,
            actor_bc_target="reference",
        )


def test_update_actor_reports_reference_bc_metrics() -> None:
    learner = _reference_learner()
    batch = _batch()
    joints = torch.randn(batch["next_obs"].shape[0], ACTION_DIM) * 0.1
    batch["next_obs"] = _plant_reference_joints(batch["next_obs"], joints)
    batch["dones"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    metrics = learner.update_actor(batch)
    assert "actor_bc_loss" in metrics
    assert "actor_bc_alpha" in metrics
    assert metrics["actor_bc_alpha"] == pytest.approx(2.5)
    assert 0.0 <= metrics["actor_bc_loss"] < 4.0


def test_replay_mode_keeps_legacy_expert_source() -> None:
    learner = _make_learner(actor_bc_alpha=1.0, actor_bc_target="replay")
    batch = _batch(batch_size=4)
    metrics = learner.update_actor(batch)
    assert "actor_bc_loss" not in metrics
    assert metrics["actor_loss"] == metrics["actor_loss"]  # finite sanity
