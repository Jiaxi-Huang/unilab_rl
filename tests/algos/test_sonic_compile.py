"""SONIC full-objective compilation: targets, fallbacks, and eager equivalence."""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch
import torch.nn.functional as F

from uni_rl.algos.sonic.auxiliary import sonic_auxiliary_losses
from uni_rl.algos.sonic.config import SonicAuxLossConfig, SonicModelConfig
from uni_rl.algos.sonic.flashsac import SonicFlashSACLearner
from uni_rl.algos.sonic.network import SonicBackbone

ACTION_DIM = 29
NUM_FUTURE_FRAMES = 2
ACTOR_OBS_DIM = 16
G1_FRAME_DIM = 64
SMPL_FRAME_DIM = 84
OBS_DIM = ACTOR_OBS_DIM + NUM_FUTURE_FRAMES * (G1_FRAME_DIM + SMPL_FRAME_DIM) + 2
CRITIC_OBS_DIM = 24


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
        "actor_group_dims": (
            ACTOR_OBS_DIM,
            NUM_FUTURE_FRAMES * G1_FRAME_DIM,
            NUM_FUTURE_FRAMES * SMPL_FRAME_DIM,
            2,
        ),
        "log_std_min": -10.0,
        "log_std_max": 2.0,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "critic_obs_dim": CRITIC_OBS_DIM,
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
    # The packed tail carries the G1/SMPL encoder masks.
    masks = torch.zeros(batch_size, 2)
    masks[torch.arange(batch_size), torch.arange(batch_size) % 2] = 1.0
    obs[:, -2:] = masks
    next_obs[:, -2:] = masks
    return {
        "obs": obs,
        "next_obs": next_obs,
        "actions": torch.tanh(torch.randn(batch_size, ACTION_DIM)),
        "critic": torch.randn(batch_size, CRITIC_OBS_DIM),
        "dones": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
        "rewards": torch.randn(batch_size),
        "next_critic": torch.randn(batch_size, CRITIC_OBS_DIM),
    }


def test_sonic_compile_full_objectives_targets_complete_forward_graphs(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = _make_learner()
    learner.device = torch.device("cuda")
    learner.compile_full_objectives = True
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "FlashSACLearner._critic_objective_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
        (
            "SonicFlashSACActor.get_mean_and_std",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
        (
            "SonicFlashSACLearner._sonic_actor_objective_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
    ]


def test_sonic_compile_bc_annealing_falls_back_to_loss_helpers(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = _reference_learner(actor_bc_alpha_end=1.0)
    learner.device = torch.device("cuda")
    learner.compile_full_objectives = True
    monkeypatch.setattr(torch, "compile", fake_compile)

    with pytest.warns(UserWarning, match="full-objective actor compilation is disabled"):
        learner._compile_training_methods()

    assert calls == [
        (
            "FlashSACLearner._critic_objective_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
        (
            "SonicFlashSACActor.get_mean_and_std",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
        (
            "FlashSACLearner._actor_loss_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
    ]


def test_sonic_actor_objective_matches_eager_decomposition() -> None:
    learner = _make_learner()
    batch = _batch()
    obs = learner._maybe_normalize_obs(batch["obs"], update=False)
    expert_actions = batch["actions"]
    critic_obs = batch["critic"]

    torch.manual_seed(123)
    actor_loss, entropy, metric_tensors = learner._sonic_actor_objective_tensors(
        obs, expert_actions, critic_obs
    )

    torch.manual_seed(123)
    with learner._autocast():
        actions, actor_info_all = learner.actor(obs, training=True)
        log_probs = actor_info_all["log_prob"]
        q_values, _ = learner.critic(critic_obs, actions, training=False)
        policy_loss, expected_entropy = learner._actor_loss_tensors(
            log_probs, q_values, actions, expert_actions, learner.temperature(), None
        )
        expected_auxiliary = actor_info_all["auxiliary_loss"]
        expected_actor_loss = policy_loss + expected_auxiliary

    torch.testing.assert_close(actor_loss, expected_actor_loss)
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(metric_tensors["actor_policy_loss"], policy_loss.detach())
    torch.testing.assert_close(metric_tensors["actor_auxiliary_loss"], expected_auxiliary.detach())
    torch.testing.assert_close(metric_tensors["action_mean"], actions.detach().mean())
    torch.testing.assert_close(metric_tensors["q_std"], q_values.detach().std(unbiased=False))
    assert "sonic_total" in metric_tensors
    assert "actor_bc_loss" not in metric_tensors


def test_sonic_update_actor_deferred_metrics_read_once() -> None:
    learner = _make_learner()
    batch = _batch()

    assert learner.update_actor(batch, read_metrics=False) == {}
    staged = learner.read_deferred_actor_metrics()

    immediate = learner.update_actor(batch)

    expected_keys = {
        "Loss/actor",
        "actor_policy_loss",
        "actor_auxiliary_loss",
        "actor_auxiliary_to_rl_ratio",
        "actor_entropy",
        "Policy/temperature",
        "Loss/temperature",
        "actor_lr",
        "action_mean",
        "action_std",
        "q_mean",
        "q_std",
        "sonic_reconstruction",
        "sonic_latent_alignment",
        "sonic_cycle_consistency",
        "sonic_total",
        "sonic_decoder_action_overflow",
        "sonic_decoder_action_abs_max",
    }
    assert set(staged) == expected_keys
    assert set(immediate) == expected_keys
    assert all(isinstance(value, float) for value in staged.values())
    # Draining twice without a new update returns nothing.
    assert learner.read_deferred_actor_metrics() == {}


def test_sonic_reference_bc_deferred_metrics_carry_alpha() -> None:
    learner = _reference_learner()
    next_obs = _batch()["next_obs"]
    joints = torch.randn(next_obs.shape[0], ACTION_DIM) * 0.2
    next_obs[:, ACTOR_OBS_DIM : ACTOR_OBS_DIM + ACTION_DIM] = joints
    batch = _batch()
    batch["next_obs"] = next_obs

    learner.update_actor(batch, read_metrics=False)
    metrics = learner.read_deferred_actor_metrics()

    assert metrics["actor_bc_alpha"] == pytest.approx(2.5)
    assert "actor_bc_loss" in metrics


def _legacy_auxiliary_reference(
    backbone: SonicBackbone,
    actor_obs: torch.Tensor,
    g1_reference: torch.Tensor,
    smpl_reference: torch.Tensor,
    encoder_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pre-mask-static formulation: boolean compression + F.mse_loss."""

    g1_active = encoder_index[:, 0].to(dtype=torch.bool)
    select_smpl = encoder_index[:, 1].to(dtype=torch.bool)
    g1_latent = backbone._encode_masked("g1", g1_reference, g1_active)
    smpl_latent = backbone._encode_masked("smpl", smpl_reference, select_smpl)
    missing_g1 = select_smpl & ~g1_active
    if torch.any(missing_g1):
        g1_latent = g1_latent.clone()
        g1_latent[missing_g1] = backbone._encode("g1", g1_reference[missing_g1])
    g1_tokens = backbone.quantize(g1_latent)
    smpl_tokens = backbone.quantize(smpl_latent)
    selected_tokens = torch.where(select_smpl[:, None, None], smpl_tokens, g1_tokens)
    reconstruction = backbone.decode_motion(selected_tokens)
    smpl_reconstruction = backbone.decode_motion(smpl_tokens[select_smpl])
    reencoded_smpl_g1 = backbone.encode_g1(smpl_reconstruction)
    selected_g1_latent = g1_latent[select_smpl]
    selected_smpl_latent = smpl_latent[select_smpl]
    reconstruction_loss = F.mse_loss(reconstruction, g1_reference)
    if selected_g1_latent.numel():
        latent_alignment = F.mse_loss(selected_smpl_latent, selected_g1_latent)
        cycle_consistency = F.mse_loss(reencoded_smpl_g1, selected_g1_latent)
    else:
        latent_alignment = selected_g1_latent.sum() + selected_smpl_latent.sum()
        cycle_consistency = selected_g1_latent.sum() + reencoded_smpl_g1.sum()
    return {
        "reconstruction": reconstruction_loss,
        "latent_alignment": latent_alignment,
        "cycle_consistency": cycle_consistency,
    }


@pytest.mark.parametrize(
    "mask_layout",
    ["all_g1", "alternating", "all_smpl", "multi_hot"],
)
def test_sonic_auxiliary_masked_matches_boolean_compression(mask_layout: str) -> None:
    torch.manual_seed(5)
    backbone = SonicBackbone(_model_config(), SonicAuxLossConfig())
    batch_size = 10
    actor_obs = torch.randn(batch_size, ACTOR_OBS_DIM)
    g1_reference = torch.randn(batch_size, NUM_FUTURE_FRAMES, G1_FRAME_DIM)
    smpl_reference = torch.randn(batch_size, NUM_FUTURE_FRAMES, SMPL_FRAME_DIM)
    encoder_index = torch.zeros(batch_size, 2)
    if mask_layout == "all_g1":
        encoder_index[:, 0] = 1.0
    elif mask_layout == "alternating":
        encoder_index[torch.arange(batch_size), torch.arange(batch_size) % 2] = 1.0
    elif mask_layout == "all_smpl":
        encoder_index[:, 1] = 1.0
    else:
        encoder_index[:, 0] = 1.0
        encoder_index[:4, 1] = 1.0

    output = backbone(
        actor_obs,
        g1_reference,
        smpl_reference,
        encoder_index,
        compute_auxiliary=True,
        compute_action_decoder=False,
    )
    reference = _legacy_auxiliary_reference(
        backbone, actor_obs, g1_reference, smpl_reference, encoder_index
    )

    for name in ("reconstruction", "latent_alignment", "cycle_consistency"):
        torch.testing.assert_close(
            output.auxiliary_losses[name], reference[name], rtol=1e-5, atol=1e-6
        )


def test_sonic_auxiliary_losses_mask_all_zero_returns_zero() -> None:
    prediction = torch.randn(4, 2, 8)
    target = torch.randn(4, 2, 8)
    zero_mask = torch.zeros(4, dtype=torch.bool)
    losses = sonic_auxiliary_losses(
        g1_reference=torch.randn(4, 2, 64),
        g1_reconstruction=torch.randn(4, 2, 64),
        g1_latent=target,
        smpl_latent=prediction,
        reencoded_smpl_g1_latent=prediction,
        row_mask=zero_mask,
        config=SonicAuxLossConfig(),
    )
    assert float(losses["latent_alignment"]) == 0.0
    assert float(losses["cycle_consistency"]) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sonic_full_objective_compile_cuda_smoke() -> None:
    torch.manual_seed(3)
    learner = _make_learner(
        device="cuda",
        use_compile=True,
        compile_full_objectives=True,
        use_amp=True,
    )
    assert learner.compile_full_objectives is True
    # torch.compile wraps the bound methods into plain callables.
    assert type(learner._sonic_actor_objective_tensors).__name__ != "method"
    assert type(learner._critic_objective_tensors).__name__ != "method"

    batch = {key: value.to("cuda") for key, value in _batch().items()}
    for _ in range(3):
        learner.update_critic(batch, read_metrics=False)
        learner.update_actor(batch, read_metrics=False)
    metrics = learner.read_deferred_actor_metrics()

    assert metrics
    assert all(isinstance(value, float) and value == value for value in metrics.values())
