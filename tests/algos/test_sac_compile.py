from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import pytest
import torch

import uni_rl.algos.common.learner_boilerplate as learner_boilerplate
import uni_rl.algos.sac.learner as sac_module
from uni_rl.algos.common.learner_boilerplate import mps_bf16_autocast_supported
from uni_rl.algos.sac.learner import (
    DistributionalQNetwork,
    SACActor,
    SACLearner,
)
from uni_rl.logging.metric_schema import normalize_metric_map


def _small_sac_learner(*, use_autotune: bool = True) -> SACLearner:
    return SACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=use_autotune,
        max_grad_norm=0.0,
    )


def _small_offpolicy_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.linspace(-0.4, 0.7, steps=batch_size * 4).view(batch_size, 4),
        "critic": torch.linspace(-0.2, 0.9, steps=batch_size * 5).view(batch_size, 5),
        "actions": torch.linspace(-0.5, 0.5, steps=batch_size * 2).view(batch_size, 2),
        "rewards": torch.linspace(-0.3, 0.6, steps=batch_size),
        "next_obs": torch.linspace(0.1, 1.2, steps=batch_size * 4).view(batch_size, 4),
        "next_critic": torch.linspace(-0.7, 0.4, steps=batch_size * 5).view(batch_size, 5),
        "dones": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "truncated": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }


def test_sac_metric_source_keys_are_canonical() -> None:
    learner = _small_sac_learner()
    batch = _small_offpolicy_batch()

    critic_metrics = learner.update_critic(batch)
    actor_metrics = learner.update_actor(batch)

    assert set(critic_metrics) == {
        "Loss/critic",
        "Train/critic_gradient_norm",
        "Train/target_q_max",
        "Train/target_q_min",
        "Loss/temperature",
        "Policy/temperature",
    }
    assert set(actor_metrics) == {
        "Loss/actor",
        "Train/actor_gradient_norm",
        "Loss/entropy",
    }
    normalize_metric_map({**critic_metrics, **actor_metrics})


def test_sac_compile_targets_training_hot_paths(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = SACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
    )
    learner.device = "cuda"
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "SACLearner._critic_loss_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
        (
            "SACLearner._actor_loss_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
    ]


def test_sac_mps_compile_targets_training_hot_paths(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = _small_sac_learner()
    learner.device = torch.device("mps")
    learner._device_type = "mps"
    monkeypatch.setattr(
        sac_module,
        "get_torch_compile_for_device",
        lambda *_args, **_kwargs: fake_compile,
    )

    learner._compile_training_methods()

    assert calls == [
        ("SACLearner._critic_loss_tensors", {"dynamic": False}),
        ("SACLearner._actor_loss_tensors", {"dynamic": False}),
    ]


def test_sac_mps_amp_and_compile_gates_are_explicit_and_mutually_exclusive() -> None:
    assert SACLearner._resolve_use_amp(False, "auto", "mps") is False
    assert SACLearner._resolve_use_amp(False, "bf16", "mps") is False
    assert SACLearner._resolve_use_amp(True, "auto", "mps") is False
    assert SACLearner._resolve_use_amp(True, "bf16", "mps") is True
    assert SACLearner._resolve_use_amp(True, "bf16", "cuda") is True

    with pytest.raises(ValueError, match="amp_dtype='bf16'"):
        SACLearner._resolve_use_amp(True, "fp16", "mps")

    with pytest.raises(ValueError, match="mutually exclusive"):
        SACLearner._validate_performance_paths("mps", use_compile=True, use_amp=True)
    SACLearner._validate_performance_paths("mps", use_compile=True, use_amp=False)
    SACLearner._validate_performance_paths("mps", use_compile=False, use_amp=True)
    SACLearner._validate_performance_paths("cuda", use_compile=True, use_amp=True)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS runtime required")
def test_sac_mps_compile_initialization_reports_torch_compile(monkeypatch) -> None:
    monkeypatch.setattr(
        sac_module,
        "get_torch_compile_for_device",
        lambda *_args, **_kwargs: lambda fn, **_kwargs: fn,
    )
    learner = SACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="mps",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        use_compile=True,
    )

    assert learner.use_compile is True
    assert learner.use_amp is False
    assert learner.compile_backend == "torch_compile"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS runtime required")
def test_sac_mps_bf16_fails_closed_without_runtime_support(monkeypatch) -> None:
    monkeypatch.setattr(sac_module, "mps_bf16_autocast_supported", lambda: False)

    with pytest.raises(RuntimeError, match="cannot execute BF16 autocast"):
        SACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=5,
            device="mps",
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            num_atoms=3,
            num_q_networks=2,
            use_layer_norm=False,
            use_autotune=False,
            use_amp=True,
            amp_dtype="bf16",
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS runtime required")
def test_sac_mps_bf16_capability_requires_autocast_linear_output(
    monkeypatch,
) -> None:
    learner_boilerplate._MPS_BF16_AUTOCAST_SUPPORTED = None

    @contextmanager
    def no_op_autocast(*_args, **_kwargs):
        yield

    monkeypatch.setattr(torch, "autocast", no_op_autocast)

    assert mps_bf16_autocast_supported() is False


@pytest.mark.parametrize(
    ("compiled_time", "expected_compile"),
    [(0.5, True), (1.0, False), (2.0, False)],
)
def test_sac_mps_compile_selection_requires_runtime_win(
    monkeypatch, compiled_time: float, expected_compile: bool
) -> None:
    learner = _small_sac_learner()
    learner._device_type = "mps"
    learner.use_compile = True
    learner.compile_backend = "torch_compile"
    learner._eager_critic_loss_tensors = SACLearner._critic_loss_tensors.__get__(learner)
    learner._eager_actor_loss_tensors = SACLearner._actor_loss_tensors.__get__(learner)

    def compiled_critic(*_args, **_kwargs):
        raise AssertionError("benchmark helper should intercept compiled loss calls")

    learner._compiled_critic_loss_tensors = compiled_critic
    learner._compiled_actor_loss_tensors = learner._eager_actor_loss_tensors
    learner.__dict__["_critic_loss_tensors"] = compiled_critic
    learner.__dict__["_actor_loss_tensors"] = learner._compiled_actor_loss_tensors

    def benchmark(self):
        return compiled_time if self._critic_loss_tensors is compiled_critic else 1.0

    monkeypatch.setattr(SACLearner, "_benchmark_critic_loss_tensors", benchmark)
    monkeypatch.setattr(
        SACLearner,
        "_benchmark_actor_loss_tensors",
        lambda self: 0.5,
    )
    batch = _small_offpolicy_batch()
    learner._select_mps_training_methods(batch, batch_size=4)

    assert learner.use_compile is expected_compile
    assert learner.compile_backend == ("torch_compile" if expected_compile else "eager")
    if expected_compile:
        assert learner._critic_loss_tensors is compiled_critic
    else:
        assert learner._critic_loss_tensors is learner._eager_critic_loss_tensors


def test_sac_mps_compile_selection_failure_falls_back_to_eager(monkeypatch, capsys) -> None:
    learner = _small_sac_learner()
    learner._device_type = "mps"
    learner.use_compile = True
    learner.compile_backend = "torch_compile"
    learner._eager_critic_loss_tensors = SACLearner._critic_loss_tensors.__get__(learner)
    learner._eager_actor_loss_tensors = SACLearner._actor_loss_tensors.__get__(learner)

    def compiled_critic(*_args, **_kwargs):
        raise RuntimeError("metal backend unavailable")

    learner._compiled_critic_loss_tensors = compiled_critic
    learner._compiled_actor_loss_tensors = lambda *_args, **_kwargs: None
    learner.__dict__["_critic_loss_tensors"] = compiled_critic
    learner.__dict__["_actor_loss_tensors"] = learner._compiled_actor_loss_tensors

    def benchmark(self):
        if self._critic_loss_tensors is compiled_critic:
            raise RuntimeError("metal backend unavailable")
        return 1.0

    monkeypatch.setattr(SACLearner, "_benchmark_critic_loss_tensors", benchmark)
    learner._select_mps_training_methods(_small_offpolicy_batch(), batch_size=4)

    assert learner.use_compile is False
    assert learner.compile_backend == "eager"
    assert learner._critic_loss_tensors is learner._eager_critic_loss_tensors
    assert "MPS torch.compile warmup failed" in capsys.readouterr().err


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS runtime required")
def test_sac_mps_compile_selection_restores_mps_rng(monkeypatch) -> None:
    learner = _small_sac_learner()
    learner.device = torch.device("mps")
    learner._device_type = "mps"
    learner.use_compile = True
    learner.compile_backend = "torch_compile"
    learner._eager_critic_loss_tensors = SACLearner._critic_loss_tensors.__get__(learner)
    learner._eager_actor_loss_tensors = SACLearner._actor_loss_tensors.__get__(learner)
    learner._compiled_critic_loss_tensors = learner._eager_critic_loss_tensors
    learner._compiled_actor_loss_tensors = learner._eager_actor_loss_tensors
    learner.__dict__["_critic_loss_tensors"] = learner._eager_critic_loss_tensors
    learner.__dict__["_actor_loss_tensors"] = learner._eager_actor_loss_tensors

    def benchmark(self):
        torch.rand(1, device="mps")
        return 0.5

    monkeypatch.setattr(SACLearner, "_benchmark_critic_loss_tensors", benchmark)
    monkeypatch.setattr(SACLearner, "_benchmark_actor_loss_tensors", benchmark)
    before = torch.mps.get_rng_state()

    learner._select_mps_training_methods(_small_offpolicy_batch(), batch_size=4)

    torch.testing.assert_close(torch.mps.get_rng_state(), before)


def test_sac_whole_cycle_uses_max_autotune_without_nested_graphs(monkeypatch) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = _small_sac_learner()
    learner.device = torch.device("cuda")
    learner._compile_full_update_cycle = True
    monkeypatch.setattr(
        sac_module,
        "get_torch_compile_for_device",
        lambda *_args, **_kwargs: fake_compile,
    )

    learner._compile_training_methods()

    assert calls == [
        (
            "SACLearner._critic_loss_tensors",
            {"dynamic": False, "mode": "max-autotune-no-cudagraphs"},
        ),
        (
            "SACLearner._actor_loss_tensors",
            {"dynamic": False, "mode": "max-autotune-no-cudagraphs"},
        ),
    ]


def test_sac_gradient_sync_leaves_whole_cycle_mode_for_dp() -> None:
    learner = _small_sac_learner()
    learner._compile_full_update_cycle = True
    learner._update_cycle_graph_cache_key = ("old",)

    def sync(_parameters):
        pass

    learner.set_gradient_sync(sync)
    assert learner._gradient_sync is sync
    assert learner._compile_full_update_cycle is False
    assert learner._update_cycle_graph_cache_key is None
    assert learner._gradient_graph_hooks is None
    assert learner.use_update_cycle is False

    learner.set_gradient_graph_hooks(lambda: None, lambda: None, lambda: None)
    learner._update_cycle_graph_cache_key = ("same",)
    learner.set_gradient_sync(sync)
    assert learner._update_cycle_graph_cache_key == ("same",)
    learner.set_gradient_sync(None)
    assert learner._update_cycle_graph_cache_key is None
    assert learner._gradient_graph_hooks is None


@pytest.mark.parametrize("optimizer_name", ["q_optimizer", "actor_optimizer", "alpha_optimizer"])
def test_dp_nonfinite_loss_with_finite_gradients_arms_same_gate(optimizer_name, monkeypatch):
    """A loss constant can overflow on one rank without corrupting its gradients."""
    for rank in range(2):
        learner = _small_sac_learner()
        learner._device_type = "cuda"  # Exercise gate math with CPU tensors.
        learner._host_finite_checks = False
        optimizer = getattr(learner, optimizer_name)
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.ones_like(parameter)
        sentinel_pointer = learner._gradient_loss_flag.grad.data_ptr()

        def average(parameters):
            parameters = list(parameters)
            assert parameters[-1] is learner._gradient_loss_flag
            flag = parameters[-1].grad
            assert flag.item() == float(rank == 0)
            flag.add_(float(rank == 1)).div_(2)

        learner.set_gradient_sync(average)
        monkeypatch.setattr(torch, "_amp_foreach_non_finite_check_and_unscale_", lambda *args: None)
        loss = torch.tensor(float("inf") if rank == 0 else 1.0)
        learner._sync_loss_gradients([parameter], loss)
        learner._arm_optimizer_finite_gate(optimizer, loss)
        assert learner._optimizer_found_inf.item() == 1.0
        assert learner._gradient_loss_flag.grad.data_ptr() == sentinel_pointer
        assert torch.isfinite(parameter.grad).all()


def test_gradient_graph_capture_hook_ends_when_warmup_fails(monkeypatch):
    learner = _small_sac_learner()
    events = []
    learner.set_gradient_graph_hooks(
        lambda: events.append("begin"), lambda: events.append("end"), lambda: None
    )

    def fail(*args, **kwargs):
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(learner, "_warm_update_cycle_graph", fail)
    with pytest.raises(RuntimeError, match="warmup failed"):
        learner._ensure_update_cycle_graph(
            {"obs": torch.zeros(2, 4)},
            updates_per_step=1,
            policy_frequency=1,
            target_frequency=1,
            policy_before_critic=False,
        )
    assert events == ["begin", "end"]


def test_sac_update_cycle_rejects_compatibility_fallback() -> None:
    learner = _small_sac_learner()

    with pytest.raises(RuntimeError, match="requires the NVIDIA CUDA whole-cycle path"):
        learner.update_cycle(
            _small_offpolicy_batch(),
            updates_per_step=1,
            policy_frequency=1,
            target_frequency=1,
            policy_before_critic=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA runtime required")
def test_sac_nvidia_cuda_fails_closed_without_inductor(monkeypatch) -> None:
    monkeypatch.setattr(sac_module, "get_torch_compile_for_device", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="requires CUDA Inductor/Triton"):
        SACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=5,
            device="cuda:0",
            use_compile=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime selection required")
def test_sac_hip_cuda_keeps_compatible_compile_opt_out(monkeypatch) -> None:
    with monkeypatch.context() as hip_runtime:
        hip_runtime.setattr(torch.version, "hip", "simulated-rocm", raising=False)
        learner = SACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=5,
            device="cuda:0",
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            num_atoms=3,
            num_q_networks=2,
            use_layer_norm=False,
            use_compile=False,
        )

    assert learner.use_compile is False
    assert learner.use_update_cycle is False
    learner.set_gradient_sync(lambda _parameters: None)
    assert learner._gradient_sync is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA runtime required")
def test_sac_nvidia_cuda_fails_closed_for_graph_incompatible_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sac_module,
        "get_torch_compile_for_device",
        lambda *_args, **_kwargs: lambda fn: fn,
    )
    common = {
        "obs_dim": 4,
        "action_dim": 2,
        "critic_obs_dim": 5,
        "device": "cuda:0",
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "num_atoms": 3,
        "num_q_networks": 2,
        "use_layer_norm": False,
    }

    with pytest.raises(ValueError, match="fp16 GradScaler"):
        SACLearner(**common, use_amp=True, amp_dtype="fp16")

    with pytest.raises(ValueError, match="obs normalization"):
        SACLearner(**common, obs_normalization=True)

    with pytest.raises(ValueError, match="NVTX ranges"):
        SACLearner(**common, nvtx_profile_ranges=True)


def test_sac_categorical_projection_preserves_rows() -> None:
    num_atoms = 5
    qnet = DistributionalQNetwork(
        obs_dim=4,
        action_dim=2,
        num_atoms=num_atoms,
        v_min=-2.0,
        v_max=2.0,
        hidden_dim=8,
        use_layer_norm=False,
    )
    # A uniform next-state distribution makes the expected projection independent
    # of the initialized hidden layers.
    with torch.no_grad():
        qnet.net[-1].weight.zero_()
        qnet.net[-1].bias.zero_()
    support = torch.linspace(qnet.v_min, qnet.v_max, num_atoms)
    rewards = torch.tensor([-0.7, 0.0, 0.3, 1.1])
    bootstrap = torch.tensor([0.0, 1.0, 1.0, 1.0])
    discount = torch.full_like(rewards, 0.9)

    projected = qnet.projection(
        torch.zeros(4, 4),
        torch.zeros(4, 2),
        rewards,
        bootstrap,
        discount,
        support,
        support.device,
    )

    expected = torch.zeros_like(projected)
    next_probability = 1.0 / num_atoms
    for row, reward in enumerate(rewards):
        target_z = (reward + bootstrap[row] * discount[row] * support).clamp(qnet.v_min, qnet.v_max)
        position = (target_z - qnet.v_min) / ((qnet.v_max - qnet.v_min) / (num_atoms - 1))
        lower = position.floor().long()
        upper = (lower + 1).clamp(max=num_atoms - 1)
        upper_weight = position - lower
        expected[row].scatter_add_(0, lower, next_probability * (1.0 - upper_weight))
        expected[row].scatter_add_(0, upper, next_probability * upper_weight)

    assert torch.allclose(projected, expected, atol=1e-6)
    assert torch.allclose(projected.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_sac_cuda_adamw_optimizers_are_capturable(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only optimizer kwargs require a CUDA-enabled torch build")

    calls: list[dict[str, Any]] = []

    class _FakeAdamW:
        def __init__(self, _params, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _FakeAdamW)
    monkeypatch.setattr(
        sac_module,
        "get_torch_compile_for_device",
        lambda *_args, **_kwargs: lambda fn: fn,
    )
    monkeypatch.setattr(SACLearner, "_materialize_capturable_optimizer_state", lambda _self: None)
    monkeypatch.setattr(SACLearner, "_compile_training_methods", lambda _self: None)

    for device in ("cuda", torch.device("cuda")):
        SACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=5,
            device=device,
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            num_atoms=3,
            num_q_networks=2,
            use_layer_norm=False,
            use_autotune=False,
            use_compile=True,
        )

    assert len(calls) == 6
    assert all(call["fused"] for call in calls)
    assert all(call["capturable"] for call in calls)


def test_sac_cpu_adamw_optimizers_keep_default_capturability(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeAdamW:
        def __init__(self, _params, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _FakeAdamW)

    SACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        use_compile=False,
    )

    assert len(calls) == 3
    assert not any(call["fused"] for call in calls)
    assert all("capturable" not in call for call in calls)


def test_sac_amp_dtype_resolution_and_scaler_rules() -> None:
    assert SACLearner._resolve_amp_dtype("auto", "cuda") is torch.bfloat16
    assert SACLearner._resolve_amp_dtype("auto", "xpu") is torch.bfloat16
    assert SACLearner._resolve_amp_dtype("fp16", "cuda") is torch.float16
    assert SACLearner._resolve_amp_dtype("bf16", "cuda") is torch.bfloat16

    assert SACLearner._should_use_grad_scaler(True, "cuda", torch.float16)
    assert not SACLearner._should_use_grad_scaler(True, "cuda", torch.bfloat16)
    assert not SACLearner._should_use_grad_scaler(True, "xpu", torch.bfloat16)
    assert not SACLearner._should_use_grad_scaler(False, "cuda", torch.float16)

    with pytest.raises(ValueError, match="amp_dtype"):
        SACLearner._resolve_amp_dtype("tf32", "cuda")


def test_sac_alpha_loss_helper_matches_reference_value_and_grad() -> None:
    learner = SACLearner(
        obs_dim=4,
        action_dim=3,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=True,
    )
    next_log_probs = torch.tensor([-1.25, -0.5, 0.25, 1.5], dtype=torch.float32)
    learner.target_entropy = -1.75
    learner.log_alpha.data.fill_(-2.0)

    reference_log_alpha = learner.log_alpha.detach().clone().requires_grad_(True)
    reference_loss = (-reference_log_alpha.exp() * (next_log_probs + learner.target_entropy)).mean()
    reference_loss.backward()

    learner.log_alpha.grad = None
    alpha_loss = learner._alpha_loss_tensor(next_log_probs)
    alpha_loss.backward()

    assert torch.allclose(alpha_loss.detach(), reference_loss.detach())
    assert learner.log_alpha.grad is not None
    assert reference_log_alpha.grad is not None
    assert torch.allclose(learner.log_alpha.grad, reference_log_alpha.grad)
    assert not next_log_probs.requires_grad


def test_sac_actor_tensor_gaussian_sampling_matches_normal_reference() -> None:
    actor = SACActor(
        obs_dim=4,
        action_dim=3,
        hidden_dim=12,
        use_layer_norm=False,
        action_scale=torch.tensor([0.5, 1.5, 2.0]),
        action_bias=torch.tensor([-0.25, 0.0, 0.75]),
    )
    obs = torch.tensor(
        [
            [-1.0, -0.25, 0.5, 1.25],
            [0.25, 0.5, -0.75, 1.0],
        ],
        dtype=torch.float32,
    )

    _, mean, log_std = actor(obs)
    std = log_std.exp()
    eps = torch.tensor(
        [
            [-0.5, 0.25, 1.0],
            [1.5, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    raw_action = mean + std * eps
    dist = torch.distributions.Normal(mean, std)
    tanh_action = torch.tanh(raw_action)
    expected_action = tanh_action * actor.action_scale + actor.action_bias
    expected_log_prob = dist.log_prob(raw_action)
    expected_log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
    expected_log_prob -= torch.log(actor.action_scale + 1e-6)
    expected_log_prob = expected_log_prob.sum(1)

    action, log_prob = actor._sample_action_and_log_prob(mean, log_std, eps=eps)

    torch.testing.assert_close(action, expected_action)
    torch.testing.assert_close(log_prob, expected_log_prob)


def test_sac_actor_tensor_gaussian_sampling_matches_normal_without_tanh() -> None:
    actor = SACActor(obs_dim=2, action_dim=3, hidden_dim=12, use_layer_norm=False, use_tanh=False)
    mean = torch.tensor(
        [[-0.5, 0.25, 1.0], [1.5, -1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    log_std = torch.tensor(
        [[-1.0, -0.25, 0.5], [0.0, -0.75, 0.25]],
        dtype=torch.float32,
        requires_grad=True,
    )
    eps = torch.tensor(
        [[0.25, -1.5, 0.75], [-0.5, 1.0, 1.5]],
        dtype=torch.float32,
    )

    action, log_prob = actor._sample_action_and_log_prob(mean, log_std, eps=eps)

    reference_mean = mean.detach().clone().requires_grad_(True)
    reference_log_std = log_std.detach().clone().requires_grad_(True)
    reference_std = reference_log_std.exp()
    reference_raw_action = reference_mean + reference_std * eps
    reference_dist = torch.distributions.Normal(reference_mean, reference_std)
    expected_log_prob = reference_dist.log_prob(reference_raw_action).sum(1)

    torch.testing.assert_close(action, reference_raw_action)
    torch.testing.assert_close(log_prob, expected_log_prob)

    loss = (action + log_prob.unsqueeze(1)).sum()
    reference_loss = (reference_raw_action + expected_log_prob.unsqueeze(1)).sum()
    loss.backward()
    reference_loss.backward()

    assert mean.grad is not None
    assert log_std.grad is not None
    assert reference_mean.grad is not None
    assert reference_log_std.grad is not None
    torch.testing.assert_close(mean.grad, reference_mean.grad)
    torch.testing.assert_close(log_std.grad, reference_log_std.grad)


def test_sac_actor_update_does_not_accumulate_critic_gradients() -> None:
    learner = _small_sac_learner()

    learner.update_actor(_small_offpolicy_batch())

    assert all(parameter.requires_grad for parameter in learner.qnet.parameters())
    assert all(parameter.grad is None for parameter in learner.qnet.parameters())


def test_sac_public_updates_can_defer_metric_reads() -> None:
    learner = _small_sac_learner()
    batch = _small_offpolicy_batch()

    assert learner.update_critic(batch, read_metrics=False) == {}
    assert learner.update_actor(batch, read_metrics=False) == {}


@pytest.mark.parametrize(
    ("policy_frequency", "target_frequency", "policy_before_critic"),
    [(2, 3, False), (2, 3, True)],
)
def test_sac_update_cycle_matches_runner_composition(
    policy_frequency: int,
    target_frequency: int,
    policy_before_critic: bool,
) -> None:
    def seeded_learner() -> SACLearner:
        torch.manual_seed(1234)
        return _small_sac_learner()

    actual = seeded_learner()
    expected = seeded_learner()
    updates_per_step = 4
    batch = _small_offpolicy_batch()
    large_batch = {
        key: value.repeat(updates_per_step, *[1] * (value.ndim - 1)) for key, value in batch.items()
    }

    rng_state = torch.random.get_rng_state()
    for update_idx in range(updates_per_step):
        start = update_idx * batch["obs"].shape[0]
        end = start + batch["obs"].shape[0]
        update_batch = {key: value[start:end] for key, value in large_batch.items()}
        do_actor_update = update_idx % policy_frequency == 0
        if policy_before_critic and do_actor_update:
            expected.update_actor(update_batch, read_metrics=False)
        expected.update_critic(
            update_batch,
            read_metrics=update_idx == updates_per_step - 1,
        )
        if not policy_before_critic and do_actor_update:
            expected.update_actor(update_batch, read_metrics=False)
        if update_idx % target_frequency == 0:
            expected.soft_update_target()

    torch.random.set_rng_state(rng_state)
    actual._run_update_cycle_core(
        large_batch,
        updates_per_step=updates_per_step,
        policy_frequency=policy_frequency,
        target_frequency=target_frequency,
        policy_before_critic=policy_before_critic,
    )

    for module_name in ("actor", "qnet", "qnet_target"):
        actual_state = getattr(actual, module_name).state_dict()
        expected_state = getattr(expected, module_name).state_dict()
        for key in actual_state:
            torch.testing.assert_close(actual_state[key], expected_state[key])
    torch.testing.assert_close(actual.log_alpha, expected.log_alpha)
    for optimizer_name in ("actor_optimizer", "q_optimizer", "alpha_optimizer"):
        actual_state = getattr(actual, optimizer_name).state_dict()
        expected_state = getattr(expected, optimizer_name).state_dict()
        assert actual_state.keys() == expected_state.keys()
        for parameter_key in actual_state["state"]:
            actual_values = actual_state["state"][parameter_key]
            expected_values = expected_state["state"][parameter_key]
            assert actual_values.keys() == expected_values.keys()
            for state_key in actual_values:
                torch.testing.assert_close(
                    actual_values[state_key],
                    expected_values[state_key],
                )


def test_sac_update_cycle_defers_metrics_until_one_read() -> None:
    learner = _small_sac_learner()
    batch = _small_offpolicy_batch()
    large_batch = {key: value.repeat(2, *[1] * (value.ndim - 1)) for key, value in batch.items()}

    learner._run_update_cycle_core(
        large_batch,
        updates_per_step=2,
        policy_frequency=1,
        target_frequency=1,
        policy_before_critic=False,
    )
    metrics = learner.read_deferred_cycle_metrics()

    assert set(metrics) == {
        "Loss/critic",
        "Train/critic_gradient_norm",
        "Train/target_q_max",
        "Train/target_q_min",
        "Loss/temperature",
        "Policy/temperature",
        "Loss/actor",
        "Train/actor_gradient_norm",
        "Loss/entropy",
    }
    assert all(math.isfinite(value) for value in metrics.values())
    assert learner.read_deferred_cycle_metrics() == {}


def test_sac_compatibility_updates_defer_metrics_until_one_read() -> None:
    learner = _small_sac_learner()
    batch = _small_offpolicy_batch()

    learner.update_critic(batch, read_metrics=False)
    learner.update_actor(batch, read_metrics=False)
    metrics = learner.read_deferred_update_metrics()

    assert set(metrics) == {
        "Loss/critic",
        "Train/critic_gradient_norm",
        "Train/target_q_max",
        "Train/target_q_min",
        "Loss/temperature",
        "Policy/temperature",
        "Loss/actor",
        "Train/actor_gradient_norm",
        "Loss/entropy",
    }
    assert all(math.isfinite(value) for value in metrics.values())
    assert learner._pending_cycle_critic_metric_values is None
    assert learner._pending_actor_metric_values is None
    assert learner.read_deferred_update_metrics() == {}


def test_sac_deferred_finite_guard_fails_at_consolidated_read() -> None:
    learner = _small_sac_learner()
    learner._metrics_finite_checks = True
    learner._host_finite_checks = True
    batch = _small_offpolicy_batch()

    learner.begin_deferred_update_metrics()
    learner.update_critic(batch, read_metrics=False)
    learner.update_actor(batch, read_metrics=False)
    assert learner._pending_update_finite_values is not None
    learner._pending_update_finite_values[0] = False

    with pytest.raises(FloatingPointError, match="non-finite critic, alpha, or actor loss"):
        learner.read_deferred_update_metrics()

    assert learner._pending_cycle_critic_metric_values is None
    assert learner._pending_actor_metric_values is None
    assert learner._pending_update_finite_values is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only whole-cycle graph")
def test_sac_update_cycle_raw_graph_replays_with_stable_inputs_and_metrics(
    monkeypatch,
) -> None:
    torch.manual_seed(123)
    learner = SACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cuda:0",
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        num_atoms=5,
        num_q_networks=2,
        use_layer_norm=False,
        use_compile=True,
    )
    torch.manual_seed(123)
    with monkeypatch.context() as hip_runtime:
        hip_runtime.setattr(torch.version, "hip", "simulated-rocm", raising=False)
        eager_learner = SACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=5,
            device="cuda:0",
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            num_atoms=5,
            num_q_networks=2,
            use_layer_norm=False,
            use_compile=False,
        )
    batch = {
        "obs": torch.randn(16, 4, device="cuda:0"),
        "critic": torch.randn(16, 5, device="cuda:0"),
        "actions": torch.rand(16, 2, device="cuda:0"),
        "rewards": torch.randn(16, device="cuda:0"),
        "next_obs": torch.randn(16, 4, device="cuda:0"),
        "next_critic": torch.randn(16, 5, device="cuda:0"),
        "dones": torch.zeros(16, device="cuda:0"),
        "truncated": torch.zeros(16, device="cuda:0"),
    }

    torch.manual_seed(321)
    learner.update_cycle(
        batch,
        updates_per_step=4,
        policy_frequency=2,
        target_frequency=1,
        policy_before_critic=False,
    )
    first_metrics = learner.read_deferred_cycle_metrics()

    torch.manual_seed(321)
    eager_learner.update_actor(batch, read_metrics=False)
    eager_learner.update_critic(batch, read_metrics=False)
    eager_learner.soft_update_target()
    for module_name in ("actor", "qnet", "qnet_target"):
        torch.testing.assert_close(
            getattr(learner, module_name).state_dict(),
            getattr(eager_learner, module_name).state_dict(),
            rtol=1e-3,
            atol=1e-3,
        )
    torch.testing.assert_close(
        learner.log_alpha,
        eager_learner.log_alpha,
        rtol=1e-3,
        atol=1e-3,
    )

    assert learner._update_cycle_graph is not None
    assert learner._update_cycle_static_batch is not None
    static_addresses = {
        key: value.data_ptr() for key, value in learner._update_cycle_static_batch.items()
    }

    changed_batch = {key: value * 0.5 for key, value in batch.items()}
    learner.update_cycle(
        changed_batch,
        updates_per_step=4,
        policy_frequency=2,
        target_frequency=1,
        policy_before_critic=False,
    )
    second_metrics = learner.read_deferred_cycle_metrics()
    assert learner._update_cycle_static_batch is not None
    assert static_addresses == {
        key: value.data_ptr() for key, value in learner._update_cycle_static_batch.items()
    }
    assert set(first_metrics) == set(second_metrics)
    assert first_metrics
    assert all(math.isfinite(value) for value in second_metrics.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only whole-cycle graph")
def test_sac_update_cycle_rekeys_on_shape_and_invalidates_checkpoint() -> None:
    torch.manual_seed(123)
    learner = SACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cuda:0",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_compile=True,
    )
    hook_events = []
    learner.set_gradient_graph_hooks(
        lambda: hook_events.append("begin"),
        lambda: hook_events.append("end"),
        lambda: hook_events.append("replay"),
    )
    batch = {
        "obs": torch.randn(16, 4, device="cuda:0"),
        "critic": torch.randn(16, 5, device="cuda:0"),
        "actions": torch.rand(16, 2, device="cuda:0"),
        "rewards": torch.randn(16, device="cuda:0"),
        "next_obs": torch.randn(16, 4, device="cuda:0"),
        "next_critic": torch.randn(16, 5, device="cuda:0"),
        "dones": torch.zeros(16, device="cuda:0"),
        "truncated": torch.zeros(16, device="cuda:0"),
    }
    learner.update_cycle(
        batch,
        updates_per_step=1,
        policy_frequency=2,
        target_frequency=1,
        policy_before_critic=False,
    )
    assert learner._update_cycle_graph is not None
    first_graph = learner._update_cycle_graph
    learner.read_deferred_cycle_metrics()

    larger_batch = {key: torch.cat((value, value)) for key, value in batch.items()}
    learner.update_cycle(
        larger_batch,
        updates_per_step=1,
        policy_frequency=2,
        target_frequency=1,
        policy_before_critic=False,
    )
    assert learner._update_cycle_graph is not None
    assert learner._update_cycle_graph is not first_graph
    assert all(
        tensor.shape == larger_batch[key].shape
        for key, tensor in learner._update_cycle_static_batch.items()
    )
    assert all(math.isfinite(value) for value in learner.read_deferred_cycle_metrics().values())

    assert hook_events == ["begin", "end", "replay", "begin", "end", "replay"]

    learner.load_state_dict(learner.get_state_dict())
    assert learner._update_cycle_graph is None
    assert learner._update_cycle_graph_cache_key is None
    assert learner._update_cycle_static_batch is None


@pytest.mark.parametrize(
    ("optimizer_name", "loss_value", "grad_value"),
    [
        ("q_optimizer", float("nan"), 1.0),
        ("q_optimizer", 1.0, float("nan")),
        ("actor_optimizer", float("nan"), 1.0),
        ("actor_optimizer", 1.0, float("nan")),
        ("alpha_optimizer", float("nan"), 1.0),
        ("alpha_optimizer", 1.0, float("nan")),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only fused optimizer gate")
def test_sac_armed_finite_gate_skips_nonfinite_optimizer_step(
    optimizer_name: str,
    loss_value: float,
    grad_value: float,
) -> None:
    learner = SACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cuda:0",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        use_compile=True,
    )
    optimizer = getattr(learner, optimizer_name)
    parameter = next(iter(optimizer.param_groups[0]["params"]))
    parameter.grad = torch.full_like(parameter, grad_value)
    learner._gradient_sync = (lambda _parameters: None) if not math.isfinite(grad_value) else None
    before = parameter.detach().clone()

    learner._arm_optimizer_finite_gate(
        optimizer,
        torch.full((), loss_value, device="cuda:0"),
    )
    optimizer.step()
    torch.cuda.synchronize()

    torch.testing.assert_close(parameter, before)
