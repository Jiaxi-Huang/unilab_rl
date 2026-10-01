"""FlashSAC adapter around the checkpoint-compatible SONIC backbone."""

from __future__ import annotations

import warnings
from dataclasses import asdict
from typing import Any

import torch
from torch import nn, optim

from uni_rl.algos.common.compile import get_torch_compile_for_cuda
from uni_rl.algos.flash_sac.layers import (
    FlashSACBlock,
    FlashSACEmbedder,
    NormalTanhPolicy,
    UnitRMSNorm,
)
from uni_rl.algos.flash_sac.learner import FlashSACLearner
from uni_rl.algos.flash_sac.update import build_lr_lambda
from uni_rl.algos.sonic.checkpoint import load_sonic_checkpoint
from uni_rl.algos.sonic.config import SonicAuxLossConfig, SonicModelConfig
from uni_rl.algos.sonic.network import SonicBackbone

SONIC_FLASHSAC_CHECKPOINT_KIND = "unilab.flashsac.sonic"
# v8: widened-embedder fields absent (single-projection entry).
# v9: adds ``sonic_actor_embedder_dim`` (None keeps the v8 trunk layout).
SONIC_FLASHSAC_CHECKPOINT_VERSION = 9
# Checkpoint versions still accepted for playback; v8 lacks the embedder
# field and therefore restores the single-projection trunk.
SONIC_FLASHSAC_COMPAT_CHECKPOINT_VERSIONS = (8, SONIC_FLASHSAC_CHECKPOINT_VERSION)
SONIC_ACTION_SCALE = 2.0
SONIC_ACTOR_GROUP_NAMES = ("obs", "g1_reference", "smpl_reference", "encoder_index")
_TERM_6D_DIM = 6


class _SonicPackedInputMixin:
    @property
    def input_dim(self) -> int:
        cfg = self.model_config
        return cfg.actor_obs_dim + cfg.g1_input_dim + cfg.smpl_input_dim + 2

    def _unpack(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.ndim != 2 or observations.shape[1] != self.input_dim:
            raise ValueError(
                f"SONIC packed actor input must have shape (B, {self.input_dim}), "
                f"got {tuple(observations.shape)}"
            )
        cfg = self.model_config
        offset = 0
        actor_obs = observations[:, offset : offset + cfg.actor_obs_dim]
        offset += cfg.actor_obs_dim
        g1_flat = observations[:, offset : offset + cfg.g1_input_dim]
        offset += cfg.g1_input_dim
        smpl_flat = observations[:, offset : offset + cfg.smpl_input_dim]
        offset += cfg.smpl_input_dim
        g1_command_dim = cfg.g1_frame_dim - _TERM_6D_DIM
        g1_command_width = cfg.num_future_frames * g1_command_dim
        g1_reference = torch.cat(
            (
                g1_flat[:, :g1_command_width].reshape(-1, cfg.num_future_frames, g1_command_dim),
                g1_flat[:, g1_command_width:].reshape(-1, cfg.num_future_frames, _TERM_6D_DIM),
            ),
            dim=-1,
        )
        smpl_human_dim = cfg.smpl_frame_dim - _TERM_6D_DIM
        smpl_human_width = cfg.num_future_frames * smpl_human_dim
        smpl_reference = torch.cat(
            (
                smpl_flat[:, :smpl_human_width].reshape(-1, cfg.num_future_frames, smpl_human_dim),
                smpl_flat[:, smpl_human_width:].reshape(-1, cfg.num_future_frames, _TERM_6D_DIM),
            ),
            dim=-1,
        )
        return actor_obs, g1_reference, smpl_reference, observations[:, offset : offset + 2]


class SonicFlashSACActor(_SonicPackedInputMixin, nn.Module):
    """Squashed Gaussian policy whose mean is produced by the SONIC decoder."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor
    zeta_cdf: torch.Tensor
    _noise: torch.Tensor
    _repeat_count: torch.Tensor
    _repeat_target: torch.Tensor

    def __init__(
        self,
        model_config: SonicModelConfig,
        auxiliary_config: SonicAuxLossConfig,
        *,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        noise_zeta_mu: float = 2.0,
        noise_zeta_max: int = 16,
        actor_hidden_dim: int | None = None,
        actor_num_blocks: int = 2,
        actor_embedder_dim: int | None = None,
        compute_action_decoder: bool | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if log_std_min >= log_std_max:
            raise ValueError("SONIC FlashSAC log_std_min must be smaller than log_std_max")
        self.model_config = model_config
        self.auxiliary_config = auxiliary_config
        self.backbone = SonicBackbone(model_config, auxiliary_config)
        legacy_decoder_head = actor_hidden_dim is None
        if compute_action_decoder is None:
            compute_action_decoder = legacy_decoder_head
        self.compute_action_decoder = bool(compute_action_decoder)
        if actor_hidden_dim is None:
            actor_hidden_dim = (
                model_config.actor_hidden_dim or model_config.g1_control_decoder_hidden_dims[-1]
            )
        if actor_hidden_dim <= 0 or actor_num_blocks <= 0:
            raise ValueError("SONIC FlashSAC actor dimensions must be positive")
        # Training uses the generic FlashSAC actor trunk on the SONIC token and
        # proprioceptive observation.  The release g1_dyn control decoder is
        # intentionally not on this path; it is retained by the backbone only
        # for official-checkpoint playback compatibility.
        self._legacy_decoder_head = legacy_decoder_head
        self.actor_hidden_dim = int(actor_hidden_dim)
        self.actor_num_blocks = int(actor_num_blocks)
        if legacy_decoder_head:
            self.policy_embedder = nn.Identity()
            self.policy_encoder = nn.ModuleList()
            self.policy_post_norm = nn.Identity()
        else:
            policy_input_dim = model_config.actor_obs_dim + model_config.token_total_dim
            self.policy_embedder = FlashSACEmbedder(
                policy_input_dim, actor_hidden_dim, widen_dim=actor_embedder_dim
            )
            self.policy_encoder = nn.ModuleList(
                [FlashSACBlock(actor_hidden_dim) for _ in range(actor_num_blocks)]
            )
            self.policy_post_norm = UnitRMSNorm(actor_hidden_dim)
        self.action_head = NormalTanhPolicy(
            hidden_dim=actor_hidden_dim,
            action_dim=model_config.action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        action_low = action_low.to(dtype=torch.float32)
        action_high = action_high.to(dtype=torch.float32)
        if tuple(action_low.shape) != (model_config.action_dim,) or not torch.all(
            action_high > action_low
        ):
            raise ValueError("SONIC FlashSAC action bounds do not match the model action dim")
        # FlashSAC policies emit normalized actions. Physical joint scaling is
        # owned by the SONIC environment action term, as in the PPO path.
        self.register_buffer("action_scale", torch.ones(model_config.action_dim))
        self.register_buffer("action_bias", torch.zeros(model_config.action_dim))
        self.noise_zeta_mu = float(noise_zeta_mu)
        self.noise_zeta_max = int(noise_zeta_max)
        ns = torch.arange(1, self.noise_zeta_max + 1, dtype=torch.float32)
        pmf = ns.pow(-self.noise_zeta_mu)
        self.register_buffer("zeta_cdf", torch.cumsum(pmf / pmf.sum(), dim=0))
        self.register_buffer("_noise", torch.zeros(0), persistent=False)
        self.register_buffer("_repeat_count", torch.zeros(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_repeat_target", torch.zeros(0, dtype=torch.int32), persistent=False)
        self.to(device)

    def _policy_parameters(
        self, observations: torch.Tensor, *, compute_auxiliary: bool
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        actor_obs, g1_reference, smpl_reference, encoder_index = self._unpack(observations)
        output = self.backbone(
            actor_obs,
            g1_reference,
            smpl_reference,
            encoder_index,
            compute_auxiliary=compute_auxiliary,
            compute_action_decoder=self.compute_action_decoder,
        )
        if self._legacy_decoder_head:
            features = output.action_features
        else:
            policy_input = torch.cat(
                (actor_obs, output.selected_tokens.flatten(start_dim=1)), dim=-1
            )
            features = self.policy_embedder(policy_input, training=compute_auxiliary)
            for block in self.policy_encoder:
                features = block(features, training=compute_auxiliary)
            features = self.policy_post_norm(features)
        pre_tanh_mean, std = self.action_head.get_mean_and_std(features)
        auxiliary = dict(output.auxiliary_losses)
        if compute_auxiliary:
            auxiliary["decoder_action_overflow"] = (pre_tanh_mean.abs() > 1.0).float().mean()
            auxiliary["decoder_action_abs_max"] = pre_tanh_mean.abs().amax()
        return pre_tanh_mean, std, auxiliary

    def get_mean_and_std(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std, _ = self._policy_parameters(observations, compute_auxiliary=False)
        return mean, std

    def forward(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, std, auxiliary = self._policy_parameters(observations, compute_auxiliary=training)
        action, info = NormalTanhPolicy.sample_from_mean_std(
            mean,
            std,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        )
        if auxiliary:
            info["auxiliary_loss"] = auxiliary["total"]
            for name, value in auxiliary.items():
                info[f"sonic_{name}"] = value
        return action, info

    def normalize_parameters(self) -> None:
        """Normalize shared FlashSAC policy parameters after optimizer steps."""
        for module in self.policy_embedder.modules():
            normalize = getattr(module, "normalize_parameters", None)
            if callable(normalize):
                normalize()
        for block in self.policy_encoder:
            for module in block.modules():
                normalize = getattr(module, "normalize_parameters", None)
                if callable(normalize):
                    normalize()
        self.action_head.mean_w.normalize_parameters()
        self.action_head.std_w.normalize_parameters()

    def _ensure_exploration_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        action_dim = self.model_config.action_dim
        if self._noise.shape == (batch_size, action_dim) and self._noise.dtype == dtype:
            return
        self._noise = torch.zeros(batch_size, action_dim, device=device, dtype=dtype)
        self._repeat_count = torch.zeros(batch_size, device=device, dtype=torch.int32)
        self._repeat_target = torch.zeros(batch_size, device=device, dtype=torch.int32)

    @torch.no_grad()
    def explore(
        self,
        observations: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        mean, std = self.get_mean_and_std(observations, training=False)
        if deterministic:
            return torch.tanh(mean)
        batch_size = int(mean.shape[0])
        self._ensure_exploration_state(batch_size, mean.device, mean.dtype)
        done_mask = (
            torch.zeros(batch_size, device=mean.device, dtype=torch.bool)
            if dones is None
            else dones.to(device=mean.device).reshape(-1) > 0.5
        )
        reinit = done_mask | (self._repeat_count <= 0) | (self._repeat_count >= self._repeat_target)
        if torch.any(reinit):
            draws = torch.rand(batch_size, device=mean.device)
            targets = torch.searchsorted(self.zeta_cdf, draws).to(torch.int32) + 1
            self._noise = torch.where(reinit[:, None], torch.randn_like(mean), self._noise)
            self._repeat_target = torch.where(reinit, targets, self._repeat_target)
            self._repeat_count = torch.where(
                reinit, torch.zeros_like(self._repeat_count), self._repeat_count
            )
        self._repeat_count += 1
        return torch.tanh(mean + std * self._noise)

    @torch.no_grad()
    def explore_native(self, observations: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for deterministic native FlashSAC playback."""
        return self.explore(observations, deterministic=True)

    def as_export_module(self) -> nn.Module:
        actor = self

        class _Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = actor

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                return self.base.explore(obs, deterministic=True)

        return _Wrapper()


class SonicReleasePPOActor(_SonicPackedInputMixin, nn.Module):
    """Official SONIC PPO playback actor.

    Release checkpoints contain the policy in the ``g1_dyn`` decoder.  This
    adapter deliberately has no FlashSAC policy head: inference returns the
    decoder's raw PPO mean, while the environment owns scaling and clipping.
    """

    def __init__(
        self,
        model_config: SonicModelConfig,
        auxiliary_config: SonicAuxLossConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.auxiliary_config = auxiliary_config
        self.backbone = SonicBackbone(model_config, auxiliary_config).to(device)

    @torch.no_grad()
    def explore_native(self, observations: torch.Tensor) -> torch.Tensor:
        actor_obs, g1_reference, smpl_reference, encoder_index = self._unpack(observations)
        output = self.backbone(
            actor_obs,
            g1_reference,
            smpl_reference,
            encoder_index,
            compute_auxiliary=False,
            compute_action_decoder=True,
        )
        return output.action_mean


class SonicFlashSACLearner(FlashSACLearner):
    """FlashSAC learner with a SONIC actor and unchanged distributional critics."""

    supports_reference_bc = True
    # The SONIC update_actor override stages its auxiliary-loss metrics on
    # device (one stacked snapshot per cycle) and drains them through
    # ``read_deferred_actor_metrics`` like the shared FlashSAC learner.
    supports_deferred_update_metrics = True
    # Actor graph inputs gain the transition dones so reference-BC can mask
    # terminal rows; the packed-staging layout already reserves a dones offset.
    _ACTOR_GRAPH_INPUT_KEYS = ("obs", "next_obs", "actions", "critic", "dones")

    def _maybe_normalize_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        """Normalize continuous packed terms while preserving encoder masks."""
        if obs.shape[-1] != self.obs_dim or self.obs_dim < 2:
            return super()._maybe_normalize_obs(obs, update=update)
        continuous = obs[..., :-2]
        masks = obs[..., -2:]
        normalized = super()._maybe_normalize_obs(
            torch.cat((continuous, torch.zeros_like(masks)), dim=-1), update=update
        )
        return torch.cat((normalized[..., :-2], masks), dim=-1)

    def normalize_observations(self, obs: torch.Tensor, *, update: bool = False) -> torch.Tensor:
        return self._maybe_normalize_obs(obs, update=update)

    def __init__(
        self,
        *,
        model_config: SonicModelConfig,
        auxiliary_config: SonicAuxLossConfig,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        actor_group_names: tuple[str, ...],
        actor_group_dims: tuple[int, ...],
        log_std_min: float,
        log_std_max: float,
        **kwargs: Any,
    ) -> None:
        pretrained_checkpoint = kwargs.pop("pretrained_checkpoint", None)
        freeze_sonic_backbone = bool(kwargs.pop("freeze_sonic_backbone", False))
        bc_joint_default = kwargs.pop("bc_joint_default", None)
        bc_action_scale = kwargs.pop("bc_action_scale", None)
        self.model_config = model_config
        raw_actor_embedder_dim = kwargs.get("actor_embedder_dim", None)
        actor = SonicFlashSACActor(
            model_config,
            auxiliary_config,
            action_low=action_low,
            action_high=action_high,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            noise_zeta_mu=float(kwargs.get("actor_noise_zeta_mu", 2.0)),
            noise_zeta_max=int(kwargs.get("actor_noise_zeta_max", 16)),
            actor_hidden_dim=int(kwargs.get("actor_hidden_dim", model_config.actor_hidden_dim)),
            actor_num_blocks=int(kwargs.get("actor_num_blocks", 2)),
            actor_embedder_dim=(
                int(raw_actor_embedder_dim) if raw_actor_embedder_dim else None
            ),
            device=kwargs.get("device", "cpu"),
        )
        super().__init__(actor_module=actor, **kwargs)
        if pretrained_checkpoint:
            report = load_sonic_checkpoint(self.actor.backbone, pretrained_checkpoint)
            self.pretrained_backbone_report = report
        else:
            self.pretrained_backbone_report = None
        if freeze_sonic_backbone:
            self.freeze_sonic_backbone()
        self.actor_group_names = tuple(actor_group_names)
        self.actor_group_dims = tuple(int(dim) for dim in actor_group_dims)
        if self.actor.input_dim != self.obs_dim:
            raise ValueError(
                f"SONIC model expects packed actor dim {self.actor.input_dim}, got {self.obs_dim}"
            )
        actor_lr = float(kwargs.get("actor_lr", 3.0e-4))
        shared_peak = float(kwargs.get("learning_rate_peak", 3.0e-4))
        if actor_lr <= 0.0:
            actor_lr = shared_peak
        if shared_peak > 0.0:
            actor_lr = min(actor_lr, shared_peak)
        if actor_lr <= 0.0:
            raise ValueError("SONIC actor learning rate must be positive")
        lr_peak = actor_lr
        optimizer_kwargs: dict[str, Any] = {"fused": self.device.type == "cuda"}
        if self.device.type == "cuda":
            optimizer_kwargs["capturable"] = True
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_peak, **optimizer_kwargs)
        schedule = build_lr_lambda(
            init_lr=float(kwargs.get("learning_rate_init", lr_peak)),
            peak_lr=lr_peak,
            end_lr=float(kwargs.get("learning_rate_end", lr_peak)),
            warmup_steps=int(kwargs.get("learning_rate_warmup_steps", 0)),
            decay_steps=int(kwargs.get("learning_rate_decay_steps", 500000)),
        )
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, schedule)
        self._init_reference_bc(bc_joint_default, bc_action_scale)
        self._pending_sonic_actor_metric_names: tuple[str, ...] | None = None
        self._pending_sonic_actor_metric_values: torch.Tensor | None = None
        self._pending_sonic_actor_scalar_extras: dict[str, float] | None = None

    def _init_reference_bc(
        self,
        bc_joint_default: torch.Tensor | None,
        bc_action_scale: torch.Tensor | None,
    ) -> None:
        """Resolve the env-owned action contract used to build reference BC targets."""
        self.bc_joint_default: torch.Tensor | None = None
        self.bc_action_scale: torch.Tensor | None = None
        if self.actor_bc_target != "reference":
            if bc_joint_default is not None or bc_action_scale is not None:
                raise ValueError(
                    "bc_joint_default/bc_action_scale are only consumed by "
                    "actor_bc_target='reference'"
                )
            return
        if not isinstance(self.obs_normalizer, nn.Identity):
            raise ValueError(
                "reference BC reads raw reference joints from the packed observations; "
                "obs_normalization must be disabled"
            )
        if bc_joint_default is None or bc_action_scale is None:
            raise ValueError(
                "actor_bc_target='reference' requires bc_joint_default and bc_action_scale"
            )
        default = torch.as_tensor(bc_joint_default, dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(bc_action_scale, dtype=torch.float32, device=self.device)
        if tuple(default.shape) != (self.action_dim,) or tuple(scale.shape) != (self.action_dim,):
            raise ValueError(
                f"reference BC contract tensors must have shape ({self.action_dim},), got "
                f"{tuple(default.shape)} and {tuple(scale.shape)}"
            )
        if not bool((scale > 0.0).all()):
            raise ValueError("bc_action_scale entries must be strictly positive")
        self.bc_joint_default = default.detach().clone()
        self.bc_action_scale = scale.detach().clone()

    def _reference_expert_actions(self, next_obs: torch.Tensor) -> torch.Tensor:
        """Convert packed next-step reference joints into normalized policy actions.

        The SONIC reference advances after the transition's reward is computed,
        so the g1 command block of ``next_obs`` starts at the joint positions
        the executed action was meant to reach (policy joint order).  The env
        action contract is ``target = default + scale * action``.
        """
        offset = self.model_config.actor_obs_dim
        ref_joints = next_obs[:, offset : offset + self.action_dim]
        expert = (ref_joints - self.bc_joint_default) / self.bc_action_scale
        return expert.clamp_(-1.0, 1.0)

    def _reference_bc_batch_terms(
        self, next_obs: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (expert actions, keep mask) for reference-mode actor BC."""
        expert_actions = self._reference_expert_actions(next_obs)
        keep = (dones.reshape(-1) < 0.5).to(dtype=expert_actions.dtype)
        return expert_actions, keep

    def freeze_sonic_backbone(self) -> None:
        """Freeze official encoders/tokenizer/kinematic decoder for finetuning."""
        for module in (
            self.actor.backbone.encoders,
            self.actor.backbone.quantizer,
            self.actor.backbone.decoders["g1_dyn"],
            self.actor.backbone.decoders["g1_kin"],
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _sonic_actor_objective_tensors(
        self,
        obs: torch.Tensor,
        expert_actions: torch.Tensor,
        critic_obs: torch.Tensor,
        bc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Full SONIC actor objective: policy + critic evaluation + auxiliary losses.

        Mirrors the eager update path exactly so the compiled and eager
        formulations stay numerically identical.  Optimizer steps, gradient
        scaling, schedulers and parameter normalization stay outside this
        region.  The critic is expected to be frozen by the caller (see
        ``_critic_parameters_frozen``), matching the shared FlashSAC actor
        objective contract.
        """
        with self._autocast():
            actions, actor_info_all = self.actor(obs, training=True)
            log_probs = actor_info_all["log_prob"]
            q_values, _ = self.critic(critic_obs, actions, training=False)
            policy_loss, entropy = self._actor_loss_tensors(
                log_probs, q_values, actions, expert_actions, self.temperature(), bc_mask
            )
            auxiliary_loss = actor_info_all["auxiliary_loss"]
            actor_loss = policy_loss + auxiliary_loss
        actions_detached = actions.detach()
        q_values_detached = q_values.detach()
        metric_tensors: dict[str, torch.Tensor] = {
            "actor_policy_loss": policy_loss.detach(),
            "actor_auxiliary_loss": auxiliary_loss.detach(),
            "actor_auxiliary_to_rl_ratio": (
                auxiliary_loss.detach().abs() / policy_loss.detach().abs().clamp_min(1.0e-8)
            ),
            "actor_entropy": entropy,
            "action_mean": actions_detached.mean(),
            "action_std": actions_detached.std(unbiased=False),
            "q_mean": q_values_detached.mean(),
            "q_std": q_values_detached.std(unbiased=False),
        }
        for name, value in actor_info_all.items():
            if name.startswith("sonic_") and value.numel() == 1:
                metric_tensors[name] = value.detach()
        if bc_mask is not None:
            row_mse = ((actions_detached - expert_actions) ** 2).mean(dim=1)
            metric_tensors["actor_bc_loss"] = (row_mse * bc_mask).sum() / bc_mask.sum().clamp_min(
                1.0
            )
        return actor_loss, entropy, metric_tensors

    def update_actor(
        self, batch: dict[str, torch.Tensor], *, read_metrics: bool = True
    ) -> dict[str, float]:
        """Run the standard FlashSAC actor update plus SONIC auxiliary losses."""

        obs = batch["obs"].to(self.device)
        expert_actions = batch["actions"].to(self.device)
        critic_obs = batch["critic"].to(self.device)
        bc_mask = None
        bc_alpha = 0.0
        if self.actor_bc_target == "reference":
            bc_alpha = self._effective_actor_bc_alpha()
            if bc_alpha > 0.0:
                # Raw (un-normalized) next observations: their g1 command block
                # holds the reference joints the executed action had to reach.
                expert_actions, bc_mask = self._reference_bc_batch_terms(
                    batch["next_obs"].to(self.device), batch["dones"].to(self.device)
                )
        # Keep SONIC's custom actor update aligned with the shared FlashSAC
        # contract: actor observations are normalized by the learner-owned
        # running statistics before both policy evaluations.
        obs = self._maybe_normalize_obs(obs, update=False)

        with self._critic_parameters_frozen():
            actor_loss, entropy, metric_tensors = self._sonic_actor_objective_tensors(
                obs, expert_actions, critic_obs, bc_mask
            )

        self.actor_optimizer.zero_grad(set_to_none=True)
        if not self._host_finite_checks or bool(torch.isfinite(actor_loss)):
            if self.scaler is not None:
                self.scaler.scale(actor_loss).backward()
                self._sync_gradients(self.actor.parameters())
                self.scaler.unscale_(self.actor_optimizer)
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                actor_loss.backward()
                self._sync_gradients(self.actor.parameters())
                with self._optimizer_finite_gate(self.actor_optimizer, actor_loss):
                    self.actor_optimizer.step()
        self.actor_scheduler.step()
        self.actor.normalize_parameters()

        temp_value = self.temperature()
        temp_loss = temp_value * (entropy - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        if not self._host_finite_checks or bool(torch.isfinite(temp_loss)):
            temp_loss.backward()
            self._sync_gradients(self.temperature.parameters())
            with self._optimizer_finite_gate(self.temperature_optimizer, temp_loss):
                self.temperature_optimizer.step()
        self.temperature_scheduler.step()

        metric_tensors["actor_loss"] = actor_loss.detach()
        metric_tensors["temperature"] = temp_value.detach()
        metric_tensors["temperature_loss"] = temp_loss.detach()
        if not read_metrics:
            # Keep a private device-side snapshot.  The eager stack copies the
            # values out of any CUDA-graph output pool, and the cycle-end drain
            # performs the only D2H read, after all compiled replays finish.
            self._pending_sonic_actor_metric_names = tuple(metric_tensors)
            self._pending_sonic_actor_metric_values = torch.stack(
                [tensor.detach().reshape(()) for tensor in metric_tensors.values()]
            )
            self._pending_sonic_actor_scalar_extras = (
                {"actor_bc_alpha": float(bc_alpha)} if bc_mask is not None else {}
            )
            return {}
        metrics = self._read_sonic_actor_metric_tensors(metric_tensors)
        metrics["actor_lr"] = float(self.actor_optimizer.param_groups[0]["lr"])
        if bc_mask is not None:
            metrics["actor_bc_alpha"] = float(bc_alpha)
        return metrics

    @staticmethod
    def _read_sonic_actor_metric_tensors(
        metric_tensors: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        names = tuple(metric_tensors)
        values = (
            torch.stack([metric_tensors[name].detach().reshape(()) for name in names])
            .cpu()
            .tolist()
        )
        return {name: float(value) for name, value in zip(names, values, strict=True)}

    def read_deferred_actor_metrics(self) -> dict[str, float]:
        names = self._pending_sonic_actor_metric_names
        values = self._pending_sonic_actor_metric_values
        extras = self._pending_sonic_actor_scalar_extras
        self._pending_sonic_actor_metric_names = None
        self._pending_sonic_actor_metric_values = None
        self._pending_sonic_actor_scalar_extras = None
        if values is None or names is None:
            if self._cuda_graph_actor_outputs is None:
                return {}
            return self._actor_graph_output_metrics(read_items=True)
        metrics = {
            name: float(value) for name, value in zip(names, values.cpu().tolist(), strict=True)
        }
        metrics["actor_lr"] = float(self.actor_optimizer.param_groups[0]["lr"])
        if extras:
            metrics.update(extras)
        return metrics

    def _sonic_actor_objective_compile_supported(self) -> bool:
        """Whether the SONIC actor objective can enter a compiled graph.

        ``_actor_loss_tensors`` resolves the reference-BC weight as a Python
        float anchored to the actor LR schedule; an annealed weight changes
        every update and would force a recompilation per step.  Constant BC
        weights (including the disabled default) compile fine.
        """
        return not (
            self.actor_bc_alpha > 0.0
            and self.actor_bc_alpha_end is not None
            and self.actor_bc_alpha_end != self.actor_bc_alpha
        )

    def _compile_training_methods(self) -> None:
        compile_fn = get_torch_compile_for_cuda(self.device, warn=True)
        if compile_fn is None:
            return

        compile_kwargs = {"options": {"triton.cudagraphs": bool(self._compile_loss_cudagraphs)}}
        if self.compile_full_objectives:
            # The inherited critic objective serves SONIC's update_critic
            # unchanged; the SONIC actor objective replaces the base actor
            # variant (this update path evaluates the actor once on ``obs``
            # and folds in the SONIC auxiliary losses).  The inference path
            # keeps its own compilation in every mode: the collector calls it
            # once per tick, and an eager heavy backbone would starve the CPU
            # threads the MuJoCo collector needs.
            self._critic_objective_tensors = compile_fn(  # type: ignore[method-assign]
                self._critic_objective_tensors, **compile_kwargs
            )
            self.actor.get_mean_and_std = compile_fn(  # type: ignore[method-assign]
                self.actor.get_mean_and_std, **compile_kwargs
            )
            if self._sonic_actor_objective_compile_supported():
                self._sonic_actor_objective_tensors = compile_fn(  # type: ignore[method-assign]
                    self._sonic_actor_objective_tensors, **compile_kwargs
                )
                return
            warnings.warn(
                "SONIC full-objective actor compilation is disabled: an annealed "
                "reference-BC weight changes every update and would force constant "
                "recompilation; the actor keeps the eager loss-only compile path.",
                stacklevel=2,
            )
            # Loss-only actor fallback: the eager actor path calls
            # ``_actor_loss_tensors`` directly and keeps its wrapper.  The
            # compiled critic objective inlines the plain ``_critic_loss_tensors``
            # at trace time, so it stays uncompiled to avoid a nested compile.
            if not self.use_cuda_graph_actor:
                self._actor_loss_tensors = compile_fn(  # type: ignore[method-assign]
                    self._actor_loss_tensors, **compile_kwargs
                )
            return
        super()._compile_training_methods()

    @staticmethod
    def _actor_graph_input_keys() -> tuple[str, ...]:
        return SonicFlashSACLearner._ACTOR_GRAPH_INPUT_KEYS

    def _prepare_actor_graph_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        prepared = {
            "obs": self._maybe_normalize_obs(batch["obs"].to(self.device), update=False),
            "next_obs": self._maybe_normalize_obs(batch["next_obs"].to(self.device), update=False),
            "actions": batch["actions"].to(self.device),
            "critic": batch["critic"].to(self.device),
            "dones": batch["dones"].to(self.device),
        }
        if "sac_graph_packed_source" in batch:
            prepared["sac_graph_packed_source"] = batch["sac_graph_packed_source"].to(self.device)
        return prepared

    def _update_actor_capture_candidate(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CUDA-graph actor update including SONIC auxiliary objectives.

        Reference-mode BC weights are captured as constants: annealing the BC
        term requires eager actor updates (``use_cuda_graph_actor=false``).
        """
        obs = self._maybe_normalize_obs(inputs["obs"], update=False)
        next_obs = self._maybe_normalize_obs(inputs["next_obs"], update=False)
        expert_actions = inputs["actions"]
        critic_obs = inputs["critic"]
        bc_mask = None
        if self.actor_bc_target == "reference" and self._effective_actor_bc_alpha() > 0.0:
            # Graph mode requires an identity obs normalizer, so the prepared
            # next_obs still carries the raw reference joints.
            expert_actions, bc_mask = self._reference_bc_batch_terms(
                inputs["next_obs"], inputs["dones"]
            )
        obs_all = torch.cat([obs, next_obs], dim=0)

        with self._autocast():
            actions_all, actor_info_all = self.actor(obs_all, training=True)
            actions = actions_all.chunk(2, dim=0)[0]
            log_probs = actor_info_all["log_prob"].chunk(2, dim=0)[0]
            self._set_requires_grad(self.critic, False)
            q_values, _ = self.critic(critic_obs, actions, training=False)
            self._set_requires_grad(self.critic, True)
            temp_value = self.temperature()
            actor_loss, entropy = self._actor_loss_tensors(
                log_probs, q_values, actions, expert_actions, temp_value, bc_mask
            )
            auxiliary_loss = actor_info_all.get("auxiliary_loss")
            if auxiliary_loss is not None:
                actor_loss = actor_loss + auxiliary_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._sync_gradients(self.actor.parameters())
        self.actor_optimizer.step()
        temp_loss = temp_value * (entropy - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temp_loss.backward()
        self._sync_gradients(self.temperature.parameters())
        self.temperature_optimizer.step()
        return actor_loss, entropy, temp_value, temp_loss

    def get_state_dict(self) -> dict[str, Any]:
        state = super().get_state_dict()
        actor = self.actor
        if not isinstance(actor, SonicFlashSACActor):
            raise TypeError("SONIC learner actor does not satisfy the checkpoint contract")
        state.update(
            {
                "checkpoint_kind": SONIC_FLASHSAC_CHECKPOINT_KIND,
                "format_version": SONIC_FLASHSAC_CHECKPOINT_VERSION,
                "sonic_model_config": asdict(actor.model_config),
                "sonic_auxiliary_config": asdict(actor.auxiliary_config),
                "actor_group_names": self.actor_group_names,
                "actor_group_dims": self.actor_group_dims,
                "sonic_log_std_min": actor.log_std_min,
                "sonic_log_std_max": actor.log_std_max,
                "sonic_std_conditioning": "flashsac_actor_hidden",
                "sonic_distribution": "native_flashsac_normal_tanh",
                "sonic_policy_head": "shared_flashsac_normal_tanh_v3",
                "sonic_noise_zeta_mu": actor.noise_zeta_mu,
                "sonic_noise_zeta_max": actor.noise_zeta_max,
                "sonic_actor_hidden_dim": actor.action_head.mean_w.w.weight.shape[1],
                "sonic_actor_num_blocks": actor.actor_num_blocks,
                # None for the legacy single-projection embedder and for the
                # Identity decoder-head path; an int records the widened first
                # stage so playback can rebuild the exact trunk shape.
                "sonic_actor_embedder_dim": getattr(
                    actor.policy_embedder, "widen_width", None
                ),
                "sonic_action_scale": SONIC_ACTION_SCALE,
            }
        )
        return state


__all__ = [
    "SONIC_ACTOR_GROUP_NAMES",
    "SONIC_FLASHSAC_CHECKPOINT_KIND",
    "SONIC_FLASHSAC_CHECKPOINT_VERSION",
    "SONIC_ACTION_SCALE",
    "SonicFlashSACActor",
    "SonicReleasePPOActor",
    "SonicFlashSACLearner",
]
