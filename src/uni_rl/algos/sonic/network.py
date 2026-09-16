"""Checkpoint-compatible G1+SMPL SONIC shared-token backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from vector_quantize_pytorch import FSQ

from uni_rl.algos.sonic.auxiliary import sonic_auxiliary_losses
from uni_rl.algos.sonic.config import SonicAuxLossConfig, SonicModelConfig


class SonicMLP(nn.Module):
    """Plain SiLU MLP with the same state-dict layout as upstream BaseModule."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(source, target))
            if index < len(dims) - 2:
                layers.append(nn.SiLU())
        self.module = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


@dataclass
class SonicForwardOutput:
    action_mean: torch.Tensor
    action_features: torch.Tensor
    selected_tokens: torch.Tensor
    g1_tokens: torch.Tensor
    smpl_tokens: torch.Tensor
    g1_latent: torch.Tensor
    smpl_latent: torch.Tensor
    g1_reconstruction: torch.Tensor | None
    auxiliary_losses: dict[str, torch.Tensor]


class SonicBackbone(nn.Module):
    """G1 and SMPL encoders sharing one FSQ and two G1 decoders.

    ``encoder_index`` stores the G1 and SMPL masks in that order. It accepts
    the release's legacy multi-hot rows; as upstream does, SMPL wins when both
    encoders are active. Paired references remain available for the release
    auxiliary objectives even when only one encoder supplies the control token.
    """

    def __init__(
        self,
        config: SonicModelConfig = SonicModelConfig(),
        auxiliary_config: SonicAuxLossConfig = SonicAuxLossConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.auxiliary_config = auxiliary_config
        self.encoders = nn.ModuleDict(
            {
                "g1": SonicMLP(
                    config.g1_input_dim,
                    config.g1_encoder_hidden_dims,
                    config.token_total_dim,
                ),
                "smpl": SonicMLP(
                    config.smpl_input_dim,
                    config.smpl_encoder_hidden_dims,
                    config.token_total_dim,
                ),
            }
        )
        self.quantizer = FSQ(
            levels=[config.fsq_levels] * config.token_dim,
            return_indices=False,
        )
        self.decoders = nn.ModuleDict(
            {
                "g1_dyn": SonicMLP(
                    config.token_total_dim + config.actor_obs_dim,
                    config.g1_control_decoder_hidden_dims,
                    config.action_dim,
                ),
                "g1_kin": SonicMLP(
                    config.token_total_dim,
                    config.g1_motion_decoder_hidden_dims,
                    config.g1_motion_output_dim,
                ),
            }
        )

    def _encode(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        batch_size = reference.shape[0]
        flat = reference.flatten(start_dim=1)
        latent = self.encoders[name](flat)
        return latent.reshape(batch_size, self.config.num_tokens, self.config.token_dim)

    def _encode_masked(
        self, name: str, reference: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode every row and zero inactive ones, returning a batch-aligned token tensor.

        Boolean-mask row compression has a data-dependent output shape, which
        Inductor lowers to ``index``/``index_put_`` calls that are illegal
        inside a CUDA graph capture.  Encoding the full batch and scaling by
        the mask keeps the shape static with identical values and gradients on
        active rows; inactive rows contribute exact zeros exactly as before.
        """
        batch_size = reference.shape[0]
        if mask.dtype != torch.bool or mask.shape != (batch_size,):
            raise ValueError(f"{name} encoder mask must have shape ({batch_size},)")
        encoded = self._encode(name, reference)
        # Autocast may run the encoder in bf16/fp16 while replay observations
        # remain fp32; scale in the encoded dtype so inactive rows become exact
        # zeros of the tensor the quantizer consumes.
        return encoded * mask[:, None, None].to(encoded.dtype)

    def encode_g1(self, reference: torch.Tensor) -> torch.Tensor:
        self._validate_reference(reference, self.config.g1_frame_dim, "g1_reference")
        return self._encode("g1", reference)

    def encode_smpl(self, reference: torch.Tensor) -> torch.Tensor:
        self._validate_reference(reference, self.config.smpl_frame_dim, "smpl_reference")
        return self._encode("smpl", reference)

    def quantize(self, latent: torch.Tensor) -> torch.Tensor:
        quantized, _ = self.quantizer(latent)
        return quantized.contiguous()

    def decode_motion(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        decoded = self.decoders["g1_kin"](tokens.flatten(start_dim=1))
        return decoded.reshape(
            batch_size,
            self.config.num_future_frames,
            self.config.g1_frame_dim,
        )

    def decode_action(self, tokens: torch.Tensor, actor_obs: torch.Tensor) -> torch.Tensor:
        decoder_input = torch.cat((tokens.flatten(start_dim=1), actor_obs), dim=-1)
        return self.decoders["g1_dyn"](decoder_input)

    def decode_action_features(self, tokens: torch.Tensor, actor_obs: torch.Tensor) -> torch.Tensor:
        """Return the penultimate G1 control-decoder representation.

        The released PPO checkpoint's final ``512 -> action_dim`` projection is
        an action-mean head.  FlashSAC keeps the released decoder trunk but owns
        a separate mean/std policy head at this boundary.
        """
        decoder_input = torch.cat((tokens.flatten(start_dim=1), actor_obs), dim=-1)
        decoder = self.decoders["g1_dyn"].module
        return decoder[:-1](decoder_input)

    def forward(
        self,
        actor_obs: torch.Tensor,
        g1_reference: torch.Tensor,
        smpl_reference: torch.Tensor,
        encoder_index: torch.Tensor,
        *,
        compute_auxiliary: bool = False,
        compute_action_decoder: bool = True,
    ) -> SonicForwardOutput:
        self._validate_inputs(actor_obs, g1_reference, smpl_reference, encoder_index)
        g1_active = encoder_index[:, 0].to(dtype=torch.bool)
        select_smpl = encoder_index[:, 1].to(dtype=torch.bool)
        # The legacy module masks each encoder by its active samples.  Keep
        # the batch-aligned tensors for selection while avoiding a full second
        # encoder pass for rows whose token is not consumed.
        g1_latent = self._encode_masked("g1", g1_reference, g1_active)
        smpl_latent = self._encode_masked("smpl", smpl_reference, select_smpl)
        if compute_auxiliary and torch.any(select_smpl & ~g1_active):
            missing_g1 = select_smpl & ~g1_active
            g1_latent[missing_g1] = self._encode("g1", g1_reference[missing_g1])
        g1_tokens = self.quantize(g1_latent)
        smpl_tokens = self.quantize(smpl_latent)
        selected_tokens = torch.where(select_smpl[:, None, None], smpl_tokens, g1_tokens)
        if compute_action_decoder:
            action_features = self.decode_action_features(selected_tokens, actor_obs)
            action_mean = self.decoders["g1_dyn"].module[-1](action_features)
        else:
            # FlashSAC uses its own policy head on actor_obs + selected_tokens.
            # Keep shape-compatible placeholders for callers that inspect the
            # structured output, without executing the legacy decoder trunk.
            batch_size = actor_obs.shape[0]
            action_features = actor_obs.new_empty((batch_size, 0))
            action_mean = actor_obs.new_empty((batch_size, self.config.action_dim))

        reconstruction = None
        losses: dict[str, torch.Tensor] = {}
        if compute_auxiliary:
            reconstruction = self.decode_motion(selected_tokens)
            smpl_reconstruction = self.decode_motion(smpl_tokens[select_smpl])
            reencoded_smpl_g1 = self.encode_g1(smpl_reconstruction)
            losses = sonic_auxiliary_losses(
                g1_reference=g1_reference,
                g1_reconstruction=reconstruction,
                g1_latent=g1_latent[select_smpl],
                smpl_latent=smpl_latent[select_smpl],
                reencoded_smpl_g1_latent=reencoded_smpl_g1,
                config=self.auxiliary_config,
            )
        return SonicForwardOutput(
            action_mean=action_mean,
            action_features=action_features,
            selected_tokens=selected_tokens,
            g1_tokens=g1_tokens,
            smpl_tokens=smpl_tokens,
            g1_latent=g1_latent,
            smpl_latent=smpl_latent,
            g1_reconstruction=reconstruction,
            auxiliary_losses=losses,
        )

    def _validate_inputs(
        self,
        actor_obs: torch.Tensor,
        g1_reference: torch.Tensor,
        smpl_reference: torch.Tensor,
        encoder_index: torch.Tensor,
    ) -> None:
        if actor_obs.ndim != 2 or actor_obs.shape[1] != self.config.actor_obs_dim:
            raise ValueError(
                f"actor_obs must have shape (B, {self.config.actor_obs_dim}), "
                f"got {tuple(actor_obs.shape)}"
            )
        self._validate_reference(g1_reference, self.config.g1_frame_dim, "g1_reference")
        self._validate_reference(smpl_reference, self.config.smpl_frame_dim, "smpl_reference")
        batch_size = actor_obs.shape[0]
        if g1_reference.shape[0] != batch_size or smpl_reference.shape[0] != batch_size:
            raise ValueError("SONIC actor and reference batch sizes must match")
        if encoder_index.shape != (batch_size, 2):
            raise ValueError(f"encoder_index must have shape (B, 2), got {tuple(encoder_index.shape)}")
        if not torch.compiler.is_compiling():
            # Value-dependent guards sync the tensor to the host, which CUDA
            # graph capture forbids; under torch.compile they fold away and the
            # packed-observation producer owns these invariants instead.
            if torch.any((encoder_index != 0) & (encoder_index != 1)):
                raise ValueError("encoder_index must contain only binary G1/SMPL masks")
            if torch.any(encoder_index.sum(dim=-1) == 0):
                raise ValueError("encoder_index must activate at least one encoder per sample")

    def _validate_reference(self, value: torch.Tensor, frame_dim: int, name: str) -> None:
        expected = (self.config.num_future_frames, frame_dim)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"{name} must have shape (B, {expected[0]}, {expected[1]})")
