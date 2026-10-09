"""Typed configuration for the G1+SMPL SONIC backbone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SonicAuxLossConfig:
    """Weights from the released SONIC G1+SMPL training configuration."""

    reconstruction: float = 0.01
    latent_alignment: float = 1.0
    cycle_consistency: float = 1.0

    def __post_init__(self) -> None:
        if min(self.reconstruction, self.latent_alignment, self.cycle_consistency) < 0.0:
            raise ValueError("SONIC auxiliary-loss weights must be non-negative")


@dataclass(frozen=True)
class SonicModelConfig:
    """Architecture of the released G1+SMPL shared-token policy.

    The defaults reproduce ``sonic_release``. ``smoke`` only reduces widths
    and token size; it keeps the same encoder -> FSQ -> decoder graph.
    """

    num_future_frames: int = 10
    num_tokens: int = 2
    token_dim: int = 32
    fsq_levels: int = 32
    # Derived from the temporal proprioceptive contract.  The optional
    # override is retained only for tiny unit-test fixtures; production Hydra
    # profiles should not expose actor_obs_dim as an independent knob.
    actor_obs_dim: int | None = None
    g1_frame_dim: int = 64
    smpl_frame_dim: int = 84
    action_dim: int = 29
    actor_hidden_dim: int | None = None
    g1_encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    smpl_encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    g1_motion_decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    g1_control_decoder_hidden_dims: tuple[int, ...] = (2048, 2048, 1024, 1024, 512, 512)
    critic_hidden_dim: int = 512

    def __post_init__(self) -> None:
        scalar_values = (
            self.num_tokens,
            self.token_dim,
            self.g1_frame_dim,
            self.smpl_frame_dim,
            self.action_dim,
        )
        if self.actor_obs_dim is None:
            object.__setattr__(self, "actor_obs_dim", self.expected_actor_obs_dim)
        if (
            min((self.num_future_frames, self.actor_obs_dim, *scalar_values)) <= 0
            or self.fsq_levels <= 1
        ):
            raise ValueError("SONIC dimensions must be positive and fsq_levels must exceed one")
        if self.g1_frame_dim <= 6 or self.smpl_frame_dim <= 6:
            raise ValueError(
                "SONIC tokenizer frames must contain a feature term followed by a 6D term"
            )
        for name in (
            "g1_encoder_hidden_dims",
            "smpl_encoder_hidden_dims",
            "g1_motion_decoder_hidden_dims",
            "g1_control_decoder_hidden_dims",
        ):
            widths = getattr(self, name)
            if not widths or min(widths) <= 0:
                raise ValueError(f"{name} must contain positive widths")
        if self.critic_hidden_dim <= 0:
            raise ValueError("critic_hidden_dim must be positive")
        if self.actor_hidden_dim is not None and self.actor_hidden_dim <= 0:
            raise ValueError("actor_hidden_dim must be positive")

    @property
    def token_total_dim(self) -> int:
        return self.num_tokens * self.token_dim

    @property
    def expected_actor_obs_dim(self) -> int:
        """Proprioceptive actor history width implied by the temporal profile."""
        return 93 * self.num_future_frames

    def validate_observation_contract(self) -> None:
        """Fail early when a profile's temporal width and actor width diverge."""
        # Tiny synthetic fixtures may intentionally use a reduced proprioceptive
        # width; production SONIC profiles (93+ features) must obey the
        # temporal contract.
        if self.actor_obs_dim >= 93 and self.actor_obs_dim != self.expected_actor_obs_dim:
            raise ValueError(
                "SONIC actor_obs_dim must equal 93 * num_future_frames: "
                f"got {self.actor_obs_dim} and {self.num_future_frames}"
            )

    def parameter_count(self) -> int:
        """Return backbone parameter count for profile diagnostics."""
        from uni_rl.algos.sonic.network import SonicBackbone

        return sum(parameter.numel() for parameter in SonicBackbone(self).parameters())

    @property
    def g1_input_dim(self) -> int:
        return self.num_future_frames * self.g1_frame_dim

    @property
    def smpl_input_dim(self) -> int:
        return self.num_future_frames * self.smpl_frame_dim

    @property
    def g1_motion_output_dim(self) -> int:
        return self.num_future_frames * self.g1_frame_dim

    @classmethod
    def smoke(cls) -> "SonicModelConfig":
        return cls(
            token_dim=8,
            g1_encoder_hidden_dims=(256, 128),
            smpl_encoder_hidden_dims=(256, 128),
            g1_motion_decoder_hidden_dims=(256, 128),
            g1_control_decoder_hidden_dims=(512, 256),
        )
