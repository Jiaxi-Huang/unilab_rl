"""Auxiliary objectives used by the SONIC shared-token backbone."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from uni_rl.algos.sonic.config import SonicAuxLossConfig


def sonic_auxiliary_losses(
    *,
    g1_reference: torch.Tensor,
    g1_reconstruction: torch.Tensor,
    g1_latent: torch.Tensor,
    smpl_latent: torch.Tensor,
    reencoded_smpl_g1_latent: torch.Tensor,
    config: SonicAuxLossConfig,
) -> dict[str, torch.Tensor]:
    """Compute the three losses retained from the G1+SMPL release graph."""

    reconstruction = F.mse_loss(g1_reconstruction, g1_reference)
    if g1_latent.numel():
        latent_alignment = F.mse_loss(smpl_latent, g1_latent)
        cycle_consistency = F.mse_loss(reencoded_smpl_g1_latent, g1_latent)
    else:
        latent_alignment = g1_latent.sum() + smpl_latent.sum()
        cycle_consistency = g1_latent.sum() + reencoded_smpl_g1_latent.sum()
    total = (
        config.reconstruction * reconstruction
        + config.latent_alignment * latent_alignment
        + config.cycle_consistency * cycle_consistency
    )
    return {
        "reconstruction": reconstruction,
        "latent_alignment": latent_alignment,
        "cycle_consistency": cycle_consistency,
        "total": total,
    }
