"""Auxiliary objectives used by the SONIC shared-token backbone."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from uni_rl.algos.sonic.config import SonicAuxLossConfig


def _masked_mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE over mask-selected rows with a batch-aligned shape.

    Matches ``F.mse_loss(prediction[rows], target[rows])`` exactly for value
    and gradient: inactive rows contribute exact zeros, and an all-inactive
    mask returns 0 instead of dividing by zero.  Boolean row compression is
    avoided because its data-dependent output shape forces torch.compile to
    recompile for every distinct active-row count.
    """
    expand = (row_mask.shape[0], *([1] * (prediction.ndim - 1)))
    mask = row_mask.reshape(expand).to(prediction.dtype)
    squared = (prediction.float() - target.float()).pow(2) * mask
    row_elements = prediction.numel() // prediction.shape[0]
    active_elements = row_mask.sum().to(squared.dtype) * row_elements
    return squared.sum() / active_elements.clamp_min(1.0)


def sonic_auxiliary_losses(
    *,
    g1_reference: torch.Tensor,
    g1_reconstruction: torch.Tensor,
    g1_latent: torch.Tensor,
    smpl_latent: torch.Tensor,
    reencoded_smpl_g1_latent: torch.Tensor,
    row_mask: torch.Tensor,
    config: SonicAuxLossConfig,
) -> dict[str, torch.Tensor]:
    """Compute the three losses retained from the G1+SMPL release graph.

    ``g1_latent``, ``smpl_latent`` and ``reencoded_smpl_g1_latent`` are
    batch-aligned full-batch tensors; ``row_mask`` selects the SMPL-paired
    rows that previously arrived boolean-compressed.
    """

    reconstruction = F.mse_loss(g1_reconstruction, g1_reference)
    latent_alignment = _masked_mean_squared_error(smpl_latent, g1_latent, row_mask)
    cycle_consistency = _masked_mean_squared_error(reencoded_smpl_g1_latent, g1_latent, row_mask)
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
