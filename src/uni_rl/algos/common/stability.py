"""Numerical stability utilities for RL training."""

import torch


def check_nan_loss(
    loss: torch.Tensor, default_metrics: dict
) -> tuple[torch.Tensor | None, dict | None]:
    """Check if loss contains NaN or Inf values.

    Args:
        loss: Loss tensor to check
        default_metrics: Default metric values to return if NaN detected

    Returns:
        (loss, None) if valid, (None, nan_metrics) if invalid
    """
    if torch.isnan(loss) or torch.isinf(loss):
        nan_metrics = {k: float("nan") for k in default_metrics}
        return None, nan_metrics
    return loss, None


def clip_gradients(parameters, max_norm: float = 10.0):
    """Clip gradients by global norm.

    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
