"""Portable torch-copy replay transfer backend."""

from __future__ import annotations

import torch


class TorchCopyReplayTransferBackend:
    """Learner-thread transfer backend for MPS."""

    h2d_submitter = "torch_copy"
    host_memory_kind = "pageable_shared"
    host_pinned = False
    direct_pinned_shared = False
    supports_timing_events = False

    def __init__(self, *, device: torch.device, ring_depth: int) -> None:
        del ring_depth
        self.device = device
        self.device_family = device.type
        if device.type != "mps":
            raise ValueError(f"Torch-copy replay transfer requires MPS, got {device.type!r}")

    def register_host_slots(self, slots: list[torch.Tensor]) -> None:
        del slots
        return None

    def allocate_device_slots(
        self,
        *,
        count: int,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        return [torch.empty(shape, dtype=dtype, device=self.device) for _ in range(count)]

    def close(self) -> None:
        return None
