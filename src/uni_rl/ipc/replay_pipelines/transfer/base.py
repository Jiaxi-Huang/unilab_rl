"""Replay transfer backend contract."""

from __future__ import annotations

from typing import Protocol

import torch


class ReplayTransferBackend(Protocol):
    """Device-specific host-to-device transfer backend."""

    device: torch.device
    device_family: str
    h2d_submitter: str
    host_memory_kind: str
    host_pinned: bool
    direct_pinned_shared: bool
    supports_timing_events: bool

    def register_host_slots(self, slots: list[torch.Tensor]) -> None: ...

    def allocate_device_slots(
        self,
        *,
        count: int,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> list[torch.Tensor]: ...

    def close(self) -> None: ...
