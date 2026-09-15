"""CUDA/ROCm replay transfer backend."""

from __future__ import annotations

from typing import Any

import torch


class CudaLikeReplayTransferBackend:
    """Pinned host to CUDA-like device transfer backend.

    PyTorch ROCm exposes the same ``torch.cuda`` surface for runtime streams
    and events, so this backend intentionally keys on the PyTorch device type
    instead of NVIDIA-specific platform names.
    """

    host_memory_kind = "registered_pinned_shared"
    supports_timing_events = True

    def __init__(self, *, device: torch.device, ring_depth: int) -> None:
        del ring_depth
        self.device = device
        torch_version = getattr(torch, "version", None)
        self.device_family = "rocm" if getattr(torch_version, "hip", None) else "cuda"
        self.h2d_submitter = "torch_copy_stream" if self.device_family == "rocm" else "pybind11"
        self._cudart: Any = torch.cuda.cudart()
        if self._cudart is None:
            raise RuntimeError("torch.cuda.cudart() is required for replay host registration")
        self._registered_shared_slots: list[torch.Tensor] = []
        self._registered_shared_ptrs: list[int] = []
        self.host_pinned = False
        self.direct_pinned_shared = False

        if self.h2d_submitter == "pybind11":
            from uni_rl.ipc.replay_pipelines.native_h2d import get_diagnostic, is_available

            if not is_available():
                import sys

                print(
                    f"[ReplayTransfer] Native H2D unavailable, using torch_copy_stream.\n"
                    f"  Reason: {get_diagnostic()}\n"
                    f"  Performance impact: negligible for pinned-memory transfers.",
                    file=sys.stderr,
                    flush=True,
                )
                self.h2d_submitter = "torch_copy_stream"

    def register_host_slots(self, slots: list[torch.Tensor]) -> None:
        for slot in slots:
            nbytes = int(slot.numel() * slot.element_size())
            result = self._cudart.cudaHostRegister(int(slot.data_ptr()), nbytes, 0)
            if result != self._cudart.cudaError.success:
                raise RuntimeError(f"cudaHostRegister failed for collector replay slot: {result}")
            if not slot.is_pinned():
                self._cudart.cudaHostUnregister(int(slot.data_ptr()))
                raise RuntimeError("cudaHostRegister did not make collector replay slot pinned")
            self._registered_shared_slots.append(slot)
            self._registered_shared_ptrs.append(int(slot.data_ptr()))
        self.host_pinned = True
        self.direct_pinned_shared = True

    def allocate_device_slots(
        self,
        *,
        count: int,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        return [torch.empty(shape, dtype=dtype, device=self.device) for _ in range(count)]

    def close(self) -> None:
        while self._registered_shared_ptrs:
            ptr = self._registered_shared_ptrs.pop()
            try:
                self._cudart.cudaHostUnregister(int(ptr))
            except Exception:
                pass
        self._registered_shared_slots.clear()
        self.host_pinned = False
        self.direct_pinned_shared = False
