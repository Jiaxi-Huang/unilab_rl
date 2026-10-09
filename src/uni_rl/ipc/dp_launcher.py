"""Single-node multi-GPU data-parallel launch and rank topology helpers.

Topology rules (config parsing, rank/device mapping, subprocess supervision)
live here so training scripts only assemble their flows. The off-policy path
uses :class:`DpRankSupervisor`; RSL-RL PPO uses PyTorch's elastic launcher and
standard ``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK`` variables. Algorithm-level
collectives remain owned by their respective learner implementations.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

UNILAB_DP_RANK = "UNILAB_DP_RANK"
UNILAB_DP_WORLD_SIZE = "UNILAB_DP_WORLD_SIZE"
UNILAB_DP_LOG_DIR = "UNILAB_DP_LOG_DIR"

_WATCHDOG_INTERVAL_S = 0.5
_COOPERATIVE_EXIT_GRACE_S = 10.0
_TERMINATE_TIMEOUT_S = 10.0
# Grace window for sibling ranks to finish their own runs once rank 0
# completed normally. Ranks run the same config, so they should land within
# startup skew of each other; exceeding the grace is treated as a failure.
_NORMAL_EXIT_GRACE_S = 600.0


def validate_dp_launchable(devices: tuple[int, ...]) -> None:
    """Fail fast at launch time when the host lacks any requested CUDA device.

    Kept as a downstream compat helper for explicit legacy device lists; the
    rank-local CUDA_VISIBLE_DEVICES path needs no validation.
    """
    import torch

    device_count = torch.cuda.device_count()
    missing = [index for index in devices if index >= device_count]
    if missing:
        raise ValueError(
            f"training.devices={list(devices)} requires CUDA device index(es) {missing}, "
            f"but torch.cuda.device_count()={device_count}"
        )


def resolve_dp_topology(devices_cfg: Any) -> tuple[int, ...] | None:
    """Compat: normalize a legacy ``training.devices`` list into CUDA indices.

    The off-policy stack now controls GPU topology exclusively through
    rank-local ``CUDA_VISIBLE_DEVICES`` (see ``reject_removed_device_config``).
    This helper remains for downstream consumers (UniLab SONIC) that still
    resolve an explicit device list from configuration.
    """
    if devices_cfg is None:
        return None
    devices = list(devices_cfg)
    if len(devices) == 0:
        return None
    normalized: list[int] = []
    for entry in devices:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(
                f"training.devices entries must be integer CUDA indices, got {entry!r}"
            )
        if entry < 0:
            raise ValueError(f"training.devices entries must be non-negative, got {entry}")
        normalized.append(int(entry))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"training.devices must not contain duplicates, got {normalized}")
    return tuple(normalized)


def reject_removed_device_config(devices_cfg: Any) -> None:
    """Fail closed if a removed CUDA-index device list is still supplied.

    GPU topology is controlled exclusively by rank-local
    ``CUDA_VISIBLE_DEVICES``. A CUDA-index list in configuration would create a
    second topology source and cannot be mapped safely when parent visibility
    contains opaque UUID/MIG tokens.
    """
    if devices_cfg is None:
        return
    if hasattr(devices_cfg, "__len__") and len(devices_cfg) == 0:
        return
    raise ValueError(
        "training.devices was removed; control GPUs exclusively with rank-local "
        "CUDA_VISIBLE_DEVICES"
    )


def current_dp_rank() -> int:
    """Data-parallel rank of this process (0 when not spawned as a rank)."""
    return int(os.environ.get(UNILAB_DP_RANK, "0"))


def visible_cuda_entries(current_visible_devices: str | None = None) -> tuple[str, ...]:
    """Return the process-local CUDA visibility entries without touching Torch.

    Entries are opaque: they may be physical indices or CUDA UUID/MIG tokens.
    An unset variable means the host namespace; an empty value (or ``-1``)
    means CUDA is deliberately hidden.  This parser is cold-path only and is
    intentionally usable before the first CUDA query.
    """
    raw = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if current_visible_devices is None
        else current_visible_devices
    )
    if raw is None:
        return ()
    entries = tuple(entry.strip() for entry in raw.split(",") if entry.strip())
    if entries == ("-1",):
        return ()
    return entries


def rank_local_cuda_device(
    *,
    rank: int = 0,
    current_visible_devices: str | None = None,
) -> str | None:
    """Resolve ``cuda:0`` when this rank owns exactly one visible GPU.

    ``CUDA_VISIBLE_DEVICES`` is the authoritative rank-local namespace.  A
    single entry can come either from an ordinary user launch or from
    :class:`DpRankSupervisor`; in both cases every in-process consumer must use
    the local index zero rather than a host-global index.

    On a host with exactly one visible CUDA device, leaving the variable unset
    is equivalent to a one-entry mask: the rank owns local ordinal zero. This
    honors the documented single-GPU launch contract without guessing on a
    multi-GPU host.
    """
    entries = visible_cuda_entries(current_visible_devices)
    if entries:
        if len(entries) != 1:
            raise ValueError(
                "A CUDA rank must own exactly one CUDA_VISIBLE_DEVICES entry; got "
                f"{','.join(entries)!r}"
            )
    else:
        try:
            visible_to_torch = torch.cuda.is_available() and torch.cuda.device_count() == 1
        except RuntimeError:
            visible_to_torch = False
        if not visible_to_torch:
            return None
    if int(rank) < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    return "cuda:0"


def resolve_dp_rank_device(rank: int = 0) -> str | None:
    """Return this rank's CUDA device from its visibility namespace.

    A CUDA rank owns exactly one visible GPU. The returned value is always the
    rank-local ``cuda:0``; opaque parent UUID/MIG tokens never become in-process
    CUDA indices.
    """
    return rank_local_cuda_device(rank=rank)


def current_dp_world_size() -> int:
    """Data-parallel world size of this process (1 when not spawned as a rank)."""
    return int(os.environ.get(UNILAB_DP_WORLD_SIZE, "1"))


def current_torch_distributed_rank() -> int:
    """Global torch-distributed rank of this process (0 outside torchrun)."""
    return int(os.environ.get("RANK", "0"))


def current_torch_distributed_local_rank() -> int:
    """Node-local torch-distributed rank of this process (0 outside torchrun)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def current_torch_distributed_world_size() -> int:
    """Torch-distributed world size of this process (1 outside torchrun)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def resolve_dp_rendezvous_path(log_dir: str, *, rank: int) -> str:
    """Shared FileStore rendezvous path for the DP parameter-sync group.

    Rank 0 anchors the store in its own run directory; spawned ranks point at
    the same canonical run root via ``UNILAB_DP_LOG_DIR``. The run directory
    is unique per run, which keeps stale FileStore state from a previous run
    out of the rendezvous.

    Cold path only: call at runner construction.
    """
    if int(rank) > 0:
        root = os.environ.get(UNILAB_DP_LOG_DIR)
        if root is None:
            raise ValueError(
                f"{UNILAB_DP_LOG_DIR} must be set for spawned data-parallel rank {rank}"
            )
    else:
        root = str(log_dir)
    return os.path.join(root, ".dp_rendezvous")


def resolve_collector_cpu_ids(
    world_size: int,
    rank: int,
    cpu_count: int | None = None,
    explicit: Any = None,
) -> list[int] | None:
    """Resolve the CPU ids exclusively owned by this rank's collector.

    Returns None for the single-rank default (``world_size <= 1``) so the
    single-card path stays bit-identical to the pre-partition behavior.
    With ``cpu_count=None`` (the normal upper-layer path), it discovers the
    current affinity and assigns complete physical-core groups, including SMT
    siblings, to each rank. ``cpu_count`` retains the deterministic logical
    CPU fallback used by compatibility callers and tests. ``explicit``
    (``training.dp_collector_cpu_ids``, one segment per rank) overrides either
    automatic route; segments must number exactly ``world_size``, be non-empty,
    contain non-negative ints, and not overlap. An empty ``explicit`` falls
    back to automatic partitioning.

    Cold path only: call at runner construction, never from the collect loop.
    """
    world_size = int(world_size)
    if world_size <= 1:
        return None
    rank = int(rank)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"data-parallel rank {rank} is out of range for world_size={world_size}")
    groups: list[list[int]] | None = None
    if cpu_count is None:
        try:
            available_ids = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_ids = list(range(os.cpu_count() or 1))
        groups = _discover_physical_cpu_groups(available_ids)
        cpu_count = len(available_ids)
    cpu_count = int(cpu_count or 0)
    if cpu_count < world_size:
        raise ValueError(
            f"cpu_count={cpu_count} cannot give each data-parallel rank its own CPU "
            f"(world_size={world_size})"
        )
    if explicit is not None and len(explicit) > 0:
        segments: list[list[int]] = [list(segment) for segment in explicit]
        if len(segments) != world_size:
            raise ValueError(
                f"training.dp_collector_cpu_ids must provide exactly world_size={world_size} "
                f"segments, got {len(segments)}"
            )
        seen: set[int] = set()
        for segment in segments:
            if not segment:
                raise ValueError(
                    f"training.dp_collector_cpu_ids segments must be non-empty, got {segments!r}"
                )
            for cpu_id in segment:
                if isinstance(cpu_id, bool) or not isinstance(cpu_id, int) or cpu_id < 0:
                    raise ValueError(
                        "training.dp_collector_cpu_ids entries must be non-negative "
                        f"integers, got {cpu_id!r}"
                    )
                if cpu_id in seen:
                    raise ValueError(
                        f"training.dp_collector_cpu_ids segments overlap on CPU {cpu_id}"
                    )
                seen.add(cpu_id)
        return segments[rank]
    if groups:
        if len(groups) < world_size:
            raise ValueError(
                f"available physical CPU groups={len(groups)} cannot serve world_size={world_size}"
            )
        group_count = len(groups)
        start = rank * group_count // world_size
        end = (rank + 1) * group_count // world_size
        if end <= start:
            raise ValueError(
                f"available physical CPU groups={group_count} cannot serve world_size={world_size}"
            )
        owned = groups[start:end]
        return [cpu_id for group in owned for cpu_id in group]
    size = cpu_count // world_size
    return list(range(rank * size, rank * size + size))


def _discover_physical_cpu_groups(cpu_ids: Sequence[int]) -> list[list[int]]:
    """Discover physical-core groups for the unified CPU partition helper."""
    ids = list(cpu_ids)
    if os.name == "posix" and os.path.isdir("/sys/devices/system/cpu"):
        groups: dict[tuple[str, str], list[int]] = {}
        try:
            for cpu_id in ids:
                with open(
                    f"/sys/devices/system/cpu/cpu{cpu_id}/topology/physical_package_id"
                ) as package_file:
                    package = package_file.read().strip()
                with open(f"/sys/devices/system/cpu/cpu{cpu_id}/topology/core_id") as core_file:
                    core = core_file.read().strip()
                groups.setdefault((package, core), []).append(cpu_id)
            if groups:
                return list(groups.values())
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            physical = int(
                subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
            )
            physical = max(1, min(physical, len(ids)))
            groups = [[] for _ in range(physical)]
            for index, cpu_id in enumerate(ids):
                groups[min(index * physical // len(ids), physical - 1)].append(cpu_id)
            return [group for group in groups if group]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return [[cpu_id] for cpu_id in ids]


def selected_visible_entries(
    *,
    world_size: int,
    current_visible_devices: str | None = None,
) -> tuple[str, ...]:
    """Return the opaque parent visibility entries selected for a DP launch.

    Parent ``CUDA_VISIBLE_DEVICES`` may contain physical indices, UUIDs, or MIG
    tokens. The first ``world_size`` entries define rank order; no integer
    configuration index is involved.
    """
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError("multi-rank visibility requires world_size >= 1")
    entries = visible_cuda_entries(current_visible_devices)
    if len(entries) != world_size:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly one entry per data-parallel "
            f"rank for this launch; got {len(entries)} entries for world_size={world_size}"
        )
    if len(set(entries)) != len(entries):
        raise ValueError(f"CUDA_VISIBLE_DEVICES contains duplicate entries: {entries}")
    return entries


def launch_torchrun_workers(
    *,
    world_size: int,
    script_path: str | os.PathLike[str],
    argv: Sequence[str],
    log_dir: str,
) -> None:
    """Launch one single-GPU torchrun worker per selected visibility entry.

    PyTorch elastic owns worker supervision and failure propagation. Each
    worker re-enters the regular training script with the original Hydra
    arguments, while ``UNILAB_DP_LOG_DIR`` supplies one canonical rank-0-owned
    run directory.
    """
    selected = selected_visible_entries(world_size=world_size)
    launch_env = os.environ.copy()
    # torchrun assigns one ordinal per worker; the worker remaps this list to
    # its own single-entry visibility before constructing CUDA resources.
    launch_env["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    launch_env.setdefault("NCCL_P2P_DISABLE", "1")
    launch_env.setdefault("NCCL_SHM_DISABLE", "1")
    launch_env[UNILAB_DP_LOG_DIR] = str(log_dir)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={world_size}",
        str(Path(script_path).resolve()),
        *argv,
    ]
    print(
        f"Launching RSL-RL data parallel training on {world_size} rank-local "
        "CUDA_VISIBLE_DEVICES entries.",
        flush=True,
    )
    completed = subprocess.run(command, env=launch_env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"torchrun workers failed with exit code {completed.returncode}")


def apply_dp_rank_config(cfg: Any, rank: int) -> str | None:
    """Apply the per-rank seed and return this rank's visibility-local device.

    Rank 0 keeps the configured seed; rank i>0 trains with ``seed + i`` until
    init broadcast lands in a later stage. The resolved rank-local device is
    returned instead of being written back into a synthetic ``training.device``
    key; with rank-local CUDA visibility this is always ``cuda:0``.
    """
    device = resolve_dp_rank_device(rank)
    if device is None:
        return None
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.algo.seed = int(cfg.algo.seed) + rank
    return device


def _current_entry_command() -> list[str]:
    """Build the command used to re-enter the current training entry point.

    The entry script belongs to the downstream owner (for example UniLab),
    while this package only owns the data-parallel supervisor.  Reusing
    ``sys.argv[0]`` keeps that ownership boundary intact and also preserves
    the original script path for source checkouts and installed consumers.
    """
    return [sys.executable, sys.argv[0], *sys.argv[1:]]


def _sigterm_system_exit(signum: int, _frame: Any) -> None:
    raise SystemExit(f"data-parallel rank 0 received signal {signum}")


def _signal_process_group(child: subprocess.Popen, signum: int) -> None:
    """Signal one rank and all of the worker processes it owns."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(child.pid, signum)
        else:  # pragma: no cover - Windows fallback for non-CUDA test hosts.
            child.send_signal(signum)
    except ProcessLookupError:
        pass


def _process_group_exists(child: subprocess.Popen) -> bool:
    """Return whether a rank process group still owns any live processes."""
    # Reap an exited group leader before probing the group. Otherwise the
    # leader may remain a zombie and make ``killpg(..., 0)`` look alive for the
    # entire escalation timeout even though no rank resources remain.
    child.poll()
    if not hasattr(os, "killpg"):  # pragma: no cover - Windows fallback.
        return child.returncode is None
    try:
        os.killpg(child.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class DpRankSupervisor:
    """Rank-0 supervisor that spawns and watches data-parallel rank subprocesses.

    Ranks 1..N-1 re-run the current entry script (``sys.argv[0]``, e.g.
    ``unilab/scripts/train_sac.py``) with the same Hydra
    argv plus the ``UNILAB_DP_*`` environment; each spawned rank builds its
    own learner+collector pair through the regular runner path. If any rank
    subprocess dies with a non-zero exit code while active, the supervisor
    delivers SIGTERM to rank 0 so the runner lifecycle unwinds through the
    normal try/finally (``runner.close()``), and ``__exit__`` tears down the
    remaining ranks. Each spawned rank owns a separate process group containing
    its collector. Rank 0 forwards Ctrl+C to those groups for cooperative
    cleanup before escalating to SIGTERM/SIGKILL.
    """

    def __init__(self, world_size: int, log_dir: str) -> None:
        self._selected = selected_visible_entries(world_size=world_size)
        self._log_dir = log_dir
        self._world_size = len(self._selected)
        self._children: list[subprocess.Popen] = []
        self._watchdog_stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._previous_signal_handlers: dict[int, Any] = {}
        self._shutdown_requested = False

    def __enter__(self) -> DpRankSupervisor:
        if self._world_size <= 1:
            # Single-device degenerate case: nothing to spawn or watch.
            return self
        base_env = os.environ | {
            UNILAB_DP_WORLD_SIZE: str(self._world_size),
            UNILAB_DP_LOG_DIR: self._log_dir,
        }
        self._install_signal_handlers()
        try:
            for rank, entry in enumerate(self._selected[1:], start=1):
                env = base_env | {
                    UNILAB_DP_RANK: str(rank),
                    "CUDA_VISIBLE_DEVICES": entry,
                }
                self._children.append(
                    subprocess.Popen(
                        _current_entry_command(),
                        env=env,
                        # Rank-local collectors inherit this group. The terminal
                        # only interrupts rank 0; the supervisor then forwards
                        # one coordinated SIGINT to each complete rank tree.
                        start_new_session=os.name == "posix",
                    )
                )
        except BaseException:
            self._request_children_shutdown()
            try:
                self._wait_for_children(_COOPERATIVE_EXIT_GRACE_S)
            finally:
                try:
                    self._terminate_children()
                finally:
                    self._restore_signal_handlers()
            raise
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="dp-rank-watchdog",
            daemon=True,
        )
        self._watchdog.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self._watchdog_stop.set()
        failed: list[tuple[int, object]] = []
        try:
            if self._watchdog is not None:
                self._watchdog.join(timeout=_TERMINATE_TIMEOUT_S)
                self._watchdog = None

            # Ranks that already exited on their own keep their exit code;
            # non-zero means the data-parallel run failed even if rank 0 is fine.
            failed = [
                (rank, child.returncode)
                for rank, child in enumerate(self._children, start=1)
                if child.returncode is not None and child.returncode != 0
            ]
            if exc_type is None and not failed:
                # Normal rank-0 completion: give sibling ranks a grace window to
                # finish their own runs instead of cutting them short.
                unfinished = self._wait_for_children(_NORMAL_EXIT_GRACE_S)
                for rank, child in unfinished:
                    print(
                        f"[dp_launcher] rank {rank} subprocess pid={child.pid} did not "
                        f"exit within {_NORMAL_EXIT_GRACE_S}s of rank 0 completion",
                        file=sys.stderr,
                    )
                    failed.append((rank, "timeout"))
                for rank, child in enumerate(self._children, start=1):
                    if child.returncode not in (None, 0) and not any(
                        failed_rank == rank for failed_rank, _ in failed
                    ):
                        failed.append((rank, child.returncode))
            else:
                # For Ctrl+C this is normally a no-op because the signal handler
                # already forwarded SIGINT. It also makes arbitrary rank-0
                # failures unwind sibling runner lifecycles cooperatively.
                self._request_children_shutdown()
                self._wait_for_children(_COOPERATIVE_EXIT_GRACE_S)
        finally:
            # Always reap complete rank process groups, including collectors
            # orphaned by a rank that exited before finishing its own cleanup.
            try:
                self._terminate_children()
            finally:
                self._restore_signal_handlers()
        if failed:
            message = "data-parallel rank subprocess(es) failed: " + ", ".join(
                f"rank {rank} exit code {code}" for rank, code in failed
            )
            if exc_type is None:
                raise RuntimeError(message)
            print(f"[dp_launcher] {message}", file=sys.stderr)
        return False

    def _install_signal_handlers(self) -> None:
        try:
            self._previous_signal_handlers[signal.SIGINT] = signal.signal(
                signal.SIGINT, self._handle_sigint
            )
            self._previous_signal_handlers[signal.SIGTERM] = signal.signal(
                signal.SIGTERM, self._handle_sigterm
            )
        except BaseException:
            self._restore_signal_handlers()
            raise

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self._previous_signal_handlers.items():
            signal.signal(signum, previous)
        self._previous_signal_handlers.clear()

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        self._request_children_shutdown()
        previous = self._previous_signal_handlers.get(signal.SIGINT, signal.default_int_handler)
        if previous == signal.SIG_IGN:
            return
        if previous == signal.SIG_DFL:
            signal.default_int_handler(signum, frame)
            return
        previous(signum, frame)

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        # A child-rank watchdog failure uses SIGTERM to unwind rank 0. Sibling
        # rank trees still receive SIGINT so their Python finally blocks run.
        self._request_children_shutdown()
        _sigterm_system_exit(signum, frame)

    def _request_children_shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        for child in self._children:
            if _process_group_exists(child):
                _signal_process_group(child, signal.SIGINT)

    def _wait_for_children(self, timeout: float) -> list[tuple[int, subprocess.Popen]]:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        pending = list(enumerate(self._children, start=1))
        while pending:
            pending = [(rank, child) for rank, child in pending if child.poll() is None]
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                pending[0][1].wait(timeout=min(_WATCHDOG_INTERVAL_S, remaining))
            except subprocess.TimeoutExpired:
                pass
        return pending

    @staticmethod
    def _wait_for_process_groups(
        children: list[subprocess.Popen], timeout: float
    ) -> list[subprocess.Popen]:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        pending = [child for child in children if _process_group_exists(child)]
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_WATCHDOG_INTERVAL_S, remaining))
            pending = [child for child in pending if _process_group_exists(child)]
        return pending

    def _terminate_children(self) -> None:
        groups = [child for child in self._children if _process_group_exists(child)]
        for child in groups:
            _signal_process_group(child, signal.SIGTERM)
        groups = self._wait_for_process_groups(groups, _TERMINATE_TIMEOUT_S)
        for child in groups:
            _signal_process_group(child, signal.SIGKILL)
        self._wait_for_process_groups(groups, _TERMINATE_TIMEOUT_S)

        # Reap every direct rank child after its complete process group has
        # stopped so no zombie or Popen handle survives supervisor teardown.
        for child in self._children:
            if child.poll() is not None:
                continue
            try:
                child.wait(timeout=_TERMINATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _signal_process_group(child, signal.SIGKILL)
                child.wait(timeout=_TERMINATE_TIMEOUT_S)

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL_S):
            for rank, child in enumerate(self._children, start=1):
                exit_code = child.poll()
                if exit_code is None:
                    continue
                if exit_code == 0:
                    # A rank that finishes cleanly (e.g. while rank 0 is still
                    # in playback) is not a failure; keep watching the rest.
                    continue
                print(
                    f"[dp_launcher] rank {rank} subprocess pid={child.pid} exited "
                    f"unexpectedly with code {exit_code}; shutting down rank 0",
                    file=sys.stderr,
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return
