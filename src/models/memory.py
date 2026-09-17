"""
Unified Memory Guard — Apple Silicon aware memory checks.
=========================================================

On a 32 GiB unified-memory Mac there is no VRAM/RAM split: the GPU and CPU
share the same physical memory. Loading Qwen (~20 GiB) and Flux (~10+ GiB)
at the same time can push the machine into swap or trigger an OOM kill.

This module answers one question: "is it safe to load model X right now?"
It deliberately FAILS CLOSED — when it cannot measure, it refuses.

Signals used:
  - hw.memsize            total physical memory (sysctl)
  - vm_stat               free/inactive/speculative pages (macOS)
  - memory_pressure -Q    system-wide free percentage (macOS)
  - process RSS           via ps, for known pid lists

No third-party dependencies: psutil is not installed in this venv.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess

from dataclasses import dataclass
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

GIB = 1024 ** 3

# Headroom that must remain AFTER a model loads, so macOS stays responsive
# and the pipeline's own compute buffers have room.
#
# CALIBRATED FROM MEASUREMENT on a 32 GiB M1 Max:
#   An 18.4 GB anonymous model load left 4.7 GB headroom and the system ran
#   fine (swap stayed at 0.0 GB, pressure fell from 95% to 35%). A 6 GB
#   reserve made the same load impossible — 24.2 GB required vs 23.6 GB
#   available on an otherwise idle machine. A guard nobody can satisfy is
#   just as broken as no guard.
#
# 4 GB is the measured working margin. Raise it if you run browsers or
# other heavy apps alongside the pipeline; do not lower it below 3 GB.
DEFAULT_RESERVED_GB = 4.0

# Absolute floor — below this macOS itself starts compressing aggressively.
MIN_FREE_AFTER_LOAD_GB = 3.0

_PAGE_SIZE = 16384  # Apple Silicon default; corrected at runtime


@dataclass
class MemorySnapshot:
    """Memory state with the semantics that matter on Apple Silicon.

    WHY THIS IS NOT "FREE RAM":
        macOS aggressively uses idle memory as file cache. `free` is
        therefore near-zero on a healthy machine. What actually matters
        when loading a model is **anonymous** memory — wired kernel pages,
        heap allocations, and compressed memory. File-backed pages (a
        mmap'd GGUF, a safetensors checkpoint, disk cache) are clean and
        can be dropped by the kernel under pressure.

        llama.cpp mmaps GGUF weights and MLX mmaps safetensors, so the
        bulk of a model's footprint is file-backed. Only the KV cache and
        compute buffers are anonymous.

    Fields:
        available_gb   total minus non-reclaimable (anonymous) memory —
                       the honest "how much can be assigned" number.
        anonymous_gb   non-reclaimable resident memory (the real consumer)
        file_cache_gb  reclaimable file-backed cache (informational)
    """

    total_gb: float
    free_gb: float
    available_gb: float
    anonymous_gb: float
    file_cache_gb: float
    swap_used_gb: float
    pressure_free_pct: Optional[float]
    source: str

    @property
    def used_gb(self) -> float:
        return round(self.anonymous_gb, 2)

    def can_allocate(self, needed_gb: float, reserved_gb: float = DEFAULT_RESERVED_GB) -> bool:
        """True when `needed_gb` of anonymous memory fits beside the reserve.

        `needed_gb` must be the *anonymous* requirement (KV cache + compute
        buffers), not the on-disk weight size.
        """
        return (self.total_gb - self.anonymous_gb) >= (needed_gb + reserved_gb)

    @property
    def headroom_gb(self) -> float:
        return round(self.total_gb - self.anonymous_gb, 2)

    def describe(self) -> str:
        pressure = f", pressure free {self.pressure_free_pct:.0f}%" if self.pressure_free_pct is not None else ""
        return (
            f"total {self.total_gb:.1f}GB, anonymous {self.anonymous_gb:.1f}GB "
            f"(headroom {self.headroom_gb:.1f}GB), file cache {self.file_cache_gb:.1f}GB, "
            f"free {self.free_gb:.1f}GB, swap {self.swap_used_gb:.1f}GB{pressure}"
        )


# ─────────────────────────────────────────────────────────────────────
# Low-level readers
# ─────────────────────────────────────────────────────────────────────

def _sysctl_int(name: str) -> Optional[int]:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return int(out.stdout.strip())
    except Exception:
        pass
    return None


def _total_memory_bytes() -> int:
    value = _sysctl_int("hw.memsize")
    if value:
        return value
    # Fallback: sysconf (works on Linux and macOS)
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def _vm_stat_pages() -> Dict[str, int]:
    """Parse `vm_stat` output into {field: page_count}."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return {}
    except Exception:
        return {}

    stats: Dict[str, int] = {}
    for line in out.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        value = re.sub(r"[^0-9]", "", raw)
        if value:
            stats[key.strip()] = int(value)
    return stats


def _memory_pressure_free_pct() -> Optional[float]:
    """`memory_pressure -Q` prints a system-wide free percentage on macOS."""
    if not shutil.which("memory_pressure"):
        return None
    try:
        out = subprocess.run(
            ["memory_pressure", "-Q"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"free percentage:\s*(\d+)%", out.stdout)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def _swap_used_gb() -> float:
    used = _sysctl_int("vm.swapusage")
    if used:
        return used / GIB
    try:
        out = subprocess.run(
            ["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"used\s*=\s*([\d.]+)([MG])", out.stdout)
        if match:
            value = float(match.group(1))
            return value / 1024 if match.group(2) == "M" else value
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def snapshot() -> MemorySnapshot:
    """Capture current memory state. Never raises; degrades to zeros."""
    total_bytes = _total_memory_bytes()
    total_gb = total_bytes / GIB if total_bytes else 0.0
    page_size = _PAGE_SIZE

    stats = _vm_stat_pages()
    if stats:
        free_pages = stats.get("Pages free", 0)
        speculative_pages = stats.get("Pages speculative", 0)
        purgeable_pages = stats.get("Pages purgeable", 0)
        wired_pages = stats.get("Pages wired down", 0)
        compressed_pages = stats.get("Pages occupied by compressor", 0)
        anonymous_pages = stats.get("Anonymous pages", 0)
        file_backed_pages = stats.get("File-backed pages", 0)

        free_gb = ((free_pages + speculative_pages) * page_size) / GIB
        file_cache_gb = ((file_backed_pages + purgeable_pages) * page_size) / GIB

        # Anonymous memory is the non-reclaimable consumer. Prefer the
        # explicit counter; fall back to wired+compressor+anonymous when a
        # macOS version omits it.
        if anonymous_pages:
            anonymous_gb = (anonymous_pages + wired_pages) * page_size / GIB
        else:
            anonymous_gb = (wired_pages + compressed_pages) * page_size / GIB

        if total_gb and anonymous_gb > total_gb:
            anonymous_gb = total_gb
        available_gb = max(total_gb - anonymous_gb, 0.0)
        source = "vm_stat"
    else:
        try:
            free_gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
        except Exception:
            free_gb = 0.0
        anonymous_gb = max(total_gb - free_gb, 0.0) if total_gb else 0.0
        available_gb = free_gb
        file_cache_gb = 0.0
        source = "sysconf"

    return MemorySnapshot(
        total_gb=round(total_gb, 2),
        free_gb=round(free_gb, 2),
        available_gb=round(available_gb, 2),
        anonymous_gb=round(anonymous_gb, 2),
        file_cache_gb=round(file_cache_gb, 2),
        swap_used_gb=round(_swap_used_gb(), 2),
        pressure_free_pct=_memory_pressure_free_pct(),
        source=source,
    )


def rss_gb(pids: List[int]) -> float:
    """Sum resident memory (GB) for the given process ids."""
    total_kb = 0
    for pid in pids:
        if not pid:
            continue
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                total_kb += int(out.stdout.strip().split()[0])
        except Exception:
            continue
    return round(total_kb / (1024 * 1024), 2)


def can_allocate(needed_gb: float, reserved_gb: float = DEFAULT_RESERVED_GB) -> bool:
    """Convenience wrapper around snapshot().can_allocate()."""
    return snapshot().can_allocate(needed_gb, reserved_gb)


def require_capacity(needed_gb: float, reserved_gb: float = DEFAULT_RESERVED_GB,
                     label: str = "model") -> MemorySnapshot:
    """Fail closed: raise RuntimeError unless `needed_gb` fits now.

    WHY FAIL CLOSED? Proceeding on an unmeasurable or tight machine is how
    you get an OOM kill mid-pipeline. Refusing with a clear message lets
    the operator free memory and retry.
    """
    snap = snapshot()
    if snap.total_gb <= 0:
        raise RuntimeError(
            f"Cannot measure system memory ({snap.source}) — refusing to load {label}."
        )
    if not snap.can_allocate(needed_gb, reserved_gb):
        raise RuntimeError(
            f"Insufficient memory for {label}: need ~{needed_gb:.1f}GB + "
            f"{reserved_gb:.1f}GB reserve, but only {snap.available_gb:.1f}GB "
            f"available ({snap.describe()})."
        )
    log.info(
        "memory.capacity_ok",
        extra={"label": label, "needed_gb": needed_gb, "available_gb": snap.available_gb},
    )
    return snap


def wait_for_capacity(
    needed_gb: float,
    reserved_gb: float = DEFAULT_RESERVED_GB,
    timeout_s: float = 45.0,
    poll_s: float = 3.0,
    label: str = "model",
) -> MemorySnapshot:
    """Poll until capacity is available or the timeout expires."""
    import time

    deadline = time.monotonic() + timeout_s
    snap = snapshot()
    while time.monotonic() < deadline:
        snap = snapshot()
        if snap.can_allocate(needed_gb, reserved_gb):
            return snap
        time.sleep(poll_s)

    raise RuntimeError(
        f"Timed out after {timeout_s:.0f}s waiting for {needed_gb:.1f}GB "
        f"(+{reserved_gb:.1f}GB reserve) to free for {label}. Last state: {snap.describe()}"
    )
