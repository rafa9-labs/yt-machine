"""
Model Runtime — sequential, memory-safe orchestration.
=======================================================

This module owns everything that can hold 10-25 GB of unified memory:

  1. PipelineLock     — one generation job at a time, across processes.
  2. ManagedProcess   — launch/adopt/terminate local model servers safely.
  3. ModelRuntime     — phase-based lifecycle enforcing "one heavy model
                        at a time" on a 32 GiB Apple Silicon Mac.

PHASES (strictly sequential)
────────────────────────────
    idle → text → image → post → idle

    text   : text model resident; image model guaranteed stopped.
    image  : text model guaranteed stopped; image child processes run one
             at a time and are reaped after every call.
    post   : neither model resident; TTS/VQA/assembly may run.

WHY PROCESS GROUPS?
    A llama-server spawns worker threads and sometimes children. Killing
    only the parent can leave orphans holding memory. We launch servers in
    their own process group (start_new_session=True) and terminate the
    whole group.

ADOPTION RULES (important):
    An already-running server is only adopted when its command line
    matches the selected model path. Anything else is left untouched —
    we never kill processes we did not start or explicitly identify.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.models import memory
from src.models.profile import ModelProfile
from src.models.registry import (
    PROVIDER_LLAMACPP,
    PROVIDER_OLLAMA,
    build_llamacpp_launch_args,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

PHASE_IDLE = "idle"
PHASE_TEXT = "text"
PHASE_IMAGE = "image"
PHASE_POST = "post"

DEFAULT_RESERVED_GB = 6.0
LLAMACPP_PORT = int(os.getenv("LLAMACPP_PORT", "8080"))
LLAMACPP_CTX = int(os.getenv("LLAMACPP_CTX", "32768"))
LLAMACPP_PARALLEL = int(os.getenv("LLAMACPP_PARALLEL", "1"))
LLAMACPP_TIMEOUT_S = float(os.getenv("LLAMACPP_START_TIMEOUT", "900"))

LOCK_PATH = Path(os.getenv("YT_PIPELINE_LOCK", "/tmp/yt-machine-pipeline.lock"))
LOCK_STALE_S = float(os.getenv("YT_PIPELINE_LOCK_STALE_S", "21600"))  # 6h


class RuntimeError_(RuntimeError):
    """Raised when the runtime cannot transition phases safely."""


class LockError(RuntimeError):
    """Raised when another pipeline process already holds the lock."""


# ─────────────────────────────────────────────────────────────────────
# 1. Cross-process pipeline lock
# ─────────────────────────────────────────────────────────────────────

class PipelineLock:
    """Advisory inter-process lock guarding heavy model usage.

    WHY A FILE LOCK (not a mutex)?
      The API server, the daily scheduler, and manual CLI runs are separate
      OS processes. Only an OS-level lock can serialize them.

    Implementation: O_CREAT|O_EXCL pid file + liveness check + stale
    takeover when the recorded process is gone.
    """

    def __init__(self, path: Path = LOCK_PATH, stale_after_s: float = LOCK_STALE_S):
        self.path = Path(path)
        self.stale_after_s = stale_after_s
        self._acquired = False
        self._fd: Optional[int] = None

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError as exc:
            return exc.errno == errno.EPERM
        except Exception:
            return False

    def _read_owner(self) -> Optional[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _is_stale(self, owner: dict) -> bool:
        pid = int(owner.get("pid") or 0)
        started = float(owner.get("started_at") or 0)
        if started and (time.time() - started) > self.stale_after_s:
            return True
        return not self._pid_alive(pid)

    # ── public API ─────────────────────────────────────────────────

    def acquire(self, job_id: str = "") -> None:
        """Take the lock or raise LockError with the current owner's info."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "pid": os.getpid(),
            "job_id": job_id,
            "started_at": time.time(),
            "host": socket.gethostname(),
        })

        for attempt in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(fd, payload.encode("utf-8"))
                os.close(fd)
                self._acquired = True
                log.info("pipeline.lock_acquired", extra={"job_id": job_id, "path": str(self.path)})
                return
            except FileExistsError:
                owner = self._read_owner() or {}
                if owner and self._is_stale(owner):
                    log.warning(
                        "pipeline.lock_stale",
                        extra={"owner_pid": owner.get("pid"), "path": str(self.path)},
                    )
                    with contextlib.suppress(Exception):
                        os.unlink(self.path)
                    continue  # retry once after clearing the stale lock
                raise LockError(
                    "Another pipeline run is already active "
                    f"(pid={owner.get('pid')}, job={owner.get('job_id') or 'unknown'}, "
                    f"since={owner.get('started_at')}). Refusing to run two "
                    "memory-heavy jobs in parallel."
                )

        raise LockError(f"Could not acquire pipeline lock at {self.path}")

    def release(self) -> None:
        if not self._acquired:
            return
        owner = self._read_owner() or {}
        if int(owner.get("pid") or 0) == os.getpid():
            with contextlib.suppress(Exception):
                os.unlink(self.path)
        self._acquired = False
        log.info("pipeline.lock_released", extra={"path": str(self.path)})

    def __enter__(self) -> "PipelineLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


# ─────────────────────────────────────────────────────────────────────
# 2. Managed subprocess
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ProcessInfo:
    pid: int
    argv: List[str]
    adopted: bool = False
    started_at: float = field(default_factory=time.time)


def _find_pid_listening_on(port: int) -> Optional[int]:
    """Return the PID listening on `port`, if any (macOS lsof)."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def _pid_command_line(pid: int) -> str:
    try:
        out = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


class ManagedProcess:
    """A local model server we either launched or verified and adopted."""

    def __init__(self, info: ProcessInfo, log_path: Optional[Path] = None,
                 process: Optional[subprocess.Popen] = None):
        self.info = info
        self.log_path = log_path
        # Retained for launched children so we can reap them. Without this,
        # an exited child stays a zombie and `os.kill(pid, 0)` keeps
        # returning success — making "not alive" undetectable.
        self._process = process

    @property
    def pid(self) -> int:
        return self.info.pid

    @property
    def adopted(self) -> bool:
        return self.info.adopted

    @property
    def alive(self) -> bool:
        """True only while the process is actually running.

        ZOMBIE TRAP:
            `os.kill(pid, 0)` succeeds for a child that has ALREADY exited
            but not been reaped, because the pid still names a zombie
            entry. Polling the Popen object is what reaps it and reveals
            the true exit status. Using os.kill alone meant shutdown always
            appeared to fail: we burned the full SIGTERM grace period, then
            the full SIGKILL timeout, on a process that was long gone
            (measured: 120s per phase transition instead of ~1s).

            Adopted processes have no Popen handle (we did not start them),
            so they fall back to the signal probe.
        """
        if self._process is not None:
            return self._process.poll() is None
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def _signal_group_or_process(pid: int, sig: int) -> None:
        """Signal a child's process group — never our own.

        THE BUG THIS PREVENTS:
            `os.killpg(os.getpgid(pid), SIGKILL)` is correct only when the
            target really is in a different group. If the group id ever
            resolves to OUR group (a child that inherited the session, a
            pid that was reaped and recycled, or a platform where
            start_new_session did not take effect) then this call kills
            the entire pipeline — including the process issuing it. The
            failure is silent: the log simply stops mid-shutdown.

            Comparing against os.getpgrp() and refusing on a match makes
            that impossible. We fall back to signalling the single pid,
            which is strictly safer.
        """
        try:
            target_pgid = os.getpgid(pid)
        except ProcessLookupError:
            return

        if target_pgid == os.getpgrp():
            log.warning(
                "process.group_is_ours_refusing_killpg",
                extra={"pid": pid, "pgid": target_pgid},
            )
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, sig)
            return

        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(target_pgid, sig)

    def terminate(self, grace_s: float = 60.0) -> None:
        """Stop the whole process group, escalating from SIGTERM to SIGKILL.

        WHY UP TO 60 SECONDS OF GRACE?
            A llama-server holding a ~18 GB Metal-resident model can take a
            while to unwind its GPU allocations on SIGTERM. SIGTERM lets
            the server release Metal buffers in order, which the kernel can
            then actually reclaim; a hard kill risks holding pages until
            the kernel gets around to it, which is worse when the next
            phase needs that memory immediately.

        The caller is expected to verify memory recovery afterwards (see
        ModelRuntime.stop_text_model), so this grace period is a ceiling,
        not a fixed cost — a fast exit returns immediately.
        """
        if not self.alive:
            return

        log.info("process.terminating", extra={"pid": self.pid, "adopted": self.adopted})
        self._signal_group_or_process(self.pid, signal.SIGTERM)

        deadline = time.monotonic() + grace_s
        terminated_cleanly = False
        while time.monotonic() < deadline:
            if not self.alive:
                terminated_cleanly = True
                break
            time.sleep(0.5)

        if not terminated_cleanly:
            log.warning("process.kill", extra={"pid": self.pid, "grace_s": grace_s})
            self._signal_group_or_process(self.pid, signal.SIGKILL)
            for _ in range(120):
                if not self.alive:
                    break
                time.sleep(0.5)
        else:
            log.info("process.terminated_cleanly", extra={"pid": self.pid})

    def wait_port_released(self, port: int, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not _port_in_use(port):
                return True
            time.sleep(0.5)
        return not _port_in_use(port)


# ─────────────────────────────────────────────────────────────────────
# 3. ModelRuntime
# ─────────────────────────────────────────────────────────────────────

class ModelRuntime:
    """Phase-based, memory-safe lifecycle for local models.

    Usage:
        runtime = ModelRuntime(profile)
        runtime.start_text_phase()
        ... text calls ...
        runtime.start_image_phase()
        ... image calls ...
        runtime.finish()
    """

    def __init__(self, profile: ModelProfile, reserved_gb: Optional[float] = None):
        self.profile = profile
        # The profile records the calibrated reserve (see tools/model_setup.py).
        # Fall back to the module default when absent.
        if reserved_gb is None:
            reserved_gb = (profile.preferences or {}).get(
                "reserved_memory_gb", DEFAULT_RESERVED_GB
            )
        self.reserved_gb = float(reserved_gb)
        self.phase = PHASE_IDLE
        self._text_process: Optional[ManagedProcess] = None
        self._owned_processes: List[ManagedProcess] = []
        self._lock = PipelineLock()

    # ── lock passthrough ───────────────────────────────────────────

    def acquire_lock(self, job_id: str = "") -> None:
        self._lock.acquire(job_id)

    def release_lock(self) -> None:
        self._lock.release()

    # ── text phase ─────────────────────────────────────────────────

    def start_text_phase(self) -> None:
        """Ensure the selected text model is serving and nothing else is loaded."""
        if self.phase == PHASE_TEXT:
            return

        spec = self.profile.text
        if spec is None:
            raise RuntimeError_("No text model configured in the model profile")

        self._ensure_heavy_models_stopped(keep="text")

        if spec.provider == PROVIDER_OLLAMA:
            # Ollama manages its own process; just verify reachability.
            from src.models.providers import OllamaTextProvider

            provider = OllamaTextProvider(spec.endpoint or "http://localhost:11434")
            if not provider.health(timeout=5):
                raise RuntimeError_(
                    f"Ollama is not reachable at {spec.endpoint}. Start it with: ollama serve"
                )
            log.info("runtime.text_phase_ollama", extra={"endpoint": spec.endpoint})
        elif spec.provider in (PROVIDER_LLAMACPP, "openai_compat"):
            self._start_or_adopt_llamacpp(spec)
        else:
            raise RuntimeError_(
                f"Text provider '{spec.provider}' cannot be managed by the runtime. "
                "Select an Ollama or llama.cpp model in model_setup."
            )

        self.phase = PHASE_TEXT

    def _start_or_adopt_llamacpp(self, spec) -> None:
        """Start llama-server for the selected GGUF, or verify a matching one."""
        model_path = spec.path or spec.id
        port = LLAMACPP_PORT
        existing_pid = _find_pid_listening_on(port)

        if existing_pid:
            cmdline = _pid_command_line(existing_pid)
            if Path(model_path).name in cmdline:
                log.info(
                    "runtime.adopted_server",
                    extra={"pid": existing_pid, "model": model_path},
                )
                self._text_process = ManagedProcess(
                    ProcessInfo(pid=existing_pid, argv=cmdline.split(), adopted=True)
                )
                return
            # Something else owns the port — we will not kill it blindly.
            raise RuntimeError_(
                f"Port {port} is already in use by pid {existing_pid} "
                f"({cmdline[:120]}), which does not match the selected text model "
                f"({Path(model_path).name}). Stop that process or select it in "
                "model_setup instead of forcing a different model."
            )

        needs_gb = spec.estimated_anonymous_gb
        memory.require_capacity(needs_gb, self.reserved_gb, label=f"text model {Path(model_path).name}")

        # Reasoning models burn their token budget on chain-of-thought and
        # return empty `content`. The pipeline expects direct answers, so
        # reasoning defaults to off unless the profile overrides it.
        reasoning = (self.profile.preferences or {}).get("reasoning", "off")

        args = build_llamacpp_launch_args(
            spec,
            port=port,
            ctx=(self.profile.preferences or {}).get("context_length")
                or spec.context_length or LLAMACPP_CTX,
            parallel=LLAMACPP_PARALLEL,
            reasoning=reasoning,
        )
        log_path = Path("output/logs/llama_server.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            "runtime.launching_server",
            extra={"model": Path(model_path).name, "port": port, "needs_gb": needs_gb},
        )
        with open(log_path, "ab") as log_file:
            process = subprocess.Popen(
                ["llama-server", *args],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # isolate into its own process group
            )

        managed = ManagedProcess(
            ProcessInfo(pid=process.pid, argv=["llama-server", *args], adopted=False),
            log_path=log_path,
            process=process,
        )
        self._text_process = managed
        self._owned_processes.append(managed)

        self._wait_for_openai_server(port, managed, log_path)

    def _wait_for_openai_server(self, port: int, managed: ManagedProcess,
                                log_path: Path, timeout_s: float = LLAMACPP_TIMEOUT_S) -> None:
        """Poll /v1/models until llama-server reports ready."""
        import requests

        url = f"http://127.0.0.1:{port}/v1/models"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not managed.alive:
                tail = ""
                with contextlib.suppress(Exception):
                    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
                raise RuntimeError_(
                    f"llama-server exited during startup. Recent log:\n{tail}"
                )
            try:
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    log.info("runtime.server_ready", extra={"port": port})
                    return
            except Exception:
                pass
            time.sleep(2)

        managed.terminate()
        raise RuntimeError_(
            f"llama-server did not become ready within {timeout_s:.0f}s. "
            f"See {log_path}."
        )

    # ── image phase ────────────────────────────────────────────────

    def start_image_phase(self) -> None:
        """Stop the text model, verify memory returned, allow image generation."""
        if self.phase == PHASE_IMAGE:
            return

        spec = self.profile.image
        if spec is None:
            raise RuntimeError_("No image model configured in the model profile")

        self._ensure_heavy_models_stopped(keep="image")

        needs_gb = spec.estimated_anonymous_gb
        memory.wait_for_capacity(
            needs_gb, self.reserved_gb, timeout_s=60.0,
            label=f"image model {Path(spec.path or spec.id).name}",
        )

        self.phase = PHASE_IMAGE
        log.info("runtime.image_phase_ready", extra={"model": spec.id, "needs_gb": needs_gb})

    def end_image_phase(self) -> None:
        """Reap any lingering image subprocesses and let memory settle."""
        self._reap_stragglers()
        # Give unified memory a moment to be reclaimed before TTS/VQA loads
        # anything small. This is informational — we do not fail here.
        deadline = time.monotonic() + 30.0
        snap = memory.snapshot()
        while time.monotonic() < deadline:
            snap = memory.snapshot()
            if snap.available_gb >= (self.reserved_gb + 2.0):
                break
            time.sleep(2)
        log.info("runtime.post_phase", extra={"available_gb": snap.available_gb})
        self.phase = PHASE_POST

    # ── teardown ───────────────────────────────────────────────────

    def stop_text_model(self) -> None:
        """Stop the managed text server and wait for its memory to return.

        VERIFICATION IS THE POINT:
            Terminating the process is not the same as reclaiming its
            memory. On unified memory, this is the barrier that makes the
            image phase safe: we sample anonymous memory before the kill
            and require it to drop close to that baseline before returning.

            If memory does not recover we do NOT silently continue — the
            caller's next phase would then load a second model on top of a
            half-reclaimed one. We log loudly; the image phase's own
            capacity check is the second line of defense.
        """
        if self._text_process is None:
            return
        process = self._text_process
        self._text_process = None

        before = memory.snapshot()

        if process.adopted:
            # Adopted servers belong to the operator's workflow. We stop them
            # only because this is an application-managed lifecycle (chosen by
            # the user), but we log it clearly.
            log.info("runtime.stopping_adopted_server", extra={"pid": process.pid})
        process.terminate()
        process.wait_port_released(LLAMACPP_PORT, timeout_s=60.0)

        # Wait for anonymous memory to fall back toward the pre-load level.
        # Target: recover at least 80% of what this model was using.
        model_anonymous_gb = 0.0
        if self.profile.text is not None:
            model_anonymous_gb = self.profile.text.estimated_anonymous_gb
        target_drop = model_anonymous_gb * 0.8

        deadline = time.monotonic() + 90.0
        last = memory.snapshot()
        while time.monotonic() < deadline:
            last = memory.snapshot()
            reclaimed = before.anonymous_gb - last.anonymous_gb
            if reclaimed >= target_drop:
                break
            time.sleep(2)

        reclaimed = round(before.anonymous_gb - last.anonymous_gb, 1)
        log.info(
            "runtime.text_model_stopped",
            extra={
                "reclaimed_gb": reclaimed,
                "available_gb": last.available_gb,
                "anonymous_gb": last.anonymous_gb,
            },
        )

        if model_anonymous_gb and reclaimed < (target_drop * 0.5):
            log.warning(
                "runtime.memory_not_reclaimed",
                extra={
                    "expected_reclaim_gb": round(target_drop, 1),
                    "actual_reclaim_gb": reclaimed,
                    "note": "model memory may still be held by the kernel",
                },
            )

    def finish(self) -> None:
        """Idempotent full teardown — safe to call from a finally block."""
        try:
            self.stop_text_model()
        except Exception as exc:  # never mask the original failure
            log.warning("runtime.stop_text_failed", extra={"error": str(exc)})

        self._reap_stragglers()
        with contextlib.suppress(Exception):
            self.release_lock()
        self.phase = PHASE_IDLE
        log.info("runtime.finished")

    # ── internals ──────────────────────────────────────────────────

    def _ensure_heavy_models_stopped(self, keep: str) -> None:
        """Guarantee the other heavy role is not resident before a phase."""
        if keep == "image":
            self.stop_text_model()
        # Image models are always subprocesses that exit per call, so when
        # entering the text phase we only need to reap leftovers.
        if keep == "text":
            self._reap_stragglers()

    def _reap_stragglers(self) -> None:
        """Kill any tracked processes that are still alive."""
        for process in list(self._owned_processes):
            if process.alive:
                process.terminate()
            self._owned_processes.remove(process)

    @property
    def status(self) -> Dict[str, object]:
        return {
            "phase": self.phase,
            "text_pid": self._text_process.pid if self._text_process else None,
            "text_adopted": self._text_process.adopted if self._text_process else None,
            "memory": memory.snapshot().describe(),
        }
