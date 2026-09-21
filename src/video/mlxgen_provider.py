"""
MLX-Gen Image Provider — subprocess-per-image generation.
==========================================================

The selected Qwen model is an MLX-Gen checkpoint:
    AbstractFramework/qwen-image-2512-4bit

It cannot be loaded by the CUDA/Diffusers code path, and it is far too
large to keep resident while other models run. So generation happens as:

    mlxgen generate --model <path> --prompt "..." --output out.png

WHY A SUBPROCESS PER IMAGE?
  - Memory is returned to the OS the moment the process exits.
  - A hung or crashed generation cannot poison the pipeline process.
  - The runtime can guarantee "no image model after the image phase".

PROCESS HYGIENE:
  Every child is started in its own process group and killed as a group on
  timeout. Nothing is left holding unified memory.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# No hardcoded install path: mlxgen is installed per machine, so it is read
# from MLXGEN_BIN (see .env.example). An empty value fails available() with
# missing_reason() instead of silently pointing at another user's home dir.
DEFAULT_MLXGEN_BIN = os.getenv("MLXGEN_BIN", "")
DEFAULT_TIMEOUT_S = float(os.getenv("MLXGEN_TIMEOUT", "900"))


class MLXGenError(RuntimeError):
    """Raised when mlxgen cannot produce an image."""


class MLXGenImageProvider:
    """Runs one `mlxgen generate` invocation per image."""

    def __init__(
        self,
        model_path: str,
        executable: str = DEFAULT_MLXGEN_BIN,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.model_path = str(model_path)
        self.executable = executable
        self.timeout_s = timeout_s
        # Populated by capabilities() — an empty dict means "not probed yet",
        # in which case we send only universally supported arguments.
        self._capabilities: Optional[dict] = None
        # Optional LoRA adapters applied to every generation. Without one,
        # any LoRA trigger words in the prompt (e.g. "Retro Pixel") are
        # inert text — the model has nothing to bind them to.
        self.lora_paths: list = []
        self.lora_scales: list = []

    # ── capability probing ─────────────────────────────────────────

    def capabilities(self, force: bool = False) -> dict:
        """Inspect the model's supported generation options.

        WHY THIS EXISTS:
            Sending an unsupported flag makes mlxgen exit with code 2
            before loading any weights. That failure is instant and easy to
            mistake for "the model could not generate", which is how a
            pipeline can quietly produce placeholder images for an entire
            run. Probing first means we simply omit flags the model does
            not accept.

            `--negative-prompt` is model-dependent. Capability probing keeps
            optional flags from being sent to routes that reject them.
        """
        if self._capabilities is not None and not force:
            return self._capabilities

        try:
            proc = subprocess.run(
                [self.executable, "capabilities", "--model", self.model_path],
                capture_output=True, text=True, timeout=120,
                start_new_session=True,
            )
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                for cap in data.get("capabilities", []):
                    if cap.get("public_task") == "text-to-image" or cap.get("id", "").endswith(".text"):
                        self._capabilities = cap
                        break
                else:
                    self._capabilities = {}
            else:
                log.warning(
                    "mlxgen.capabilities_failed",
                    extra={"returncode": proc.returncode, "tail": proc.stderr[-200:]},
                )
                self._capabilities = {}
        except Exception as exc:
            log.warning("mlxgen.capabilities_error", extra={"error": str(exc)})
            self._capabilities = {}

        return self._capabilities

    def supports(self, option: str) -> bool:
        """True when the model advertises support for `option`.

        Unknown (unprobed/empty) capabilities return False so we err on the
        side of omitting optional flags.
        """
        return bool(self.capabilities().get(f"supports_{option}"))

    # ── availability ───────────────────────────────────────────────

    def set_loras(self, paths: list, scales: list = None) -> None:
        """Attach LoRA adapters, applied on every subsequent generation."""
        self.lora_paths = [str(p) for p in (paths or []) if p]
        self.lora_scales = [float(s) for s in (scales or [])][:len(self.lora_paths)]

    def _executable_ok(self) -> bool:
        """True when the mlxgen executable is configured and present.

        An empty value must NOT be treated as a path: Path("").exists() is
        True (it resolves to "."), which would make a missing MLXGEN_BIN look
        installed right up to the subprocess launch.
        """
        return bool(self.executable) and Path(self.executable).is_file()

    def available(self) -> bool:
        return self._executable_ok() and Path(self.model_path).exists()

    def missing_reason(self) -> str:
        if not self._executable_ok():
            if not self.executable:
                return ("MLXGEN_BIN is not set — point it at your mlxgen "
                        "executable (see .env.example)")
            return f"mlxgen executable not found at {self.executable}"
        if not Path(self.model_path).exists():
            return f"model folder not found at {self.model_path}"
        return ""

    # ── generation ─────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        output_path: Path,
        width: int = 512,
        height: int = 512,
        steps: int = 28,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        extra_args: Optional[list] = None,
    ) -> dict:
        """Generate a single image. Returns a result dict (never None).

        Raises MLXGenError only for caller-fixable problems (missing binary,
        missing model). Generation failures return {"success": False}.
        """
        if not prompt or not prompt.strip():
            raise MLXGenError("Prompt cannot be empty")

        if not Path(self.executable).exists():
            raise MLXGenError(f"mlxgen executable not found at {self.executable}")
        if not Path(self.model_path).exists():
            raise MLXGenError(f"MLX-Gen model folder not found at {self.model_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        caps = self.capabilities()

        args = [
            self.executable, "generate",
            "--model", self.model_path,
            "--prompt", prompt.strip(),
            "--output", str(output_path),
            "--width", str(width),
            "--height", str(height),
            "--steps", str(steps),
            "--replace",
            "--no-progress",
        ]
        # Guidance is only meaningful on models that run a CFG branch.
        if guidance is not None and (not caps or caps.get("supports_guidance")):
            args.extend(["--guidance", str(guidance)])
        if seed is not None:
            args.extend(["--seed", str(seed)])
        # Negative prompts require a guidance branch. Omit rather than send
        # an unsupported flag — sending one makes mlxgen exit(2) instantly.
        if negative_prompt and caps.get("supports_negative_prompt"):
            args.extend(["--negative-prompt", negative_prompt])
        elif negative_prompt and caps and not caps.get("supports_negative_prompt"):
            log.info(
                "mlxgen.negative_prompt_unsupported",
                extra={"model": Path(self.model_path).name},
            )
        # Style LoRAs. Only meaningful when the model advertises support;
        # omitted otherwise (mlxgen rejects unknown combinations).
        if self.lora_paths and caps.get("supports_lora"):
            args.extend(["--lora-paths", *self.lora_paths])
            if self.lora_scales:
                args.extend(["--lora-scales", ",".join(str(s) for s in self.lora_scales)])
        elif self.lora_paths:
            log.info("mlxgen.lora_unsupported", extra={"model": Path(self.model_path).name})

        if extra_args:
            args.extend(extra_args)

        log.info(
            "mlxgen.generate_start",
            extra={
                "output": str(output_path),
                "seed": seed,
                "steps": steps,
                "loras": len(self.lora_paths),
            },
        )
        started = time.monotonic()

        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,  # own process group → killable as a unit
            )
        except OSError as exc:
            raise MLXGenError(f"Failed to start mlxgen: {exc}") from exc

        try:
            stdout, _ = process.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            self._kill_group(process)
            raise MLXGenError(
                f"mlxgen timed out after {self.timeout_s:.0f}s and was terminated"
            )

        elapsed = round(time.monotonic() - started, 1)
        tail = (stdout or "")[-400:]

        if process.returncode != 0:
            # Surface the real reason. `mlxgen` writes actionable messages
            # (unsupported flag, missing weights) to stderr, and an earlier
            # version of this code logged only the exit code — which turned
            # a precise "negative-prompt is not supported" error into an
            # opaque "exited with code 2" and sent debugging in circles.
            reason = self._extract_error(tail) or f"mlxgen exited with code {process.returncode}"
            log.warning(
                "mlxgen.failed",
                extra={"returncode": process.returncode, "reason": reason},
            )
            return {
                "success": False,
                "error": reason,
                "returncode": process.returncode,
                "stdout_tail": tail,
                "source": "mlxgen",
                "provider": "mlxgen",
            }

        if not output_path.exists():
            log.warning("mlxgen.no_output", extra={"output": str(output_path), "tail": tail})
            return {
                "success": False,
                "error": "mlxgen reported success but produced no output file",
                "stdout_tail": tail,
                "source": "mlxgen",
                "provider": "mlxgen",
            }

        log.info("mlxgen.generate_complete", extra={"output": str(output_path), "seconds": elapsed})
        return {
            "success": True,
            "filename": output_path.name,
            "path": str(output_path),
            "prompt_used": prompt,
            "source": "mlxgen",
            "provider": "mlxgen",
            "width": width,
            "height": height,
            "steps": steps,
            "seconds": elapsed,
        }

    # ── error parsing ──────────────────────────────────────────────

    @staticmethod
    def _extract_error(output: str) -> Optional[str]:
        """Pull the actionable line out of mlxgen's argparse output.

        Examples of what this recovers:
            "mlxgen generate: error: --negative-prompt is not supported ..."
            "error: unrecognized arguments: --foo"
        """
        if not output:
            return None
        for line in reversed(output.splitlines()):
            stripped = line.strip()
            if "error:" in stripped.lower():
                # Trim the leading "mlxgen generate: " noise if present.
                idx = stripped.lower().rfind("error:")
                return stripped[idx:].strip()
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        return lines[-1] if lines else None

    # ── hygiene ────────────────────────────────────────────────────

    @staticmethod
    def _kill_group(process: subprocess.Popen) -> None:
        """Kill the child's process group, never our own.

        Signalling our own group would terminate the pipeline process that
        is performing the cleanup, with no error raised.
        """
        def _send(sig: int) -> None:
            try:
                target_pgid = os.getpgid(process.pid)
            except ProcessLookupError:
                return
            if target_pgid == os.getpgrp():
                log.warning(
                    "mlxgen.group_is_ours_refusing_killpg",
                    extra={"pid": process.pid},
                )
                with contextlib.suppress(ProcessLookupError):
                    os.kill(process.pid, sig)
                return
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(target_pgid, sig)

        _send(signal.SIGTERM)
        try:
            process.wait(timeout=10)
            return
        except Exception:
            pass
        _send(signal.SIGKILL)
        with contextlib.suppress(Exception):
            process.wait(timeout=10)
