"""
Lifecycle integration tests — process adoption, phase ordering, teardown.
Run: .venv/bin/python -m pytest tests/test_runtime_lifecycle.py -v

These tests never start a real model server. They mock the process layer so
we can verify the *safety logic*: refusing foreign servers on our port,
stopping the text model before images, and always cleaning up.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from src.models.profile import ModelProfile
from src.models.registry import (
    PROVIDER_LLAMACPP,
    PROVIDER_MLXGEN,
    PROVIDER_OLLAMA,
    ModelSpec,
)
from src.models.runtime import (
    LLAMACPP_PORT,
    ModelRuntime,
    PHASE_IDLE,
    PHASE_IMAGE,
    PHASE_POST,
    PHASE_TEXT,
)


def _llamacpp_profile():
    return ModelProfile(
        text=ModelSpec(
            id="/models/qwen.gguf",
            provider=PROVIDER_LLAMACPP,
            capabilities=["text"],
            endpoint=f"http://127.0.0.1:{LLAMACPP_PORT}",
            path="/models/qwen.gguf",
            size_bytes=18 * 1024 ** 3,
            metadata={"served_model_name": "qwen.gguf"},
        ),
        image=ModelSpec(
            id="/models/flux", provider=PROVIDER_MLXGEN,
            capabilities=["image"], path="/models/flux",
        ),
    )


class TestServerAdoption:
    """The port-ownership rules that keep us from killing a stranger."""

    def test_no_server_means_launch(self):
        runtime = ModelRuntime(_llamacpp_profile())

        with patch("src.models.runtime._find_pid_listening_on", return_value=None), \
             patch("src.models.runtime.memory.require_capacity") as cap, \
             patch("subprocess.Popen") as popen, \
             patch.object(runtime, "_wait_for_openai_server") as wait:
            popen.return_value = MagicMock(pid=4242)
            runtime.start_text_phase()

        cap.assert_called_once()
        popen.assert_called_once()
        wait.assert_called_once()
        assert runtime.phase == PHASE_TEXT

    def test_matching_server_is_adopted(self):
        runtime = ModelRuntime(_llamacpp_profile())

        with patch("src.models.runtime._find_pid_listening_on", return_value=111), \
             patch("src.models.runtime._pid_command_line",
                   return_value="llama-server --model /models/qwen.gguf"), \
             patch("subprocess.Popen") as popen:
            runtime.start_text_phase()

        # No new server launched — we reused the operator's.
        popen.assert_not_called()
        assert runtime._text_process.pid == 111
        assert runtime._text_process.adopted is True

    def test_foreign_server_on_port_is_refused(self):
        """A different model on our port must never be silently killed."""
        runtime = ModelRuntime(_llamacpp_profile())

        with patch("src.models.runtime._find_pid_listening_on", return_value=222), \
             patch("src.models.runtime._pid_command_line",
                   return_value="llama-server --model /models/some_other.gguf"):
            with pytest.raises(RuntimeError, match="already in use"):
                runtime.start_text_phase()

        assert runtime.phase == PHASE_IDLE


class TestPhaseOrdering:
    """Heavy models must never coexist in memory."""

    def _ollama_profile(self):
        return ModelProfile(
            text=ModelSpec(id="gemma3:4b", provider=PROVIDER_OLLAMA,
                           capabilities=["text"], endpoint="http://127.0.0.1:59999"),
            image=ModelSpec(id="/flux", provider=PROVIDER_MLXGEN,
                            capabilities=["image"], path="/flux"),
        )

    def test_image_phase_stops_text_model_before_proceeding(self):
        runtime = ModelRuntime(_llamacpp_profile())
        calls = []

        def track_stop():
            calls.append("stop_text")

        runtime._text_process = MagicMock()
        runtime.stop_text_model = track_stop

        def track_capacity(*args, **kwargs):
            calls.append("check_capacity")
            return MagicMock()

        with patch("src.models.runtime.memory.wait_for_capacity", side_effect=track_capacity):
            runtime.start_image_phase()

        # Order matters: stop the text model, THEN verify memory.
        assert calls == ["stop_text", "check_capacity"]
        assert runtime.phase == PHASE_IMAGE

    def test_image_phase_aborts_when_memory_is_short(self):
        runtime = ModelRuntime(_llamacpp_profile())
        runtime.stop_text_model = lambda: None

        with patch("src.models.runtime.memory.wait_for_capacity",
                   side_effect=RuntimeError("Insufficient memory for image model")):
            with pytest.raises(RuntimeError, match="Insufficient memory"):
                runtime.start_image_phase()

        # Still idle — we did NOT enter the image phase.
        assert runtime.phase == PHASE_IDLE

    def test_text_phase_reaps_leftover_image_processes(self):
        runtime = ModelRuntime(_llamacpp_profile())
        leftover = MagicMock()
        leftover.alive = True
        runtime._owned_processes = [leftover]

        with patch("src.models.runtime._find_pid_listening_on", return_value=None), \
             patch("src.models.runtime.memory.require_capacity"), \
             patch("subprocess.Popen") as popen, \
             patch.object(runtime, "_wait_for_openai_server"):
            popen.return_value = MagicMock(pid=5)
            runtime.start_text_phase()

        leftover.terminate.assert_called_once()

    def test_end_image_phase_moves_to_post(self):
        runtime = ModelRuntime(_llamacpp_profile())
        runtime.phase = PHASE_IMAGE
        runtime._reap_stragglers = MagicMock()

        with patch("src.models.runtime.memory.snapshot", return_value=MagicMock(available_gb=25.0)):
            runtime.end_image_phase()

        assert runtime.phase == PHASE_POST
        runtime._reap_stragglers.assert_called()


class TestTeardown:
    def test_finish_stops_text_and_reaps(self):
        runtime = ModelRuntime(_llamacpp_profile())
        text_process = MagicMock()
        text_process.adopted = True
        text_process.pid = 999
        runtime._text_process = text_process
        runtime._owned_processes = []

        with patch.object(runtime, "stop_text_model") as stop, \
             patch.object(runtime, "release_lock") as release:
            runtime.finish()

        stop.assert_called_once()
        release.assert_called_once()
        assert runtime.phase == PHASE_IDLE

    def test_finish_survives_stop_failure(self):
        """Teardown must never raise — it runs in finally/atexit paths."""
        runtime = ModelRuntime(_llamacpp_profile())

        with patch.object(runtime, "stop_text_model", side_effect=OSError("boom")), \
             patch.object(runtime, "release_lock"):
            runtime.finish()  # must not raise

        assert runtime.phase == PHASE_IDLE

    def test_owned_processes_are_killed_on_reap(self):
        runtime = ModelRuntime(_llamacpp_profile())
        alive = MagicMock()
        alive.alive = True
        dead = MagicMock()
        dead.alive = False
        runtime._owned_processes = [alive, dead]

        runtime._reap_stragglers()

        alive.terminate.assert_called_once()
        dead.terminate.assert_not_called()
        assert runtime._owned_processes == []


class TestProcessGroupKill:
    def test_terminate_escalates_to_sigkill(self):
        from src.models.runtime import ManagedProcess, ProcessInfo

        process = ManagedProcess(ProcessInfo(pid=1234, argv=["llama-server"]))
        with patch("os.kill", return_value=None), \
             patch("os.killpg") as killpg, \
             patch("os.getpgid", return_value=1234), \
             patch("time.sleep"):
            process.terminate(grace_s=0.01)

        # SIGTERM then SIGKILL, both to the process group.
        assert killpg.call_count == 2
        signals = [call.args[1] for call in killpg.call_args_list]
        assert signals[0] == 15  # SIGTERM
        assert signals[1] == 9   # SIGKILL

    def test_terminate_never_signals_our_own_group(self):
        """A group id matching ours must degrade to a single-pid signal.

        Signalling our own group would silently kill the pipeline process
        that is performing the shutdown.
        """
        from src.models.runtime import ManagedProcess, ProcessInfo

        process = ManagedProcess(ProcessInfo(pid=1234, argv=["llama-server"]))
        with patch("os.kill") as kill, \
             patch("os.killpg") as killpg, \
             patch("os.getpgid", return_value=os.getpgrp()), \
             patch("time.sleep"):
            process.terminate(grace_s=0.01)

        killpg.assert_not_called()
        assert kill.call_count >= 1

    def test_zombie_child_is_not_reported_alive(self):
        """Regression: an unreaped exited child must read as not-alive.

        `os.kill(pid, 0)` succeeds for a zombie, so a Popen-backed
        ManagedProcess must poll() to reap and report the true status.
        Otherwise shutdown burns the full grace + kill timeouts on a
        process that already exited (measured at 120s per transition).
        """
        import subprocess
        from src.models.runtime import ManagedProcess, ProcessInfo

        child = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        child.wait()  # exited, and reaped by wait()

        managed = ManagedProcess(
            ProcessInfo(pid=child.pid, argv=["python"]), process=child,
        )
        assert managed.alive is False

    def test_zombie_without_popen_falls_back_to_signal_probe(self):
        """Adopted processes have no Popen; use the signal probe carefully."""
        from src.models.runtime import ManagedProcess, ProcessInfo

        managed = ManagedProcess(ProcessInfo(pid=999999, argv=["foreign"]))
        assert managed.alive is False
