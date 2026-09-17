"""
Tests for the model registry, profile, providers, memory guard, and runtime.
Run: .venv/bin/python -m pytest tests/test_model_registry.py -v
"""

import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from src.models.registry import (
    ModelSpec,
    PROVIDER_GGUF,
    PROVIDER_LLAMACPP,
    PROVIDER_MLXGEN,
    PROVIDER_OLLAMA,
    build_llamacpp_launch_args,
    probe_ollama,
    scan_gguf,
    specs_for_capability,
)
from src.models.profile import (
    ModelProfile,
    ProfileError,
    ROLE_IMAGE,
    ROLE_TEXT,
)
from src.models.providers import (
    NullEmbeddingProvider,
    OllamaEmbeddingProvider,
    OllamaTextProvider,
    OpenAITextProvider,
    ProviderError,
    build_embedding_provider,
    build_text_provider,
)


# ════════════════════════════════════════════════════════════════════
# ModelSpec
# ════════════════════════════════════════════════════════════════════

class TestModelSpec:
    def test_roundtrip(self):
        spec = ModelSpec(
            id="qwen.gguf",
            provider=PROVIDER_LLAMACPP,
            capabilities=["text"],
            endpoint="http://127.0.0.1:8080",
            path="/models/qwen.gguf",
            size_bytes=18 * 1024 ** 3,
            quantization="Q4_K_M",
        )
        restored = ModelSpec.from_dict(spec.to_dict())
        assert restored.id == spec.id
        assert restored.provider == spec.provider
        assert restored.supports("text")

    def test_unknown_keys_are_ignored(self):
        spec = ModelSpec.from_dict({
            "id": "x", "provider": "ollama", "capabilities": ["text"],
            "future_field": "ignored",
        })
        assert spec.id == "x"

    def test_anonymous_estimate_lower_than_total_for_mmap_providers(self):
        """GGUF weights are mmap'd, so anonymous memory must be below total."""
        spec = ModelSpec(
            id="m.gguf", provider=PROVIDER_LLAMACPP, capabilities=["text"],
            size_bytes=18 * 1024 ** 3,
        )
        assert spec.estimated_anonymous_gb < spec.estimated_memory_gb

    def test_ollama_anonymous_covers_weights(self):
        """Ollama loads weights into anonymous memory, not mmap."""
        spec = ModelSpec(
            id="gemma3:4b", provider=PROVIDER_OLLAMA, capabilities=["text"],
            size_bytes=3 * 1024 ** 3,
        )
        assert spec.estimated_anonymous_gb >= 3.0


# ════════════════════════════════════════════════════════════════════
# Discovery
# ════════════════════════════════════════════════════════════════════

class TestDiscovery:
    def test_probe_ollama_unreachable_returns_empty(self):
        with patch("src.models.registry._http_get_json", return_value=None):
            assert probe_ollama("http://127.0.0.1:1") == []

    def test_probe_ollama_maps_capabilities(self):
        tags = {"models": [{
            "name": "gemma3:4b", "size": 3338801804,
            "details": {"family": "gemma3", "quantization_level": "Q4_K_M"},
        }]}
        show = {
            "capabilities": ["completion", "vision"],
            "model_info": {"general.architecture": "gemma3", "general.context_length": 131072},
        }

        def fake_get(url, timeout=3.0):
            return tags if url.endswith("/api/tags") else None

        def fake_post(url, payload, timeout=5.0):
            return show

        with patch("src.models.registry._http_get_json", side_effect=fake_get), \
             patch("src.models.registry._http_post_json", side_effect=fake_post):
            specs = probe_ollama("http://localhost:11434")

        assert len(specs) == 1
        caps = specs[0].capabilities
        assert "text" in caps and "vision" in caps
        assert specs[0].context_length == 131072
        assert specs[0].provider == PROVIDER_OLLAMA

    def test_scan_gguf_finds_files(self, tmp_path):
        gguf = tmp_path / "model-Q4_K_M.gguf"
        gguf.write_bytes(b"\x00" * 1024)
        specs = scan_gguf([str(tmp_path)])
        assert len(specs) == 1
        assert specs[0].provider == PROVIDER_GGUF
        assert specs[0].quantization == "Q4_K_M"

    def test_scan_gguf_skips_mmproj_but_marks_vision(self, tmp_path):
        (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"\x00" * 1024)
        (tmp_path / "mmproj-model-f16.gguf").write_bytes(b"\x00" * 128)
        specs = scan_gguf([str(tmp_path)])
        assert len(specs) == 1
        assert "vision" in specs[0].capabilities

    def test_specs_for_capability_filters(self):
        discovered = {
            "a": [ModelSpec(id="t", provider="ollama", capabilities=["text"])],
            "b": [ModelSpec(id="i", provider="mlxgen", capabilities=["image"])],
        }
        assert [s.id for s in specs_for_capability(discovered, "text")] == ["t"]
        assert [s.id for s in specs_for_capability(discovered, "image")] == ["i"]


class TestLaunchArgs:
    def test_launch_args_force_single_slot(self):
        spec = ModelSpec(id="m.gguf", provider=PROVIDER_LLAMACPP, path="/m.gguf")
        args = build_llamacpp_launch_args(spec, port=8080, ctx=32768, parallel=1)
        assert "--parallel" in args
        assert args[args.index("--parallel") + 1] == "1"
        assert "--no-cont-batching" in args
        assert "/m.gguf" in args


# ════════════════════════════════════════════════════════════════════
# Profile
# ════════════════════════════════════════════════════════════════════

class TestModelProfile:
    def _profile(self):
        return ModelProfile(
            text=ModelSpec(id="qwen", provider=PROVIDER_LLAMACPP, capabilities=["text"],
                           endpoint="http://127.0.0.1:8080", path="/q.gguf"),
            image=ModelSpec(id="/flux", provider=PROVIDER_MLXGEN, capabilities=["image"],
                            path="/flux"),
        )

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "profile.json"
        profile = self._profile()
        profile.save(path)

        loaded = ModelProfile.load(path)
        assert loaded.text.id == "qwen"
        assert loaded.image.id == "/flux"
        assert loaded.text_endpoint == "http://127.0.0.1:8080"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(ProfileError):
            ModelProfile.load(tmp_path / "nope.json")

    def test_load_or_none_returns_none(self, tmp_path):
        assert ModelProfile.load_or_none(tmp_path / "nope.json") is None

    def test_validate_rejects_missing_text(self):
        profile = ModelProfile(image=ModelSpec(id="x", provider="mlxgen", capabilities=["image"]))
        with pytest.raises(ProfileError):
            profile.validate()

    def test_validate_rejects_wrong_capability(self):
        profile = ModelProfile(
            text=ModelSpec(id="t", provider="ollama", capabilities=["text"]),
            embedding=ModelSpec(id="not-emb", provider="ollama", capabilities=["text"]),
        )
        with pytest.raises(ProfileError):
            profile.validate()

    def test_save_is_atomic(self, tmp_path):
        """A .tmp file must not be left behind after a successful save."""
        path = tmp_path / "profile.json"
        self._profile().save(path)
        assert path.exists()
        assert not path.with_suffix(".json.tmp").exists()


# ════════════════════════════════════════════════════════════════════
# Providers
# ════════════════════════════════════════════════════════════════════

class TestProviderFactory:
    def test_build_llamacpp_provider(self):
        profile = ModelProfile(
            text=ModelSpec(id="q", provider=PROVIDER_LLAMACPP, capabilities=["text"],
                           endpoint="http://127.0.0.1:8080"),
        )
        provider = build_text_provider(profile)
        assert isinstance(provider, OpenAITextProvider)
        assert provider.base_url == "http://127.0.0.1:8080"

    def test_build_ollama_provider(self):
        profile = ModelProfile(
            text=ModelSpec(id="g", provider=PROVIDER_OLLAMA, capabilities=["text"],
                           endpoint="http://localhost:11434"),
        )
        assert isinstance(build_text_provider(profile), OllamaTextProvider)

    def test_build_provider_requires_endpoint(self):
        profile = ModelProfile(
            text=ModelSpec(id="q", provider=PROVIDER_LLAMACPP, capabilities=["text"]),
        )
        with pytest.raises(ProviderError):
            build_text_provider(profile)

    def test_unsupported_provider_raises(self):
        profile = ModelProfile(
            text=ModelSpec(id="m", provider=PROVIDER_MLXGEN, capabilities=["text"],
                           endpoint="http://x"),
        )
        with pytest.raises(ProviderError):
            build_text_provider(profile)

    def test_embedding_falls_back_to_null(self):
        profile = ModelProfile(
            text=ModelSpec(id="t", provider=PROVIDER_OLLAMA, capabilities=["text"]),
        )
        provider = build_embedding_provider(profile)
        assert isinstance(provider, NullEmbeddingProvider)
        assert provider.available is False

    def test_null_embedding_raises_on_use(self):
        with pytest.raises(ProviderError):
            NullEmbeddingProvider("no model").embed("hello")


class TestProviderStreaming:
    def test_openai_extract_sse_delta(self):
        line = b'data: {"choices":[{"delta":{"content":"hello"}}]}'
        assert OpenAITextProvider._extract(line) == "hello"

    def test_openai_extract_ignores_done(self):
        assert OpenAITextProvider._extract(b"data: [DONE]") is None

    def test_openai_extract_surfaces_errors(self):
        with pytest.raises(ProviderError):
            OpenAITextProvider._extract(b'data: {"error":"boom"}')

    def test_ollama_extract_response(self):
        assert OllamaTextProvider._extract(b'{"response":"hi","done":false}') == "hi"

    def test_ollama_extract_chat_message(self):
        line = b'{"message":{"role":"assistant","content":"yo"},"done":false}'
        assert OllamaTextProvider._extract(line) == "yo"

    def test_connection_error_becomes_provider_error(self):
        provider = OpenAITextProvider("http://127.0.0.1:1")
        with patch("requests.post", side_effect=__import__("requests").exceptions.ConnectionError("x")):
            with pytest.raises(ProviderError):
                provider.chat([{"role": "user", "content": "hi"}], model="m")


# ════════════════════════════════════════════════════════════════════
# Memory guard
# ════════════════════════════════════════════════════════════════════

class TestMemoryGuard:
    def test_snapshot_reads_real_values(self):
        from src.models import memory
        snap = memory.snapshot()
        assert snap.total_gb > 0
        assert snap.available_gb >= 0
        assert snap.anonymous_gb >= 0

    def test_can_allocate_respects_reserve(self):
        from src.models.memory import MemorySnapshot
        snap = MemorySnapshot(
            total_gb=32.0, free_gb=1.0, available_gb=20.0,
            anonymous_gb=12.0, file_cache_gb=18.0,
            swap_used_gb=0.0, pressure_free_pct=90.0, source="test",
        )
        assert snap.can_allocate(10.0, reserved_gb=6.0) is True
        assert snap.can_allocate(16.0, reserved_gb=6.0) is False

    def test_require_capacity_raises_when_tight(self):
        from src.models import memory
        from src.models.memory import MemorySnapshot
        tight = MemorySnapshot(
            total_gb=32.0, free_gb=0.5, available_gb=2.0,
            anonymous_gb=30.0, file_cache_gb=1.0,
            swap_used_gb=8.0, pressure_free_pct=5.0, source="test",
        )
        with patch("src.models.memory.snapshot", return_value=tight):
            with pytest.raises(RuntimeError, match="Insufficient memory"):
                memory.require_capacity(20.0, reserved_gb=6.0, label="big model")

    def test_require_capacity_fails_closed_when_unmeasurable(self):
        from src.models import memory
        from src.models.memory import MemorySnapshot
        unknown = MemorySnapshot(
            total_gb=0.0, free_gb=0.0, available_gb=0.0,
            anonymous_gb=0.0, file_cache_gb=0.0,
            swap_used_gb=0.0, pressure_free_pct=None, source="none",
        )
        with patch("src.models.memory.snapshot", return_value=unknown):
            with pytest.raises(RuntimeError, match="Cannot measure"):
                memory.require_capacity(5.0)


# ════════════════════════════════════════════════════════════════════
# Runtime
# ════════════════════════════════════════════════════════════════════

class TestPipelineLock:
    def test_acquire_and_release(self, tmp_path):
        from src.models.runtime import PipelineLock
        lock = PipelineLock(tmp_path / "test.lock")
        lock.acquire("job-1")
        assert lock.path.exists()
        lock.release()
        assert not lock.path.exists()

    def test_second_acquire_fails_while_held(self, tmp_path):
        from src.models.runtime import PipelineLock, LockError
        path = tmp_path / "test.lock"
        first = PipelineLock(path)
        first.acquire("job-1")
        try:
            second = PipelineLock(path)
            with pytest.raises(LockError, match="already active"):
                second.acquire("job-2")
        finally:
            first.release()

    def test_stale_lock_is_reclaimed(self, tmp_path):
        from src.models.runtime import PipelineLock
        path = tmp_path / "test.lock"
        path.write_text(json.dumps({"pid": 999999, "job_id": "dead", "started_at": 1}))
        lock = PipelineLock(path)
        lock.acquire("fresh")
        assert lock._acquired
        lock.release()

    def test_context_manager_releases_on_error(self, tmp_path):
        from src.models.runtime import PipelineLock
        path = tmp_path / "test.lock"
        with pytest.raises(ValueError):
            with PipelineLock(path):
                raise ValueError("boom")
        assert not path.exists()


class TestModelRuntime:
    def _profile(self):
        return ModelProfile(
            text=ModelSpec(id="g", provider=PROVIDER_OLLAMA, capabilities=["text"],
                           endpoint="http://127.0.0.1:59999"),
            image=ModelSpec(id="/flux", provider=PROVIDER_MLXGEN, capabilities=["image"],
                            path="/flux"),
        )

    def test_initial_phase_is_idle(self):
        from src.models.runtime import ModelRuntime, PHASE_IDLE
        assert ModelRuntime(self._profile()).phase == PHASE_IDLE

    def test_text_phase_fails_when_ollama_unreachable(self):
        from src.models.runtime import ModelRuntime
        runtime = ModelRuntime(self._profile())
        with pytest.raises(RuntimeError, match="Ollama is not reachable"):
            runtime.start_text_phase()

    def test_finish_is_idempotent(self):
        from src.models.runtime import ModelRuntime, PHASE_IDLE
        runtime = ModelRuntime(self._profile())
        runtime.finish()
        runtime.finish()
        assert runtime.phase == PHASE_IDLE

    def test_finish_releases_lock(self, tmp_path):
        from src.models.runtime import ModelRuntime, PipelineLock
        runtime = ModelRuntime(self._profile())
        runtime._lock = PipelineLock(tmp_path / "r.lock")
        runtime.acquire_lock("job")
        runtime.finish()
        assert not (tmp_path / "r.lock").exists()

    def test_image_phase_stops_text_first(self):
        """Entering the image phase must stop the text model."""
        from src.models.runtime import ModelRuntime
        runtime = ModelRuntime(self._profile())
        stopped = {"called": False}

        def fake_stop():
            stopped["called"] = True

        runtime.stop_text_model = fake_stop
        with patch("src.models.runtime.memory.wait_for_capacity") as wait:
            wait.return_value = MagicMock()
            runtime.start_image_phase()

        assert stopped["called"] is True
        assert runtime.phase == "image"

    def test_require_capacity_blocks_image_phase(self):
        from src.models.runtime import ModelRuntime
        runtime = ModelRuntime(self._profile())
        runtime.stop_text_model = lambda: None
        with patch("src.models.runtime.memory.wait_for_capacity",
                   side_effect=RuntimeError("Insufficient memory")):
            with pytest.raises(RuntimeError, match="Insufficient memory"):
                runtime.start_image_phase()
        assert runtime.phase == "idle"


# ════════════════════════════════════════════════════════════════════
# MLX-Gen provider
# ════════════════════════════════════════════════════════════════════

class TestMLXGenProvider:
    def test_missing_binary_reports_reason(self, tmp_path):
        from src.video.mlxgen_provider import MLXGenImageProvider
        provider = MLXGenImageProvider(model_path=str(tmp_path), executable="/nope/mlxgen")
        assert provider.available() is False
        assert "executable not found" in provider.missing_reason()

    def test_missing_model_reports_reason(self, tmp_path):
        from src.video.mlxgen_provider import MLXGenImageProvider
        binary = tmp_path / "mlxgen"
        binary.write_text("#!/bin/sh\n")
        provider = MLXGenImageProvider(model_path="/nope/model", executable=str(binary))
        assert provider.available() is False
        assert "model folder not found" in provider.missing_reason()

    def test_empty_prompt_raises(self, tmp_path):
        from src.video.mlxgen_provider import MLXGenImageProvider, MLXGenError
        binary = tmp_path / "mlxgen"
        binary.write_text("#!/bin/sh\n")
        model = tmp_path / "model"
        model.mkdir()
        provider = MLXGenImageProvider(model_path=str(model), executable=str(binary))
        with pytest.raises(MLXGenError, match="empty"):
            provider.generate("  ", tmp_path / "out.png")

    def test_no_output_file_returns_failure(self, tmp_path):
        """A zero-exit run that produces no file must not report success."""
        from src.video.mlxgen_provider import MLXGenImageProvider
        binary = tmp_path / "mlxgen"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        model = tmp_path / "model"
        model.mkdir()

        provider = MLXGenImageProvider(model_path=str(model), executable=str(binary))
        result = provider.generate("a scene", tmp_path / "missing.png")
        assert result["success"] is False
        assert "no output file" in result["error"]

    def test_failed_returncode_is_reported(self, tmp_path):
        from src.video.mlxgen_provider import MLXGenImageProvider
        binary = tmp_path / "mlxgen"
        binary.write_text("#!/bin/sh\necho 'boom' >&2\nexit 3\n")
        binary.chmod(0o755)
        model = tmp_path / "model"
        model.mkdir()

        provider = MLXGenImageProvider(model_path=str(model), executable=str(binary))
        result = provider.generate("a scene", tmp_path / "out.png")
        assert result["success"] is False
        assert result["provider"] == "mlxgen"

    def test_extract_error_recovers_actionable_message(self):
        """The abort message must name the real cause, not just an exit code."""
        from src.video.mlxgen_provider import MLXGenImageProvider

        output = (
            "usage: mlxgen generate [-h] [--model MODEL]\n"
            "mlxgen generate: error: --negative-prompt is not supported "
            "on flux2.text for /path/model. Drop --negative-prompt."
        )
        msg = MLXGenImageProvider._extract_error(output)
        assert msg is not None
        assert "negative-prompt is not supported" in msg

    def test_extract_error_falls_back_to_last_line(self):
        from src.video.mlxgen_provider import MLXGenImageProvider
        assert MLXGenImageProvider._extract_error("some traceback\nlast line") == "last line"

    def test_extract_error_returns_none_for_empty(self):
        from src.video.mlxgen_provider import MLXGenImageProvider
        assert MLXGenImageProvider._extract_error("") is None

    def test_negative_prompt_omitted_when_unsupported(self, tmp_path):
        """Unsupported flags must be dropped, not sent.

        Sending `--negative-prompt` to a model that does not advertise
        support makes mlxgen exit(2) immediately, which previously produced
        placeholder images for an entire run.
        """
        from src.video.mlxgen_provider import MLXGenImageProvider

        binary = tmp_path / "mlxgen"
        binary.write_text(
            "#!/bin/sh\n"
            'for a in "$@"; do\n'
            '  if [ "$a" = "--negative-prompt" ]; then echo "bad flag" >&2; exit 2; fi\n'
            "done\n"
            'echo ok > /dev/null\n'
            "exit 0\n"
        )
        binary.chmod(0o755)
        model = tmp_path / "model"
        model.mkdir()

        provider = MLXGenImageProvider(model_path=str(model), executable=str(binary))
        # Simulate a probed capability set that rejects negative prompts.
        provider._capabilities = {"supports_negative_prompt": False, "supports_guidance": True}

        out = tmp_path / "out.png"
        out.write_bytes(b"fake")  # satisfy the output-exists check
        result = provider.generate("scene", out, negative_prompt="blurry, bad")
        assert result["success"] is True, result.get("error")

    def test_negative_prompt_sent_when_supported(self, tmp_path):
        from src.video.mlxgen_provider import MLXGenImageProvider

        binary = tmp_path / "mlxgen"
        # Fails unless --negative-prompt is present.
        binary.write_text(
            "#!/bin/sh\n"
            'found=0\n'
            'for a in "$@"; do [ "$a" = "--negative-prompt" ] && found=1; done\n'
            '[ "$found" = "1" ] || { echo "missing flag" >&2; exit 9; }\n'
            "exit 0\n"
        )
        binary.chmod(0o755)
        model = tmp_path / "model"
        model.mkdir()

        provider = MLXGenImageProvider(model_path=str(model), executable=str(binary))
        provider._capabilities = {"supports_negative_prompt": True, "supports_guidance": True}

        out = tmp_path / "out.png"
        out.write_bytes(b"fake")
        result = provider.generate("scene", out, negative_prompt="blurry")
        assert result["success"] is True, result.get("error")
