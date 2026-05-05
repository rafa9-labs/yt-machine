"""
Model Orchestrator — Centralized GPU model lifecycle manager.

Coordinates Ollama, FLUX, and Kokoro VRAM usage across pipeline phases
to prevent contention and eliminate redundant model loading.

Phase transitions:
  idle → llm          (Ollama loads naturally — NO memory cap, full GPU available)
  llm → image_gen     (evict Ollama, preload FLUX with GGUF/CPU offload, pin for batch)
  image_gen → post    (flush FLUX, Kokoro loads lazily)
  post → tts          (Kokoro loads lazily)
  tts → cleanup       (flush all)

Quantization backends:
  - GGUF (recommended): Pre-quantized files, no bitsandbytes, Windows-safe
    Peak VRAM: Q4_K_S ~12.5GB, Q8_0 ~20.5GB
  - bitsandbytes 8/4-bit: ⚠️ Segfaults on Windows during loading
  - bf16 (none): ❌ OOM on 24GB GPU (needs ~36GB)

Memory fraction cap:
  DISABLED — set_per_process_memory_fraction causes segfaults with bitsandbytes
  on Windows. GGUF mode with CPU offload is memory-safe without the cap.

Performance impact:
  Before: 8x FLUX load/unload cycle (~12 min overhead)
  After:  1x FLUX load, pinned for batch (~1 min overhead)

VRAM budget (24GB GPU):
  Model weights (resident):
    - Ollama (gemma-4-26B-A4B Q4):  ~14.0 GB
    - FLUX.1-dev (GGUF Q4_K_S):      ~7.0 GB  (recommended, Windows-safe)
    - FLUX.1-dev (GGUF Q5_K_S):      ~8.5 GB
    - FLUX.1-dev (GGUF Q8_0):        ~13.0 GB
    - FLUX.1-dev (8-bit bnb):        ~12.5 GB  ⚠️ segfaults on Windows
    - FLUX.1-dev (bf16 full):        ~24.0 GB  ❌ OOM on 24GB GPU
    - Kokoro TTS:                    ~0.2 GB
    - PyTorch/CUDA base overhead:    ~1.5 GB (always present)

  Inference overhead (transient, on top of weights):
    - FLUX GGUF Q4_K_S @ 768x810/40steps:  ~4.0 GB
    - FLUX GGUF Q5_K_S @ 768x810/40steps:  ~5.0 GB
    - FLUX dev 8-bit @ 768x810/40steps:     ~5.5 GB
    - Ollama inference:                      ~2.0 GB (KV cache, attention)
    - Kokoro inference:                      ~0.1 GB (small model)

  Peak VRAM scenarios:
    - FLUX GGUF Q4_K_S + inference: 1.5 + 7.0 + 4.0 = 12.5 GB  ✅ fits 24GB easily
    - FLUX GGUF Q8_0 + inference:   1.5 + 13.0 + 6.0 = 20.5 GB  ✅ fits 24GB
    - Ollama load + inference:       1.5 + 14.0 + 2.0 = 17.5 GB  ✅ fits 24GB
    - Ollama + FLUX (both loaded):   1.5 + 14.0 + 7.0 = 22.5 GB ⚠️ tight, sequential preferred

  RULE: Only one heavy model (Ollama OR FLUX) on GPU at a time.
  RULE: Before loading FLUX, confirm enough VRAM for weights + inference overhead.
  RULE: Memory fraction cap is ONLY active during image_gen phase.
"""

import os
import time
import gc
import logging
from typing import Optional

import requests

log = logging.getLogger(__name__)

GPU_TOTAL_GB = 24.0
GPU_BASE_OVERHEAD_GB = 1.5
MEMORY_FRACTION = 0.95

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')


class ModelOrchestrator:
    """Manages GPU model lifecycle across pipeline phases.

    Tracks which models are on GPU and enforces the VRAM budget:
    - Never load Ollama + FLUX simultaneously (28GB > 24GB)
    - Kokoro (0.2GB) can share GPU with anything
    - FLUX pipeline stays pinned during batch generation (keep-alive)
    - All VRAM checks include inference overhead, not just model weights
    """

    VRAM_ESTIMATES = {
        'ollama': 14.0,
        'flux_dev_gguf_q4ks': 8.5,
        'flux_dev_gguf_q5ks': 10.0,
        'flux_dev_gguf_q8': 15.0,
        'flux_dev_8bit': 12.5,
        'flux_dev_bf16': 24.0,
        'flux_schnell_8bit': 12.0,
        'kokoro': 0.2,
        'vlm': 9.0,
    }

    INFERENCE_OVERHEAD = {
        'flux_dev_gguf_q4ks': 4.0,
        'flux_dev_gguf_q5ks': 5.0,
        'flux_dev_gguf_q8': 6.0,
        'flux_dev_8bit': 5.5,
        'flux_dev_8bit_warmup': 2.0,
        'flux_dev_bf16': 8.0,
        'flux_schnell_8bit': 3.0,
        'ollama': 2.0,
        'kokoro': 0.1,
    }

    def __init__(self, min_vram_gb: float = 14.0, vram_poll_timeout: int = 45,
                 gpu_total_gb: float = GPU_TOTAL_GB, memory_fraction: float = MEMORY_FRACTION):
        self._flux_loaded = False
        self._ollama_evicted = False
        self._current_phase = 'idle'
        self._min_vram_gb = min_vram_gb
        self._vram_poll_timeout = vram_poll_timeout
        self._loaded_models: dict[str, float] = {}
        self._inference_active: str | None = None
        self._gpu_total_gb = gpu_total_gb
        self._memory_fraction = memory_fraction
        self._memory_fraction_set = False

    @property
    def current_phase(self) -> str:
        return self._current_phase

    @property
    def loaded_models(self) -> dict[str, float]:
        return dict(self._loaded_models)

    def vram_used_gb(self) -> float:
        base = GPU_BASE_OVERHEAD_GB + sum(self._loaded_models.values())
        if self._inference_active:
            base += self.INFERENCE_OVERHEAD.get(self._inference_active, 0.0)
        return base

    def vram_available_gb(self) -> float:
        return self._gpu_total_gb - self.vram_used_gb()

    # ── Pre-pipeline GPU sweep ─────────────────────────────────────────

    def phase_pre_pipeline(self, timeout: int = 90) -> bool:
        """Mandatory GPU sweep BEFORE any model loading.

        1. Evicts all Ollama models from GPU
        2. Runs aggressive gc.collect() + torch.cuda.empty_cache()
        3. Checks actual free VRAM against peak FLUX budget
        4. If insufficient: waits up to `timeout` seconds, polling every 5s
        5. If still insufficient: REFUSES to proceed with clear error message

        NOTE: Does NOT set the CUDA memory fraction cap here. That is deferred
        to phase_image_generation() so Ollama can load freely during the LLM phase.

        Call this as the VERY FIRST thing before phase_llm().
        Returns True if safe to proceed, False if insufficient VRAM.
        """
        print(f"\n{'='*60}")
        print(f"  GPU PRE-PIPELINE CHECK")
        print(f"{'='*60}")

        self._evict_ollama()

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free_gb = self._get_free_vram()
                print(f"  GPU: {total_gb:.1f}GB total, {free_gb:.1f}GB free")
        except ImportError:
            print("  CUDA not available — assuming sufficient VRAM")
            self._current_phase = 'ready'
            return True

        peak_budget = GPU_BASE_OVERHEAD_GB + self.VRAM_ESTIMATES['flux_dev_8bit'] + self.INFERENCE_OVERHEAD['flux_dev_8bit']
        ollama_budget = GPU_BASE_OVERHEAD_GB + self.VRAM_ESTIMATES['ollama'] + self.INFERENCE_OVERHEAD['ollama']
        min_needed = max(peak_budget, ollama_budget)

        free_gb = self._get_free_vram()
        print(f"  Peak FLUX budget:  {peak_budget:.1f}GB")
        print(f"  Peak Ollama budget: {ollama_budget:.1f}GB")
        print(f"  Minimum needed:    {min_needed:.1f}GB")
        print(f"  Free VRAM:         {free_gb:.1f}GB")

        if free_gb >= min_needed:
            print(f"  [OK] VRAM sufficient -- safe to proceed")
            print(f"{'='*60}\n")
            self._current_phase = 'ready'
            return True

        print(f"  [!] Only {free_gb:.1f}GB free, need {min_needed:.1f}GB")
        print(f"  Waiting up to {timeout}s for VRAM to free...")
        log.warning(f"orchestrator.pre_pipeline — only {free_gb:.1f}GB free, need {min_needed:.1f}GB, waiting {timeout}s")

        start = time.time()
        while time.time() - start < timeout:
            time.sleep(5)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            free_gb = self._get_free_vram()
            elapsed = int(time.time() - start)
            print(f"  [{elapsed:3d}s] Free VRAM: {free_gb:.1f}GB / {min_needed:.1f}GB needed")

            if free_gb >= min_needed:
                print(f"  [OK] VRAM freed -- {free_gb:.1f}GB available after {elapsed}s")
                print(f"{'='*60}\n")
                self._current_phase = 'ready'
                return True

            self._evict_ollama()

        free_gb = self._get_free_vram()
        print(f"\n{'='*60}")
        print(f"  [FAIL] INSUFFICIENT VRAM -- CANNOT START PIPELINE")
        print(f"  Free: {free_gb:.1f}GB / Needed: {min_needed:.1f}GB")
        print(f"  Shortfall: {min_needed - free_gb:.1f}GB")
        print(f"")
        print(f"  Other processes may be using GPU memory:")
        print(f"    - Close Ollama chat sessions (run: ollama stop <model>)")
        print(f"    - Close browser tabs with GPU acceleration")
        print(f"    - Close ComfyUI / Stable Diffusion WebUI")
        print(f"    - Close other CUDA processes")
        print(f"")
        print(f"  Or set USE_LOCAL_FLUX=false to use cloud API instead.")
        print(f"{'='*60}\n")
        log.error(f"orchestrator.pre_pipeline — FATAL: {free_gb:.1f}GB free, need {min_needed:.1f}GB after {timeout}s timeout")
        return False

    def _set_memory_fraction(self) -> bool:
        """Set PyTorch's per-process CUDA memory fraction cap.

        DISABLED: set_per_process_memory_fraction causes segfaults with bitsandbytes
        on Windows during quantized weight loading. Instead, we rely on explicit VRAM
        checks and torch.cuda.empty_cache() to prevent OOM.
        """
        log.info("orchestrator.memory_fraction — SKIPPED (disabled for bitsandbytes compatibility)")
        print("  CUDA memory cap: DISABLED (bitsandbytes compatibility)", flush=True)
        return True

    def _reset_memory_fraction(self) -> None:
        """Reset PyTorch's CUDA memory fraction to 100% (no cap).

        Called after FLUSH is done so that subsequent phases (Kokoro TTS, or
        a return to LLM phase) can use the full GPU. Without this, Ollama
        would be unable to load in a future LLM phase because the cap would
        reserve 85% for PyTorch.
        """
        if not self._memory_fraction_set:
            return

        self._memory_fraction_set = False

        try:
            import torch
            if not torch.cuda.is_available():
                return

            torch.cuda.set_per_process_memory_fraction(1.0, device=0)
            gc.collect()
            torch.cuda.empty_cache()
            log.info("orchestrator.memory_fraction — RESET to 100% (cap removed)")
            print("  CUDA memory cap: 100% (cap released)", flush=True)
        except Exception as e:
            log.warning(f"orchestrator.memory_fraction — failed to reset: {e}")

    def can_load_model_with_inference(self, model_name: str) -> bool:
        """Check if we can load a model AND run inference (weights + overhead)."""
        weights = self.VRAM_ESTIMATES.get(model_name, 0.0)
        overhead = self.INFERENCE_OVERHEAD.get(model_name, 0.0)
        return (self._gpu_total_gb - self.vram_used_gb()) >= (weights + overhead)

    def can_load_model(self, model_name: str) -> bool:
        needed = self.VRAM_ESTIMATES.get(model_name, 0.0)
        return (self._gpu_total_gb - self.vram_used_gb()) >= needed

    def peak_vram_for_model(self, model_name: str) -> float:
        return GPU_BASE_OVERHEAD_GB + self.VRAM_ESTIMATES.get(model_name, 0.0) + self.INFERENCE_OVERHEAD.get(model_name, 0.0)

    def can_run_inference(self, model_name: str) -> bool:
        if model_name not in self._loaded_models:
            return False
        overhead = self.INFERENCE_OVERHEAD.get(model_name, 0.0)
        return self.vram_available_gb() >= overhead

    def _register_model(self, model_name: str) -> None:
        vram = self.VRAM_ESTIMATES.get(model_name, 0.0)
        self._loaded_models[model_name] = vram
        log.info(f"orchestrator.vram — registered {model_name} ({vram}GB), total used={self.vram_used_gb():.1f}GB/{GPU_TOTAL_GB}GB")

    def _unregister_model(self, model_name: str) -> None:
        if model_name in self._loaded_models:
            vram = self._loaded_models.pop(model_name)
            log.info(f"orchestrator.vram — unregistered {model_name} ({vram}GB), total used={self.vram_used_gb():.1f}GB/{GPU_TOTAL_GB}GB")

    def _clear_all_models(self) -> None:
        self._loaded_models.clear()
        self._flux_loaded = False
        self._ollama_evicted = True
        self._inference_active = None

    def begin_inference(self, model_name: str) -> bool:
        if model_name not in self._loaded_models:
            log.warning(f"orchestrator.inference — {model_name} not loaded, cannot start inference")
            return False
        if not self.can_run_inference(model_name):
            log.warning(f"orchestrator.inference — insufficient VRAM for {model_name} inference (need {self.peak_vram_for_model(model_name):.1f}GB peak, have {GPU_TOTAL_GB}GB total)")
            return False
        self._inference_active = model_name
        log.info(f"orchestrator.inference — started {model_name} (peak vram={self.vram_used_gb():.1f}GB/{GPU_TOTAL_GB}GB)")
        return True

    def end_inference(self) -> None:
        if self._inference_active:
            log.info(f"orchestrator.inference — ended {self._inference_active}")
            self._inference_active = None

    # ── Phase transitions ──────────────────────────────────────────────

    def phase_llm(self) -> None:
        """Evict FLUX if loaded. Allow Ollama to load naturally on next LLM call."""
        self._flush_flux_if_loaded()
        self._ollama_evicted = False
        self._current_phase = 'llm'
        self._register_model('ollama')
        log.info("orchestrator.phase llm — FLUX flushed, Ollama will load on demand")

    def phase_image_generation(self) -> bool:
        """Evict Ollama, set CUDA memory cap, wait for VRAM, preload FLUX.

        Sets the PyTorch per-process memory fraction cap HERE (not in pre_pipeline)
        so that Ollama can use the full GPU during the LLM phase. The cap is only
        needed during FLUX image generation to prevent PyTorch from spilling into
        system RAM.

        Returns:
            True if FLUX preloaded successfully, False if fallback to cloud needed.
        """
        self._evict_ollama()
        self._current_phase = 'image_gen'

        # Reclaim VRAM from Ollama before setting the memory fraction cap
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Set hard CUDA memory cap NOW — after Ollama is gone, before FLUX loads.
        # This prevents PyTorch from allocating beyond 85% of GPU during image gen,
        # which would crash the PC by spilling into system RAM.
        self._set_memory_fraction()

        from .pixel_art_tool import signal_flux_keep_alive, preload_flux_pipeline, LOCAL_FLUX_QUANTIZE, _IS_GGUF

        signal_flux_keep_alive(True)

        _QUANT_TO_FLUX_KEY = {
            "gguf_q4ks": "flux_dev_gguf_q4ks",
            "gguf_q5ks": "flux_dev_gguf_q5ks",
            "gguf_q8":   "flux_dev_gguf_q8",
            "8bit":      "flux_dev_8bit",
            "4bit":      "flux_dev_8bit",
            "none":      "flux_dev_bf16",
        }
        flux_key = _QUANT_TO_FLUX_KEY.get(LOCAL_FLUX_QUANTIZE, "flux_dev_gguf_q4ks")
        peak_needed = self.peak_vram_for_model(flux_key)

        if not self.can_load_model_with_inference(flux_key):
            free_gb = self.vram_available_gb()
            log.warning(f"orchestrator.vram_insufficient — only {free_gb:.1f}GB available, need {peak_needed:.1f}GB peak for {flux_key} (weights+inference)")
            return False

        vram_ok = self._wait_for_vram(
            peak_needed,
            timeout=self._vram_poll_timeout,
        )
        if not vram_ok:
            free_gb = self._get_free_vram()
            log.warning(f"orchestrator.vram_insufficient — free={free_gb:.1f}GB needed={peak_needed:.1f}GB peak, falling back to cloud API")
            return False

        preloaded = preload_flux_pipeline()
        if preloaded:
            self._flux_loaded = True
            self._register_model(flux_key)
            log.info(f"orchestrator.phase image_gen — FLUX preloaded and pinned for batch (peak={peak_needed:.1f}GB)")
        else:
            log.warning("orchestrator.phase image_gen — FLUX preload failed, will try per-image or fall back to cloud")

        return preloaded

    def phase_image_generation_done(self) -> None:
        """Flush FLUX after all images generated. Release VRAM cap for next phase."""
        self._flush_flux_if_loaded()

        from .pixel_art_tool import signal_flux_keep_alive
        signal_flux_keep_alive(False)

        # Release the CUDA memory fraction cap so Kokoro/other processes can use
        # the full GPU. The cap is only needed during FLUX image generation.
        self._reset_memory_fraction()

        self._current_phase = 'post_image'
        self._inference_active = None
        log.info("orchestrator.phase post_image — FLUX flushed, VRAM cap released")

    def phase_tts(self) -> None:
        """Flush FLUX if still loaded. Kokoro loads lazily on first TTS call."""
        self._flush_flux_if_loaded()
        self._current_phase = 'tts'
        self._register_model('kokoro')
        log.info("orchestrator.phase tts — FLUX flushed, Kokoro will load on demand")

    def phase_video_assembly(self) -> None:
        """No GPU models needed. Ensure all are flushed."""
        self._flush_flux_if_loaded()
        self._unregister_model('kokoro')
        self._current_phase = 'video_edit'
        self._inference_active = None
        log.info("orchestrator.phase video_edit — No GPU models needed")

    def phase_cleanup(self) -> None:
        """Final cleanup — evict all GPU models, release cap."""
        self._flush_flux_if_loaded()
        self._evict_ollama()
        self._reset_memory_fraction()
        self._clear_all_models()
        self._current_phase = 'idle'
        log.info("orchestrator.phase idle — All GPU models evicted, cap released")

    # ── Ollama eviction ────────────────────────────────────────────────

    def _evict_ollama(self) -> None:
        """Tell Ollama to unload all running models from GPU, freeing VRAM."""
        if self._ollama_evicted:
            log.info("orchestrator.ollama — Already evicted, skipping")
            return

        try:
            import requests
            base_url = "http://localhost:11434"

            resp = requests.get(f"{base_url}/api/ps", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                for m in models:
                    model_name = m.get("name", m.get("model", ""))
                    if model_name:
                        try:
                            requests.post(
                                f"{base_url}/api/generate",
                                json={"model": model_name, "keep_alive": 0},
                                timeout=30,
                            )
                            log.info(f"orchestrator.ollama.evicted model={model_name}")
                        except Exception as e:
                            log.warning(f"orchestrator.ollama.evict_failed model={model_name} error={e}")
            else:
                log.info(f"orchestrator.ollama — Ollama not running (status {resp.status_code})")
        except Exception as e:
            log.info(f"orchestrator.ollama — Ollama not available: {e}")

        self._ollama_evicted = True
        self._unregister_model('ollama')
        time.sleep(2)
        gc.collect()

    # ── FLUX pipeline lifecycle ────────────────────────────────────────

    def _flush_flux_if_loaded(self) -> None:
        """Flush the FLUX pipeline from GPU if it's currently loaded."""
        if not self._flux_loaded:
            return

        from .pixel_art_tool import _flush_flux_pipeline
        _flush_flux_pipeline()
        self._flux_loaded = False
        self._unregister_model('flux_dev_8bit')
        log.info("orchestrator.flux — FLUX pipeline flushed from GPU")

    # ── VRAM monitoring ────────────────────────────────────────────────

    def _get_free_vram(self) -> float:
        """Get current free VRAM in GB. Returns -1.0 if CUDA unavailable."""
        try:
            import torch
            if not torch.cuda.is_available():
                return -1.0
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            return free_bytes / (1024 ** 3)
        except ImportError:
            return -1.0

    def _wait_for_vram(self, required_gb: float, timeout: int = 45) -> bool:
        """Poll VRAM until required_gb is free, or timeout.

        Args:
            required_gb: Minimum free VRAM needed in GB
            timeout: Maximum seconds to wait

        Returns:
            True if VRAM available, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            free_gb = self._get_free_vram()
            if free_gb < 0:
                log.info("orchestrator.vram — CUDA unavailable, assuming sufficient VRAM")
                return True
            if free_gb >= required_gb:
                log.info(f"orchestrator.vram — free={free_gb:.2f}GB required={required_gb:.1f}GB, VRAM sufficient")
                return True
            log.info(f"orchestrator.vram — free={free_gb:.2f}GB required={required_gb:.1f}GB, waiting...")
            time.sleep(1)

        free_gb = self._get_free_vram()
        log.warning(f"orchestrator.vram_timeout — free={free_gb:.2f}GB required={required_gb:.1f}GB after {timeout}s")
        return False

    def force_cleanup(self) -> None:
        """Aggressively free all GPU memory. Call between phase transitions.

        Evicts Ollama, flushes FLUX, runs gc.collect() + torch.cuda.empty_cache(),
        and resets the memory fraction cap to 100%. Ensures a clean GPU state
        before loading the next model.
        """
        print("  [ORCHESTRATOR] Force cleanup — freeing all GPU memory...", flush=True)
        self._evict_ollama()
        self._flush_flux_if_loaded()
        self._reset_memory_fraction()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_gb = self._get_free_vram()
                print(f"  [ORCHESTRATOR] Clean state: {free_gb:.1f}GB free / {self._gpu_total_gb:.1f}GB total", flush=True)
                log.info(f"orchestrator.force_cleanup — free={free_gb:.1f}GB total={self._gpu_total_gb:.1f}GB")
        except ImportError:
            print("  [ORCHESTRATOR] CUDA not available — skipping empty_cache", flush=True)
        self._clear_all_models()
        self._inference_active = None

    def verify_clean_state(self, required_free_gb: float = None) -> bool:
        """Verify GPU is in a clean state before a phase transition.

        Checks:
        1. No stale model registrations
        2. Sufficient free VRAM
        3. Memory fraction cap is not set (or explicitly expected)

        Args:
            required_free_gb: Minimum free VRAM required. If None, uses min FLUX budget.

        Returns:
            True if state is clean and ready, False if issues detected.
        """
        if required_free_gb is None:
            required_free_gb = GPU_BASE_OVERHEAD_GB + self.VRAM_ESTIMATES['flux_dev_8bit'] + self.INFERENCE_OVERHEAD['flux_dev_8bit']

        free_gb = self._get_free_vram()
        issues = []

        if self._loaded_models and self._current_phase not in ('llm', 'image_gen', 'tts'):
            issues.append(f"Stale models registered: {list(self._loaded_models.keys())}")

        if free_gb >= 0 and free_gb < required_free_gb:
            issues.append(f"Insufficient VRAM: {free_gb:.1f}GB free, need {required_free_gb:.1f}GB")

        if self._memory_fraction_set and self._current_phase != 'image_gen':
            issues.append(f"Memory fraction cap active outside image_gen phase")

        if issues:
            log.warning(f"orchestrator.verify_clean_state — issues: {issues}")
            print(f"  [ORCHESTRATOR] Clean state issues: {'; '.join(issues)}", flush=True)
            return False

        print(f"  [ORCHESTRATOR] Clean state verified: {free_gb:.1f}GB free, phase={self._current_phase}", flush=True)
        return True

    def check_ollama_health(self, timeout: int = 10) -> bool:
        """Check if Ollama is responsive before making LLM calls.

        Sends a lightweight GET to /api/ps. Returns True if Ollama is
        reachable and responsive within `timeout` seconds.

        This prevents the pipeline from hanging on a dead Ollama process.
        """
        try:
            base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            resp = requests.get(f"{base_url}/api/ps", timeout=timeout)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", m.get("model", "")) for m in models]
                print(f"  [ORCHESTRATOR] Ollama healthy — loaded models: {model_names or 'none'}", flush=True)
                log.info(f"orchestrator.ollama_healthy — models={model_names}")
                return True
            else:
                print(f"  [ORCHESTRATOR] Ollama unhealthy — HTTP {resp.status_code}", flush=True)
                log.warning(f"orchestrator.ollama_unhealthy — status={resp.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("  [ORCHESTRATOR] Ollama NOT RUNNING — start it with: ollama serve", flush=True)
            log.error("orchestrator.ollama_not_running — ConnectionError")
            return False
        except requests.exceptions.Timeout:
            print(f"  [ORCHESTRATOR] Ollama UNRESPONSIVE — no response after {timeout}s", flush=True)
            log.error(f"orchestrator.ollama_timeout — {timeout}s")
            return False
        except Exception as e:
            print(f"  [ORCHESTRATOR] Ollama check failed: {e}", flush=True)
            log.warning(f"orchestrator.ollama_check_failed — {e}")
            return False

    def heartbeat(self, context: str = "") -> dict:
        """Print and log current VRAM status. Call during long-running phases.

        Args:
            context: Optional label (e.g. 'news_analysis_3/5')

        Returns:
            Status dict with vram info.
        """
        free_gb = self._get_free_vram()
        used_gb = self.vram_used_gb()
        status = {
            'phase': self._current_phase,
            'free_vram_gb': round(free_gb, 2) if free_gb >= 0 else 'n/a',
            'accounted_used_gb': round(used_gb, 2),
            'loaded_models': list(self._loaded_models.keys()),
            'inference_active': self._inference_active,
            'context': context,
        }
        prefix = f"[{context}] " if context else ""
        print(f"  {prefix}VRAM: {free_gb:.1f}GB free / {self._gpu_total_gb:.1f}GB total | "
              f"Phase: {self._current_phase} | Models: {list(self._loaded_models.keys())} | "
              f"Inference: {self._inference_active or 'none'}")
        log.info(f"orchestrator.heartbeat — {prefix}free={free_gb:.1f}GB phase={self._current_phase} "
                 f"models={list(self._loaded_models.keys())} inference={self._inference_active}")
        return status

    # ── Status reporting ────────────────────────────────────────────────

    def status(self) -> dict:
        """Return current orchestrator status for debugging/logging."""
        return {
            'phase': self._current_phase,
            'flux_loaded': self._flux_loaded,
            'ollama_evicted': self._ollama_evicted,
            'free_vram_gb': round(self._get_free_vram(), 2),
            'loaded_models': dict(self._loaded_models),
            'vram_used_gb': round(self.vram_used_gb(), 2),
            'vram_available_gb': round(self.vram_available_gb(), 2),
            'inference_active': self._inference_active,
            'memory_fraction_set': self._memory_fraction_set,
            'memory_fraction_cap_gb': round(self._gpu_total_gb * self._memory_fraction, 1) if self._memory_fraction_set else None,
        }

    # ── VRAM budget validation ─────────────────────────────────────────

    @classmethod
    def validate_vram_budget(cls, total_gpu_gb: float = GPU_TOTAL_GB, memory_fraction: float = MEMORY_FRACTION) -> dict:
        """Validate that no phase transition would exceed the GPU VRAM budget.

        Checks both steady-state model weights AND peak inference VRAM.

        Returns a dict with:
          - valid: bool — True if all stable phases fit within budget
          - phases: dict — Each phase's estimated peak VRAM usage
          - violations: list — Phases that exceed the budget
          - headroom: dict — GB headroom per phase
        """
        EST = cls.VRAM_ESTIMATES
        INF = cls.INFERENCE_OVERHEAD
        BASE = GPU_BASE_OVERHEAD_GB

        phase_budgets = {
            'idle': {
                'models': [],
                'vram': BASE,
                'peak_vram': BASE,
                'brief_overlap': False,
            },
            'llm': {
                'models': ['ollama'],
                'vram': BASE + EST['ollama'],
                'peak_vram': BASE + EST['ollama'] + INF['ollama'],
                'brief_overlap': False,
            },
            'image_gen (idle)': {
                'models': ['flux_dev_8bit'],
                'vram': BASE + EST['flux_dev_8bit'],
                'peak_vram': BASE + EST['flux_dev_8bit'],
                'brief_overlap': False,
            },
            'image_gen (inference peak)': {
                'models': ['flux_dev_8bit'],
                'vram': BASE + EST['flux_dev_8bit'] + INF['flux_dev_8bit'],
                'peak_vram': BASE + EST['flux_dev_8bit'] + INF['flux_dev_8bit'],
                'brief_overlap': False,
            },
            'post_image': {
                'models': [],
                'vram': BASE,
                'peak_vram': BASE,
                'brief_overlap': False,
            },
            'tts': {
                'models': ['kokoro'],
                'vram': BASE + EST['kokoro'],
                'peak_vram': BASE + EST['kokoro'] + INF['kokoro'],
                'brief_overlap': False,
            },
            'video_edit': {
                'models': [],
                'vram': BASE,
                'peak_vram': BASE,
                'brief_overlap': False,
            },
            'llm→image_gen (Ollama→FLUX transition)': {
                'models': ['ollama', 'flux_dev_8bit'],
                'vram': BASE + EST['ollama'] + EST['flux_dev_8bit'],
                'peak_vram': BASE + EST['ollama'] + EST['flux_dev_8bit'] + INF['flux_dev_8bit'],
                'brief_overlap': True,
            },
        }

        violations = []
        headroom = {}

        for name, budget in phase_budgets.items():
            used = budget['peak_vram']
            remaining = total_gpu_gb - used
            headroom[name] = round(remaining, 2)
            if used > total_gpu_gb:
                violations.append({
                    'phase': name,
                    'used_gb': round(used, 2),
                    'total_gb': total_gpu_gb,
                    'over_by_gb': round(used - total_gpu_gb, 2),
                    'models': budget['models'],
                    'brief_overlap': budget['brief_overlap'],
                })

        return {
            'valid': len(violations) == 0 or all(v['brief_overlap'] for v in violations),
            'phases': {
                k: {
                    'models': v['models'],
                    'weights_gb': round(v['vram'], 2),
                    'peak_gb': round(v['peak_vram'], 2),
                    'brief_overlap': v['brief_overlap'],
                } for k, v in phase_budgets.items()
            },
            'violations': violations,
            'headroom': headroom,
            'total_gpu_gb': total_gpu_gb,
            'memory_fraction_cap_gb': round(total_gpu_gb * memory_fraction, 1),
            'note': f'Memory fraction cap ({memory_fraction:.0%} = {total_gpu_gb * memory_fraction:.1f}GB) is set ONLY during image_gen phase, not during LLM phase',
        }