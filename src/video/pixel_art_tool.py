import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional
from src.collector.geopolitical_validator import GeopoliticalValidator

# Load environment variables from .env file
load_dotenv()

# Reduce CUDA memory fragmentation and prevent spill to system RAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Load image style config — single source of truth
_STYLE_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "image_style.json"
with open(_STYLE_CONFIG_PATH, 'r', encoding='utf-8') as _f:
    IMAGE_STYLE_CONFIG = json.load(_f)

server = FastMCP("pixel-art-tool")

FAL_KEY = os.getenv("FAL_KEY")
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PRIMARY: fal-ai/flux/dev — high quality (~$0.025/img), 28 steps
# FALLBACK: fal-ai/flux/schnell — cheapest (~$0.003/img), 4 steps
# ============================================================================

MAX_PIXELS = 1_300_000

PIXEL_ART_ENFORCEMENT_PREFIX = (
    "Clean 32-bit pixel art, high contrast news graphic, isometric style, "
    "uniform grid-aligned pixels, no anti-aliasing, no color bleeding, "
    "no soft gradients, sharp focus, detailed scene composition"
)

PIXEL_ART_STRICT_NEGATIVE = (
    "anti-aliasing, soft edges, smooth gradients, color bleeding, "
    "inconsistent pixel sizes, subpixel rendering, motion blur, "
    "depth of field, photorealistic shading"
)

IMAGE_SIZE_MAP = {
    "PinguPlay":  {"width": 512,  "height": 512},
    "PinguQuest": {"width": 768,  "height": 768},
    "PinguHero":  {"width": 1024, "height": 1024},
    "Default":    {"width": 1088, "height": 1152},
}

PIXEL_ART_MODEL = "fal-ai/flux/dev"
PIXEL_ART_MODEL_ENDPOINT = "https://fal.run/fal-ai/flux/dev"
PIXEL_ART_MODEL_CONFIG = {
    "model": "fal-ai/flux/dev",
    "api_endpoint": "https://fal.run/fal-ai/flux/dev",
    "auth_type": "key",
    "content_type": "application/json",
    "optimized_for": ["pixel_art", "isometric", "16bit", "retro_style"],
    "supports_reference": False,
    "default_params": {
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "enable_safety_checker": False,
        "output_format": "png"
    }
}

MODEL_STEP_CONFIG = {
    "fal-ai/flux/dev": 40,
    "fal-ai/flux/schnell": 4,
    "fal-ai/flux-lora": 20,
    "fal-ai/flux-pro/v1.1-ultra": 28,
}

LORA_SCALE = 0.75

FAL_MODEL = "fal-ai/flux/dev"
FAL_FALLBACK_MODELS = ["fal-ai/flux/schnell"]

USE_LOCAL_FLUX = os.getenv("USE_LOCAL_FLUX", "auto").lower() in ("true", "1", "yes", "auto")
LOCAL_FLUX_MODEL = os.getenv("LOCAL_FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
LOCAL_FLUX_FALLBACK_MODEL = "black-forest-labs/FLUX.1-schnell"
LOCAL_FLUX_QUANTIZE = os.getenv("LOCAL_FLUX_QUANTIZE", "gguf_q4ks").lower()
LOCAL_FLUX_COMPILE = os.getenv("LOCAL_FLUX_COMPILE", "none").lower()
LOCAL_FLUX_MIN_VRAM_GB = int(os.getenv("LOCAL_FLUX_MIN_VRAM_GB", "14"))
LOCAL_FLUX_EVICT_OLLAMA = os.getenv("LOCAL_FLUX_EVICT_OLLAMA", "true").lower() in ("true", "1", "yes")

_GGUF_MODEL_MAP = {
    "gguf_q2k":  ("city96/FLUX.1-dev-gguf", "flux1-dev-Q2_K.gguf"),
    "gguf_q3ks": ("city96/FLUX.1-dev-gguf", "flux1-dev-Q3_K_S.gguf"),
    "gguf_q4ks": ("city96/FLUX.1-dev-gguf", "flux1-dev-Q4_K_S.gguf"),
    "gguf_q5ks": ("city96/FLUX.1-dev-gguf", "flux1-dev-Q5_K_S.gguf"),
    "gguf_q6k":  ("city96/FLUX.1-dev-gguf", "flux1-dev-Q6_K.gguf"),
    "gguf_q8":   ("city96/FLUX.1-dev-gguf", "flux1-dev-Q8_0.gguf"),
    "gguf_f16":  ("city96/FLUX.1-dev-gguf", "flux1-dev-F16.gguf"),
}

def _resolve_gguf_path() -> str:
    repo_id, gguf_filename = _GGUF_MODEL_MAP[LOCAL_FLUX_QUANTIZE]
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=gguf_filename)

_IS_GGUF = LOCAL_FLUX_QUANTIZE in _GGUF_MODEL_MAP
_IS_BNB = LOCAL_FLUX_QUANTIZE in ("8bit", "4bit")
_KEEP_ALIVE = LOCAL_FLUX_COMPILE not in ("none", "")
_BATCH_KEEP_ALIVE = False  # Set by ModelOrchestrator during batch image generation
_flux_pipeline = None


def signal_flux_keep_alive(keep_alive: bool) -> None:
    """Signal whether the FLUX pipeline should stay loaded between image generations.
    
    Called by ModelOrchestrator at phase boundaries:
      - True before batch generation (pin pipeline for all 8 images)
      - False after batch generation (allow cleanup)
    """
    global _BATCH_KEEP_ALIVE
    _BATCH_KEEP_ALIVE = keep_alive
    effective_keep_alive = _KEEP_ALIVE or _BATCH_KEEP_ALIVE
    print(f"  [IMG] FLUX keep_alive={keep_alive} (compile={_KEEP_ALIVE}, batch={_BATCH_KEEP_ALIVE}, effective={effective_keep_alive})")


def preload_flux_pipeline() -> bool:
    """Preload the FLUX pipeline onto GPU without generating an image.
    
    Called by ModelOrchestrator at the image_generation phase boundary
    to eagerly load FLUX and verify it works before the generation loop.
    
    Returns:
        True if pipeline loaded successfully, False otherwise.
    """
    global _flux_pipeline

    if _flux_pipeline is not None:
        print("  [IMG] FLUX pipeline already loaded — skipping preload")
        return True

    if not USE_LOCAL_FLUX:
        print("  [IMG] Local FLUX disabled — skipping preload")
        return False

    try:
        import torch
        if not torch.cuda.is_available():
            print("  [IMG] CUDA not available — skipping FLUX preload")
            return False
    except ImportError:
        return False

    vram_ok, free_gb = _check_vram_available(LOCAL_FLUX_MIN_VRAM_GB)
    if not vram_ok:
        print(f"  [IMG] Insufficient VRAM for FLUX preload: {free_gb}GB free, need {LOCAL_FLUX_MIN_VRAM_GB}GB")
        return False

    print(f"  [IMG] Preloading FLUX pipeline (VRAM: {free_gb}GB free)...")

    result = _generate_local_flux(
        prompt="warmup pixel art scene, isometric perspective, test image",
        output_path=OUTPUT_DIR / "_warmup_preload.png",
        size={"width": 256, "height": 256},
        seed=42,
        steps=1,
        guidance_scale=3.5,
        negative_prompt=NEGATIVE_PROMPT,
    )

    if result and result.get('success'):
        try:
            warmup_path = OUTPUT_DIR / "_warmup_preload.png"
            if warmup_path.exists():
                warmup_path.unlink()
        except Exception:
            pass

        if _BATCH_KEEP_ALIVE:
            print("  [IMG] FLUX pipeline preloaded and PINNED for batch generation")
        else:
            print("  [IMG] FLUX pipeline preloaded (will be flushed after generation)")
        return True
    else:
        print("  [IMG] FLUX preload failed — will attempt per-image load or fall back to cloud")
        return False


def _check_vram_available(min_gb: int) -> tuple:
    """Check if enough VRAM is available on the default CUDA device.

    Returns (has_enough, free_gb) where has_enough is True when free_gb >= min_gb.
    Falls back to (True, -1) if CUDA is not yet initialised (allows first load to proceed).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0.0
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        return free_gb >= min_gb, round(free_gb, 2)
    except Exception:
        return True, -1.0


def _evict_ollama_models() -> None:
    """Tell Ollama to unload all running models from GPU, freeing VRAM.

    Models reload automatically on next LLM call. Safe to call at any time.
    """
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/ps", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            for m in models:
                model_name = m.get("name", "")
                if model_name:
                    try:
                        requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": model_name, "keep_alive": 0},
                            timeout=30,
                        )
                        print(f"  [IMG] Evicted Ollama model: {model_name}")
                    except Exception:
                        print(f"  [IMG] Failed to evict Ollama model: {model_name}")
        else:
            print(f"  [IMG] Ollama API returned status {resp.status_code} — skipping eviction")
    except requests.exceptions.ConnectionError:
        print("  [IMG] Ollama not running — no models to evict")
    except Exception as e:
        print(f"  [IMG] Could not evict Ollama models ({e}) — continuing anyway")


def _flush_flux_pipeline() -> None:
    """Manually free the FLUX pipeline from GPU memory.

    Call this after a batch of image generations when LOCAL_FLUX_COMPILE
    keeps the pipeline alive between calls. Safe to call even if no
    pipeline is loaded.
    """
    global _flux_pipeline
    import gc
    try:
        import torch
    except ImportError:
        _flux_pipeline = None
        return

    if _flux_pipeline is not None:
        print("  [IMG] Flushing FLUX pipeline — releasing GPU memory...")
        try:
            if _IS_BNB or _IS_GGUF:
                del _flux_pipeline
            else:
                _flux_pipeline.to("cpu")
                del _flux_pipeline
        except Exception:
            try:
                del _flux_pipeline
            except Exception:
                pass
        _flux_pipeline = None
        torch.cuda.empty_cache()
        gc.collect()
        print("  [IMG] FLUX pipeline flushed — GPU memory released")


def _generate_local_flux(prompt: str, output_path: Path, size: dict, seed: int,
                          steps: int, guidance_scale: float, negative_prompt: str) -> Optional[dict]:
    """Generate image using local FLUX.1-dev on GPU. Returns result dict or None."""
    global _flux_pipeline

    try:
        import torch
        if not torch.cuda.is_available():
            print("  [IMG] CUDA not available — skipping local FLUX")
            return None
    except ImportError:
        return None

    # Pre-flight VRAM check — skip if pipeline already loaded (orchestrator preloaded it)
    # or if _BATCH_KEEP_ALIVE is active (orchestrator manages VRAM lifecycle)
    if _flux_pipeline is None and not _BATCH_KEEP_ALIVE:
        vram_ok, free_gb = _check_vram_available(LOCAL_FLUX_MIN_VRAM_GB)
        if not vram_ok:
            print(f"  [IMG] Insufficient VRAM: {free_gb}GB free, need {LOCAL_FLUX_MIN_VRAM_GB}GB")
            if LOCAL_FLUX_EVICT_OLLAMA:
                print("  [IMG] Attempting to free VRAM by evicting Ollama models...")
                _evict_ollama_models()
                print("  [IMG] Waiting for GPU memory to reclaim (polling)...")
                deadline = time.time() + 15
                while time.time() < deadline:
                    time.sleep(2)
                    vram_ok, free_gb = _check_vram_available(LOCAL_FLUX_MIN_VRAM_GB)
                    print(f"  [IMG]   VRAM poll: {free_gb:.1f}GB free / {LOCAL_FLUX_MIN_VRAM_GB}GB needed")
                    if vram_ok:
                        break
            if not vram_ok:
                print(f"  [IMG] Still only {free_gb:.1f}GB free after {15}s — falling back to fal.ai")
                return None
            print(f"  [IMG] VRAM freed: {free_gb}GB now available (needed {LOCAL_FLUX_MIN_VRAM_GB}GB)")
        else:
            print(f"  [IMG] VRAM check passed: {free_gb}GB free (need {LOCAL_FLUX_MIN_VRAM_GB}GB)")
    elif _flux_pipeline is None and _BATCH_KEEP_ALIVE:
        print(f"  [IMG] Batch keep-alive active — skipping VRAM check (orchestrator manages lifecycle)")

    try:
        from diffusers import FluxPipeline

        _is_quantized = _IS_BNB
        _is_compiled = LOCAL_FLUX_COMPILE not in ("none", "")

        if _flux_pipeline is None:
            model_id = LOCAL_FLUX_MODEL
            print(f"  [IMG] Loading {model_id} pipeline onto GPU (quantize={LOCAL_FLUX_QUANTIZE}, compile={LOCAL_FLUX_COMPILE})...")

            if _IS_GGUF:
                from diffusers import GGUFQuantizationConfig, FluxTransformer2DModel

                gguf_path = _resolve_gguf_path()
                print(f"  [IMG] GGUF model: {gguf_path}")
                quant_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
                transformer = FluxTransformer2DModel.from_single_file(
                    gguf_path,
                    quantization_config=quant_config,
                    torch_dtype=torch.bfloat16,
                )
                _flux_pipeline = FluxPipeline.from_pretrained(
                    model_id,
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                )
            elif _is_quantized:
                from diffusers.quantizers import PipelineQuantizationConfig

                if LOCAL_FLUX_QUANTIZE == "8bit":
                    quant_config = PipelineQuantizationConfig(
                        quant_backend="bitsandbytes_8bit",
                        quant_kwargs={"load_in_8bit": True},
                        components_to_quantize=["transformer", "text_encoder_2"],
                    )
                else:
                    quant_config = PipelineQuantizationConfig(
                        quant_backend="bitsandbytes_4bit",
                        quant_kwargs={
                            "load_in_4bit": True,
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_compute_dtype": "bfloat16",
                        },
                        components_to_quantize=["transformer", "text_encoder_2"],
                    )

                try:
                    _flux_pipeline = FluxPipeline.from_pretrained(
                        model_id,
                        quantization_config=quant_config,
                        torch_dtype=torch.bfloat16,
                    )
                except Exception as auth_err:
                    if "401" in str(auth_err) or "access" in str(auth_err).lower() or "gated" in str(auth_err).lower():
                        print(f"  [IMG] {model_id} is gated or auth required — trying {LOCAL_FLUX_FALLBACK_MODEL}")
                        model_id = LOCAL_FLUX_FALLBACK_MODEL
                        steps = min(steps, 4)
                        _flux_pipeline = FluxPipeline.from_pretrained(
                            model_id,
                            quantization_config=quant_config,
                            torch_dtype=torch.bfloat16,
                        )
                    else:
                        raise
            else:
                try:
                    _flux_pipeline = FluxPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                    )
                except Exception as auth_err:
                    if "401" in str(auth_err) or "access" in str(auth_err).lower() or "gated" in str(auth_err).lower():
                        print(f"  [IMG] {model_id} is gated or auth required — trying {LOCAL_FLUX_FALLBACK_MODEL}")
                        model_id = LOCAL_FLUX_FALLBACK_MODEL
                        steps = min(steps, 4)
                        _flux_pipeline = FluxPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch.bfloat16,
                        )
                    else:
                        raise

            if _resolved_lora_path and LORA_SCALE:
                print(f"  [IMG] Loading LoRA: {_resolved_lora_path} (scale={LORA_SCALE})")
                _flux_pipeline.load_lora_weights(_resolved_lora_path)
                if not _IS_GGUF:
                    _flux_pipeline.fuse_lora(lora_scale=LORA_SCALE)
                else:
                    print(f"  [IMG] LoRA loaded (unfused — GGUF weights are read-only, scale applied at inference)")

            if _IS_GGUF:
                _free_vram_bytes, _ = torch.cuda.mem_get_info()
                _free_vram_gb = _free_vram_bytes / (1024 ** 3)
                if _free_vram_gb >= 12:
                    _flux_pipeline.to("cuda")
                    print(f"  [IMG] {model_id} pipeline ready on GPU (GGUF {LOCAL_FLUX_QUANTIZE}, {_free_vram_gb:.1f}GB free)")
                else:
                    _flux_pipeline.enable_model_cpu_offload()
                    print(f"  [IMG] {model_id} pipeline ready (GGUF {LOCAL_FLUX_QUANTIZE}, CPU offload enabled — {_free_vram_gb:.1f}GB free)")
            elif _is_quantized:
                for name, component in _flux_pipeline.components.items():
                    if component is not None and hasattr(component, 'to'):
                        try:
                            component.to('cuda')
                        except Exception:
                            pass
                print(f"  [IMG] {model_id} pipeline ready on GPU (quantized: {LOCAL_FLUX_QUANTIZE})")
            else:
                _flux_pipeline.to("cuda")
                print(f"  [IMG] {model_id} pipeline ready on GPU")

            try:
                _flux_pipeline.enable_xformers_memory_efficient_attention()
                print("  [IMG] xformers memory efficient attention enabled")
            except ImportError:
                print("  [IMG] xformers not installed — using PyTorch SDPA (automatic)")
            except AttributeError:
                print("  [IMG] xformers not available for this model — using default attention")
            except Exception as e:
                print(f"  [IMG] xformers unavailable ({e}) — using default attention")

            if _is_compiled:
                try:
                    torch.set_float32_matmul_precision("high")
                    _flux_pipeline.transformer = torch.compile(
                        _flux_pipeline.transformer,
                        mode=LOCAL_FLUX_COMPILE,
                        fullgraph=True,
                    )
                    print(f"  [IMG] transformer compiled (mode={LOCAL_FLUX_COMPILE})")
                except Exception as compile_err:
                    print(f"  [IMG] torch.compile failed ({compile_err}) — continuing uncompiled")
                    _is_compiled = False

                if _is_compiled:
                    print("  [IMG] Running warmup inference to trigger compilation...")
                    try:
                        warmup_size = {"height": 256, "width": 256}
                        _warmup_kwargs = {
                            "prompt": "warmup",
                            "num_inference_steps": 1,
                            "guidance_scale": 1.0,
                            **warmup_size,
                        }
                        _flux_pipeline(**_warmup_kwargs)
                        print("  [IMG] Warmup complete — compiled kernels cached")
                    except Exception as warmup_err:
                        print(f"  [IMG] Warmup failed ({warmup_err}) — first real generation will be slower")

        w, h = size["width"], size["height"]
        print(f"  [IMG] LOCAL flux/dev | {w}x{h} | steps={steps} | guidance={guidance_scale} | seed={seed} | quantize={LOCAL_FLUX_QUANTIZE} | compile={LOCAL_FLUX_COMPILE}")

        gen_kwargs = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }
        if _IS_GGUF and _resolved_lora_path and LORA_SCALE:
            gen_kwargs["joint_attention_kwargs"] = {"scale": LORA_SCALE}
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
            gen_kwargs["generator"] = generator

        if _IS_GGUF:
            image = _flux_pipeline(**gen_kwargs).images[0]
        else:
            _cudnn_was_enabled = torch.backends.cudnn.enabled
            try:
                torch.backends.cudnn.enabled = False
                image = _flux_pipeline(**gen_kwargs).images[0]
            finally:
                torch.backends.cudnn.enabled = _cudnn_was_enabled
        image.save(str(output_path))
        print(f"  [IMG] Local FLUX generation complete: {output_path.name}")

        if _KEEP_ALIVE or _BATCH_KEEP_ALIVE:
            effective_reason = "compile mode" if _KEEP_ALIVE else "batch mode (orchestrator)"
            print(f"  [IMG] Pipeline kept alive ({effective_reason}) — VRAM still in use, call _flush_flux_pipeline() to release")
        elif _is_quantized or _IS_GGUF:
            del _flux_pipeline
            _flux_pipeline = None
            torch.cuda.empty_cache()
            print("  [IMG] FLUX pipeline offloaded — GPU memory freed")
        else:
            _flux_pipeline.to("cpu")
            del _flux_pipeline
            _flux_pipeline = None
            torch.cuda.empty_cache()
            print("  [IMG] FLUX pipeline offloaded — GPU memory freed")

        try:
            _upscale_pixel_art(str(output_path))
        except Exception as upscale_err:
            print(f"  [IMG] Warning: Upscale failed: {upscale_err}")

        is_failed, fail_reason = _detect_failed_image(str(output_path))
        if is_failed:
            print(f"  [IMG] Detected failed image: {fail_reason}")
            return {
                "success": True,
                "filename": output_path.name,
                "path": str(output_path),
                "prompt_used": prompt,
                "source": "local_flux",
                "provider": "local",
                "width": w,
                "height": h,
                "steps": steps,
                "detected_failure": fail_reason,
            }

        return {
            "success": True,
            "filename": output_path.name,
            "path": str(output_path),
            "prompt_used": prompt,
            "source": "local_flux",
            "provider": "local",
            "width": w,
            "height": h,
            "steps": steps,
        }

    except torch.cuda.OutOfMemoryError:
        print("  [IMG] CUDA OOM — FLUX too large for available VRAM, falling back to cloud API")
        if _flux_pipeline is not None:
            try:
                if not _is_quantized and not _IS_GGUF:
                    _flux_pipeline.to("cpu")
                del _flux_pipeline
            except Exception:
                pass
        _flux_pipeline = None
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"  [IMG] Local FLUX failed: {type(e).__name__}: {e}")
        if _flux_pipeline is not None:
            try:
                del _flux_pipeline
            except Exception:
                pass
        _flux_pipeline = None
        return None


# Prompt hash cache — avoids regenerating identical/near-identical scenes
_prompt_cache: Dict[str, str] = {}


def _check_prompt_cache(prompt: str) -> Optional[str]:
    """Check if we already generated an image for this exact prompt. Returns path or None."""
    import hashlib
    cache_key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
    return _prompt_cache.get(cache_key)


def _store_prompt_cache(prompt: str, image_path: str) -> None:
    """Store a prompt→image mapping in the cache."""
    import hashlib
    cache_key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
    _prompt_cache[cache_key] = image_path


def _upscale_pixel_art(input_path: str, render_size: tuple = None,
                       target_size: tuple = None) -> str:
    """
    Upscale a small pixel-art image using nearest-neighbor interpolation.
    This preserves hard pixel edges — the key to TRUE pixel art vs blurry faux-pixel.
    SKIPS upscale when the image is already at or above target resolution.
    """
    if render_size is None:
        render_size = tuple(GENERATION_PARAMS.get('render_resolution', [256, 256]))
    if target_size is None:
        target_size = tuple(GENERATION_PARAMS.get('target_resolution', [1024, 1792]))
    img = Image.open(input_path)
    img_w, img_h = img.size

    if img_w >= target_size[0] and img_h >= target_size[1]:
        img.close()
        print(f"  [IMG] Skipping upscale — already at {img_w}x{img_h} >= {target_size[0]}x{target_size[1]}")
        return input_path

    if img.size != render_size:
        img = img.resize(render_size, Image.NEAREST)
    upscaled = img.resize(target_size, Image.NEAREST)
    upscaled.save(input_path, format='PNG')
    print(f"  [IMG] Upscaled {render_size} -> {target_size} (nearest-neighbor)")
    try:
        _apply_sharpening(input_path)
    except Exception as sharpen_err:
        print(f"  [IMG] Warning: Post-sharpening failed: {sharpen_err}")
    return input_path


def _apply_sharpening(input_path: str) -> str:
    """
    Apply unsharp mask sharpening to an upscaled pixel-art image.
    Enhances edge crispness without sacrificing the pixel art aesthetic.
    """
    try:
        from PIL import ImageFilter
        img = Image.open(input_path)
        sharpened = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        sharpened.save(input_path, format='PNG')
        print(f"  [IMG] Applied unsharp mask sharpening")
        return input_path
    except Exception as e:
        print(f"  [IMG] Warning: Sharpening failed: {e}")
        return input_path


def _detect_failed_image(image_path: str) -> tuple:
    """
    Detect visually failed images: solid-color, near-monochrome, or extremely low variance.
    Returns (is_failed: bool, reason: str).
    is_failed=True means the image is likely a failed generation (not usable).
    """
    try:
        import numpy as np
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)

        std_val = arr.std()

        if std_val < 8.0:
            return True, f"near_monochrome (std={std_val:.1f})"

        r_std = arr[:, :, 0].std()
        g_std = arr[:, :, 1].std()
        b_std = arr[:, :, 2].std()
        if r_std < 5.0 and g_std < 5.0 and b_std < 5.0:
            return True, f"flat_color (r_std={r_std:.1f}, g_std={g_std:.1f}, b_std={b_std:.1f})"

        h, w = arr.shape[:2]
        corners = [
            arr[:h//8, :w//8],
            arr[:h//8, -w//8:],
            arr[-h//8:, :w//8],
            arr[-h//8:, -w//8:],
        ]
        corner_means = [c.mean() for c in corners]
        corner_spread = max(corner_means) - min(corner_means)
        if corner_spread < 3.0 and std_val < 15.0:
            return True, f"uniform_color (corner_spread={corner_spread:.1f}, std={std_val:.1f})"

        edge_row_diff = np.abs(np.diff(arr[h//2, :, 0], axis=0)).mean()
        edge_col_diff = np.abs(np.diff(arr[:, w//2, 0], axis=0)).mean()
        if edge_row_diff < 3.0 and edge_col_diff < 3.0 and std_val < 25.0:
            return True, f"low_detail (edge_row={edge_row_diff:.1f}, edge_col={edge_col_diff:.1f}, std={std_val:.1f})"

        return False, "ok"
    except Exception as e:
        print(f"  [IMG] Warning: Failed image detection error: {e}")
        return False, f"detection_error ({e})"


def _get_pixel_art_model_headers() -> Dict[str, str]:
    if not FAL_KEY:
        raise ValueError("FAL_KEY environment variable not set")
    return {
        "Authorization": f"Key {FAL_KEY}",
        "Content-Type": "application/json"
    }

# LoRA configuration — loaded from config, with custom LoRA override support
_CUSTOM_LORA_PATH = Path(__file__).parent.parent.parent / "config" / "custom_lora.json"
_custom_lora = None
if _CUSTOM_LORA_PATH.exists():
    with open(_CUSTOM_LORA_PATH, 'r', encoding='utf-8') as _f:
        _custom_lora = json.load(_f)
    print(f"  [IMG] Custom LoRA loaded: {_custom_lora.get('trigger_word', 'N/A')}")

_lora_defaults = IMAGE_STYLE_CONFIG.get('lora_defaults', {})


def _resolve_lora_path(custom_lora: dict) -> str:
    """
    Resolve the effective LoRA path for fal.ai.
    fal.ai requires a URL or HuggingFace repo ID — it cannot load local files directly.
    If the stored path is a local .safetensors file, upload it to fal.ai CDN once
    and cache the returned URL back into custom_lora.json.
    """
    lora_url = custom_lora.get('lora_url', '')
    hub_url = custom_lora.get('hub_url')

    # Prefer HuggingFace Hub URL if available (permanent, fastest)
    if hub_url and hub_url.startswith('https://huggingface.co'):
        return hub_url

    # If it's already an HTTPS URL, use directly
    if lora_url.startswith('https://') or lora_url.startswith('http://'):
        return lora_url

    # If it's a HuggingFace repo ID (e.g. "username/sentinel-pixel-lora")
    if '/' in lora_url and not lora_url.startswith('/') and not ':' in lora_url:
        return lora_url

    # Local file path — upload to fal.ai CDN and cache the URL
    local_path = Path(lora_url) if lora_url else None
    if not local_path:
        local_path_str = custom_lora.get('lora_local_path', '')
        local_path = Path(local_path_str) if local_path_str else None

    if local_path and local_path.exists() and local_path.suffix == '.safetensors':
        try:
            import fal_client
            print(f"  [IMG] Uploading local LoRA to fal.ai CDN: {local_path.name}")
            cdn_url = fal_client.upload_file(str(local_path))
            print(f"  [IMG] LoRA CDN URL: {cdn_url}")
            # Cache URL back so we don't re-upload every run
            custom_lora['lora_url'] = cdn_url
            _CUSTOM_LORA_PATH.write_text(
                json.dumps(custom_lora, indent=2), encoding='utf-8'
            )
            return cdn_url
        except Exception as e:
            print(f"  [IMG] WARNING: Could not upload LoRA to CDN: {e}")
            print(f"  [IMG] Falling back to default HuggingFace LoRA")

    # Fallback to default
    return _lora_defaults.get('path', 'prithivMLmods/Retro-Pixel-Flux-LoRA')


_resolved_lora_path = (
    _resolve_lora_path(_custom_lora) if _custom_lora
    else _lora_defaults.get('path', 'prithivMLmods/Retro-Pixel-Flux-LoRA')
)

PIXEL_ART_LORA = {
    "path": _resolved_lora_path,
    "scale": _lora_defaults.get('scale', 0.85)
}

# ============================================================================
# PHASE 3: Accuracy Refinement - Strength, Noise, and Guidance Parameters
# ============================================================================

# Image-to-Image accuracy refinement configuration
# These parameters control how closely output matches reference vs prompt
I2I_REFINEMENT_CONFIG = {
    # Strength: How much to transform the reference image (0.0 = unchanged, 1.0 = completely new)
    "strength": {
        "low": 0.45,      # Subtle changes, keeps reference structure
        "medium": 0.75,   # Balanced transformation (default)
        "high": 0.95,     # Major transformation, keeps only composition
        "default": 0.75
    },
    
    # Guidance Scale: How closely to follow the text prompt (higher = more prompt adherence)
    "guidance_scale": {
        "low": 2.0,       # More creative freedom, less prompt adherence
        "medium": 3.0,    # Balanced (default)
        "high": 5.0,      # Strict prompt adherence
        "pixel_art_optimized": 3.5,  # Optimized for pixel-art detail preservation
        "default": 3.0
    },
    
    # Inference Steps: Quality vs speed tradeoff (more steps = higher quality, slower)
    "num_inference_steps": {
        "fast": 20,       # Quick generation, acceptable quality
        "balanced": 35,   # Good quality/speed balance (default)
        "quality": 50,    # Maximum quality, slower
        "pixel_art_max": 45,  # Optimized for pixel-art clarity
        "default": 35
    },
    
    # Control modes for different generation scenarios
    "control_modes": {
        "strict_reference": {
            "strength": 0.45,
            "guidance_scale": 2.5,
            "description": "Keep reference image mostly intact, subtle pixel-art styling"
        },
        "balanced": {
            "strength": 0.75,
            "guidance_scale": 3.0,
            "description": "Balanced reference influence and prompt adherence (default)"
        },
        "prompt_dominant": {
            "strength": 0.95,
            "guidance_scale": 5.0,
            "description": "Follow prompt closely, use reference for composition only"
        },
        "pixel_art_precise": {
            "strength": 0.70,
            "guidance_scale": 3.5,
            "num_inference_steps": 45,
            "description": "Optimized for pixel-art accuracy with reference guidance"
        }
    }
}


def _get_refinement_params(
    mode: str = "balanced",
    custom_strength: Optional[float] = None,
    custom_guidance: Optional[float] = None,
    custom_steps: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get accuracy refinement parameters for Image-to-Image generation.
    
    Args:
        mode: Predefined control mode ("strict_reference", "balanced", "prompt_dominant", "pixel_art_precise")
        custom_strength: Override strength value (0.0-1.0)
        custom_guidance: Override guidance scale value
        custom_steps: Override inference steps
        
    Returns:
        Dict with strength, guidance_scale, and num_inference_steps
    """
    # Get base params from control mode
    mode_config = I2I_REFINEMENT_CONFIG["control_modes"].get(mode, I2I_REFINEMENT_CONFIG["control_modes"]["balanced"])
    
    params = {
        "strength": custom_strength if custom_strength is not None else mode_config.get("strength", I2I_REFINEMENT_CONFIG["strength"]["default"]),
        "guidance_scale": custom_guidance if custom_guidance is not None else mode_config.get("guidance_scale", I2I_REFINEMENT_CONFIG["guidance_scale"]["default"]),
        "num_inference_steps": custom_steps if custom_steps is not None else mode_config.get("num_inference_steps", I2I_REFINEMENT_CONFIG["num_inference_steps"]["default"]),
        "mode": mode,
        "mode_description": mode_config.get("description", "")
    }
    
    # Validate ranges
    params["strength"] = max(0.0, min(1.0, params["strength"]))
    params["guidance_scale"] = max(1.0, min(20.0, params["guidance_scale"]))
    params["num_inference_steps"] = max(1, min(100, params["num_inference_steps"]))
    
    return params


def _print_refinement_settings(params: Dict[str, Any]) -> None:
    """Print current refinement settings for debugging."""
    print(f"  [IMG] I2I Refinement: {params['mode']} - {params['mode_description']}")
    print(f"  [IMG]   Strength: {params['strength']:.2f} (transform amount)")
    print(f"  [IMG]   Guidance: {params['guidance_scale']:.1f} (prompt adherence)")
    print(f"  [IMG]   Steps: {params['num_inference_steps']} (quality level)")


# Build STYLE_LORAS from config
_lora_by_type = IMAGE_STYLE_CONFIG.get('lora_by_visual_type', {})
STYLE_LORAS = {}
for _vtype, _vcfg in _lora_by_type.items():
    STYLE_LORAS[_vtype] = {
        'path': _resolved_lora_path,
        'scale': _vcfg.get('scale', 0.85),
        'trigger': _custom_lora['trigger_word'] if _custom_lora else _vcfg.get('trigger', 'Retro Pixel'),
        'additional_prompts': _vcfg.get('additional_prompts', '')
    }

BRAND_COLORS = IMAGE_STYLE_CONFIG.get('brand_colors', {})

STYLE_SUFFIX = IMAGE_STYLE_CONFIG.get('style_suffix', 'Retro Pixel, true 16-bit pixel art, retro SNES style, isometric perspective, hard pixel edges, limited color palette, detailed proportions, flat colors, dramatic lighting')
CLIP_STYLE_TAG = IMAGE_STYLE_CONFIG.get('clip_style_tag', '16-bit isometric pixel art, isometric perspective, vibrant colors')
COLOR_PALETTE_PROMPT = IMAGE_STYLE_CONFIG.get('color_palette_prompt', '')
GENERATION_PARAMS = IMAGE_STYLE_CONFIG.get('generation_params', {})

# Render-small-then-upscale for TRUE pixel art (must be after GENERATION_PARAMS)
RENDER_RESOLUTION = tuple(GENERATION_PARAMS.get('render_resolution', [256, 256]))
TARGET_RESOLUTION = tuple(GENERATION_PARAMS.get('target_resolution', [1024, 1792]))
UPSCALE_METHOD = GENERATION_PARAMS.get('upscale_method', 'nearest')

# Phase 5.1: Negative prompt — loaded from config, sent to FAL API
NEGATIVE_PROMPT = IMAGE_STYLE_CONFIG.get('negative_prompt',
    "text, words, letters, watermark, signature, logo, UI elements, HUD, "
    "speech bubbles, captions, subtitles, labels, annotations, "
    "blurry, low quality, jpeg artifacts, pixelated noise, distorted proportions, "
    "photorealistic, 3D render, CGI, anime style, cartoon, sketch, pencil drawing, "
    "nsfw, gore, violence close-up, human faces distorted"
)

# Phase 5.1: Quality check keywords — prompt must contain enough specificity
MIN_SPECIFICITY_WORDS = 4  # Below this, prompt is flagged as too generic

# FAL.ai content-policy safe substitutions - MULTI-LAYER DEFENSE
# Layer 1: Extreme terms (always applied via _sanitize_prompt_for_api)
# Layer 2: Named people + military actions (applied via _progressive_content_scrub level 1)
# Layer 3: Visual-only extraction (applied via _progressive_content_scrub level 2)
_SAFE_SUBSTITUTIONS = [
    # Extreme/violent terms
    (r'\bgore\b',                     'aftermath'),
    (r'\bblood(?:y|ied)?\b',          'impact scene'),
    (r'\bcasualt(?:y|ies)\b',         'aftermath scene'),
    (r'\bdead bodies\b',              'aftermath scene'),
    (r'\bcorpse(?:s)?\b',             'aftermath scene'),
    (r'\bexecution(?:s)?\b',          'confrontation'),
    (r'\bmassacre\b',                 'aftermath scene'),
    (r'\bgenocide\b',                 'aftermath scene'),
    (r'\btortur(?:e|ed|ing)\b',       'confrontation'),
    (r'\bmass\s+grave\b',             'memorial scene'),
    (r'\bbody\s+bag(?:s)?\b',         'aftermath scene'),
    (r'\bdeath(?:s)?\b',              'aftermath'),
    (r'\bdyi(?:ng|ed)\b',             'fallen'),
    (r'\bwounded\b',                  'affected'),
    (r'\bdestroyed\b',                'damaged'),
    (r'\bdestroy(?:ing)?\b',          'damaging'),
    (r'\bkill(?:ed|ing|er)?\b',       'affected'),
    (r'\bmurd(?:er|ered|ering)\b',    'confrontation'),
    (r'\bassassina\w+\b',             'confrontation'),
    (r'\bsuicide\b',                  'incident'),
    (r'\bterroris\w+\b',             'militant'),
    (r'\bexplosi\w+\b',              'detonation scene'),
    (r'\bnuclear\s+weapon(?:s)?\b',   'strategic installation'),
    (r'\bweapon(?:s)?\b',             'equipment'),
    (r'\barmed\b',                    'equipped'),
]

# Layer 2 substitutions — named people + military actions (applied on retry)
_NAMED_ENTITY_SUBSTITUTIONS = [
    # Political figures
    (r'\bTrump\b',                    'a president'),
    (r'\bPutin\b',                    'a world leader'),
    (r'\bXi(?:\s+Jinping)?\b',        'a world leader'),
    (r'\b[Zz]elenskyy?\b',            'a president'),
    (r'\bBiden\b',                    'a president'),
    (r'\bNetanyahu\b',                'a prime minister'),
    (r'\bKim\s+Jong\s+Un\b',         'a leader'),
    (r'\bJD\s+Vance\b',              'a vice president'),
    (r'\bEric\s+Cheng\b',            'a politician'),
    (r'\bScholz\b',                   'a chancellor'),
    (r'\bMacron\b',                   'a president'),
    (r'\bStarmer\b',                  'a prime minister'),
    (r'\bModi\b',                     'a prime minister'),
    # Military action softening
    (r'\bwarship(?:s)?\b',            'naval vessel'),
    (r'\bwar\b',                      'conflict'),
    (r'\bmissile(?:s)?\b',            'equipment'),
    (r'\bdrone\s+strike(?:s)?\b',     'aerial operation'),
    (r'\battack(?:ing|ed|s)?\b',      'maneuver'),
    (r'\binvasi\w+\b',               'advance'),
    (r'\bstrike(?:s)?\b',             'operation'),
    (r'\bbomb(?:ing|ed|s)?\b',        'operation'),
    (r'\bshoot(?:ing|s|er)?\b',       'confrontation'),
    (r'\bbl[oe]w\w*\b',              'struck'),
    (r'\bfire(?:d|s)?\b',            'launched'),
    (r'\bcombat\b',                   'engagement'),
    (r'\bbattle(?:s|field)?\b',       'engagement zone'),
    (r'\boffensive\b',               'operation'),
    (r'\bsiege\b',                    'blockade'),
    (r'\bfrontline\b',               'front'),
    (r'\bammunit\w+\b',              'supplies'),
]

# Pre-approved safe prompts per visual category — last resort before placeholder
_CATEGORY_SAFE_PROMPTS = {
    'warfare': '16-bit isometric pixel art scene: tactical landscape with smoke in the distance, military terrain at dusk, dramatic orange sky, trenches and fortifications, atmospheric haze',
    'naval': '16-bit isometric pixel art scene: ocean harbor with docked vessels at sunset, calm waves reflecting orange sky, maritime flags on masts, nautical atmosphere',
    'aerial': '16-bit isometric pixel art scene: open sky with cloud formations at golden hour, distant contrails, dramatic sunset colors of orange and purple, atmospheric depth',
    'arms_defense': '16-bit isometric pixel art scene: military installation with radar equipment at dusk, technical markings visible, national insignia on structures, moody lighting',
    'markets': '16-bit isometric pixel art scene: trading floor with screens and displays, financial charts in blue and green, professional atmosphere, dramatic lighting',
    'trade_sanctions': '16-bit isometric pixel art scene: cargo port with container stacks and cranes, ships docked at harbor, industrial atmosphere, overcast sky',
    'energy': '16-bit isometric pixel art scene: industrial complex with smokestacks and flames at night, pipeline infrastructure, orange glow on horizon, moody atmosphere',
    'commodities': '16-bit isometric pixel art scene: warehouse with supply crates and storage, resource materials stacked, industrial lighting, scarcity atmosphere',
    'diplomacy': '16-bit isometric pixel art scene: grand meeting hall with flags of many nations, formal table setting, chandeliers, diplomatic atmosphere, warm lighting',
    'political': '16-bit isometric pixel art scene: government building interior with marble columns, parliamentary seating, formal atmosphere, warm golden lighting',
    'espionage': '16-bit isometric pixel art scene: dark office with surveillance screens, classified documents on desk, shadows and moody lighting, mystery atmosphere',
    'protests': '16-bit isometric pixel art scene: city square with crowds and banners, buildings in background, dramatic sky, urban atmosphere',
    'humanitarian': '16-bit isometric pixel art scene: refugee camp with tents and supplies, mountain backdrop, humanitarian atmosphere, dramatic sunset lighting',
    'border': '16-bit isometric pixel art scene: border checkpoint with fence stretching to horizon, guard towers in distance, dramatic sky, frontier atmosphere',
    'cyber': '16-bit isometric pixel art scene: server room with glowing blue screens, digital data visualization, holographic displays, cyber atmosphere',
    'megaprojects': '16-bit isometric pixel art scene: massive construction site with cranes and scaffolding, engineering scale, dramatic lighting, industrial atmosphere',
    'general': '16-bit isometric pixel art scene: dramatic landscape at golden hour, strategic overview perspective, balanced composition, atmospheric lighting',
}


def _sanitize_prompt_for_api(prompt: str) -> str:
    """
    Replace terms that trigger FAL.ai content policy with safe visual equivalents.
    Preserves the visual intent while avoiding safety checker rejections.
    """
    import re
    result = prompt
    for pattern, replacement in _SAFE_SUBSTITUTIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _sanitize_visual_prompt(prompt: str) -> str:
    """
    Light content scrubbing for VISUAL prompts (image generation).
    Only strips extreme/gore terms that FAL.ai will reject — keeps country
    names, equipment names, and military terms that make images RELEVANT.
    Visual prompts describe scenes, not claims, so geographic and equipment
    specificity is essential for image-video alignment.
    """
    import re
    _VISUAL_SAFE_ONLY = [
        (r'\bgore\b', 'aftermath'),
        (r'\bblood(?:y|ied)?\b', 'impact scene'),
        (r'\bdead bodies\b', 'aftermath scene'),
        (r'\bcorpse(?:s)?\b', 'aftermath scene'),
        (r'\bmassacre\b', 'aftermath scene'),
        (r'\bgenocide\b', 'aftermath scene'),
        (r'\btortur(?:e|ed|ing)\b', 'confrontation'),
        (r'\bmass\s+grave\b', 'memorial scene'),
        (r'\bbody\s+bag(?:s)?\b', 'aftermath scene'),
        (r'\bsuicide\b', 'incident'),
        (r'\bnsfw\b', 'scene'),
        (r'\bnude\b', 'scene'),
    ]
    result = prompt
    for pattern, replacement in _VISUAL_SAFE_ONLY:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _progressive_content_scrub(prompt: str, visual_type: str, level: int = 1) -> str:
    """
    Progressively scrub a prompt to bypass FAL.ai content policy rejections.
    
    Level 1: Remove named people + soften military actions
    Level 2: Extract visual-only terms (strip all narrative, keep colors/shapes/lighting)
    Level 3: Use pre-approved category-safe prompt (last resort)
    
    Args:
        prompt: The original prompt that was rejected
        visual_type: Detected visual category (warfare, naval, etc.)
        level: Scrubbing aggressiveness (1-3)
    
    Returns:
        Scrubbed prompt string
    """
    import re
    
    if level >= 1:
        # Level 1: Named entity removal + military action softening
        result = prompt
        for pattern, replacement in _NAMED_ENTITY_SUBSTITUTIONS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        # Also apply L1 substitutions
        for pattern, replacement in _SAFE_SUBSTITUTIONS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        print(f"  [IMG] 🧹 Content scrub level 1: removed named entities + softened military terms")
        if level == 1:
            return result
    
    if level >= 2:
        # Level 2: Visual-only extraction — strip narrative, keep only visual elements
        # Extract: colors, lighting, perspective, composition, architecture, atmosphere
        visual_keywords = [
            'pixel art', 'isometric', '16-bit', 'snes', 'retro', 'dramatic',
            'sunset', 'dawn', 'dusk', 'night', 'golden hour', 'overcast',
            'orange', 'blue', 'red', 'green', 'grey', 'dark', 'warm', 'cool',
            'smoke', 'fire', 'fog', 'haze', 'clouds', 'storm', 'rain',
            'foreground', 'midground', 'background', 'perspective',
            'silhouette', 'horizon', 'skyline', 'landscape', 'terrain',
            'urban', 'industrial', 'harbor', 'ocean', 'mountain', 'desert',
            'forest', 'city', 'building', 'structure', 'vehicle', 'vessel',
            'flag', 'uniform', 'helmet', 'equipment', 'radar', 'screen',
            'table', 'chair', 'document', 'crane', 'tower', 'fence',
            'lighting', 'atmosphere', 'composition', 'shadows', 'glow',
            'reflection', 'waves', 'trenches', 'fortifications',
        ]
        
        # Rebuild prompt preserving only visual terms
        words = re.findall(r'\b\w+\b', prompt.lower())
        preserved = [w for w in words if any(vk.startswith(w) or w in vk for vk in visual_keywords)]
        
        # Build a visual-only prompt with the style suffix
        if preserved:
            visual_fragment = ', '.join(set(preserved[:15]))
            result = f"16-bit isometric pixel art scene: {visual_fragment}"
        else:
            result = _CATEGORY_SAFE_PROMPTS.get(visual_type, _CATEGORY_SAFE_PROMPTS['general'])
        
        print(f"  [IMG] 🧹 Content scrub level 2: visual-only extraction")
        if level == 2:
            return result
    
    if level >= 3:
        # Level 3: Pre-approved safe prompt — guaranteed to pass
        result = _CATEGORY_SAFE_PROMPTS.get(visual_type, _CATEGORY_SAFE_PROMPTS['general'])
        print(f"  [IMG] 🧹 Content scrub level 3: category-safe fallback ({visual_type})")
        return result
    
    return result


def _detect_visual_type(prompt: str) -> str:
    """
    16-category geopolitical visual type detection.
    Returns one of: warfare, naval, aerial, arms_defense, markets,
    trade_sanctions, energy, commodities, diplomacy, political,
    espionage, protests, humanitarian, border, cyber, megaprojects, general
    """
    prompt_lower = prompt.lower()

    CATEGORY_KEYWORDS = {
        'warfare': [
            'troops', 'infantry', 'ground forces', 'invasion', 'artillery',
            'tank', 'battle', 'frontline', 'combat', 'offensive', 'siege',
            'ground war', 'soldiers', 'march', 'advancing', 'retreating',
            'firefight', 'explosion', 'bombing', 'smoke', 'battlefield',
            'military', 'forces', 'strike', 'attack', 'war', 'weapon',
            'deployment', 'f-35', 'f-16', 's-400', 'ah-64', 'mq-9', 'uav'
        ],
        'naval': [
            'warship', 'destroyer', 'frigate', 'carrier', 'submarine',
            'naval', 'fleet', 'navy', 'strait', 'hormuz', 'blockade',
            'patrol boat', 'convoy', 'ship', 'vessel', 'port', 'harbor',
            'maritime', 'sealane', 'ocean', 'sea', 'flotilla'
        ],
        'aerial': [
            'aircraft', 'fighter jet', 'drone', 'airstrike', 'bomber',
            'air force', 'jet', 'plane', 'helicopter', 'sortie',
            'aerial', 'air defense', 'missile intercept', 'contrail',
            'uav', 'stealth', 'airspace', 'no-fly zone', 'sky', 'squadron'
        ],
        'arms_defense': [
            'missile defense', 'iron dome', 'patriot', 's-300', 's-400',
            'missile', 'defense system', 'radar', 'military hardware',
            'weapons expo', 'arms deal', 'defense contract', 'nuclear',
            'ballistic', 'hypersonic', 'warhead', 'arsenal', 'munitions'
        ],
        'markets': [
            'stock', 'trading floor', 'wall street', 'market crash',
            'ticker', 'traders', 'dow', 'nasdaq', 's&p', 'bear market',
            'bull market', 'portfolio', 'hedge fund', 'market panic',
            'price', 'economy', 'market', 'inflation', 'trading',
            'financial', 'economic', 'revenue', 'budget'
        ],
        'trade_sanctions': [
            'sanctions', 'tariff', 'trade war', 'embargo', 'import ban',
            'export control', 'cargo', 'container', 'shipping', 'port congestion',
            'customs', 'trade barrier', 'duties', 'supply chain', 'semiconductor ban',
            'oil ban', 'swift', 'decoupling'
        ],
        'energy': [
            'oil', 'gas', 'pipeline', 'refinery', 'lng', 'natural gas',
            'opec', 'energy', 'petroleum', 'crude', 'shale', 'fracking',
            'nuclear plant', 'power grid', 'electricity', 'fuel', 'diesel',
            'petrol', 'energy crisis', 'blackout', 'power plant'
        ],
        'commodities': [
            'wheat', 'grain', 'food shortage', 'famine', 'bread lines',
            'commodity', 'gold', 'copper', 'lithium', 'rare earth',
            'mining', 'agricultural', 'crop failure', 'drought',
            'supply shortage', 'queue', 'shelves', 'rationing'
        ],
        'diplomacy': [
            'diplomatic', 'summit', 'treaty', 'negotiation', 'agreement',
            'minister', 'ambassador', 'embassy', 'talks', 'officials',
            'foreign', 'envoy', 'delegation', 'ceasefire', 'accord',
            'handshake', 'meeting', 'peace deal', 'diplomacy', 'joint statement'
        ],
        'political': [
            'election', 'parliament', 'vote', 'president', 'prime minister',
            'congress', 'senate', 'government', 'regime change', 'coup',
            'political', 'opposition', 'rally', 'campaign', 'inauguration',
            'resignation', 'impeachment', 'ballot', 'podium', 'legislature'
        ],
        'espionage': [
            'spy', 'espionage', 'intelligence', 'surveillance', 'cia',
            'mi6', 'fsb', 'mossad', 'intercept', 'classified', 'leak',
            'whistleblower', 'cyber espionage', 'wiretap', 'agent',
            'covert', 'undercover', 'espionage', 'secret service'
        ],
        'protests': [
            'protest', 'demonstration', 'riot', 'unrest', 'march',
            'crowd', 'banners', 'signs', 'police crackdown', 'tear gas',
            'occupation', 'sit-in', 'strike', 'uprising', 'revolt',
            'civil disobedience', 'clash with police', 'barricades'
        ],
        'humanitarian': [
            'refugees', 'displaced', 'humanitarian', 'aid', 'un camp',
            'fleeing', 'evacuation', 'rescue', 'relief', 'crisis',
            'civilian', 'families', 'victims', 'shelter', 'tent city',
            'convoy', 'medical', 'hospital', 'wounded', 'children'
        ],
        'border': [
            'border', 'checkpoint', 'crossing', 'wall', 'fence',
            'frontier', 'immigration', 'migrant', 'deportation',
            'no-mans land', 'guard tower', 'territory', 'annexation',
            'demilitarized zone', 'dmz', 'buffer zone', 'border patrol'
        ],
        'cyber': [
            'cyber', 'hack', 'ransomware', 'data breach', 'server',
            'network', 'digital', 'online', 'internet shutdown',
            'firewall', 'malware', 'phishing', 'ddos', 'encryption',
            'code', 'holographic', 'virtual', 'ai-powered'
        ],
        'megaprojects': [
            'belt and road', 'canal', 'megaproject', 'construction',
            'infrastructure', 'high-speed rail', 'bridge', 'dam',
            'tunnel', 'crane', 'building', 'development project',
            'port expansion', 'airport', 'smart city', 'industrial zone'
        ],
    }

    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return 'general'

    max_score = max(scores.values())
    winners = [cat for cat, s in scores.items() if s == max_score]

    if len(winners) == 1:
        return winners[0]

    # Tiebreaker: prefer more specific categories over general ones
    specificity_order = [
        'warfare', 'naval', 'aerial', 'arms_defense',
        'espionage', 'cyber', 'protests', 'humanitarian', 'border',
        'energy', 'commodities', 'trade_sanctions', 'markets',
        'diplomacy', 'political', 'megaprojects'
    ]
    for cat in specificity_order:
        if cat in winners:
            return cat

    return winners[0]


def _get_adaptive_enrichment(visual_type: str) -> str:
    """
    Get adaptive enrichment text based on visual type.
    Supports all 16 geopolitical visual categories with per-category emphasis.
    """
    enrichments = {
        'warfare': 'tactical positioning, battlefield terrain, smoke and fire, military tension',
        'naval': 'naval formation, ocean waves, maritime flags, fleet maneuvers',
        'aerial': 'altitude perspective, contrails visible, sky domination, payload detail',
        'arms_defense': 'hardware detail, technical markings, defense system, national insignia',
        'markets': 'market indicators visible, price displays, trading floor, financial panic',
        'trade_sanctions': 'cargo containers, port bottleneck, trade barrier, nation flags',
        'energy': 'industrial infrastructure, flames and smoke, pipeline terrain, workers',
        'commodities': 'resource scarcity, supply queue, raw materials, shortage indicators',
        'diplomacy': 'formal meeting setting, official flags, professional atmosphere, summit table',
        'political': 'government building, political rally, ballot or podium, crowd energy',
        'espionage': 'surveillance screens, classified documents, shadows and secrecy, tech equipment',
        'protests': 'demonstration crowd, protest banners, police barricades, city backdrop',
        'humanitarian': 'civilian perspective, displacement crisis, aid supplies, distress indicators',
        'border': 'border fence, checkpoint guards, vehicle queue, terrain context',
        'cyber': 'server infrastructure, digital data, warning alerts, holographic displays',
        'megaprojects': 'massive construction, heavy machinery, engineering scale, route maps',
        'general': 'dramatic composition, strategic perspective, balanced lighting, professional atmosphere'
    }
    
    return enrichments.get(visual_type, enrichments['general'])


# ============================================================================
# PHASE 2: Reference Image Pipeline - Image-to-Image Generation Support
# ============================================================================

def _upload_reference_image_to_fal(image_path: str) -> str:
    """
    Upload a local reference image to FAL.ai CDN and return the URL.
    Required for Image-to-Image generation with the pixel-art optimized model.
    
    Args:
        image_path: Local path to reference image file
        
    Returns:
        str: CDN URL of uploaded image
        
    Raises:
        ValueError: If image file doesn't exist or upload fails
    """
    import fal_client
    
    local_path = Path(image_path)
    if not local_path.exists():
        raise ValueError(f"Reference image not found: {image_path}")
    
    if not local_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
        raise ValueError(f"Unsupported image format: {local_path.suffix}")
    
    try:
        print(f"  [IMG] Uploading reference image to FAL.ai CDN: {local_path.name}")
        cdn_url = fal_client.upload_file(str(local_path))
        print(f"  [IMG] Reference image CDN URL: {cdn_url}")
        return cdn_url
    except Exception as e:
        raise ValueError(f"Failed to upload reference image: {str(e)}")


def _validate_reference_image_url(url: str) -> bool:
    """
    Validate that a reference image URL is accessible and valid.
    
    Args:
        url: Image URL to validate
        
    Returns:
        bool: True if valid and accessible
    """
    import requests
    
    if not url or not isinstance(url, str):
        return False
    
    # Must be HTTP(S) URL
    if not url.startswith(('http://', 'https://')):
        return False
    
    try:
        # Head request to check accessibility without downloading
        response = requests.head(url, timeout=10, allow_redirects=True)
        content_type = response.headers.get('content-type', '')
        
        # Must be image content type
        if not content_type.startswith('image/'):
            print(f"  [IMG] Warning: Reference URL is not an image: {content_type}")
            return False
        
        return response.status_code == 200
    except Exception as e:
        print(f"  [IMG] Warning: Cannot validate reference image: {e}")
        return False


def _prepare_reference_image(
    reference_image: Optional[str] = None,
    auto_upload: bool = True
) -> Optional[str]:
    """
    Prepare reference image for Image-to-Image generation.
    Handles both local file paths and remote URLs.
    
    Args:
        reference_image: Local path or URL to reference image
        auto_upload: If True, automatically upload local files to FAL CDN
        
    Returns:
        str: Valid image URL ready for API call, or None if invalid
    """
    if not reference_image:
        return None
    
    # Check if it's already a URL
    if reference_image.startswith(('http://', 'https://')):
        if _validate_reference_image_url(reference_image):
            return reference_image
        else:
            print(f"  [IMG] Warning: Invalid reference image URL, proceeding without reference")
            return None
    
    # It's a local file path
    if auto_upload:
        try:
            return _upload_reference_image_to_fal(reference_image)
        except Exception as e:
            print(f"  [IMG] Warning: Could not upload reference: {e}, proceeding without reference")
            return None
    else:
        # Don't upload, just validate path exists
        if Path(reference_image).exists():
            # Return as-is; caller must handle upload
            return reference_image
        else:
            print(f"  [IMG] Warning: Reference image not found: {reference_image}")
            return None


def _build_i2i_generation_args(
    prompt: str,
    reference_image_url: str,
    strength: float = 0.75,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 35,
    seed: Optional[int] = None,
    negative_prompt: str = None
) -> Dict[str, Any]:
    """
    Build API arguments for Image-to-Image generation.
    
    Args:
        prompt: Text prompt for generation
        reference_image_url: URL of reference image (must be accessible)
        strength: How much to transform reference (0.0-1.0, higher = more change)
        guidance_scale: How closely to follow prompt (higher = more adherence)
        num_inference_steps: Number of denoising steps
        seed: Optional seed for reproducibility
        negative_prompt: Optional negative prompt
        
    Returns:
        Dict of arguments for FAL API call
    """
    args = {
        "prompt": prompt,
        "image_url": reference_image_url,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "image_size": {"width": RENDER_RESOLUTION[0], "height": RENDER_RESOLUTION[1]},
        "enable_safety_checker": PIXEL_ART_MODEL_CONFIG["default_params"]["enable_safety_checker"],
        "output_format": PIXEL_ART_MODEL_CONFIG["default_params"]["output_format"],
        "num_images": 1,
    }
    if PIXEL_ART_LORA and PIXEL_ART_LORA.get("path"):
        args["lora"] = [{"path": PIXEL_ART_LORA["path"], "scale": PIXEL_ART_LORA.get("scale", 0.85)}]
    
    if seed is not None:
        args["seed"] = seed
    
    if negative_prompt:
        args["negative_prompt"] = negative_prompt
    
    return args


def _select_style_lora(visual_type: str) -> Dict[str, Any]:
    """
    Select appropriate LoRA configuration based on visual type.
    Returns the LoRA config to use for generation.
    """
    return STYLE_LORAS.get(visual_type, STYLE_LORAS['general'])


def _enhance_prompt_with_lora_trigger(prompt: str, visual_type: str) -> str:
    lora_config = _select_style_lora(visual_type)
    default_lora = IMAGE_STYLE_CONFIG.get('lora_defaults', {})
    trigger = default_lora.get('trigger_word', 'Retro Pixel')
    additional = lora_config.get('additional_prompts', '')

    if trigger.lower() not in prompt.lower():
        prompt = f"{trigger}, {prompt}"

    if additional and additional.lower() not in prompt.lower():
        prompt = f"{prompt}, {additional}"

    return prompt


def _score_prompt_specificity(prompt: str) -> int:
    """
    Score how specific / scene-relevant a prompt is (0-100).
    Based on presence of locations, actions, subjects, and era cues.
    Higher = more specific = better image relevance expected.
    """
    score = 0
    prompt_lower = prompt.lower()

    # Locations add 20 pts
    location_markers = [
        "strait", "gulf", "sea", "ocean", "desert", "city", "capital",
        "port", "base", "border", "valley", "mountain", "coast", "plain"
    ]
    if any(m in prompt_lower for m in location_markers):
        score += 20

    # Named geography adds 15 pts
    named_geo = [
        "hormuz", "ukraine", "gaza", "taiwan", "syria", "iran", "russia",
        "china", "israel", "korea", "persian", "red sea", "arctic"
    ]
    if any(g in prompt_lower for g in named_geo):
        score += 15

    # Action verbs add 20 pts
    actions = [
        "intercept", "launch", "strike", "deploy", "blockade", "patrol",
        "advancing", "retreating", "firing", "bombing", "crossing", "signing"
    ]
    if any(a in prompt_lower for a in actions):
        score += 20

    # Military/subject keywords add 15 pts
    military_subjects = [
        "missile", "tank", "warship", "aircraft", "soldier", "drone",
        "fighter", "destroyer", "submarine", "convoy", "carrier", "artillery"
    ]
    if any(s in prompt_lower for s in military_subjects):
        score += 15
    
    # Economic subjects add 15 pts (equal weight)
    economic_subjects = [
        "trading", "market", "price", "inflation", "shortage", "queue",
        "gas station", "stock", "economy", "dollar", "oil prices", "shelves"
    ]
    if any(s in prompt_lower for s in economic_subjects):
        score += 15
    
    # Diplomatic subjects add 15 pts (equal weight)
    diplomatic_subjects = [
        "summit", "treaty", "negotiation", "embassy", "minister",
        "diplomat", "agreement", "talks", "meeting", "officials"
    ]
    if any(s in prompt_lower for s in diplomatic_subjects):
        score += 15
    
    # Human impact subjects add 15 pts (equal weight)
    human_subjects = [
        "civilian", "family", "families", "protest", "refugee", "crowd",
        "people", "residents", "evacuees", "victims"
    ]
    if any(s in prompt_lower for s in human_subjects):
        score += 15

    # Lighting/atmosphere adds 10 pts
    atmosphere = [
        "dusk", "dawn", "night", "smoke", "explosion", "fire", "storm",
        "dramatic", "silhouette", "horizon", "sunset", "overcast"
    ]
    if any(a in prompt_lower for a in atmosphere):
        score += 10

    # Era/period cues add 10 pts
    eras = ["1980s", "1990s", "2000s", "2010s", "2020s", "cold war", "modern", "historical"]
    if any(e in prompt_lower for e in eras):
        score += 10

    # Penalise generic filler phrases
    generic = ["dramatic confrontation", "strategic forces", "strategic location"]
    for g in generic:
        if g in prompt_lower:
            score -= 15

    return max(0, min(100, score))


def _generate_placeholder(prompt: str, output_path: Path) -> None:
    width, height = 1024, 1792
    img = Image.new("RGB", (width, height), color=(10, 5, 25))
    draw = ImageDraw.Draw(img)

    grid_color = (0, 40, 80)
    spacing = 32
    for x in range(0, width, spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    border_color = (0, 180, 255)
    draw.rectangle([8, 8, width - 8, height - 8], outline=border_color, width=3)
    draw.rectangle([16, 16, width - 16, height - 16], outline=(0, 100, 180), width=1)

    cx, cy = width // 2, height // 2
    draw.rectangle([cx - 180, cy - 180, cx + 180, cy + 180], outline=(0, 255, 180), width=2)
    draw.line([(cx - 200, cy), (cx + 200, cy)], fill=(0, 255, 180), width=1)
    draw.line([(cx, cy - 200), (cx, cy + 200)], fill=(0, 255, 180), width=1)

    for rx, ry, rw, rh, col in [
        (40, 40, 120, 60, (0, 120, 200)),
        (width - 160, 40, 120, 60, (200, 0, 120)),
        (40, height - 100, 120, 60, (0, 200, 120)),
        (width - 160, height - 100, 120, 60, (200, 120, 0)),
    ]:
        draw.rectangle([rx, ry, rx + rw, ry + rh], outline=col, width=2)

    try:
        font_large = ImageFont.truetype("arial.ttf", 36)
        font_small = ImageFont.truetype("arial.ttf", 22)
        font_tiny  = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = font_large
        font_tiny  = font_large

    draw.text((cx, cy - 280), "GENERIC PIXEL ART ASSET", font=font_large,
              fill=(0, 255, 255), anchor="mm")
    draw.text((cx, cy - 240), "[ PLACEHOLDER — NO FAL_KEY SET ]", font=font_small,
              fill=(255, 80, 80), anchor="mm")

    max_chars = 60
    words = prompt[:300].split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    for i, line in enumerate(lines[:6]):
        draw.text((cx, cy + 320 + i * 28), line, font=font_tiny,
                  fill=(160, 160, 200), anchor="mm")

    draw.text((cx, height - 60), "YT-MACHINE  |  SENTINEL v2.1", font=font_tiny,
              fill=(0, 100, 160), anchor="mm")

    img.save(str(output_path), format="PNG")


def _extract_visual_terms_from_narration(script_text: str) -> str:
    """
    Extract concrete visual terms from narration text for grounded enrichment.
    Scans for proper nouns, locations, equipment, and numbers — then builds
    weighted prompt tokens that reinforce what the narrator is talking about.
    Returns empty string if nothing concrete found (caller falls back to category enrichment).
    """
    if not script_text:
        return ''
    
    import re
    
    # Known geopolitical locations (partial match)
    geo_locations = [
        'strait of hormuz', 'hormuz', 'south china sea', 'taiwan strait',
        'persian gulf', 'red sea', 'suez canal', 'panama canal',
        'ukraine', 'gaza', 'syria', 'yemen', 'iran', 'israel',
        'russia', 'china', 'taiwan', 'korea', 'kiev', 'kyiv',
        'moscow', 'beijing', 'tehran', 'washington', 'brussels',
        'arctic', 'sahara', 'sahel', 'himalaya', 'indian ocean',
        'pacific', 'atlantic', 'mediterranean', 'black sea',
        'crimea', 'donbas', 'golan heights', 'west bank',
        'south ossetia', 'abkhazia', 'kashmir', 'kuril islands'
    ]
    
    # Known equipment/hardware terms
    equipment_terms = [
        'f-35', 'f-16', 'f-22', 'su-35', 'su-57', 'j-20',
        'missile', 'drone', 'tank', 'warship', 'destroyer',
        'submarine', 'carrier', 'aircraft carrier', 'frigate',
        'artillery', 'helicopter', 'fighter jet', 'bomber',
        'patrol boat', 'oil tanker', 'cargo ship', 'container ship',
        'pipeline', 'refinery', 'nuclear plant', 'power grid',
        's-400', 's-300', 'patriot', 'iron dome', 'thaad',
        'himars', 'm142', 'iskander', 'kinzhal', 'dagger',
        'reaper', 'mq-9', 'bayraktar', 'tb2', 'switchblade'
    ]
    
    # Known economic/infrastructure terms
    economic_terms = [
        'oil', 'gas', 'pipeline', 'lng', 'crude', 'petroleum',
        'wheat', 'grain', 'semiconductor', 'chip', 'rare earth',
        'lithium', 'copper', 'gold', 'uranium', 'cobalt',
        'sanctions', 'tariff', 'embargo', 'trade war', 'swift',
        'trading floor', 'stock market', 'inflation', 'price'
    ]
    
    text_lower = script_text.lower()
    found_terms = []
    
    # Extract locations (highest priority → weight 1.3)
    for loc in geo_locations:
        if loc in text_lower:
            found_terms.append((loc, 1.3))
            if len(found_terms) >= 3:
                break
    
    # Extract equipment (second priority → weight 1.2)
    for equip in equipment_terms:
        if equip in text_lower:
            found_terms.append((equip, 1.2))
            if len(found_terms) >= 5:
                break
    
    # Extract economic terms (third priority → weight 1.1)
    for econ in economic_terms:
        if econ in text_lower and not any(econ in ft[0] for ft in found_terms):
            found_terms.append((econ, 1.1))
            if len(found_terms) >= 6:
                break
    
    # Extract numbers (billions, millions, percentages)
    number_patterns = re.findall(r'\b(\d+)\s*(?:billion|million|percent|%|trillion)\b', text_lower)
    for num in number_patterns[:2]:
        found_terms.append((num, 1.1))
    
    if not found_terms:
        return ''
    
    # Build plain-text prompt fragment for FLUX (no Compel weighting)
    # Take top 4 terms max to avoid overloading
    top_terms = found_terms[:4]
    weighted_parts = [f"{term}" for term, weight in top_terms]
    
    return ', '.join(weighted_parts)


def _inject_geopolitical_context(prompt: str, issues: list, script_text: str) -> Optional[str]:
    """
    Inject geopolitical context into a prompt that failed validation.
    Tries to fix specific issues (missing location, missing country context, etc.)
    by extracting relevant terms from the narration text.
    Returns the enriched prompt, or None if no improvement could be made.
    """
    if not script_text:
        return None
    
    issues_text = ' '.join(issues).lower()
    enrichment_parts = []
    
    # If missing location context — extract from narration
    if any(word in issues_text for word in ['location', 'geography', 'region', 'country']):
        narration_terms = _extract_visual_terms_from_narration(script_text)
        if narration_terms:
            enrichment_parts.append(narration_terms)
    
    # If missing subject/equipment — extract key nouns from narration
    if any(word in issues_text for word in ['subject', 'equipment', 'specific', 'vague']):
        # Look for capitalized proper nouns in the narration
        import re
        proper_nouns = re.findall(r'\b([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})?)\b', script_text)
        # Filter common non-visual words
        skip_words = {'The', 'This', 'That', 'These', 'Those', 'They', 'Their',
                      'Masker', 'Today', 'Meanwhile', 'Actually', 'Believe', 'Simply',
                      'Welcome', 'Afternoon', 'Evening', 'Morning', 'Subscribe', 'Story'}
        visual_nouns = [n for n in proper_nouns if n not in skip_words][:3]
        if visual_nouns:
            enrichment_parts.append(', '.join(visual_nouns))
    
    if not enrichment_parts:
        return None
    
    # Inject before the style suffix
    enrichment = ', '.join(enrichment_parts)
    enriched = f"{prompt}, geopolitical context: {enrichment}"
    return enriched


def _truncate_prompt_for_resolution(prompt: str, width: int, height: int) -> str:
    """
    Truncate overly long prompts for small render resolutions.
    At 272x288, FLUX needs concise prompts — 2-3 sentences max.
    Longer prompts at tiny resolutions produce muddled, unfocused images.
    """
    MAX_SENTENCES_SMALL = 3
    MAX_WORDS_SMALL = 50
    SMALL_THRESHOLD = 400 * 400

    if width * height >= SMALL_THRESHOLD:
        return prompt

    import re
    sentences = re.split(r'(?<=[.!?])\s+', prompt)
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= MAX_SENTENCES_SMALL:
        word_count = len(prompt.split())
        if word_count <= MAX_WORDS_SMALL:
            return prompt

    kept = sentences[:MAX_SENTENCES_SMALL]
    truncated = ' '.join(kept)

    words = truncated.split()
    if len(words) > MAX_WORDS_SMALL:
        truncated = ' '.join(words[:MAX_WORDS_SMALL])
        if not truncated.endswith(('.', '!', '?')):
            truncated += '.'

    if len(sentences) > MAX_SENTENCES_SMALL:
        print(f"  [IMG] Prompt truncated for {width}x{height}: {len(sentences)} → {MAX_SENTENCES_SMALL} sentences, {len(prompt.split())} → {len(truncated.split())} words")

    return truncated


def _resolve_size(target_app: str = "Default") -> Dict[str, int]:
    return IMAGE_SIZE_MAP.get(target_app, IMAGE_SIZE_MAP["Default"])


@server.tool()
def generate_pixel_art(
    prompt: str, 
    script_text: str = None, 
    seed: int = None,
    reference_image: str = None,
    i2i_mode: str = "balanced",
    i2i_strength: float = None,
    i2i_guidance: float = None,
    i2i_steps: int = None,
    use_pixel_art_model: bool = True
) -> dict:
    """
    Generate a 16-bit cyberpunk pixel art image for a given scene prompt.
    Enhanced with visual type detection, structured prompting, and Image-to-Image support.
    
    PHASE 1-3 INTEGRATION:
    - Pixel-art optimized model (use_pixel_art_model=True)
    - Reference image guidance (reference_image parameter)
    - Accuracy refinement (i2i_mode, i2i_strength, i2i_guidance, i2i_steps)

    Args:
        prompt: Scene description (e.g. "Strait of Hormuz blockade at dusk")
        script_text: Original script text for geopolitical context (optional)
        seed: Optional seed for reproducible style. Use base_seed + scene_index for batch consistency.
        reference_image: Path or URL to reference image for I2I generation (optional)
        i2i_mode: Refinement mode ("strict_reference", "balanced", "prompt_dominant", "pixel_art_precise")
        i2i_strength: Override strength (0.0-1.0) - how much to transform reference
        i2i_guidance: Override guidance scale - how closely to follow prompt
        i2i_steps: Override inference steps
        use_pixel_art_model: If True, use pixel-art optimized model endpoint

    Returns:
        dict with success status, image path, and metadata including i2i_params
    """
    if not prompt or not prompt.strip():
        return {"success": False, "error": "Prompt cannot be empty"}

    # Geopolitical accuracy validation — 3-tier: "strict", "log_only", "disabled"
    VALIDATION_MODE = "log_only"
    geo_validator = GeopoliticalValidator()
    geo_validator.validation_rules['strict_mode'] = (VALIDATION_MODE == "strict")
    geo_validator.validation_rules['min_accuracy_score'] = 50  # Lower threshold — let enrichment fix it
    
    should_proceed, geo_error = geo_validator.validate_before_generation(script_text or "", prompt)
    
    if not should_proceed and VALIDATION_MODE == "strict":
        return {
            "success": False, 
            "error": f"Geopolitical accuracy validation failed: {geo_error}",
            "validation_type": "geopolitical",
            "accuracy_blocked": True
        }
    elif not should_proceed:
        print(f"  [IMG] ⚠️ Pre-generation validation: {geo_error} (proceeding anyway)")

    # Enhanced visual type detection
    visual_type = _detect_visual_type(prompt)
    print(f"  [IMG] Visual type detected: {visual_type}")

    # Phase 5.1: Score and optionally enrich low-specificity prompts
    specificity = _score_prompt_specificity(prompt)
    print(f"  [IMG] Specificity score: {specificity}/100")

    # Always enrich with narration-grounded terms when available
    enriched_prompt = prompt.strip()
    narration_enrichment = _extract_visual_terms_from_narration(script_text) if script_text else ''
    if narration_enrichment:
        enriched_prompt += f", {narration_enrichment}"
        print(f"  [IMG] Narration-grounded enrichment applied")
    
    # Additional category enrichment for very low specificity
    if specificity < 35 and not narration_enrichment:
        enrichment = _get_adaptive_enrichment(visual_type)
        enriched_prompt += f", {enrichment}"
        print(f"  [IMG] Low specificity ({specificity}/100) — {visual_type} category enrichment")

    # Sanitize for FAL.ai content policy before building final prompt
    # Use light scrubbing for visual prompts — keep country/equipment names for relevance
    sanitized_prompt = _sanitize_visual_prompt(enriched_prompt)
    
    # Enhance with LoRA trigger and additional prompts
    enhanced_prompt = _enhance_prompt_with_lora_trigger(sanitized_prompt, visual_type)
    
    # Build final prompt: clip_style_tag first (CLIP 77-token window), then color palette + full style suffix (T5 sees all)
    full_prompt = f"{enhanced_prompt}, {CLIP_STYLE_TAG}"
    if COLOR_PALETTE_PROMPT:
        full_prompt = f"{full_prompt}, {COLOR_PALETTE_PROMPT}"
    full_prompt = f"{full_prompt}, {STYLE_SUFFIX}"
    
    # Final geopolitical validation — log warnings, retry enrichment if score is low
    final_geo_validation = geo_validator.validate_prompt_geopolitical_accuracy(full_prompt, script_text)
    print(f"  [IMG] Geopolitical accuracy score: {final_geo_validation['accuracy_score']}%")
    
    if not final_geo_validation['passed']:
        if VALIDATION_MODE == "strict":
            return {
                "success": False,
                "error": f"Final prompt failed geopolitical validation: {'; '.join(final_geo_validation['issues'][:2])}",
                "validation_type": "geopolitical_final",
                "accuracy_score": final_geo_validation['accuracy_score'],
                "issues": final_geo_validation['issues']
            }
        elif VALIDATION_MODE == "log_only":
            print(f"  [IMG] ⚠️ Geopolitical issues: {'; '.join(final_geo_validation['issues'][:3])}")
            # Try to inject geopolitical context as a retry
            if final_geo_validation['accuracy_score'] < 50 and script_text:
                retry_enrichment = _inject_geopolitical_context(full_prompt, final_geo_validation['issues'], script_text)
                if retry_enrichment:
                    full_prompt = retry_enrichment
                    retry_validation = geo_validator.validate_prompt_geopolitical_accuracy(full_prompt, script_text)
                    print(f"  [IMG] 🔄 Retry accuracy: {retry_validation['accuracy_score']}%")
    
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:40])
    filename = f"pixel_art_{safe_name}_{hash(prompt) % 100000}.png"
    output_path = OUTPUT_DIR / filename

    # Truncate overly long prompts for small render resolutions
    render_w = RENDER_RESOLUTION[0]
    render_h = RENDER_RESOLUTION[1]
    full_prompt = _truncate_prompt_for_resolution(full_prompt, render_w, render_h)

    # ============================================================================
    # PHASE 2 & 3: Reference Image Preparation and Refinement Parameters
    # ============================================================================
    
    reference_image_url = None
    i2i_params = None
    
    if reference_image:
        print(f"  [IMG] Preparing reference image for I2I generation...")
        reference_image_url = _prepare_reference_image(reference_image, auto_upload=True)
        
        if reference_image_url:
            # Get refinement parameters (Phase 3)
            i2i_params = _get_refinement_params(
                mode=i2i_mode,
                custom_strength=i2i_strength,
                custom_guidance=i2i_guidance,
                custom_steps=i2i_steps
            )
            _print_refinement_settings(i2i_params)
        else:
            print(f"  [IMG] Proceeding with text-to-image generation (no reference)")

    # ========== PRIMARY: Local FLUX (GPU, free, no API key needed) ==========
    if USE_LOCAL_FLUX and not reference_image_url:
        render_size = IMAGE_STYLE_CONFIG.get("generation_params", {}).get("render_resolution", [512, 512])
        local_size = {"width": render_size[0], "height": render_size[1]}
        if local_size["width"] * local_size["height"] > MAX_PIXELS:
            local_size = {"width": 512, "height": 512}

        local_steps = int(os.environ.get("LOCAL_FLUX_STEPS", "40"))
        local_guidance = 3.5

        local_result = _generate_local_flux(
            prompt=full_prompt,
            output_path=output_path,
            size=local_size,
            seed=seed,
            steps=local_steps,
            guidance_scale=local_guidance,
            negative_prompt=NEGATIVE_PROMPT,
        )

        if local_result:
            _store_prompt_cache(full_prompt, str(output_path))
            local_result.update({
                "specificity_score": specificity,
                "visual_type": visual_type,
                "pixel_art_model_used": use_pixel_art_model,
                "geopolitical_validation": final_geo_validation,
                "accuracy_score": final_geo_validation['accuracy_score'],
                "countries_detected": list(final_geo_validation['country_analysis'].keys()),
                "equipment_validated": not any(analysis['issues'] for analysis in final_geo_validation['equipment_analysis'].values()),
                "target_app": os.environ.get("TARGET_APP", "Default"),
                "i2i_used": False,
                "i2i_params": None,
                "reference_image_url": None,
            })
            return local_result

        print("  [IMG] Local FLUX failed — falling back to fal.ai")

    # ========== FALLBACK: fal.ai cloud API ==========
    if FAL_KEY:
        try:
            import fal_client
            import requests

            os.environ["FAL_KEY"] = FAL_KEY

            target_app = os.environ.get("TARGET_APP", "Default")
            size = {"width": RENDER_RESOLUTION[0], "height": RENDER_RESOLUTION[1]}

            if size["width"] * size["height"] > MAX_PIXELS:
                print(f"  [IMG] WARNING: {size['width']}x{size['height']} exceeds 1MP cost cap — clamping to 512x512")
                size = {"width": 512, "height": 512}

            print(f"  [IMG] FAL flux/dev | {size['width']}x{size['height']} (render) → {TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]} (upscale) | steps={MODEL_STEP_CONFIG['fal-ai/flux/dev']} | guidance=4.0 | app={target_app}")

            models_to_try = [FAL_MODEL] + FAL_FALLBACK_MODELS

            result = None
            model_used = None
            last_error = None

            for model in models_to_try:
                try:
                    print(f"  [IMG] Trying model: {model}")

                    if reference_image_url and i2i_params:
                        arguments = _build_i2i_generation_args(
                            prompt=full_prompt,
                            reference_image_url=reference_image_url,
                            strength=i2i_params["strength"],
                            guidance_scale=i2i_params["guidance_scale"],
                            num_inference_steps=i2i_params["num_inference_steps"],
                            seed=seed,
                            negative_prompt=NEGATIVE_PROMPT
                        )
                        print(f"  [IMG] I2I params: strength={i2i_params['strength']:.2f}, guidance={i2i_params['guidance_scale']:.1f}")
                    else:
                        base_args = {
                            "prompt": full_prompt,
                            "image_size": {"width": size["width"], "height": size["height"]},
                            "num_images": 1,
                            "num_inference_steps": MODEL_STEP_CONFIG.get(model, 40),
                            "guidance_scale": 4.0,
                            "enable_safety_checker": False,
                            "negative_prompt": NEGATIVE_PROMPT,
                            "output_format": "png",
                        }
                        if PIXEL_ART_LORA and PIXEL_ART_LORA.get("path"):
                            base_args["lora"] = [{"path": PIXEL_ART_LORA["path"], "scale": PIXEL_ART_LORA.get("scale", 0.85)}]
                            print(f"  [IMG] LoRA: {PIXEL_ART_LORA['path']} (scale={PIXEL_ART_LORA.get('scale', 0.85)})")
                        print(f"  [IMG] Dimensions: {size['width']}x{size['height']}")
                        if seed is not None:
                            base_args["seed"] = seed

                        arguments = base_args

                    result = fal_client.run(model, arguments=arguments)
                    model_used = model
                    print(f"  [IMG] Success with {model}")
                    break

                except Exception as model_error:
                    last_error = str(model_error)
                    print(f"  [IMG] {model} failed: {last_error[:100]}")
                    if model == models_to_try[-1]:
                        raise
                    continue

            if not result:
                raise Exception(f"All models failed. Last error: {last_error}")

            image_url = result["images"][0]["url"]

            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(img_response.content)

            try:
                _upscale_pixel_art(str(output_path))
            except Exception as upscale_err:
                print(f"  [IMG] Warning: Upscale failed, using raw output: {upscale_err}")

            _store_prompt_cache(full_prompt, str(output_path))

            is_failed, fail_reason = _detect_failed_image(str(output_path))
            if is_failed:
                print(f"  [IMG] Detected failed image: {fail_reason}")
                return {
                    "success": True,
                    "filename": filename,
                    "path": str(output_path),
                    "prompt_used": full_prompt,
                    "specificity_score": specificity,
                    "visual_type": visual_type,
                    "source": model_used,
                    "image_url": image_url,
                    "output_directory": str(OUTPUT_DIR),
                    "i2i_used": reference_image_url is not None,
                    "i2i_params": i2i_params,
                    "reference_image_url": reference_image_url,
                    "pixel_art_model_used": use_pixel_art_model,
                    "geopolitical_validation": final_geo_validation,
                    "accuracy_score": final_geo_validation['accuracy_score'],
                    "countries_detected": list(final_geo_validation['country_analysis'].keys()),
                    "equipment_validated": not any(analysis['issues'] for analysis in final_geo_validation['equipment_analysis'].values()),
                    "target_app": target_app,
                    "provider": "fal_ai",
                    "detected_failure": fail_reason,
                    "width": size["width"],
                    "height": size["height"],
                    "steps": MODEL_STEP_CONFIG.get(model_used, 28),
                }

            return {
                "success": True,
                "filename": filename,
                "path": str(output_path),
                "prompt_used": full_prompt,
                "specificity_score": specificity,
                "visual_type": visual_type,
                "source": model_used,
                "image_url": image_url,
                "output_directory": str(OUTPUT_DIR),
                "i2i_used": reference_image_url is not None,
                "i2i_params": i2i_params,
                "reference_image_url": reference_image_url,
                "pixel_art_model_used": use_pixel_art_model,
                "geopolitical_validation": final_geo_validation,
                "accuracy_score": final_geo_validation['accuracy_score'],
                "countries_detected": list(final_geo_validation['country_analysis'].keys()),
                "equipment_validated": not any(analysis['issues'] for analysis in final_geo_validation['equipment_analysis'].values()),
                "provider": "fal_ai",
                "target_app": target_app,
                "width": size["width"],
                "height": size["height"],
                "steps": MODEL_STEP_CONFIG.get(model_used, 28),
            }

        except Exception as e:
            error_msg = str(e)
            is_content_policy = any(kw in error_msg.lower() for kw in [
                'content', 'safety', 'policy', 'inappropriate', 'nsfw',
                'sensitive', 'filtered', 'blocked', 'moderation', 'flagged'
            ])

            print(f"  FAL.ai generation failed: {error_msg[:120]}")

            if is_content_policy:
                import time
                for scrub_level in [1, 2, 3]:
                    print(f"  [IMG] Content policy detected — retrying with scrub level {scrub_level}/3...")

                    scrubbed = _progressive_content_scrub(
                        enriched_prompt, visual_type, level=scrub_level
                    )
                    scrubbed_enhanced = _enhance_prompt_with_lora_trigger(scrubbed, visual_type)
                    scrubbed_full = f"{scrubbed_enhanced}, {CLIP_STYLE_TAG}"
                    if COLOR_PALETTE_PROMPT:
                        scrubbed_full = f"{scrubbed_full}, {COLOR_PALETTE_PROMPT}"
                    scrubbed_full = f"{scrubbed_full}, {STYLE_SUFFIX}"

                    # Try FLUX/dev first for better quality, then fall to schnell
                    for retry_model in [FAL_MODEL, "fal-ai/flux/schnell"]:
                        try:
                            retry_args = {
                                "prompt": scrubbed_full,
                                "image_size": {"width": size["width"], "height": size["height"]},
                                "num_images": 1,
                                "num_inference_steps": MODEL_STEP_CONFIG.get(retry_model, 40),
                                "guidance_scale": 4.0,
                                "enable_safety_checker": False,
                                "negative_prompt": NEGATIVE_PROMPT,
                                "output_format": "png",
                            }
                            if PIXEL_ART_LORA and PIXEL_ART_LORA.get("path"):
                                retry_args["lora"] = [{"path": PIXEL_ART_LORA["path"], "scale": PIXEL_ART_LORA.get("scale", 0.85)}]
                            if seed is not None:
                                retry_args["seed"] = seed + scrub_level

                            retry_result = fal_client.run(retry_model, arguments=retry_args)
                            retry_image_url = retry_result["images"][0]["url"]

                            img_response = requests.get(retry_image_url, timeout=30)
                            img_response.raise_for_status()

                            with open(output_path, "wb") as f:
                                f.write(img_response.content)

                            try:
                                _upscale_pixel_art(str(output_path))
                            except Exception:
                                pass

                            is_retry_failed, retry_fail_reason = _detect_failed_image(str(output_path))
                            if is_retry_failed:
                                print(f"  [IMG] Scrub level {scrub_level} ({retry_model}) produced failed image: {retry_fail_reason}")
                                continue

                            print(f"  [IMG] Scrub level {scrub_level} succeeded with {retry_model} — image saved")
                            return {
                                "success": True,
                                "filename": filename,
                                "path": str(output_path),
                                "prompt_used": scrubbed_full,
                                "original_prompt": full_prompt,
                                "visual_type": visual_type,
                                "source": f"{retry_model} (scrub level {scrub_level})",
                                "scrub_level": scrub_level,
                                "note": f"Content policy retry succeeded at scrub level {scrub_level} with {retry_model}",
                                "output_directory": str(OUTPUT_DIR),
                                "geopolitical_validation": final_geo_validation,
                                "i2i_used": False,
                                "i2i_params": None,
                                "provider": "fal_ai",
                                "target_app": target_app,
                                "width": size["width"],
                                "height": size["height"],
                            }

                        except Exception as retry_model_err:
                            print(f"  [IMG] Scrub level {scrub_level} with {retry_model} failed: {str(retry_model_err)[:80]}")
                            continue

                    time.sleep(1)

                print(f"  [IMG] All scrub levels failed — falling to placeholder")
            else:
                print(f"   Non-content error — generating placeholder image as fallback.")

            _generate_placeholder(prompt, output_path)
            return {
                "success": True,
                "filename": filename,
                "path": str(output_path),
                "prompt_used": full_prompt,
                "visual_type": visual_type,
                "source": "placeholder",
                "note": f"FAL.ai fallback: {error_msg[:80]}",
                "output_directory": str(OUTPUT_DIR),
                "geopolitical_validation": final_geo_validation,
                "i2i_used": reference_image_url is not None,
                "i2i_params": i2i_params,
                "provider": "fal_ai",
            }

    else:
        try:
            _generate_placeholder(prompt, output_path)

            return {
                "success": True,
                "filename": filename,
                "path": str(output_path),
                "prompt_used": full_prompt,
                "visual_type": visual_type,
                "source": "placeholder",
                "note": "FAL_KEY not set — placeholder image generated",
                "output_directory": str(OUTPUT_DIR),
                "geopolitical_validation": final_geo_validation,
                "i2i_used": False,
                "i2i_params": None
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Placeholder generation failed: {str(e)}",
            }
