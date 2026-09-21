"""
Model Registry — Discovery and normalization of local model providers.
=====================================================================

Discovers text/vision/embedding/image models from the local machine:

  - Ollama servers        (GET /api/tags + POST /api/show)
  - llama.cpp servers     (GET /v1/models, OpenAI compatible)
  - GGUF files on disk    (*.gguf plus sibling mmproj files)
  - MLX-Gen model folders (config.json / model_index.json + weights)

Everything is normalized into a single `ModelSpec` so the rest of the
pipeline never needs to know how a model is served.

WHY A REGISTRY?
  Before: model names were hard-coded in config/system_prompts.json and
  every module built its own endpoint. Now discovery happens once, the
  user selects roles in tools/model_setup.py, and every provider adapter
  consumes the same ModelSpec.

DESIGN RULES:
  - Discovery never loads a model. It only reads metadata/filesystem info.
  - Capability guessing is conservative and always overridable by the user.
  - Nothing here performs inference or starts a server.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

CAP_TEXT = "text"
CAP_VISION = "vision"
CAP_EMBEDDING = "embedding"
CAP_IMAGE = "image"

KNOWN_CAPABILITIES = (CAP_TEXT, CAP_VISION, CAP_EMBEDDING, CAP_IMAGE)

PROVIDER_OLLAMA = "ollama"
PROVIDER_LLAMACPP = "llamacpp"
PROVIDER_OPENAI = "openai_compat"
PROVIDER_GGUF = "gguf"
PROVIDER_MLXGEN = "mlxgen"

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_LLAMACPP_URL = "http://127.0.0.1:8080"
# No hardcoded install path: the mlxgen executable is machine-specific, so it
# is read from MLXGEN_BIN (see .env.example). An empty value fails the
# availability check with an actionable message rather than pointing at
# somebody else's home directory.
DEFAULT_MLXGEN_BIN = os.getenv("MLXGEN_BIN", "")

# Regex for GGUF quantization tags in file names (Q4_K_M, IQ3_XXS, F16 ...)
_QUANT_RE = re.compile(r"\b(IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|F16|F32|BF16)\b", re.IGNORECASE)

GIB = 1024 ** 3


# ─────────────────────────────────────────────────────────────────────
# ModelSpec
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Normalized description of one discoverable model.

    A spec is metadata only — it does not own a process or a connection.
    """

    id: str
    provider: str
    capabilities: List[str] = field(default_factory=lambda: [CAP_TEXT])
    endpoint: Optional[str] = None
    path: Optional[str] = None
    size_bytes: int = 0
    quantization: Optional[str] = None
    family: Optional[str] = None
    context_length: Optional[int] = None
    managed: bool = False
    launch_args: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── serialization ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in (data or {}).items() if k in known}
        spec = cls(**clean)
        if not spec.capabilities:
            spec.capabilities = [CAP_TEXT]
        return spec

    # ── helpers ────────────────────────────────────────────────────

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / GIB, 2)

    @property
    def estimated_memory_gb(self) -> float:
        """Total resident footprint (weights + runtime overhead), in GB.

        This is the number the user should think about, not what the memory
        guard enforces — see `estimated_anonymous_gb`.
        """
        weights = self.size_bytes / GIB
        if self.provider == PROVIDER_OLLAMA:
            return round(max(weights * 1.15, 1.0) + 1.5, 1)
        if self.provider == PROVIDER_MLXGEN:
            return round(max(weights * 1.15, 2.0) + 2.5, 1)
        return round(max(weights * 1.15, 1.0) + 2.5, 1)

    @property
    def estimated_anonymous_gb(self) -> float:
        """Non-reclaimable (anonymous) memory this model will consume.

        CALIBRATED FROM MEASUREMENT, not from theory:
          A 18.5 GB Q4_K_M GGUF loaded by llama-server with
          `--n-gpu-layers 999 --ctx-size 32768` on Apple Silicon measured
          +18.4 GB anonymous. The weights are resident (not evictable file
          cache) even though llama.cpp mmaps them, because Metal uploads
          them into the unified-memory working set. The KV cache at 32k
          context is small relative to the weights.

          So for this machine the weight size IS the anonymous estimate.

        Do not "optimize" this to a small fraction of the weights without
        measuring first — an under-estimate here is exactly what causes a
        mid-run OOM kill instead of a clean refusal.
        """
        weights = self.size_bytes / GIB
        if self.provider == PROVIDER_MLXGEN:
            # MLX streams the text encoder and denoiser; safetensors stay
            # file-backed, so anonymous use is well below the folder size.
            return round(max(weights * 0.65, 6.0), 1)
        if self.provider == PROVIDER_OLLAMA:
            return round(max(weights * 1.1, 3.0) + 1.0, 1)
        # llama.cpp / GGUF on Metal: essentially the full weight footprint
        # plus a small KV-cache allowance.
        return round(max(weights * 1.0 + 1.0, 4.0), 1)

    def describe(self) -> str:
        bits = [f"[{self.provider}]", self.id]
        if self.quantization:
            bits.append(f"({self.quantization})")
        if self.size_gb:
            bits.append(f"{self.size_gb}GB")
        bits.append(f"caps={','.join(self.capabilities)}")
        return " ".join(bits)


# ─────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────

def _http_get_json(url: str, timeout: float = 3.0) -> Optional[Any]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Any]:
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Discovery — Ollama
# ─────────────────────────────────────────────────────────────────────

_OLLAMA_CAP_MAP = {
    "completion": CAP_TEXT,
    "tools": CAP_TEXT,
    "insert": CAP_TEXT,
    "vision": CAP_VISION,
    "embedding": CAP_EMBEDDING,
}


def probe_ollama(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 3.0) -> List[ModelSpec]:
    """List models installed in an Ollama server. Never loads a model."""
    base_url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    data = _http_get_json(f"{base_url}/api/tags", timeout=timeout)
    if not data or "models" not in data:
        return []

    specs: List[ModelSpec] = []
    for entry in data.get("models", []):
        name = entry.get("name") or entry.get("model")
        if not name:
            continue

        details = entry.get("details") or {}
        caps: List[str] = []

        show = _http_post_json(f"{base_url}/api/show", {"model": name}, timeout=timeout)
        if isinstance(show, dict):
            for raw in show.get("capabilities") or []:
                mapped = _OLLAMA_CAP_MAP.get(str(raw).lower())
                if mapped and mapped not in caps:
                    caps.append(mapped)
            model_info = show.get("model_info") or {}
            ctx = model_info.get("general.context_length") or model_info.get(
                f"{details.get('family', '')}.context_length"
            )
            family = model_info.get("general.architecture") or details.get("family")
        else:
            ctx = None
            family = details.get("family")

        if not caps:
            # Ollama did not report capabilities — infer conservatively.
            caps = [CAP_TEXT]
            if "embed" in name.lower() or "nomic" in name.lower():
                caps = [CAP_EMBEDDING]

        specs.append(
            ModelSpec(
                id=name,
                provider=PROVIDER_OLLAMA,
                capabilities=caps,
                endpoint=base_url,
                size_bytes=int(entry.get("size") or 0),
                quantization=details.get("quantization_level"),
                family=family,
                context_length=ctx,
                metadata={"modified_at": entry.get("modified_at")},
            )
        )
    return specs


# ─────────────────────────────────────────────────────────────────────
# Discovery — llama.cpp / generic OpenAI compatible servers
# ─────────────────────────────────────────────────────────────────────

def probe_openai_compatible(
    base_url: str = DEFAULT_LLAMACPP_URL,
    timeout: float = 2.0,
    provider: str = PROVIDER_LLAMACPP,
) -> List[ModelSpec]:
    """List models from any OpenAI-compatible /v1/models endpoint."""
    base_url = (base_url or DEFAULT_LLAMACPP_URL).rstrip("/")
    data = _http_get_json(f"{base_url}/v1/models", timeout=timeout)
    if not data or "data" not in data:
        return []

    specs: List[ModelSpec] = []
    for entry in data.get("data", []):
        model_id = entry.get("id")
        if not model_id:
            continue

        # llama.cpp reports the loaded model; try to locate the GGUF on disk
        # by scanning configured roots for a matching file name.
        path = None
        size_bytes = 0
        for root in default_scan_roots():
            candidate = Path(root) / Path(model_id).name
            if candidate.exists():
                path = str(candidate)
                size_bytes = candidate.stat().st_size
                break

        specs.append(
            ModelSpec(
                id=model_id,
                provider=provider,
                capabilities=[CAP_TEXT],
                endpoint=base_url,
                path=path,
                size_bytes=size_bytes,
                quantization=_guess_quantization(model_id),
                metadata={"source": "v1/models"},
            )
        )
    return specs


# ─────────────────────────────────────────────────────────────────────
# Discovery — GGUF files on disk
# ─────────────────────────────────────────────────────────────────────

def default_scan_roots() -> List[str]:
    """Configured directories to scan for model files.

    WHY CONFIGURED ROOTS? Scanning the entire disk is slow and finds
    unrelated models. Users opt in via YT_MODEL_ROOTS=/a:/b or the
    defaults below.
    """
    env = os.getenv("YT_MODEL_ROOTS", "").strip()
    if env:
        return [p for p in env.split(os.pathsep) if p]

    candidates = [
        Path.home() / "AI",
        Path.home() / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    return [str(p) for p in candidates if p.exists()]


def _guess_quantization(text: str) -> Optional[str]:
    match = _QUANT_RE.search(text or "")
    return match.group(1).upper() if match else None


def _looks_like_vision_mmproj(path: Path) -> bool:
    return "mmproj" in path.name.lower() or "projector" in path.name.lower()


def scan_gguf(roots: Optional[Iterable[str]] = None, max_files: int = 200) -> List[ModelSpec]:
    """Find .gguf model files under the given roots (shallow, bounded)."""
    roots = list(roots or default_scan_roots())
    specs: List[ModelSpec] = []
    seen: set[str] = set()

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        try:
            candidates = sorted(root_path.rglob("*.gguf"))
        except Exception:
            continue
        for gguf in candidates[:max_files]:
            real = str(gguf.resolve())
            if real in seen or _looks_like_vision_mmproj(gguf):
                continue
            seen.add(real)

            caps = [CAP_TEXT]
            # A sibling mmproj file means this is a vision-capable model.
            try:
                for sibling in gguf.parent.glob("*.gguf"):
                    if _looks_like_vision_mmproj(sibling):
                        caps.append(CAP_VISION)
                        break
            except Exception:
                pass

            try:
                size_bytes = gguf.stat().st_size
            except OSError:
                size_bytes = 0

            specs.append(
                ModelSpec(
                    id=str(gguf),
                    provider=PROVIDER_GGUF,
                    capabilities=caps,
                    path=real,
                    size_bytes=size_bytes,
                    quantization=_guess_quantization(gguf.name),
                )
            )

    specs.sort(key=lambda s: s.size_bytes, reverse=True)
    return specs


# ─────────────────────────────────────────────────────────────────────
# Discovery — MLX-Gen image model folders
# ─────────────────────────────────────────────────────────────────────

_MLXGEN_WEIGHT_SUFFIXES = (".safetensors", ".npz", ".bin", ".gguf")


def _dir_size(path: Path, limit_files: int = 5000) -> int:
    total = 0
    count = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
                count += 1
                if count >= limit_files:
                    break
    except Exception:
        return total
    return total


_MLXGEN_COMPONENTS = ("transformer", "vae")


def _is_mlxgen_model_dir(path: Path) -> bool:
    """Heuristic for a prepared MLX-Gen checkpoint root.

    A real checkpoint has component subfolders (transformer+vae) containing
    weights, or a README declaring `library_name: mlx-gen`. Component
    subfolders themselves (…/transformer, …/vae) are rejected so discovery
    reports one model, not five fragments.
    """
    if not path.is_dir():
        return False

    # Reject component directories: a transformer/vae/tokenizer folder is
    # never a model root.
    if path.name in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
        return False
    if "snapshots" in path.parts:
        # HF cache snapshot dirs hold a complete repo — valid roots.
        pass

    has_components = all((path / c).is_dir() for c in _MLXGEN_COMPONENTS)
    if has_components:
        for component in _MLXGEN_COMPONENTS:
            try:
                if any(f.suffix in _MLXGEN_WEIGHT_SUFFIXES
                       for f in (path / component).iterdir() if f.is_file()):
                    return True
            except Exception:
                continue

    readme = path / "README.md"
    if readme.exists():
        try:
            head = readme.read_text(errors="replace")[:2000]
            if "mlx-gen" in head or "mlx_gen" in head or "mflux" in head:
                return True
        except Exception:
            pass

    return False


def scan_mlxgen_models(roots: Optional[Iterable[str]] = None, max_dirs: int = 50) -> List[ModelSpec]:
    """Find prepared MLX-Gen model folders.

    Detection is structural (component dirs + weights) or metadata-based
    (README declaring the mlx-gen library). Nothing is loaded.
    """
    roots = list(roots or default_scan_roots())
    specs: List[ModelSpec] = []
    seen_roots: set[str] = set()

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue

        # Look up to 3 levels deep: root/<model>, root/<org>/<model>,
        # root/models/<model>. Deeper structures are almost always
        # components, not checkpoints.
        candidates: List[Path] = [root_path]
        for depth in range(3):
            new_candidates: List[Path] = []
            for candidate in candidates:
                try:
                    for child in candidate.iterdir():
                        if child.is_dir() and not child.name.startswith("."):
                            new_candidates.append(child)
                except Exception:
                    continue
            candidates.extend(new_candidates)

        for candidate in candidates:
            if len(specs) >= max_dirs:
                break
            if candidate in (root_path,) and candidate.name in ("hub", "models"):
                continue
            if not _is_mlxgen_model_dir(candidate):
                continue
            real = str(candidate.resolve())
            if real in seen_roots:
                continue
            seen_roots.add(real)

            name_lower = candidate.name.lower()
            quant = _guess_quantization(candidate.name)
            if not quant:
                for token in ("8bit", "4bit", "6bit", "bf16"):
                    if token in name_lower:
                        quant = token
                        break

            specs.append(
                ModelSpec(
                    id=real,
                    provider=PROVIDER_MLXGEN,
                    capabilities=[CAP_IMAGE],
                    path=real,
                    size_bytes=_dir_size(candidate),
                    quantization=quant,
                    family="mlx-gen",
                    metadata={"executable": DEFAULT_MLXGEN_BIN},
                )
            )

    specs.sort(key=lambda s: s.size_bytes, reverse=True)
    return specs


# ─────────────────────────────────────────────────────────────────────
# Aggregate discovery
# ─────────────────────────────────────────────────────────────────────

def discover_all(
    ollama_url: str = DEFAULT_OLLAMA_URL,
    llamacpp_url: str = DEFAULT_LLAMACPP_URL,
    scan_roots: Optional[Iterable[str]] = None,
    include_mlxgen: bool = True,
    verbose: bool = False,
) -> Dict[str, List[ModelSpec]]:
    """Discover every reachable model source, grouped by provider.

    Never raises: an unreachable server simply yields an empty list.
    """
    result: Dict[str, List[ModelSpec]] = {
        PROVIDER_OLLAMA: [],
        PROVIDER_LLAMACPP: [],
        PROVIDER_GGUF: [],
        PROVIDER_MLXGEN: [],
    }

    result[PROVIDER_OLLAMA] = probe_ollama(ollama_url)
    if verbose:
        print(f"  [discover] ollama {ollama_url}: {len(result[PROVIDER_OLLAMA])} models")

    # The llama.cpp server may not be running; that is expected and fine.
    llamacpp = probe_openai_compatible(llamacpp_url)
    if llamacpp:
        result[PROVIDER_LLAMACPP] = llamacpp
    if verbose:
        print(f"  [discover] llamacpp {llamacpp_url}: {len(llamacpp)} models")

    result[PROVIDER_GGUF] = scan_gguf(scan_roots)
    if verbose:
        print(f"  [discover] gguf files: {len(result[PROVIDER_GGUF])}")

    if include_mlxgen:
        result[PROVIDER_MLXGEN] = scan_mlxgen_models(scan_roots)
        if verbose:
            print(f"  [discover] mlxgen models: {len(result[PROVIDER_MLXGEN])}")

    return result


def all_specs(discovered: Dict[str, List[ModelSpec]]) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for specs in discovered.values():
        out.extend(specs)
    return out


def specs_for_capability(discovered: Dict[str, List[ModelSpec]], capability: str) -> List[ModelSpec]:
    return [s for s in all_specs(discovered) if s.supports(capability)]


def find_spec(capability: str, provider: Optional[str] = None,
              identifier: Optional[str] = None) -> Optional[ModelSpec]:
    """Locate one discovered spec by capability + optional provider/id filters."""
    discovered = discover_all()
    for spec in specs_for_capability(discovered, capability):
        if provider and spec.provider != provider:
            continue
        if identifier and identifier not in (spec.id, spec.path or ""):
            continue
        return spec
    return None


# ─────────────────────────────────────────────────────────────────────
# Cached MLX-Gen resolution
# ─────────────────────────────────────────────────────────────────────

# WHY A CACHE: the pipeline resolves the image model once per generated image
# (eight times per run, plus retries). A full discover_all() probes Ollama over
# HTTP and walks the model roots — measured at ~0.3s here — so resolving
# uncached would add seconds to every run for an answer that cannot change
# while the process is alive. Models are not hot-plugged mid-run, so a
# process-lifetime cache is safe.
_MLXGEN_SPEC_CACHE: Dict[str, Optional[ModelSpec]] = {}


def resolve_mlxgen_model(match: str, *, use_cache: bool = True) -> Optional[ModelSpec]:
    """Resolve a substring to one discovered MLX-Gen image model.

    The generation profile names its model by a short identifier (for example
    ``qwen-image-2512-4bit``) rather than an absolute path, because the path
    differs between machines and between an HF cache snapshot and a local
    checkpoint folder. Resolution matches against both ``id`` and ``path``.

    Returns None when nothing matches. Callers decide whether that is fatal.
    """
    key = (match or "").strip().lower()
    if not key:
        return None
    if use_cache and key in _MLXGEN_SPEC_CACHE:
        return _MLXGEN_SPEC_CACHE[key]

    discovered = discover_all()
    found: Optional[ModelSpec] = None
    for spec in specs_for_capability(discovered, CAP_IMAGE):
        if spec.provider != PROVIDER_MLXGEN:
            continue
        haystack = f"{spec.id} {spec.path or ''}".lower()
        if key in haystack:
            found = spec
            break

    if use_cache:
        _MLXGEN_SPEC_CACHE[key] = found
    return found


def clear_mlxgen_cache() -> None:
    """Drop the resolver cache. Used by tests that change discovery inputs."""
    _MLXGEN_SPEC_CACHE.clear()


def list_mlxgen_models() -> List[ModelSpec]:
    """Every discovered MLX-Gen image model, largest first.

    Used to tell an operator what a profile *could* have matched when its
    configured model is missing.
    """
    discovered = discover_all()
    return [s for s in specs_for_capability(discovered, CAP_IMAGE)
            if s.provider == PROVIDER_MLXGEN]



# ─────────────────────────────────────────────────────────────────────
# Payload builders ("prepare", not "launch")
# ─────────────────────────────────────────────────────────────────────

def build_llamacpp_launch_args(
    spec: ModelSpec,
    port: int = 8080,
    ctx: int = 32768,
    parallel: int = 1,
    kv_cache_type: str = "q8_0",
    batch_size: int = 512,
    reasoning: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Command-line args for a managed llama-server on Apple Silicon.

    WHY THESE SPECIFIC DEFAULTS (all measured on a 32 GiB M1 Max):

      --parallel 1 + --no-cont-batching
          One slot, no batching. Keeps memory predictable and prevents a
          second request from allocating a second KV cache.

      --flash-attn on
          Required for quantized KV cache, and reduces attention memory.

      --cache-type-k/v q8_0
          THE CRITICAL FLAG. At f16, a 32k context KV cache pushed an
          18.5 GB Q4_K_M model past the Metal wired limit and every
          inference failed with "Insufficient Memory
          (kIOGPUCommandBufferCallbackErrorOutOfMemory)" — the server
          started and reported healthy, then errored on the first token.
          Quantizing the KV cache to q8_0 cuts it roughly in half and
          32k context fits comfortably. Quality impact at q8_0 is
          negligible.

      --batch-size 512
          Smaller compute graph buffers than the default 2048, which
          lowers peak Metal allocation.

      --reasoning off  (when requested)
          This Qwen build emits chain-of-thought into `reasoning_content`
          BEFORE its answer. With a small max_tokens the budget is consumed
          by thinking and `content` comes back empty — the pipeline sees a
          silent failure. Disabling reasoning makes the model answer
          directly, which is what every prompt in this pipeline expects.
          Pass reasoning="auto" to leave the model's template default.

    Do not remove the cache-type flags without re-measuring: the failure
    mode is a healthy-looking server that cannot generate a single token.
    """
    model_path = spec.path or spec.id
    args = [
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(ctx),
        "--parallel", str(parallel),
        "--no-cont-batching",
        "--n-gpu-layers", "999",
        "--flash-attn", "on",
        "--cache-type-k", kv_cache_type,
        "--cache-type-v", kv_cache_type,
        "--batch-size", str(batch_size),
        "--log-verbosity", "1",
        "--alias", Path(model_path).name,
    ]
    if reasoning in ("on", "off", "auto"):
        args.extend(["--reasoning", reasoning])
    if extra_args:
        args.extend(extra_args)
    return args
