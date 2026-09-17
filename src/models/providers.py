"""
Provider Adapters — one interface, multiple local backends.
============================================================

The pipeline speaks to every text model through `TextProvider` and every
embedding model through `EmbeddingProvider`. Concrete adapters translate
that interface to a specific server API:

    OllamaTextProvider   → native Ollama  /api/generate + /api/chat (NDJSON)
    OpenAITextProvider   → OpenAI-compatible /v1/chat/completions (SSE)

WHY ADAPTERS?
  Before, LLMInterface hard-coded Ollama's endpoint format. The confirmed
  Qwen GGUF actually runs on llama-server (OpenAI-compatible). Now the
  active profile selects the adapter and the rest of the code is unchanged.

NO SILENT FALLBACK:
  Providers raise ProviderError when the server is unreachable or returns
  nothing usable. Callers decide whether to retry or abort — we never
  quietly swap in a different model.

STREAMING TIMEOUTS:
  Both adapters stream and enforce two wall-clock limits:
    hard_timeout  — total request duration
    idle_timeout  — max gap between tokens
  When exceeded, the HTTP response is closed immediately.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

from src.models.registry import (
    DEFAULT_OLLAMA_URL,
    PROVIDER_LLAMACPP,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
)

DEFAULT_HARD_TIMEOUT = 600.0
DEFAULT_IDLE_TIMEOUT = 120.0


class ProviderError(RuntimeError):
    """Raised when a provider cannot complete a request."""


# ─────────────────────────────────────────────────────────────────────
# Text providers
# ─────────────────────────────────────────────────────────────────────

class TextProvider:
    """Interface every text backend must implement."""

    name = "base"

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        num_ctx: Optional[int] = None,
        hard_timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        num_ctx: Optional[int] = None,
        hard_timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

    def health(self, timeout: float = 5.0) -> bool:
        raise NotImplementedError

    def list_models(self, timeout: float = 5.0) -> List[str]:
        raise NotImplementedError


def _stream_with_deadlines(
    response: requests.Response,
    extract: "callable",
    hard_timeout: float,
    idle_timeout: float,
) -> str:
    """Consume a streaming response while enforcing hard + idle deadlines.

    `extract(line)` returns the next text delta for a line, or None to skip.
    """
    full: List[str] = []
    start = time.monotonic()
    last_data = time.monotonic()

    try:
        for raw_line in response.iter_lines():
            now = time.monotonic()
            if now - start > hard_timeout:
                raise ProviderError(
                    f"hard timeout exceeded ({hard_timeout:.0f}s) — request aborted"
                )
            if now - last_data > idle_timeout:
                raise ProviderError(
                    f"idle timeout exceeded ({idle_timeout:.0f}s with no data) — request aborted"
                )
            if not raw_line:
                continue
            text = extract(raw_line)
            if text:
                last_data = now
                full.append(text)
    finally:
        try:
            response.close()
        except Exception:
            pass

    return "".join(full).strip()


class OllamaTextProvider(TextProvider):
    """Native Ollama API adapter (NDJSON streaming)."""

    name = PROVIDER_OLLAMA

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 10.0):
        self.base_url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        self.timeout = timeout

    # ── HTTP plumbing ──────────────────────────────────────────────

    def _post_stream(self, endpoint: str, payload: Dict[str, Any]) -> requests.Response:
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=(self.timeout, self.timeout * 30),
                stream=True,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ProviderError(
                f"Ollama not reachable at {self.base_url}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        if response.status_code != 200:
            body = ""
            try:
                body = response.text[:300]
            except Exception:
                pass
            response.close()
            raise ProviderError(
                f"Ollama HTTP {response.status_code} for {endpoint}: {body}"
            )
        return response

    @staticmethod
    def _extract(line: bytes) -> Optional[str]:
        try:
            chunk = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if isinstance(chunk.get("error"), str):
            raise ProviderError(f"Ollama error: {chunk['error']}")
        return chunk.get("response") or (chunk.get("message") or {}).get("content")

    # ── TextProvider ───────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        num_ctx: Optional[int] = None,
        hard_timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx
        if system_prompt:
            payload["system"] = system_prompt

        response = self._post_stream("/api/generate", payload)
        return _stream_with_deadlines(
            response,
            self._extract,
            hard_timeout or DEFAULT_HARD_TIMEOUT,
            idle_timeout or DEFAULT_IDLE_TIMEOUT,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        num_ctx: Optional[int] = None,
        hard_timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx

        response = self._post_stream("/api/chat", payload)
        return _stream_with_deadlines(
            response,
            self._extract,
            hard_timeout or DEFAULT_HARD_TIMEOUT,
            idle_timeout or DEFAULT_IDLE_TIMEOUT,
        )

    def unload(self, model: str, timeout: float = 10.0) -> bool:
        """Ask Ollama to evict the model from memory (keep_alive=0)."""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=timeout,
            )
            return True
        except Exception:
            return False

    # ── health ─────────────────────────────────────────────────────

    def health(self, timeout: float = 5.0) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self, timeout: float = 5.0) -> List[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            if resp.status_code != 200:
                return []
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return []

    def loaded_models(self, timeout: float = 5.0) -> List[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=timeout)
            if resp.status_code != 200:
                return []
            return [m.get("name", m.get("model", "")) for m in resp.json().get("models", [])]
        except Exception:
            return []


class OpenAITextProvider(TextProvider):
    """OpenAI-compatible adapter — llama.cpp server, vLLM, LM Studio, MLX."""

    name = PROVIDER_OPENAI

    # ── HTTP plumbing ──────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_stream(self, endpoint: str, payload: Dict[str, Any]) -> requests.Response:
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                headers=self._headers(),
                timeout=(self.timeout, self.timeout * 30),
                stream=True,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ProviderError(
                f"OpenAI-compatible server not reachable at {self.base_url}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"Request to {self.base_url} failed: {exc}") from exc

        if response.status_code != 200:
            body = ""
            try:
                body = response.text[:300]
            except Exception:
                pass
            response.close()
            raise ProviderError(
                f"HTTP {response.status_code} from {self.base_url}{endpoint}: {body}"
            )
        return response

    def __init__(self, base_url: str, timeout: float = 10.0, api_key: Optional[str] = None):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        # Reasoning models stream chain-of-thought into `reasoning_content`
        # before producing `content`. Calling code only wants the answer, so
        # we separate the two channels. When a response contains ONLY
        # reasoning (the token budget ran out mid-think), that is reported as
        # an explicit error rather than a silent empty string — otherwise the
        # pipeline treats "model is thinking" as "model returned nothing".
        self._reasoning_chars = 0

    @staticmethod
    def _extract_reasoning(line: bytes) -> Optional[str]:
        """Return the reasoning_content delta from an SSE line, if any."""
        if not line.startswith(b"data:"):
            return None
        data = line[5:].strip()
        if data == b"[DONE]":
            return None
        try:
            chunk = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        choices = chunk.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        return delta.get("reasoning_content")

    @staticmethod
    def _extract(line: bytes) -> Optional[str]:
        if not line.startswith(b"data:"):
            return None
        data = line[5:].strip()
        if data == b"[DONE]":
            return None
        try:
            chunk = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if isinstance(chunk.get("error"), str):
            raise ProviderError(f"Server error: {chunk['error']}")
        choices = chunk.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is None and choice.get("text"):
            content = choice["text"]
        return content

    # ── TextProvider ───────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        num_ctx: Optional[int] = None,
        hard_timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._post_stream("/v1/chat/completions", payload)

        # Track reasoning tokens separately. A response that contains only
        # chain-of-thought (token budget exhausted mid-think) must not look
        # like an empty success.
        reasoning_chars = 0

        def _extract(line: bytes) -> Optional[str]:
            nonlocal reasoning_chars
            reasoning = self._extract_reasoning(line)
            if reasoning:
                reasoning_chars += len(reasoning)
            return self._extract(line)

        text = _stream_with_deadlines(
            response,
            _extract,
            hard_timeout or DEFAULT_HARD_TIMEOUT,
            idle_timeout or DEFAULT_IDLE_TIMEOUT,
        )

        if not text and reasoning_chars:
            raise ProviderError(
                f"Model produced only reasoning content ({reasoning_chars} chars) "
                f"and no answer within max_tokens={max_tokens}. This model emits "
                "chain-of-thought before its answer — raise max_tokens or disable "
                "thinking (llama.cpp: --reasoning off or a reasoning-budget)."
            )
        return text

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        num_ctx: Optional[int] = None,
        hard_timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            hard_timeout=hard_timeout,
            idle_timeout=idle_timeout,
        )

    # ── health ─────────────────────────────────────────────────────

    def health(self, timeout: float = 3.0) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self, timeout: float = 5.0) -> List[str]:
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=timeout)
            if resp.status_code != 200:
                return []
            return [m.get("id", "") for m in resp.json().get("data", [])]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────
# Embedding providers
# ─────────────────────────────────────────────────────────────────────

class EmbeddingProvider:
    """Interface for embedding backends."""

    name = "base"

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    def health(self) -> bool:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        return 768


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama /api/embeddings adapter."""

    name = PROVIDER_OLLAMA

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = "nomic-embed-text",
                 dimension: int = 768):
        self.base_url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        self.model = model
        self._dimension = dimension

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text.strip()},
                timeout=30,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ProviderError(
                f"Embedding server not reachable at {self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise ProviderError(
                f"Embedding failed (HTTP {response.status_code}): {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"Embedding response was not JSON: {exc}") from exc

        if isinstance(data.get("embeddings"), list) and data["embeddings"]:
            vector = data["embeddings"][0]
        else:
            vector = data.get("embedding")

        if not vector or not isinstance(vector, list):
            raise ProviderError(f"Bad embedding response: {str(data)[:200]}")

        self._dimension = len(vector)
        return vector

    def health(self) -> bool:
        return bool(self.list_models())

    def list_models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return []
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return []

    @property
    def dimension(self) -> int:
        return self._dimension


class NullEmbeddingProvider(EmbeddingProvider):
    """Explicitly unavailable embeddings — fails loudly instead of faking zeros.

    Callers that treat embeddings as optional should check `available` first.
    """

    name = "none"

    def __init__(self, reason: str = "no embedding model configured"):
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    def embed(self, text: str) -> List[float]:
        raise ProviderError(f"Embeddings unavailable: {self.reason}")

    def health(self) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────

def build_text_provider(profile) -> TextProvider:
    """Create the adapter matching the profile's selected text model."""
    spec = profile.text
    if spec is None:
        raise ProviderError("No text model selected in the model profile")

    provider = spec.provider
    if provider == PROVIDER_OLLAMA:
        return OllamaTextProvider(spec.endpoint or DEFAULT_OLLAMA_URL)

    if provider in (PROVIDER_LLAMACPP, PROVIDER_OPENAI):
        if not spec.endpoint:
            raise ProviderError(
                f"Text model {spec.id} uses provider '{provider}' but has no endpoint"
            )
        return OpenAITextProvider(spec.endpoint, api_key=(spec.metadata or {}).get("api_key"))

    raise ProviderError(
        f"Unsupported text provider '{provider}' for model {spec.id}. "
        "Select an Ollama or llama.cpp/OpenAI-compatible model in model_setup."
    )


def build_embedding_provider(profile) -> EmbeddingProvider:
    """Create the embedding adapter, or a Null embedder when unconfigured."""
    spec = profile.embedding
    if spec is None:
        return NullEmbeddingProvider("no embedding role configured")

    if spec.provider == PROVIDER_OLLAMA:
        return OllamaEmbeddingProvider(
            spec.endpoint or DEFAULT_OLLAMA_URL,
            model=spec.metadata.get("served_model_name") or spec.id,
        )

    return NullEmbeddingProvider(
        f"provider '{spec.provider}' is not supported for embeddings"
    )
