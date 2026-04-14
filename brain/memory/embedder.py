"""
Embedder Module — Text → Vector via Ollama
============================================

WHAT IS AN EMBEDDING?
─────────────────────
An embedding is a list of floating-point numbers (a "vector") that
represents the SEMANTIC MEANING of a piece of text.

Example: The sentence "Iran sanctions escalate" might become:
    [0.12, -0.45, 0.89, 0.33, -0.67, ..., 0.21]  ← 768 numbers

These numbers capture meaning such that:
  - Similar meanings → vectors close together (high cosine similarity)
  - Different meanings → vectors far apart (low cosine similarity)

This is the foundation of semantic search. Instead of comparing text
character-by-character (like your old substring matching), we compare
the mathematical representations of meaning.

WHY nomic-embed-text?
─────────────────────
It's a 274MB model that runs locally via Ollama:
  - 768-dimensional vectors (good quality, reasonable size)
  - Runs in ~50ms per text on CPU
  - No API key needed (unlike OpenAI embeddings)
  - Matches OpenAI text-embedding-3-small quality on benchmarks

WHY NOT USE THE LLM (llama3.1) FOR EMBEDDINGS?
─────────────────────────────────────────────────
LLMs generate text. Embedding models generate vectors. They're
architecturally different:
  - LLM: decoder-only transformer (predicts next token)
  - Embedding model: encoder transformer (captures full input meaning)

Using an LLM for embeddings would be like using a race car to
plow a field — wrong tool, expensive, and poor results.
"""

import logging
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Ollama API endpoint — same server that runs your LLMs
OLLAMA_BASE_URL = "http://localhost:11434"

# The embedding model — must be pulled first: ollama pull nomic-embed-text
DEFAULT_MODEL = "nomic-embed-text"

# nomic-embed-text produces 768-dimensional vectors.
# This must match the VECTOR(768) column in PostgreSQL.
EMBEDDING_DIMENSION = 768


class Embedder:
    """
    Converts text into vector embeddings using Ollama's local API.

    USAGE:
        embedder = Embedder()
        vector = embedder.embed("Iran sanctions escalate tensions")
        # → [0.12, -0.45, 0.89, ...] (768 floats)

        # Batch embedding (more efficient for multiple texts):
        vectors = embedder.embed_batch(["topic 1", "topic 2", "topic 3"])
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimension: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """
        Convert a single text string into a vector embedding.

        HOW IT WORKS (under the hood):
        1. We send a POST request to Ollama's /api/embeddings endpoint
        2. Ollama tokenizes the text (splits into subwords)
        3. The model processes tokens through its transformer layers
        4. The final hidden state is pooled into a single vector
        5. We receive a list of 768 floats back

        Args:
            text: The text to embed (topic, title, article summary, etc.)

        Returns:
            List of 768 floats representing the text's meaning.

        Raises:
            ConnectionError: If Ollama is not running.
            ValueError: If the text is empty or model returns bad data.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # ── CALL OLLAMA API ──
        # Ollama exposes a REST API at localhost:11434
        # The /api/embeddings endpoint takes a model name and prompt text
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text.strip(),
            },
            timeout=30,  # 30s timeout — embedding should be fast
        )

        if response.status_code != 200:
            raise ConnectionError(
                f"Ollama embedding failed (HTTP {response.status_code}): "
                f"{response.text}"
            )

        data = response.json()

        # The response contains "embedding" — a list of floats
        vector = data.get("embedding")
        if not vector or not isinstance(vector, list):
            raise ValueError(f"Bad embedding response: {data}")

        # Validate dimension matches our PostgreSQL column
        if len(vector) != EMBEDDING_DIMENSION:
            logger.warning(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
                f"got {len(vector)}. Model may have changed."
            )

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts sequentially.

        WHY NOT PARALLEL? Ollama processes requests sequentially anyway
        (single GPU/CPU). Sending 10 concurrent requests just creates
        overhead. Sequential is actually faster for local models.

        For cloud APIs (OpenAI), parallel requests DO help because
        the server has massive parallel capacity.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of vectors (one per input text).
        """
        vectors = []
        for i, text in enumerate(texts):
            try:
                vector = self.embed(text)
                vectors.append(vector)
            except Exception as e:
                logger.error(f"Failed to embed text {i}/{len(texts)}: {e}")
                # Use zero vector as fallback — will have 0 similarity
                # to anything, so it won't cause false dedup matches
                vectors.append([0.0] * EMBEDDING_DIMENSION)

        return vectors

    def check_health(self) -> bool:
        """
        Check if Ollama is running and the embedding model is available.

        Call this at startup to fail fast with a clear error message
        instead of cryptic errors during embedding.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Ollama stores models as "name:tag" (e.g., "nomic-embed-text:latest")
            return any(self.model in name for name in model_names)

        except requests.ConnectionError:
            return False
        except Exception:
            return False

    @property
    def dimension(self) -> int:
        """The number of floats in each embedding vector."""
        return EMBEDDING_DIMENSION