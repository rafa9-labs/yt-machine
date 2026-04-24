"""
Vector Memory Module
====================
Provides semantic memory for the yt-machine pipeline using
pgvector + Ollama embeddings for topic deduplication.

Import specific modules directly:
    from src.brain.memory.embedder import Embedder
    from src.brain.memory.vector_store import VectorStore
    from src.brain.memory.deduplication import DeduplicationChecker
"""

__all__ = ["Embedder", "VectorStore", "DeduplicationChecker"]
