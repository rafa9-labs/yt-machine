"""
Vector Store Module — pgvector CRUD Operations
================================================

HOW PGVECTOR WORKS (under the hood):
─────────────────────────────────────
pgvector adds a VECTOR column type to PostgreSQL. When you insert
a vector like [0.12, -0.45, 0.89, ...], pgvector stores it as a
compact binary blob (not as 768 separate FLOAT columns).

For similarity search, pgvector computes distances between vectors:

  COSINE DISTANCE (1 - cosine_similarity):
    Measures the ANGLE between two vectors, ignoring magnitude.
    Range: 0 (identical) to 2 (opposite).
    
    Example: "Iran sanctions" vs "Tehran economic penalties"
    → distance ≈ 0.06 (very similar, angle between vectors is tiny)
    
    Example: "Iran sanctions" vs "Ukraine counteroffensive"  
    → distance ≈ 0.69 (very different, vectors point apart)

  WHY COSINE (not Euclidean or Dot Product)?
    - Text embeddings are normalized (unit length), so cosine ≈ dot product
    - Cosine is standard for text similarity in NLP literature
    - pgvector's <=> operator gives cosine distance directly

  THE QUERY:
    SELECT topic, embedding <=> '[0.12, -0.45, ...]' AS distance
    FROM topic_embeddings
    ORDER BY embedding <=> '[0.12, -0.45, ...]'
    LIMIT 5;

    This finds the 5 closest vectors in ~2ms for <10K rows.
    The <=> operator is the COSINE DISTANCE operator from pgvector.

WHY STORE VECTORS IN POSTGRESQL (not a separate vector DB)?
──────────────────────────────────────────────────────────
  - Your embeddings table is JOINED to your videos table via project_id
  - One database = one backup, one connection string, one Docker container
  - At your scale (~1800 vectors/year), pgvector is faster than any
    cloud vector DB (no network latency)
"""

import logging
from typing import List, Dict, Any, Optional

from psycopg2.extras import RealDictCursor

from db.connection import get_connection
from brain.memory.embedder import Embedder, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Stores and queries vector embeddings in PostgreSQL via pgvector.

    This replaces the JSON-file-based memory in open-viking/.

    USAGE:
        store = VectorStore()
        store.store_embedding(
            project_id="video_123",
            topic="Iran sanctions escalate tensions",
            category="geopolitics",
            vector=[0.12, -0.45, 0.89, ...]
        )
        results = store.search_similar("Tehran economic penalties", top_k=5)
    """

    def __init__(self, embedder: Optional[Embedder] = None):
        """
        Args:
            embedder: An Embedder instance for auto-embedding queries.
                     If None, creates one with default settings.
        """
        # The embedder is used by search_similar() to convert the
        # query text into a vector before searching pgvector.
        self.embedder = embedder or Embedder()

    # ─────────────────────────────────────────────────────────────────
    # WRITE OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_embedding(
        self,
        project_id: str,
        topic: str,
        category: str,
        vector: List[float],
    ) -> int:
        """
        Store a topic embedding in PostgreSQL.

        Args:
            project_id: The video project ID (foreign key to videos table).
            topic: The topic text (e.g., "Iran sanctions escalate").
            category: The content category (e.g., "geopolitics").
            vector: The 768-float embedding from Ollama.

        Returns:
            The database row ID of the stored embedding.

        WHY STORE THE TOPIC TEXT ALONGSIDE THE VECTOR?
        When we find a duplicate via similarity search, we need to
        show the user WHAT the duplicate is. Storing topic text lets
        us return "Duplicate of: 'Iran sanctions' (video_456, 2 days ago)"
        without a JOIN query.
        """
        conn = get_connection()
        cur = conn.cursor()

        try:
            # Convert Python list → PostgreSQL vector string
            # pgvector expects the format: '[0.12, -0.45, 0.89, ...]'
            vector_str = self._vector_to_pgstr(vector)

            cur.execute(
                """
                INSERT INTO topic_embeddings (project_id, topic, category, embedding)
                VALUES (%s, %s, %s, %s::vector)
                RETURNING id
                """,
                (project_id, topic, category, vector_str),
            )

            row_id = cur.fetchone()["id"]
            conn.commit()
            return row_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store embedding for '{topic}': {e}")
            raise
        finally:
            cur.close()
            conn.close()

    def store_embedding_from_text(
        self,
        project_id: str,
        topic: str,
        category: str,
    ) -> int:
        """
        Convenience method: embed text and store in one call.

        This is what you'll use most often in the pipeline:
            store.store_embedding_from_text(
                project_id="video_123",
                topic="Iran sanctions escalate",
                category="geopolitics"
            )
        """
        vector = self.embedder.embed(topic)
        return self.store_embedding(project_id, topic, category, vector)

    # ─────────────────────────────────────────────────────────────────
    # READ / SEARCH OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the most semantically similar past topics.

        HOW THIS QUERY WORKS:
        1. query_text → vector (via Ollama embedding)
        2. PostgreSQL computes cosine distance between query vector
           and EVERY stored vector (with an index, it's approximate NN)
        3. Returns the top_k closest matches with distance scores

        Args:
            query_text: The new topic to check against past topics.
            top_k: How many results to return (default 5).
            category: Optional filter to only check within a category.

        Returns:
            List of dicts with keys: id, project_id, topic, category,
            distance, created_at
            Sorted by distance (closest first).

        INTERPRETING DISTANCE VALUES:
            distance = 0.00 → exact same meaning (same topic)
            distance < 0.15 → very similar (likely duplicate)
            distance < 0.30 → related (same region/conflict, different angle)
            distance > 0.50 → different topic entirely
        """
        query_vector = self.embedder.embed(query_text)
        vector_str = self._vector_to_pgstr(query_vector)

        conn = get_connection()
        cur = conn.cursor()

        try:
            # ── COSINE DISTANCE SEARCH ──
            # The <=> operator is pgvector's COSINE DISTANCE operator.
            # It computes: 1 - cos(A, B) where A and B are vectors.
            # Lower distance = more similar.
            if category:
                cur.execute(
                    """
                    SELECT 
                        id, project_id, topic, category, created_at,
                        embedding <=> %s::vector AS distance
                    FROM topic_embeddings
                    WHERE category = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_str, category, vector_str, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        id, project_id, topic, category, created_at,
                        embedding <=> %s::vector AS distance
                    FROM topic_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_str, vector_str, top_k),
                )

            results = []
            for row in cur.fetchall():
                results.append(dict(row))

            return results

        finally:
            cur.close()
            conn.close()

    def get_all_embeddings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch all stored embeddings (for debugging/migration)."""
        conn = get_connection()
        cur = conn.cursor()

        try:
            cur.execute(
                """
                SELECT id, project_id, topic, category, created_at
                FROM topic_embeddings
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            cur.close()
            conn.close()

    def get_embedding_count(self) -> int:
        """Count total stored embeddings."""
        conn = get_connection()
        cur = conn.cursor()

        try:
            cur.execute("SELECT COUNT(*) as count FROM topic_embeddings")
            return cur.fetchone()["count"]
        finally:
            cur.close()
            conn.close()

    # ─────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _vector_to_pgstr(vector: List[float]) -> str:
        """
        Convert a Python list of floats to PostgreSQL vector string.

        Python:     [0.12, -0.45, 0.89]
        PostgreSQL: '[0.12,-0.45,0.89]'

        The ::vector cast in SQL tells PostgreSQL to interpret this
        string as a pgvector VECTOR type.
        """
        return "[" + ",".join(str(v) for v in vector) + "]"