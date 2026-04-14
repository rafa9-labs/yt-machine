"""
Deduplication Module — Semantic Duplicate Detection
====================================================

THIS IS THE BRAIN OF THE MEMORY SYSTEM.
─────────────────────────────────────────
It replaces open-viking/memory_reader.py's check_topic_coverage(),
which used SUBSTRING MATCHING to detect duplicates:

    # OLD (misses semantic duplicates):
    if "iran sanctions" in "tehran economic penalties":  # False! Missed!
        return duplicate

    # NEW (catches semantic duplicates via vector similarity):
    similarity = cosine_distance("iran sanctions", "tehran economic penalties")
    # similarity ≈ 0.06 → very similar → DUPLICATE DETECTED!

HOW THRESHOLDS WORK:
────────────────────
Cosine DISTANCE (not similarity) ranges from 0 to 2:
  - 0.00 = identical meaning
  - 0.15 = very similar (rephrased same topic)
  - 0.30 = related but different angle
  - 0.50+ = different topic entirely

We set a THRESHOLD: if distance < threshold → it's a duplicate.
  - STRICT (0.10): only catches near-exact rephrases
  - BALANCED (0.15): catches rephrases + different wording ← recommended
  - LOOSE (0.25): also catches related topics (might skip valid content)
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from brain.memory.embedder import Embedder
from brain.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DeduplicationResult(BaseModel):
    """Result of a semantic deduplication check."""
    is_duplicate: bool = Field(description="True if this topic was already covered")
    query_topic: str = Field(description="The topic we checked")
    matched_topic: Optional[str] = Field(default=None, description="The existing topic that matched")
    matched_project_id: Optional[str] = Field(default=None, description="The project ID of the matching video")
    distance: Optional[float] = Field(default=None, description="Cosine distance (0=identical, 2=opposite)")
    days_since_match: Optional[int] = Field(default=None, description="Days ago the matching video was created")
    all_candidates: List[Dict[str, Any]] = Field(default_factory=list, description="Top similar results")


DEFAULT_DUPLICATE_THRESHOLD = 0.15
DEFAULT_RECENCY_DAYS = 14


class DeduplicationChecker:
    """
    Checks whether a proposed video topic was already covered.

    USAGE:
        checker = DeduplicationChecker()
        result = checker.check_topic("Iran sanctions escalate")
        if result.is_duplicate:
            print(f"Already covered as: {result.matched_topic}")
    """

    def __init__(
        self,
        threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        recency_days: int = DEFAULT_RECENCY_DAYS,
        embedder: Optional[Embedder] = None,
    ):
        self.threshold = threshold
        self.recency_days = recency_days
        self.embedder = embedder or Embedder()
        self.store = VectorStore(embedder=self.embedder)

    def check_topic(self, topic: str, category: Optional[str] = None) -> DeduplicationResult:
        """Check if a topic has already been covered."""
        results = self.store.search_similar(query_text=topic, top_k=5, category=category)

        if not results:
            return DeduplicationResult(is_duplicate=False, query_topic=topic)

        closest = results[0]
        distance = closest.get("distance", 1.0)
        matched_topic = closest.get("topic", "")
        matched_project_id = closest.get("project_id", "")
        created_at = closest.get("created_at")

        is_recent = True
        days_since = None
        if created_at:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            days_since = (datetime.utcnow() - created_at).days
            is_recent = days_since <= self.recency_days

        is_duplicate = distance < self.threshold and is_recent

        if is_duplicate:
            logger.info(f"[DEDUP] DUPLICATE: '{topic}' ≈ '{matched_topic}' (dist={distance:.4f}, {days_since}d ago)")
        else:
            reason = "not similar enough" if distance >= self.threshold else "too old"
            logger.info(f"[DEDUP] OK: '{topic}' — {reason} (closest: '{matched_topic}', dist={distance:.4f})")

        return DeduplicationResult(
            is_duplicate=is_duplicate,
            query_topic=topic,
            matched_topic=matched_topic if is_duplicate else None,
            matched_project_id=matched_project_id if is_duplicate else None,
            distance=round(distance, 4),
            days_since_match=days_since,
            all_candidates=[
                {"topic": r.get("topic"), "distance": round(r.get("distance", 1.0), 4), "project_id": r.get("project_id")}
                for r in results[:3]
            ],
        )

    def check_topic_with_override(self, topic: str, force: bool = False, category: Optional[str] = None) -> DeduplicationResult:
        """Check topic but allow force-override."""
        if force:
            logger.info(f"[DEDUP] FORCE OVERRIDE — skipping check for '{topic}'")
            return DeduplicationResult(is_duplicate=False, query_topic=topic)
        return self.check_topic(topic, category=category)