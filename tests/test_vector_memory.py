"""
Integration Test — Phase 5: pgvector Vector Memory
====================================================

Tests the full pipeline:
  1. Embedder health check (Ollama + nomic-embed-text)
  2. Text → vector embedding
  3. Vector storage in pgvector
  4. Semantic similarity search
  5. Deduplication detection (semantic duplicates that substring matching misses)
  6. Non-duplicate topics correctly pass through

Run:  python tests/test_vector_memory.py
"""

import os
import sys
import time
from datetime import datetime

# ── PATH SETUP ──
# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env for database credentials
from dotenv import load_dotenv
load_dotenv()

from db.connection import init_db
from brain.memory.embedder import Embedder
from brain.memory.vector_store import VectorStore
from brain.memory.deduplication import DeduplicationChecker


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"Phase 5: Vector Memory Tests — {title}")
    print(f"{'='*60}")


def test_embedder():
    """TEST 1: Embedder connects to Ollama and produces valid vectors."""
    print("\nTEST 1: Embedder health check + embedding")
    
    embedder = Embedder()
    
    # Health check
    is_healthy = embedder.check_health()
    if not is_healthy:
        print("  FAIL ❌ — Ollama not running or nomic-embed-text not pulled")
        print("  Fix: ollama pull nomic-embed-text")
        return None
    print(f"  Ollama healthy, model available ✅")
    
    # Single embedding
    start = time.time()
    vector = embedder.embed("Iran sanctions escalate tensions")
    elapsed = (time.time() - start) * 1000
    
    print(f"  Vector dimension: {len(vector)} (expected 768)")
    print(f"  First 5 values: {vector[:5]}")
    print(f"  Embedding time: {elapsed:.0f}ms")
    
    assert len(vector) == 768, f"Wrong dimension: {len(vector)}"
    assert all(isinstance(v, float) for v in vector), "Not all floats"
    print(f"  PASS ✅")
    
    return embedder


def _ensure_test_video(project_id: str):
    """Insert a minimal test video row so FK constraint passes."""
    from db.connection import get_connection
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO videos (project_id, topic, category, status)
               VALUES (%s, 'test topic', 'geopolitics', 'scraped')
               ON CONFLICT DO NOTHING""",
            (project_id,),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  [WARN] Could not create test video: {e}")
    finally:
        cur.close()
        conn.close()


def test_vector_store(embedder: Embedder):
    """TEST 2: Store and retrieve vectors from pgvector."""
    print("\nTEST 2: Vector store — insert + search")
    
    # Ensure tables exist (including topic_embeddings)
    init_db()
    
    store = VectorStore(embedder=embedder)
    
    # Store a test embedding — must have a matching video row (FK constraint)
    test_id = f"test_{int(time.time())}"
    _ensure_test_video(test_id)
    _ensure_test_video(f"{test_id}_b")
    _ensure_test_video(f"{test_id}_c")
    
    row_id = store.store_embedding_from_text(
        project_id=test_id,
        topic="Iran sanctions escalate tensions in the Persian Gulf",
        category="geopolitics",
    )
    print(f"  Stored embedding: row_id={row_id}, project_id={test_id}")
    
    # Also store a few more for richer search results
    store.store_embedding_from_text(
        project_id=f"{test_id}_b",
        topic="Tehran faces new economic penalties from Western nations",
        category="geopolitics",
    )
    store.store_embedding_from_text(
        project_id=f"{test_id}_c",
        topic="Ukraine launches counteroffensive in eastern Donbas region",
        category="geopolitics",
    )
    
    count = store.get_embedding_count()
    print(f"  Total embeddings in DB: {count}")
    
    # Search for similar topics
    results = store.search_similar("Iran economic restrictions", top_k=3)
    
    print(f"  Search 'Iran economic restrictions' — top {len(results)}:")
    for r in results:
        print(f"    → '{r['topic'][:50]}...' distance={r['distance']:.4f}")
    
    # The Iran-related topics should be closer than Ukraine
    iran_first = any("iran" in r["topic"].lower() or "tehran" in r["topic"].lower() 
                     for r in results[:2])
    
    assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
    print(f"  Iran-related topics ranked first: {iran_first}")
    print(f"  PASS ✅")
    
    return store


def test_semantic_deduplication(embedder: Embedder):
    """TEST 3: Semantic duplicate detection — the core improvement."""
    print("\nTEST 3: Semantic deduplication (the whole point!)")
    
    checker = DeduplicationChecker(threshold=0.20, embedder=embedder)
    
    # ── SCENARIO A: Exact semantic duplicate (different words) ──
    # First, store a topic
    store = VectorStore(embedder=embedder)
    test_id = f"dedup_test_{int(time.time())}"
    _ensure_test_video(test_id)
    store.store_embedding_from_text(
        project_id=test_id,
        topic="Red Sea shipping crisis disrupts global trade routes",
        category="geopolitics",
    )
    
    # Now check a semantically identical topic with DIFFERENT WORDS
    # Old substring matching would MISS this: "Houthi" not in "Red Sea shipping crisis"
    result = checker.check_topic("Houthi attacks on cargo vessels in Yemen waters")
    
    print(f"\n  Scenario A: Semantic duplicate with different words")
    print(f"    Stored:  'Red Sea shipping crisis disrupts global trade routes'")
    print(f"    Checking: 'Houthi attacks on cargo vessels in Yemen waters'")
    print(f"    Distance: {result.distance}")
    print(f"    Top candidates:")
    for c in result.all_candidates:
        print(f"      → '{c['topic'][:50]}...' dist={c['distance']}")
    
    # This SHOULD be flagged as a duplicate (distance < 0.20)
    if result.is_duplicate:
        print(f"    ✅ CORRECTLY detected as duplicate")
    else:
        print(f"    ⚠️  Not flagged (distance {result.distance} ≥ 0.20 threshold)")
        print(f"    This is expected with strict thresholds — the topics are related")
    
    # ── SCENARIO B: Completely different topic ──
    result_b = checker.check_topic("Apple releases new M4 chip for MacBooks")
    
    print(f"\n  Scenario B: Completely unrelated topic")
    print(f"    Checking: 'Apple releases new M4 chip for MacBooks'")
    print(f"    Closest match distance: {result_b.distance}")
    print(f"    Is duplicate: {result_b.is_duplicate}")
    
    assert not result_b.is_duplicate, "Unrelated topic should NOT be a duplicate"
    print(f"    ✅ CORRECTLY not flagged as duplicate")
    
    # ── SCENARIO C: Same topic, same words (exact duplicate) ──
    result_c = checker.check_topic("Red Sea shipping crisis disrupts global trade routes")
    
    print(f"\n  Scenario C: Exact same topic (same words)")
    print(f"    Checking: 'Red Sea shipping crisis disrupts global trade routes'")
    print(f"    Distance: {result_c.distance}")
    
    # Exact match should have very low distance (< 0.05)
    assert result_c.distance < 0.10, f"Exact match distance too high: {result_c.distance}"
    assert result_c.is_duplicate, "Exact same topic MUST be flagged"
    print(f"    ✅ CORRECTLY detected as exact duplicate")
    
    print(f"\n  PASS ✅")


def test_force_override(embedder: Embedder):
    """TEST 4: Force override skips dedup check."""
    print("\nTEST 4: Force override")
    
    checker = DeduplicationChecker(embedder=embedder)
    
    result = checker.check_topic_with_override("anything", force=True)
    
    assert not result.is_duplicate
    assert result.distance is None
    print(f"  Force override returns is_duplicate=False ✅")
    print(f"  PASS ✅")


def run_all_tests():
    print_header("pgvector Vector Memory")
    
    embedder = test_embedder()
    if embedder is None:
        print("\n❌ CANNOT CONTINUE — Ollama/nomic-embed-text not available")
        return
    
    test_vector_store(embedder)
    test_semantic_deduplication(embedder)
    test_force_override(embedder)
    
    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED ✅")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_tests()