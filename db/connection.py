"""
PostgreSQL Database Connection Module
======================================

WHY POSTGRESQL INSTEAD OF JSON FILES? (Educational)
────────────────────────────────────────────────────
Your old system stored data in:
  - output/projects/video_*/manifest.json  (per-project data)
  - data/videos.json                        (deduplication history)

Problems with JSON file storage:
  1. CONCURRENT WRITES: If two pipeline runs overlap, both read
     videos.json, modify it, and write it back. The second write
     OVERWRITES the first → data loss. PostgreSQL uses ACID
     transactions so concurrent writes are safe.

  2. NO QUERYING: "Show me all geopolitics videos from last month
     with impact_score > 7" requires loading ALL data into Python
     and filtering manually. PostgreSQL does this in milliseconds
     with indexed SQL queries.

  3. NO SCHEMA ENFORCEMENT: A typo like {"impcat_score": 8} silently
     corrupts your data. PostgreSQL columns have types (INTEGER,
     VARCHAR, TIMESTAMP) that reject bad data at the database level.

  4. NO RELATIONSHIPS: Want to link articles → analyses → scripts?
     With JSON, you manually match IDs across files. PostgreSQL
     FOREIGN KEY constraints guarantee referential integrity.

  5. NO BACKUP/RECOVERY: Delete a JSON file and it's gone.
     PostgreSQL has WAL (Write-Ahead Logging) and point-in-time
     recovery.

WHY psycopg2 (asyncpg later)? 
  psycopg2 is the most mature PostgreSQL driver for Python.
  It's synchronous (blocking), which is fine for our pipeline.
  When we move to FastAPI (Phase 4), we'll add asyncpg for
  non-blocking database calls in async endpoints.
"""

import os
import logging
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# CONNECTION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────

def get_connection_string() -> str:
    """
    Build a PostgreSQL connection string from environment variables.

    WHY ENV VARS? Hardcoding credentials in source code is a security
    risk (they end up in Git). The .env file stores secrets locally,
    and environment variables override them in production/Docker.

    The connection string format:
        postgresql://username:password@host:port/database_name

    psycopg2 uses this exact format to connect.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "yt_machine")
    password = os.getenv("POSTGRES_PASSWORD", "changeme")
    database = os.getenv("POSTGRES_DB", "yt_machine")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def get_connection():
    """
    Create and return a new PostgreSQL connection.

    WHY RealDictCursor? By default, psycopg2 returns rows as tuples:
        row = ("video_123", "geopolitics", "published")
        # Access by index: row[0], row[1] — unreadable!

    RealDictCursor returns rows as dictionaries:
        row = {"project_id": "video_123", "category": "geopolitics", ...}
        # Access by name: row["project_id"] — readable!

    This pairs perfectly with Pydantic's .model_validate(dict) which
    accepts dicts natively.
    """
    conn = psycopg2.connect(get_connection_string(), cursor_factory=RealDictCursor)
    return conn


def init_db():
    """
    Initialize the database — create tables if they don't exist.

    This is called once at application startup. The IF NOT EXISTS
    clause makes it safe to run multiple times (idempotent).

    WHY SEPARATE TABLES INSTEAD OF ONE BIG JSON?
    Normalization. Each table holds ONE type of entity:
      - videos: core project tracking
      - articles: scraped news articles
      - analyses: LLM analysis results
      - scripts: generated video scripts
      - topic_embeddings: vector data for deduplication

    Benefits: no duplicate data, efficient queries, easy to add
    new fields to one entity without affecting others.
    """
    conn = get_connection()
    cur = conn.cursor()

    # ── VIDEOS TABLE ──
    # This replaces your videos.json file
    cur.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            -- PRIMARY KEY: unique identifier for each video project
            -- WHY TEXT not INT? Your project IDs are like "video_1775852604"
            project_id TEXT PRIMARY KEY,
            
            -- Core fields
            topic TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'geopolitics',
            status TEXT NOT NULL DEFAULT 'scraped',
            
            -- Check constraints enforce valid values at the DB level
            -- Even if your Python code has a bug, the DB rejects bad data
            CHECK (status IN (
                'scraped', 'analyzed', 'scripted',
                'voiceover_generated', 'assembled', 'published', 'failed'
            )),
            CHECK (category IN (
                'geopolitics', 'military_tech', 'cybersecurity',
                'ai_regulation', 'financial', 'energy'
            )),
            
            -- Output file paths
            voiceover_path TEXT,
            video_path TEXT,
            thumbnail_path TEXT,
            subtitle_path TEXT,
            
            -- Publishing
            platforms TEXT[] DEFAULT '{}',
            youtube_video_id TEXT,
            published_at TIMESTAMP,
            
            -- Metadata
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            error_message TEXT,
            processing_time_seconds REAL DEFAULT 0.0
        );
    """)

    # ── INDEXES ──
    # WHY INDEXES? Without an index, finding videos by category requires
    # scanning EVERY row (full table scan). With an index, PostgreSQL
    # jumps directly to the relevant rows — like a book's index page.
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_videos_category 
        ON videos(category);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_videos_status 
        ON videos(status);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_videos_created_at 
        ON videos(created_at DESC);
    """)

    # ── ARTICLES TABLE ──
    # Stores scraped news articles for reference and deduplication
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            link TEXT NOT NULL,
            summary TEXT DEFAULT '',
            source TEXT DEFAULT 'unknown',
            published TIMESTAMP,
            full_text TEXT DEFAULT '',
            scraped_at TIMESTAMP DEFAULT NOW(),
            
            -- Prevent duplicate articles from the same URL
            UNIQUE(link)
        );
    """)

    # ── ANALYSES TABLE ──
    # Stores LLM analysis results linked to videos
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id SERIAL PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES videos(project_id)
                ON DELETE CASCADE,
            topic TEXT NOT NULL,
            impact_score INTEGER NOT NULL CHECK (impact_score >= 1 AND impact_score <= 10),
            key_facts TEXT[] DEFAULT '{}',
            angle TEXT DEFAULT '',
            category TEXT DEFAULT 'geopolitics',
            second_order_consequence TEXT DEFAULT '',
            sources_involved TEXT[] DEFAULT '{}',
            confidence REAL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # ── SCRIPTS TABLE ──
    # Stores generated video scripts linked to videos
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scripts (
            id SERIAL PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES videos(project_id)
                ON DELETE CASCADE,
            title TEXT NOT NULL,
            greeting TEXT DEFAULT 'Hello Maskers',
            intro_hook TEXT NOT NULL,
            full_text TEXT NOT NULL,
            closing TEXT DEFAULT '',
            target_word_count INTEGER DEFAULT 500,
            estimated_duration_seconds REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # ── TOPIC EMBEDDINGS TABLE ──
    # WHY SEPARATE FROM VIDEOS? Vector data is large (~1536 floats per row).
    # Storing it in the videos table would slow down all video queries.
    # Separate table = fetch vectors only when needed for deduplication.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS topic_embeddings (
            id SERIAL PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES videos(project_id)
                ON DELETE CASCADE,
            topic TEXT NOT NULL,
            category TEXT NOT NULL,
            embedding VECTOR(1536),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database initialized successfully — all tables ready.")


# ─────────────────────────────────────────────────────────────────────
# CRUD HELPERS — Bridge between Pydantic models and PostgreSQL
# ─────────────────────────────────────────────────────────────────────
# WHY THESE HELPERS? They translate between Pydantic model objects
# and SQL queries. Your pipeline code works with Pydantic models
# (type-safe, validated), and these helpers handle the SQL underneath.

def save_project(project_dict: dict) -> None:
    """
    Insert or update a video project in the database.

    WHY UPSERT (INSERT ... ON CONFLICT)?
    Your pipeline creates a project early (status='scraped') then
    updates it multiple times as it progresses. UPSERT handles both
    the initial insert AND subsequent updates in one safe operation.

    The ON CONFLICT clause says: "if project_id already exists,
    update ALL the listed columns with the new values instead."
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO videos (
            project_id, topic, category, status,
            voiceover_path, video_path, thumbnail_path, subtitle_path,
            platforms, youtube_video_id, published_at,
            updated_at, error_message, processing_time_seconds
        ) VALUES (
            %(project_id)s, %(topic)s, %(category)s, %(status)s,
            %(voiceover_path)s, %(video_path)s, %(thumbnail_path)s, %(subtitle_path)s,
            %(platforms)s, %(youtube_video_id)s, %(published_at)s,
            NOW(), %(error_message)s, %(processing_time_seconds)s
        )
        ON CONFLICT (project_id) DO UPDATE SET
            topic = EXCLUDED.topic,
            category = EXCLUDED.category,
            status = EXCLUDED.status,
            voiceover_path = EXCLUDED.voiceover_path,
            video_path = EXCLUDED.video_path,
            thumbnail_path = EXCLUDED.thumbnail_path,
            subtitle_path = EXCLUDED.subtitle_path,
            platforms = EXCLUDED.platforms,
            youtube_video_id = EXCLUDED.youtube_video_id,
            published_at = EXCLUDED.published_at,
            updated_at = NOW(),
            error_message = EXCLUDED.error_message,
            processing_time_seconds = EXCLUDED.processing_time_seconds
    """, project_dict)

    conn.commit()
    cur.close()
    conn.close()


def get_project(project_id: str) -> Optional[dict]:
    """
    Fetch a single video project by ID.
    Returns a dict (from RealDictCursor) or None if not found.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM videos WHERE project_id = %s", (project_id,))
    row = cur.fetchone()

    cur.close()
    conn.close()
    return dict(row) if row else None


def get_recent_topics(limit: int = 50) -> list:
    """
    Fetch recent video topics for deduplication checks.

    This replaces loading the entire videos.json file and scanning
    every entry. With an indexed query on created_at, PostgreSQL
    returns the latest N topics in microseconds.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT project_id, topic, category, created_at 
        FROM videos 
        ORDER BY created_at DESC 
        LIMIT %s
    """, (limit,))

    results = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results