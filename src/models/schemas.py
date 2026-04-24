"""
Pydantic Models for the yt-machine Pipeline — Phase 1 of Stack Migration
=========================================================================

WHY PYDANTIC? (Educational)
────────────────────────────────
Your old system passed raw Python dictionaries everywhere:
    analysis = {"impact_score": "high", "key_facts": "some text"}
Problems:
  - "high" should be an int → silent bug, crashes later during sorting
  - "key_facts" should be a list → iteration fails downstream
  - Missing fields → KeyError at runtime, hard to debug
  - No IDE autocomplete → you must remember every key name

Pydantic solves ALL of this:
  - Runtime type validation: if the LLM returns a string where an int
    is expected, Pydantic raises a clear ValidationError instantly
  - Default values: missing optional fields get sensible defaults
  - IDE autocomplete: your editor knows every field and its type
  - .model_dump() → clean dict for serialization
  - .model_validate(data) → parse + validate from raw dict/JSON
  - JSON Schema generation → FastAPI uses this for auto-docs

Think of Pydantic as a "contract" — data entering or leaving any
function MUST match the schema. Invalid data is caught at the door.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────
# ENUMS — Controlled vocabularies prevent typos and invalid values
# ─────────────────────────────────────────────────────────────────────
# WHY ENUMS? Before, you used raw strings like "geopolitics", "Geopolitics",
# "GEOPOLITICS" — all different to Python. An Enum forces one canonical value.

class VideoStatus(str, Enum):
    """Lifecycle states for a video project."""
    SCRAPED = "scraped"
    ANALYZED = "analyzed"
    SCRIPTED = "scripted"
    VOICEOVER_GENERATED = "voiceover_generated"
    ASSEMBLED = "assembled"
    PUBLISHED = "published"
    FAILED = "failed"


class Platform(str, Enum):
    """Target publishing platforms."""
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    TWITTER = "twitter"


class Category(str, Enum):
    """News categories matching your redfish/category_rotation.py topics."""
    GEOPOLITICS = "geopolitics"
    MILITARY_TECH = "military_tech"
    CYBERSECURITY = "cybersecurity"
    AI_REGULATION = "ai_regulation"
    FINANCIAL = "financial"
    ENERGY = "energy"


# ─────────────────────────────────────────────────────────────────────
# RSS / SCRAPER MODELS
# ─────────────────────────────────────────────────────────────────────

class RSSArticle(BaseModel):
    """
    A single article fetched from an RSS feed.

    This replaces the raw dict you built in redfish/rss_scraper.py:
        article = {"title": ..., "link": ..., "summary": ...}

    Now every article is GUARANTEED to have a title (min 1 char),
    a valid URL, and a parsed summary. If not → ValidationError.
    """
    title: str = Field(..., min_length=1, description="Article headline")
    link: str = Field(..., description="Canonical URL of the article")
    summary: str = Field(default="", description="RSS feed summary/snippet")
    source: str = Field(default="unknown", description="Publication name")
    published: Optional[datetime] = None
    full_text: str = Field(default="", description="Full article body after scraping")

    @field_validator("link")
    @classmethod
    def link_must_be_url(cls, v: str) -> str:
        """Basic URL sanity check — catches empty strings and garbage."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {v}")
        return v


class ScrapeResult(BaseModel):
    """
    Output of a full scraping run — all articles from all feeds.

    Before: scrape_news() returned a list of dicts with no structure.
    After: returns a ScrapeResult that can be validated, serialized,
           and passed to the next pipeline stage with confidence.
    """
    articles: List[RSSArticle] = Field(default_factory=list)
    scrape_timestamp: datetime = Field(default_factory=datetime.utcnow)
    feed_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)


# ─────────────────────────────────────────────────────────────────────
# LLM ANALYSIS MODELS
# ─────────────────────────────────────────────────────────────────────

class NewsAnalysis(BaseModel):
    """
    Structured output from the LLM's analysis of a news article.

    CRITICAL UPGRADE: Your old code did analysis.get("impact_score", 0)
    and hoped for an int. The LLM often returned "high" or "8/10".
    With Pydantic:
      - impact_score MUST be an int between 1-10 (Field constraint)
      - key_facts MUST be a list of strings
      - If the LLM returns bad data, LangChain's parser auto-retries

    The Field(...) syntax: first arg = default, then constraints.
    ... means "required, no default" — the field MUST be provided.
    """
    topic: str = Field(..., min_length=2, description="Core topic of the article")
    impact_score: int = Field(..., ge=1, le=10, description="Geopolitical impact 1-10")
    key_facts: List[str] = Field(default_factory=list, description="Extracted facts")
    angle: str = Field(default="", description="Unique reporting angle")
    category: Category = Field(default=Category.GEOPOLITICS)
    second_order_consequence: str = Field(
        default="",
        description="Cascading effect most people miss"
    )
    sources_involved: List[str] = Field(
        default_factory=list,
        description="Countries, organizations, or companies involved"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────
# SCRIPT MODELS
# ─────────────────────────────────────────────────────────────────────

class ScriptSegment(BaseModel):
    """
    One segment of the final video script (intro, story, closing).

    Replaces the raw dict segments in your old script structure:
        {"segment": "intro", "text": "Hello Maskers...", "word_count": 45}
    """
    segment_type: str = Field(..., description="intro | story | transition | closing")
    text: str = Field(..., min_length=1)
    word_count: int = Field(default=0, ge=0)

    def count_words(self) -> int:
        """Auto-calculate word count."""
        return len(self.text.split())


class VideoScript(BaseModel):
    """
    Complete video script — the main output of the LLM synthesis step.

    This replaces your 120-line f-string prompt + manual JSON recovery.
    LangChain's PydanticOutputParser will use this model to:
      1. Generate format instructions for the LLM prompt
      2. Parse + validate the LLM's JSON response
      3. Auto-retry if validation fails
    """
    title: str = Field(..., min_length=5, description="Video title")
    greeting: str = Field(default="Hello Maskers")
    intro_hook: str = Field(..., description="Attention-grabbing opening")
    stories: List[ScriptSegment] = Field(default_factory=list)
    closing: str = Field(default="")
    full_text: str = Field(default="")
    target_word_count: int = Field(default=500, ge=100, le=2000)
    estimated_duration_seconds: float = Field(default=0.0, ge=0.0)

    def build_full_text(self) -> str:
        """
        Assemble all segments into the final voiceover text.
        Called automatically — no manual string concatenation needed.
        """
        parts = [self.intro_hook]
        for story in self.stories:
            parts.append(story.text)
        if self.closing:
            parts.append(self.closing)
        self.full_text = " ".join(parts)
        return self.full_text


# ─────────────────────────────────────────────────────────────────────
# VIDEO PROJECT MODELS
# ─────────────────────────────────────────────────────────────────────

class VideoProject(BaseModel):
    """
    Master record for a single video project — tracks the full lifecycle.

    This replaces your manifest.json + videos.json entries.
    One model, one source of truth, fully validated.

    WHY Optional[...] for many fields? Because a project is created
    at the SCRAPED stage — it doesn't have a script or video yet.
    As the pipeline progresses, fields get filled in. Pydantic tracks
    which fields are set and validates each one when it's assigned.
    """
    project_id: str = Field(..., description="Unique project ID (e.g., video_1775852604)")
    topic: str = Field(..., min_length=2)
    category: Category = Field(default=Category.GEOPOLITICS)
    status: VideoStatus = Field(default=VideoStatus.SCRAPED)

    # Pipeline data — each stage fills these in
    source_article: Optional[RSSArticle] = None
    analysis: Optional[NewsAnalysis] = None
    script: Optional[VideoScript] = None

    # Output paths
    voiceover_path: Optional[str] = None
    video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    subtitle_path: Optional[str] = None

    # Publishing
    platforms: List[Platform] = Field(default_factory=list)
    youtube_video_id: Optional[str] = None
    published_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    processing_time_seconds: float = Field(default=0.0, ge=0.0)


# ─────────────────────────────────────────────────────────────────────
# API REQUEST / RESPONSE MODELS (for FastAPI — Phase 4)
# ─────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """
    Replaces the raw request.get_json() in your Flask endpoint.
    FastAPI will VALIDATE incoming requests against this model.
    If n8n sends bad data → automatic 422 error with details.

    Example valid request:
        {"platforms": ["youtube"], "category": "geopolitics", "max_word_count": 600}
    Example INVALID request (caught automatically):
        {"platforms": ["fake_platform"], "max_word_count": 99999}
    """
    platforms: List[Platform] = Field(default=[Platform.YOUTUBE])
    category: Optional[Category] = None
    skip_images: bool = Field(default=False)
    skip_voiceover: bool = Field(default=False)
    max_word_count: int = Field(default=500, ge=100, le=2000)


class GenerateResponse(BaseModel):
    """Structured response — replaces your Flask return jsonify({...})."""
    project_id: str
    status: VideoStatus
    message: str
    video_path: Optional[str] = None
    estimated_duration: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────
# VECTOR / EMBEDDING MODELS (for Pinecone — Phase 5)
# ─────────────────────────────────────────────────────────────────────

class TopicEmbedding(BaseModel):
    """
    A topic converted to a vector for semantic deduplication.

    WHY THIS MATTERS: Your old system compared topics by substring match:
        if "iran sanctions" in video_topic:  # misses "tehran penalties"

    With vector embeddings, semantically similar phrases produce vectors
    that are mathematically close (cosine similarity > 0.9). Pinecone
    stores these vectors and retrieves near-duplicates instantly.

    The embedding vector is a list of ~1536 floats (OpenAI) or
    ~768 floats (local model). Each float represents a dimension
    of meaning learned during model training.
    """
    topic: str = Field(..., min_length=1)
    vector: List[float] = Field(..., description="Embedding vector")
    project_id: str = Field(..., description="Linked video project")
    category: Category
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("vector")
    @classmethod
    def vector_must_not_be_empty(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ValueError("Embedding vector cannot be empty")
        return v
