"""Small, validated runtime knobs for the production pipeline."""

from __future__ import annotations

import re


DEFAULT_NUM_STORIES = 2
IMAGES_PER_STORY = 4
DEFAULT_NUM_IMAGES = DEFAULT_NUM_STORIES * IMAGES_PER_STORY


# ── script length budget ──────────────────────────────────────────────
#
# WHY THIS LIVES HERE AND NOT IN llm_interface
# The synthesizer previously hardcoded MIN_WORDS=130 / MAX_WORDS=170 while the
# prompt asked for "150-170 words", and the per-segment limits summed to
# 112-197. Three budgets, mutually inconsistent, so a script could satisfy one
# and violate another — which is how real runs came out at 296-325 words
# (~118-130s) while the enforcement believed it was targeting 130-170.
#
# One derivation, one band. Changing the target duration changes the word
# budget automatically, so the two cannot drift apart again.
#
# 2.5 words/second is the conversion the pipeline already uses everywhere
# (`word_count / 2.5`), so the budget and the duration estimate agree by
# construction rather than by coincidence.
WORDS_PER_SECOND = 2.5
TARGET_VIDEO_SECONDS = (60, 70)
MIN_WORDS = int(TARGET_VIDEO_SECONDS[0] * WORDS_PER_SECOND)   # 150
MAX_WORDS = int(TARGET_VIDEO_SECONDS[1] * WORDS_PER_SECOND)   # 175


# Per-beat word budget, per story. Ranges, not fixed values — the synthesizer
# needs room to write naturally.
#
# WHY A TABLE
# The previous compression instruction trimmed part_1/part_2 while telling the
# model to leave real_talk, fallout and segue "as-is". Those locked fields were
# 106 words on the reference run and the minimum trimmed parts were 80, so the
# best achievable total was 186 against a 170 ceiling — the instruction could
# never reach its own target, which is why every retry failed. Every field now
# has an explicit range, and `beat_budget_bounds()` proves the ranges are
# compatible with the global band below.
BEAT_WORD_RANGES = {
    "part_1_narration": (20, 23),
    "part_2_narration": (24, 29),
    "real_talk": (14, 17),
    "fallout": (12, 15),
    "segue": (6, 10),          # story 1 only
}


def _beat_budget_arithmetic():
    """Import-time check: the per-beat ranges must fit the global band.

    A mismatch here is a configuration error that would silently make
    enforcement unsatisfiable, which is exactly the defect this table replaced.
    Raising at import is deliberate: it fails on the first test run rather than
    producing over-long scripts months later.
    """
    stories = DEFAULT_NUM_STORIES
    per_story_lo = sum(
        lo for field, (lo, _) in BEAT_WORD_RANGES.items() if field != "segue"
    )
    per_story_hi = sum(
        hi for field, (_, hi) in BEAT_WORD_RANGES.items() if field != "segue"
    )
    segue_lo, segue_hi = BEAT_WORD_RANGES["segue"]
    # Segues exist between stories: one fewer than the story count.
    segue_count = max(stories - 1, 0)

    lowest = per_story_lo * stories + segue_lo * segue_count
    highest = per_story_hi * stories + segue_hi * segue_count

    if highest < MAX_WORDS or lowest > MIN_WORDS:
        raise ValueError(
            "beat word ranges cannot reach the global band: beats span "
            f"{lowest}-{highest} words but the target band is "
            f"{MIN_WORDS}-{MAX_WORDS}. Adjust BEAT_WORD_RANGES or "
            "TARGET_VIDEO_SECONDS."
        )


def beat_budget_bounds() -> dict:
    """Lowest/highest totals the per-beat ranges can produce, and the band."""
    stories = DEFAULT_NUM_STORIES
    per_story_lo = sum(
        lo for field, (lo, _) in BEAT_WORD_RANGES.items() if field != "segue"
    )
    per_story_hi = sum(
        hi for field, (_, hi) in BEAT_WORD_RANGES.items() if field != "segue"
    )
    segue_lo, segue_hi = BEAT_WORD_RANGES["segue"]
    segue_count = max(stories - 1, 0)
    return {
        "lowest": per_story_lo * stories + segue_lo * segue_count,
        "highest": per_story_hi * stories + segue_hi * segue_count,
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
    }


_beat_budget_arithmetic()


# Fields that are SPOKEN. Order mirrors the segment timeline.
#
# The closing is included deliberately. It was previously omitted from the
# enforcement counter while `full_text` (built from the segment timeline)
# includes it, so the counter and the artifact disagreed by the length of the
# closing — 20 words on the reference run. Counting what is actually narrated
# is the only way the enforcement can bound the real duration.
_NARRATED_FIELDS = ("part_1_narration", "part_2_narration", "real_talk",
                    "fallout", "segue")

# Story separator markers ("....") carry a pause, not words. They are present
# in `full_text` and in the segment timeline, so every word count must exclude
# them or the reported duration is inflated by the number of separators.
_SEPARATOR_RE = re.compile(r"\.{3,}")


def count_spoken_words(text: str | None) -> int:
    """Words in a narration string, excluding separator markers.

    Use this when only the assembled text is available (the pipeline rebuilds
    `full_text` at several stages). Use :func:`count_narrated_words` when the
    structured script is available.
    """
    if not text:
        return 0
    return len(_SEPARATOR_RE.sub(" ", text).split())


def count_narrated_words(script: dict | None) -> int:
    """Words that will be spoken, including the closing.

    Single source of truth for "how long is this script". Used by the
    synthesizer's enforcement and by the pipeline when it records
    ``script['word_count']``, so the number being validated and the number
    being reported are the same quantity.

    Prefers the segment timeline when it exists: that is the authoritative
    narration order, and it is what `full_text` is derived from. Before the
    timeline is built (inside the synthesizer's enforcement) the structured
    story fields are counted instead. The two agree — verified on real
    manifests — because the timeline is generated from those fields.
    """
    if not isinstance(script, dict):
        return 0

    timeline = script.get("segment_timeline")
    if isinstance(timeline, list) and timeline:
        joined = " ".join(
            seg.get("text", "") for seg in timeline if isinstance(seg, dict)
        )
        return count_spoken_words(joined)

    parts = [script.get("greeting", ""), script.get("intro_hook", "")]
    for story in script.get("stories", []) or []:
        if not isinstance(story, dict):
            continue
        for field in _NARRATED_FIELDS:
            parts.append(story.get(field, ""))
    parts.append(script.get("closing", ""))
    return count_spoken_words(" ".join(p for p in parts if p))


def resolve_image_limit(raw: str | None, maximum: int = DEFAULT_NUM_IMAGES) -> int:
    """Resolve the optional image-count override used by smoke runs.

    The production default remains ``maximum``. A value outside the complete
    pipeline's supported range is rejected so a typo cannot silently create a
    partial or empty video.
    """
    if maximum < 1:
        raise ValueError("maximum image count must be positive")
    if raw is None or not raw.strip():
        return maximum

    try:
        value = int(raw.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"YT_IMAGE_LIMIT must be an integer from 1 to {maximum}; got {raw!r}"
        ) from exc

    if not 1 <= value <= maximum:
        raise ValueError(
            f"YT_IMAGE_LIMIT must be an integer from 1 to {maximum}; got {raw!r}"
        )
    return value


# ── delivery copy ─────────────────────────────────────────────────────
#
# The assembler produces a lossless master (~38 Mbit/s). Delivery providers
# impose hard ceilings — Telegram refuses anything over 50 MB — so a separate
# size-bounded copy is produced for upload. See src/video/media_export.py.
DEFAULT_DELIVERY_MAX_MB = 50
DEFAULT_DELIVERY_CRF = 20

# Telegram's Bot API cap is 50 MB and is the tightest of the consumers, which
# is why DEFAULT_DELIVERY_MAX_MB is that number rather than something larger.
_DELIVERY_MAX_MB_CEILING = 2000


def resolve_delivery_enabled(raw: str | None) -> bool:
    """Whether to produce a size-bounded delivery copy.

    Defaults to on. Anything in the falsey set disables it, matching the
    convention of the other boolean toggles (USE_LOCAL_FLUX, USE_KOKORO), so
    an operator is not surprised by a new value being truthy.
    """
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() in ("true", "1", "yes", "on")


def resolve_delivery_max_mb(raw: str | None, default: int = DEFAULT_DELIVERY_MAX_MB) -> int:
    """Resolve the delivery size ceiling in megabytes.

    Rejects non-integers and out-of-range values so a typo cannot silently
    disable the size guard or produce an unusable copy.
    """
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"YT_DELIVERY_MAX_MB must be an integer from 1 to "
            f"{_DELIVERY_MAX_MB_CEILING}; got {raw!r}"
        ) from exc
    if not 1 <= value <= _DELIVERY_MAX_MB_CEILING:
        raise ValueError(
            f"YT_DELIVERY_MAX_MB must be an integer from 1 to "
            f"{_DELIVERY_MAX_MB_CEILING}; got {raw!r}"
        )
    return value


def resolve_delivery_crf(raw: str | None, default: int = DEFAULT_DELIVERY_CRF) -> int:
    """Resolve the delivery CRF. 0-51 is the x264 range; 18-28 is useful."""
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"YT_DELIVERY_CRF must be an integer from 0 to 51; got {raw!r}"
        ) from exc
    if not 0 <= value <= 51:
        raise ValueError(f"YT_DELIVERY_CRF must be an integer from 0 to 51; got {raw!r}")
    return value


def needs_delivery(master_bytes: int, max_bytes: int) -> bool:
    """Whether a master needs a delivery copy.

    Below the ceiling the master is served directly and no second file is
    written, so short runs keep the exact behaviour they had before the
    delivery stage existed.
    """
    return master_bytes > max_bytes


# ── retention ─────────────────────────────────────────────────────────
#
# Storage lifecycle for output/projects/ and output/images/. See
# src/video/retention.py for the policy and its safety rules.
#
# DEFAULT IS REPORT-ONLY ("report"): a run reports what could be reclaimed and
# deletes nothing. Deletion requires an explicit mode or the --cleanup CLI
# action. This matches the project's stance against silent destructive
# behaviour — deleting generated output is the sharpest version of that.
DEFAULT_RETENTION_DAYS = 30
DEFAULT_RETENTION_KEEP_LAST = 3
DEFAULT_RETENTION_MODE = "report"

_RETENTION_MODES = ("off", "report", "full", "delivery_only")
_RETENTION_DAYS_CEILING = 3650
_RETENTION_KEEP_LAST_CEILING = 1000


def resolve_retention_enabled(raw: str | None) -> bool:
    """Whether retention participates at all. Defaults to on (in report mode)."""
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() in ("true", "1", "yes", "on")


def resolve_retention_days(raw: str | None, default: int = DEFAULT_RETENTION_DAYS) -> int:
    """Age window in days."""
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"YT_RETENTION_DAYS must be an integer from 0 to "
            f"{_RETENTION_DAYS_CEILING}; got {raw!r}"
        ) from exc
    if not 0 <= value <= _RETENTION_DAYS_CEILING:
        raise ValueError(
            f"YT_RETENTION_DAYS must be an integer from 0 to "
            f"{_RETENTION_DAYS_CEILING}; got {raw!r}"
        )
    return value


def resolve_retention_keep_last(
    raw: str | None, default: int = DEFAULT_RETENTION_KEEP_LAST
) -> int:
    """Minimum number of most-recent projects to retain regardless of age."""
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"YT_RETENTION_KEEP_LAST must be an integer from 0 to "
            f"{_RETENTION_KEEP_LAST_CEILING}; got {raw!r}"
        ) from exc
    if not 0 <= value <= _RETENTION_KEEP_LAST_CEILING:
        raise ValueError(
            f"YT_RETENTION_KEEP_LAST must be an integer from 0 to "
            f"{_RETENTION_KEEP_LAST_CEILING}; got {raw!r}"
        )
    return value


def resolve_retention_mode(raw: str | None, default: str = DEFAULT_RETENTION_MODE) -> str:
    """One of off | report | full | delivery_only."""
    if raw is None or not str(raw).strip():
        return default
    value = str(raw).strip().lower()
    if value not in _RETENTION_MODES:
        raise ValueError(
            f"YT_RETENTION_MODE must be one of "
            f"{', '.join(_RETENTION_MODES)}; got {raw!r}"
        )
    return value
