"""Small, validated runtime knobs for the production pipeline."""

from __future__ import annotations


DEFAULT_NUM_STORIES = 2
IMAGES_PER_STORY = 4
DEFAULT_NUM_IMAGES = DEFAULT_NUM_STORIES * IMAGES_PER_STORY


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
