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
