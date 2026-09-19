"""
SceneSpec — structured prompt construction for pixel-art generation.
====================================================================

WHY THIS REPLACES FREE-TEXT ENRICHMENT
--------------------------------------
The previous builder appended category keywords to whatever the LLM wrote.
For any "warfare" story it injected:

    tactical map layout, territory indicators, resource flow arrows,
    data tension visualization

Those are *visual instructions*, not neutral enrichment. They fired on
category alone, regardless of whether the scene wanted an infographic. That
is why unrelated stories converged on the same composition: a large display
on one side, environment in the middle, a figure on the other side, crates
in the foreground.

This module makes composition an explicit, authored decision.

WHY FIELDS AND NOT A LONGER SENTENCE
------------------------------------
Each field answers one question the model must resolve. Writing them
separately keeps the prompt honest: if there is no action, the field is
empty rather than padded. It also makes the text policy enforceable — the
`text_requirements` field is classified, not silently rewritten.

TEXT POLICY
-----------
    A) not visually necessary  -> omitted entirely
    B) contextual/state       -> expressed visually by the model
    C) exact value required   -> NOT sent to the model; rendered
                                 deterministically after generation

The old sanitizer rewrote every quoted literal into generic phrasing, which
deleted meaning and could promote the replacement object to focal point
("$33,000,000,000" -> "a large red indicator" made red gauges dominate).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ── Text classification ────────────────────────────────────────────────

TEXT_OMIT = "omit"           # A: not needed in the artwork
TEXT_CONTEXTUAL = "contextual"  # B: short label the model may render
TEXT_EXACT = "exact"         # C: must be rendered deterministically

# Anything that looks like a precise figure, code, or identifier must not be
# trusted to diffusion. Qwen-Image-2512 renders text far better than FLUX,
# but "better" is not "correct", and a wrong number in an editorial graphic
# is a factual error, not a style problem.
# NOTE on equipment designators (F-35, S-400, ATACMS):
# These are NOT classified as exact text. They are commonly rendered as
# recognisable shapes by the model, and treating them as deferred would
# strip meaningful subject identity from the prompt. Only values where a
# wrong rendering is a *factual error* (money, percentages, tallies) are
# deferred. See `_EXACT_PATTERNS`.
_EXACT_PATTERNS = [
    re.compile(r"[$€£¥]\s?\d[\d,\.]*"),          # currency amounts
    re.compile(r"\b\d{1,3}(?:,\d{3})+\b"),        # 33,000,000 (grouped thousands)
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),           # percentages
    re.compile(r"\b\d{4,}\b"),                    # bare long tallies
]


@dataclass
class TextRequirement:
    """One piece of text the scene implies, with its disposition."""

    text: str
    disposition: str = TEXT_OMIT
    rationale: str = ""

    def to_dict(self) -> dict:
        return {"text": self.text, "disposition": self.disposition,
                "rationale": self.rationale}


def classify_text(raw: str) -> TextRequirement:
    """Decide how to handle a piece of text found in a scene description."""
    if not raw or not raw.strip():
        return TextRequirement(raw, TEXT_OMIT, "empty")

    value = raw.strip()

    for pattern in _EXACT_PATTERNS:
        if pattern.search(value):
            return TextRequirement(
                value, TEXT_EXACT,
                "precise figure/value - must be rendered deterministically, not by diffusion",
            )

    # Short, simple words are the only thing worth letting the model attempt,
    # and even then it is benchmarked rather than assumed.
    if len(value) <= 24 and len(value.split()) <= 3:
        return TextRequirement(
            value, TEXT_CONTEXTUAL,
            "short contextual label - may be rendered by the model",
        )

    return TextRequirement(value, TEXT_OMIT, "too long/ambiguous to render reliably")


# ── SceneSpec ──────────────────────────────────────────────────────────

@dataclass
class SceneSpec:
    """Structured description of one image.

    Field order mirrors the prompt order: a reader (human or model) learns
    what the scene is before how it is styled.
    """

    subject: str = ""
    action: str = ""
    environment: str = ""
    camera: str = "Isometric three-quarter view"
    hierarchy: str = ""
    props: List[str] = field(default_factory=list)
    text_requirements: List[TextRequirement] = field(default_factory=list)
    style: str = ""
    lora_trigger: str = ""

    # ── construction ───────────────────────────────────────────────

    def add_text(self, raw: str) -> "SceneSpec":
        self.text_requirements.append(classify_text(raw))
        return self

    @property
    def renderable_text(self) -> List[str]:
        """Text the model is allowed to attempt (contextual only)."""
        return [t.text for t in self.text_requirements
                if t.disposition == TEXT_CONTEXTUAL]

    @property
    def deferred_text(self) -> List[str]:
        """Text that must be drawn after generation."""
        return [t.text for t in self.text_requirements
                if t.disposition == TEXT_EXACT]

    # ── assembly ───────────────────────────────────────────────────

    def to_prompt(self) -> str:
        """Assemble the generation prompt.

        Order is deliberate:
            trigger -> subject/action -> environment -> props ->
            camera/hierarchy -> style

        The subject leads (after the trigger) so the model weights it most
        heavily. Style is last and short; a long style block placed first
        was measured to crowd out the scene description.
        """
        parts: List[str] = []

        if self.lora_trigger:
            parts.append(self.lora_trigger.rstrip(".") + ".")

        # Subject + action + environment form ONE sentence. They are a
        # single idea ("who does what, where") and the environment usually
        # begins with a prepositional phrase that only reads correctly when
        # attached — "stands beside two damaged hangars. at a desert base"
        # is broken English, and a malformed prompt measurably hurts output.
        lead_parts = [p.strip().rstrip(".") for p in
                      (self.subject, self.action, self.environment) if p]
        lead = " ".join(lead_parts).strip()
        if lead:
            parts.append(lead + ".")

        if self.props:
            # Deduplicate while preserving order.
            seen = set()
            props = [p for p in self.props if not (p.lower() in seen or seen.add(p.lower()))]
            parts.append("Included are " + ", ".join(p.strip().rstrip(".") for p in props) + ".")

        composition = ", ".join(
            p.strip().rstrip(".") for p in (self.camera, self.hierarchy) if p
        ).strip()
        if composition:
            parts.append(composition.capitalize() + ".")

        if self.renderable_text:
            labels = ", ".join(f'"{t}"' for t in self.renderable_text)
            parts.append(f"Visible labelling reads {labels}.")

        if self.style:
            style = self.style.strip().rstrip(".")
            parts.append(style[:1].upper() + style[1:] + ".")

        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "action": self.action,
            "environment": self.environment,
            "camera": self.camera,
            "hierarchy": self.hierarchy,
            "props": list(self.props),
            "text_requirements": [t.to_dict() for t in self.text_requirements],
            "style": self.style,
            "lora_trigger": self.lora_trigger,
            "assembled_prompt": self.to_prompt(),
            "deferred_text": self.deferred_text,
        }


# ── Presets ────────────────────────────────────────────────────────────

# Style tails are intentionally short. The measured failure mode was a
# ~100-word triple style block competing with the ~55-word scene.
STYLE_PIXEL_ART = (
    "crisp pixel art, limited palette, hard pixel edges, flat shading, "
    "readable silhouettes, consistent pixel scale"
)

STYLE_PIXEL_SCENE = (
    "crisp game-art pixel art, limited purposeful palette, hard pixel edges, "
    "flat shading, clean readable silhouettes, consistent pixel scale"
)


def from_visual_scene(scene: dict, trigger: str = "Pixel Art",
                      style: str = STYLE_PIXEL_SCENE) -> SceneSpec:
    """Build a SceneSpec from a pipeline visual-scene dict.

    The pipeline supplies a single prose `description`. We route it into the
    subject/environment slots rather than dumping it wholesale, and we lift
    any quoted literals into the text policy so they are classified instead
    of blindly rewritten.
    """
    description = (scene.get("description") or "").strip()

    # Lift quoted literals before they reach the prompt.
    quoted = re.findall(r"['\"]([^'\"]{1,60})['\"]", description)
    cleaned = re.sub(r"['\"][^'\"]{1,60}['\"]", "", description)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")

    # Drop boilerplate medium prefixes; the style field states the medium.
    cleaned = re.sub(
        r"^\s*(?:\d+-bit\s+)?(?:isometric\s+)?pixel\s+art(?:\s+scene)?\s*[:\-]\s*",
        "", cleaned, flags=re.IGNORECASE,
    )

    spec = SceneSpec(
        subject=cleaned,
        environment="",
        camera="Isometric three-quarter view",
        style=style,
        lora_trigger=trigger,
    )
    for item in quoted:
        spec.add_text(item)
    return spec
