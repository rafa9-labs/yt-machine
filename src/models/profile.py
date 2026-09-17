"""
Model Profile — persisted role→model selections.
=================================================

A profile maps pipeline roles to concrete ModelSpecs:

    text       → the LLM used for analysis/scriptwriting  (required)
    image      → the image generator                      (required unless skipped)
    vision     → optional VQA/vision model                (nullable)
    embedding  → optional embedding model                 (nullable)

WHY A SEPARATE FILE?
  config/system_prompts.json holds prompts (hand-edited). The profile is
  machine-written by tools/model_setup.py. Keeping them separate means
  re-running setup never clobbers prompt engineering.

LOCATION:
  Default: config/model_profile.json (gitignored).
  Override: YT_MODEL_PROFILE=/path/to/profile.json

SECURITY:
  No secrets are stored here. `metadata` may carry env var *names*, never
  values.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.models.registry import (
    CAP_EMBEDDING,
    CAP_IMAGE,
    CAP_TEXT,
    CAP_VISION,
    ModelSpec,
)

log = logging.getLogger(__name__)

ROLE_TEXT = "text"
ROLE_IMAGE = "image"
ROLE_VISION = "vision"
ROLE_EMBEDDING = "embedding"

ROLE_CAPABILITY = {
    ROLE_TEXT: CAP_TEXT,
    ROLE_IMAGE: CAP_IMAGE,
    ROLE_VISION: CAP_VISION,
    ROLE_EMBEDDING: CAP_EMBEDDING,
}

REQUIRED_ROLES = (ROLE_TEXT, ROLE_IMAGE)
OPTIONAL_ROLES = (ROLE_VISION, ROLE_EMBEDDING)

PROFILE_VERSION = 1

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "model_profile.json"


def profile_path() -> Path:
    override = os.getenv("YT_MODEL_PROFILE")
    return Path(override) if override else DEFAULT_PROFILE_PATH


class ProfileError(RuntimeError):
    """Raised when the persisted profile is missing or unusable."""


@dataclass
class ModelProfile:
    """Role → ModelSpec selections plus runtime preferences."""

    text: Optional[ModelSpec] = None
    image: Optional[ModelSpec] = None
    vision: Optional[ModelSpec] = None
    embedding: Optional[ModelSpec] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    version: int = PROFILE_VERSION

    # ── serialization ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "roles": {
                ROLE_TEXT: self.text.to_dict() if self.text else None,
                ROLE_IMAGE: self.image.to_dict() if self.image else None,
                ROLE_VISION: self.vision.to_dict() if self.vision else None,
                ROLE_EMBEDDING: self.embedding.to_dict() if self.embedding else None,
            },
            "preferences": self.preferences,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelProfile":
        roles = (data or {}).get("roles") or {}
        return cls(
            text=ModelSpec.from_dict(roles[ROLE_TEXT]) if roles.get(ROLE_TEXT) else None,
            image=ModelSpec.from_dict(roles[ROLE_IMAGE]) if roles.get(ROLE_IMAGE) else None,
            vision=ModelSpec.from_dict(roles[ROLE_VISION]) if roles.get(ROLE_VISION) else None,
            embedding=ModelSpec.from_dict(roles[ROLE_EMBEDDING]) if roles.get(ROLE_EMBEDDING) else None,
            preferences=(data or {}).get("preferences") or {},
            version=int((data or {}).get("version", PROFILE_VERSION)),
        )

    # ── persistence ────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else profile_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        log.info("profile.saved", extra={"path": str(path)})
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ModelProfile":
        path = Path(path) if path else profile_path()
        if not path.exists():
            raise ProfileError(
                f"Model profile not found at {path}. "
                "Run: .venv/bin/python tools/model_setup.py"
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"Model profile at {path} is unreadable: {exc}") from exc

        profile = cls.from_dict(data)
        profile.validate()
        return profile

    @classmethod
    def load_or_none(cls, path: Optional[Path] = None) -> Optional["ModelProfile"]:
        try:
            return cls.load(path)
        except ProfileError as exc:
            log.info("profile.missing", extra={"reason": str(exc)})
            return None

    # ── accessors ──────────────────────────────────────────────────

    def get(self, role: str) -> Optional[ModelSpec]:
        return getattr(self, role, None)

    @property
    def text_model(self) -> str:
        if not self.text:
            raise ProfileError("No text model configured in the model profile")
        return self.text.metadata.get("served_model_name") or self.text.id

    @property
    def text_endpoint(self) -> str:
        if not self.text or not self.text.endpoint:
            raise ProfileError("Text model has no endpoint configured")
        return self.text.endpoint.rstrip("/")

    @property
    def text_provider(self) -> str:
        if not self.text:
            raise ProfileError("No text model configured in the model profile")
        return self.text.provider

    def is_local_generation_role(self, role: str) -> bool:
        spec = self.get(role)
        return bool(spec and spec.provider in ("mlxgen", "gguf"))

    # ── validation ─────────────────────────────────────────────────

    def validate(self) -> None:
        errors = []
        if not self.text:
            errors.append("text role is empty")
        elif CAP_TEXT not in self.text.capabilities and CAP_VISION not in self.text.capabilities:
            errors.append(f"text model {self.text.id} lacks text capability")

        if self.image and CAP_IMAGE not in self.image.capabilities:
            errors.append(f"image model {self.image.id} lacks image capability")

        if self.embedding and CAP_EMBEDDING not in self.embedding.capabilities:
            errors.append(f"embedding model {self.embedding.id} lacks embedding capability")

        if errors:
            raise ProfileError("Invalid model profile: " + "; ".join(errors))
