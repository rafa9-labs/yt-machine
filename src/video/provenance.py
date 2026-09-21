"""Provenance records for generated assets and pipeline runs.

WHY THIS MODULE EXISTS
======================
A generated image is only reproducible if the inputs that produced it are
recorded next to it: model, sampling, adapter, and the processed output. An
image without that record cannot be re-derived, compared, or audited.

Two record kinds are produced:

  * per-image sidecars, written by ``src.video.pixel_art_tool`` and propagated
    into the project folder by the pipeline (``<image>.provenance.json``);
  * a run-level record, assembled at manifest time, covering the identifiers
    that belong to the whole run (LLM, TTS, assembly, git commit).

SECRETS
=======
A provenance record is written to disk and may be copied, shared, or attached
to a bug report. It must never contain credentials. ``assert_no_secrets`` is a
deny-list guard applied on every write, so a future caller that carelessly
passes ``os.getenv("FAL_KEY")`` fails loudly here instead of leaking it into
an artifact. The guard is deliberately value-based as well as key-based: a
secret pasted under an innocent key name is still caught.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


# ── secret detection ──────────────────────────────────────────────────
#
# KEY NAMES: matched case-insensitively as a substring, so "FAL_KEY",
# "fal_key", "apiKey" and "X-API-Key" are all caught. Deliberately broad:
# a false positive costs one renamed dict key, a false negative leaks a
# credential into a published artifact.
_SECRET_KEY_PATTERNS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
    "authorization",
    "auth_key",
    "bearer",
    "session_id",
    "client_secret",
    "refresh_token",
    "hf_",       # HuggingFace tokens; also covered by value patterns
)

# KEY SHAPES: any key ending in "key" is treated as credential-bearing, so a
# provider-specific name that was never enumerated (FAL_KEY, GEMINI_KEY, ...)
# is still caught. Redacting a benign "*_key" is an acceptable cost.
_KEY_SUFFIX_RE = re.compile(r"(^|_)key$", re.IGNORECASE)

# VALUE SHAPES: provider-specific formats that must never be serialized,
# regardless of the key they arrived under.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),                  # GitHub classic PAT
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),          # GitHub fine-grained
    re.compile(r"gho_[A-Za-z0-9]{20,}"),                  # GitHub OAuth
    re.compile(r"BSA[A-Za-z0-9]{20,}"),                   # Brave Search
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),                 # OpenAI / generic
    re.compile(r"hf_[A-Za-z0-9]{20,}"),                   # HuggingFace
    re.compile(r"AKIA[0-9A-Z]{16}"),                      # AWS access key id
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),          # Slack
    re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_-]{30,}"),       # Telegram bot token
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),    # PEM key material
)

_REDACTED = "[REDACTED]"


class ProvenanceSecretError(ValueError):
    """Raised when a provenance payload appears to contain a credential."""


def _is_secret_key(key: str) -> bool:
    lowered = str(key).lower()
    if any(pattern in lowered for pattern in _SECRET_KEY_PATTERNS):
        return True
    return bool(_KEY_SUFFIX_RE.search(lowered))


def _looks_like_secret_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS)


def find_secrets(payload: Any, _path: str = "") -> list:
    """Return dotted paths of every credential-shaped entry in ``payload``.

    Keys whose *name* implies a secret are reported, and so are values that
    match a known provider format under any key. Lists and nested dicts are
    walked, so a secret cannot hide one level down.
    """
    findings = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            location = f"{_path}.{key}" if _path else str(key)
            if _is_secret_key(key):
                findings.append(location)
            findings.extend(find_secrets(value, location))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            findings.extend(find_secrets(value, f"{_path}[{index}]"))
    else:
        if _looks_like_secret_value(payload):
            findings.append(_path or "<value>")
    return findings


def redact_secrets(payload: Any) -> Any:
    """Return a deep copy with credential-shaped entries replaced.

    Used by the writer so an accidental secret degrades to a redaction
    marker rather than aborting a run that has already produced images.
    ``assert_no_secrets`` is the strict form, for tests and CI.
    """
    if isinstance(payload, dict):
        result = {}
        for key, value in payload.items():
            if _is_secret_key(key):
                result[key] = _REDACTED
            else:
                result[key] = redact_secrets(value)
        return result
    if isinstance(payload, list):
        return [redact_secrets(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(redact_secrets(item) for item in payload)
    if _looks_like_secret_value(payload):
        return _REDACTED
    return payload


def assert_no_secrets(payload: Any) -> None:
    """Raise if ``payload`` contains anything credential-shaped.

    The strict counterpart to ``redact_secrets``; used by tests so a
    regression in what callers pass is a test failure, not a silent leak.
    """
    findings = find_secrets(payload)
    if findings:
        raise ProvenanceSecretError(
            "Provenance payload contains credential-shaped entries at: "
            + ", ".join(findings)
        )


# ── reproducibility helpers ───────────────────────────────────────────

def sha256_file(path: Path) -> Optional[str]:
    """SHA-256 of a file, or None when it does not exist / cannot be read.

    Streaming read: asset hashes are taken for images and videos that can be
    hundreds of megabytes, so the file is never loaded into memory.
    """
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    """Current git commit SHA, or None when unavailable.

    Returns None rather than raising: provenance must not fail a run that is
    otherwise healthy, and images can legitimately be generated from an
    exported tarball with no repository present.
    """
    repo = Path(cwd) if cwd else Path(__file__).resolve().parent.parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def build_run_provenance(
    *,
    project_id: int,
    profile: Optional[Dict[str, Any]] = None,
    tts_result: Optional[Dict[str, Any]] = None,
    llm_model: Optional[str] = None,
    assembly: Optional[Dict[str, Any]] = None,
    assets: Optional[Dict[str, Any]] = None,
    status: str = "complete",
) -> Dict[str, Any]:
    """Assemble the run-level provenance record.

    Only identifiers are recorded — never credentials. Every field is
    optional so a partial or failed run still yields a truthful record of
    what was attempted.
    """
    record: Dict[str, Any] = {
        "kind": "run",
        "project_id": project_id,
        "status": status,
        "git_commit": git_commit(),
    }

    profile = profile or {}
    if profile:
        record["generation_profile"] = {
            "name": profile.get("name"),
            "provider": profile.get("provider"),
            "model": (profile.get("model") or {}).get("match")
            if isinstance(profile.get("model"), dict)
            else profile.get("model_id"),
            "steps": profile.get("steps"),
            "guidance": profile.get("guidance"),
            "lora": (profile.get("lora") or {}).get("name"),
            "postprocess": profile.get("postprocess"),
        }

    if llm_model:
        record["llm_model"] = llm_model

    if tts_result:
        record["tts"] = {
            "engine": tts_result.get("engine"),
            "voice": tts_result.get("voice"),
            "estimated_duration_seconds": tts_result.get(
                "estimated_duration_seconds"
            ),
        }

    if assembly:
        record["assembly"] = {
            key: assembly.get(key)
            for key in (
                "duration_seconds",
                "resolution",
                "fps",
                "scenes",
                "render_method",
            )
            if assembly.get(key) is not None
        }

    if assets:
        record["assets"] = assets

    return record
