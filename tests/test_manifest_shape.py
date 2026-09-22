"""Regression tests for manifest shape and the metadata contract.

The manifest previously carried two fields that no code ever read:

  * ``format: "multi_news_3"`` — named a 3-story / 2-scene format replaced by
    2 stories x 4 beats (ADR-004) on 2026-04-29.
  * ``pipeline_version: "unified"`` — named a v1/v2 merge whose ``generate_v2.py``
    was deleted on 2026-04-24.

Both were removed rather than replaced. A derived label would still be wrong
for ``YT_IMAGE_LIMIT`` smoke runs, and the manifest's own structural fields
(``script.stories``, ``assets.images``) are truthful by construction.

These tests lock the removal and pin the fields that consumers actually depend
on, so a future edit that drops a relied-upon key fails here rather than in the
publisher.
"""

import json
from pathlib import Path


# Keys the manifest must NOT contain. Verified absent from every reader in the
# repository (including git history) before removal.
RETIRED_FIELDS = ("format", "pipeline_version")


def _manifest_signature() -> dict:
    """The manifest dict as constructed by tools/generate_complete_video.py.

    Parsed from source rather than imported: the module executes the pipeline
    at import time, so importing it would start a run.
    """
    source = (
        Path(__file__).parent.parent / "tools" / "generate_complete_video.py"
    ).read_text(encoding="utf-8")
    start = source.index("\nmanifest = {")
    end = source.index("\n}\n", start) + 2
    return source[start:end]


class TestRetiredFieldsAreGone:
    def test_manifest_literal_has_no_retired_keys(self):
        block = _manifest_signature()
        for field in RETIRED_FIELDS:
            assert f"'{field}'" not in block, (
                f"manifest must not define {field!r} — it was removed because "
                f"nothing read it and its value was stale"
            )

    def test_no_code_reads_the_retired_fields(self):
        """A *manifest* reader appearing anywhere means removal must be revisited.

        Naive substring matching is not enough: ``ffprobe`` output also has a
        ``['format']`` key (``src/video/tts_tool.py`` reads ``['format']['duration']``
        from probe JSON). This looks for manifest access specifically — either a
        `manifest` variable on the same line, or the multi-key manifest pattern.
        """
        repo = Path(__file__).parent.parent
        offenders = []
        for path in list((repo / "src").rglob("*.py")) + list(
            (repo / "tools").rglob("*.py")
        ):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if "manifest" not in line.lower():
                    continue
                for field in RETIRED_FIELDS:
                    if f"['{field}']" in line or f".get('{field}')" in line:
                        offenders.append(
                            f"{path.relative_to(repo)}:{lineno} {line.strip()[:70]}"
                        )
        assert not offenders, (
            "retired manifest fields now have manifest readers; revisit the "
            "removal: " + ", ".join(offenders)
        )


class TestStructuralSourceOfTruth:
    def test_manifest_still_carries_real_structure(self):
        """The fields that replaced the stale label must be present."""
        block = _manifest_signature()
        for key in ("'project_id'", "'created_at'", "'script'", "'assets'",
                    "'articles'", "'analyses'", "'platform_metadata'"):
            assert key in block, f"manifest must still define {key}"

    def test_script_and_assets_are_the_structure(self):
        """script.stories and assets.images describe the run; no label needed."""
        block = _manifest_signature()
        assert "'script': script" in block
        assert "'images':" in block


class TestGeneratedManifestShape:
    """Against a real project if one is present; otherwise skip.

    Scoped to manifests generated AFTER the removal: projects written by
    earlier versions legitimately contain the retired fields, and asserting
    their absence there would fail on historical output. The marker is the
    removal itself — a new manifest has no ``format`` key but does have
    ``provenance`` (added in the same series of changes), whereas a legacy one
    has ``format`` and no ``provenance``.
    """

    def _current_manifests(self):
        projects = Path(__file__).parent.parent / "output" / "projects"
        if not projects.is_dir():
            return []
        current = []
        for path in sorted(projects.glob("*/manifest.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            # Skip historical output written before the metadata change.
            if "provenance" in data:
                current.append((path, data))
        return current

    def test_current_manifests_have_no_retired_fields(self):
        manifests = self._current_manifests()
        if not manifests:
            import pytest

            pytest.skip("no post-change projects on disk")
        for path, data in manifests:
            for field in RETIRED_FIELDS:
                assert field not in data, f"{path.parent.name} still has {field}"

    def test_current_manifests_expose_structure(self):
        manifests = self._current_manifests()
        if not manifests:
            import pytest

            pytest.skip("no post-change projects on disk")
        for path, data in manifests:
            stories = data.get("script", {}).get("stories", [])
            images = data.get("assets", {}).get("images", [])
            assert stories, f"{path.parent.name} has no script.stories"
            assert images, f"{path.parent.name} has no assets.images"

    def test_legacy_manifests_are_not_required_to_match(self):
        """Documents why the assertions above are scoped.

        Historical projects predate the removal; failing on them would make the
        suite unusable on any machine with existing output.
        """
        projects = Path(__file__).parent.parent / "output" / "projects"
        if not projects.is_dir():
            import pytest

            pytest.skip("no output tree")
        legacy = []
        for path in sorted(projects.glob("*/manifest.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if "provenance" not in data:
                legacy.append(path.parent.name)
        # No assertion about their contents — the point is that they are
        # tolerated. This test exists to make the scoping intentional.
        assert isinstance(legacy, list)

