"""Tests for the real-script image acceptance runner."""

import pytest

from tools.run_image_acceptance import parse_seeds, select_scenes


def test_parse_seeds():
    assert parse_seeds("42, 137,891") == [42, 137, 891]


def test_select_scenes_uses_saved_real_scene_descriptions():
    payload = {
        "all_visual_scenes": [
            {"scene": f"story_{index}", "description": f"scene {index}"}
            for index in range(1, 6)
        ]
    }
    selected = select_scenes(payload, 3)
    assert [scene["scene"] for scene in selected] == ["story_1", "story_2", "story_3"]


@pytest.mark.parametrize("raw", ["", "abc", "42,nope"])
def test_parse_seeds_rejects_invalid_input(raw):
    with pytest.raises(ValueError):
        parse_seeds(raw)


def test_select_scenes_requires_enough_real_scenes():
    with pytest.raises(ValueError, match="visual scenes"):
        select_scenes({"all_visual_scenes": [{"description": "one"}]}, 2)
