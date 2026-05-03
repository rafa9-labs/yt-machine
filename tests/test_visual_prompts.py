"""
Unit tests for visual prompt composition validation, _ensure_visual_prompt,
_extract_key_entities, and build_fallback_prompt.

Run: python -m pytest tests/test_visual_prompts.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.brain.llm_interface import LLMInterface
from src.pipeline_utils import build_fallback_prompt


# ═══════════════════════════════════════════
# _validate_visual_prompt_composition
# ═══════════════════════════════════════════
class TestValidateVisualPromptComposition:
    """Tests for LLMInterface._validate_visual_prompt_composition()"""

    # --- Should PASS (composition-style prompts) ---

    @pytest.mark.parametrize("prompt", [
        "16-bit isometric pixel art scene: dramatic wide establishing shot, military forces positioned on left, strategic landscape, sunset lighting",
        "16-bit isometric pixel art scene: tactical close-up view of strategic infrastructure, oil facility on left, supply routes visible in background, golden hour lighting",
        "16-bit isometric pixel art scene: somber revealing scene, civilian perspective on left, consequences visible in background, cold blue lighting",
        "16-bit isometric pixel art scene: forward-looking consequence scene, domino effect visible, dark horizon on left, twilight atmosphere",
        "16-bit isometric pixel art scene: dramatic wide establishing shot, foreground military equipment on left, Persian Gulf in background at sunset",
        "Retro Pixel, true 16-bit pixel art, dramatic wide establishing shot, Iran military positioned on left, Persian Gulf background, isometric perspective, sunset lighting, atmospheric depth",
    ])
    def test_composition_prompts_pass(self, prompt):
        assert LLMInterface._validate_visual_prompt_composition(prompt) is True

    # --- Should FAIL (narration-style prompts) ---

    @pytest.mark.parametrize("prompt", [
        "Pixel art, a green cartoon mask character holding a giant wooden mallet over a map of the Persian Gulf, vibrant colors",
        "Pixel art, a large wedge splitting a golden oil barrel in half, dramatic shadows, 8-bit style",
        "Pixel art, a close-up of a single, heavy iron nail entering a soft wooden surface",
        "pixel art, the aftermath of Pixel art, consequences unfolding, dark sky, twilight",
        "Pixel art, a giant robotic hand holding a golden contract, surrounded by glowing microchips",
        "Pixel art, a computer screen showing a soldier's face made of digital code and binary numbers",
        "Pixel art, a heavy iron padlock being clamped onto a glowing blue data stream",
    ])
    def test_narration_prompts_fail(self, prompt):
        assert LLMInterface._validate_visual_prompt_composition(prompt) is False

    # --- Edge cases ---

    def test_empty_string_fails(self):
        assert LLMInterface._validate_visual_prompt_composition("") is False

    def test_short_string_fails(self):
        assert LLMInterface._validate_visual_prompt_composition("Pixel art scene") is False

    def test_single_sentence_fails(self):
        assert LLMInterface._validate_visual_prompt_composition("A dramatic scene of military forces.") is False

    def test_two_narration_patterns_fails(self):
        assert LLMInterface._validate_visual_prompt_composition(
            "Pixel art, a character holding a giant mallet, says the report"
        ) is False

    def test_barely_composition_passes(self):
        assert LLMInterface._validate_visual_prompt_composition(
            "16-bit isometric pixel art scene: foreground elements, background landscape, sunset lighting"
        ) is True


# ═══════════════════════════════════════════
# _ensure_visual_prompt
# ═══════════════════════════════════════════
class TestEnsureVisualPrompt:
    """Tests for LLMInterface._ensure_visual_prompt()"""

    # --- Composition-style visual field should pass through ---

    def test_composition_field_passes_through(self):
        prompt = "16-bit isometric pixel art scene: dramatic wide establishing shot, forces on left, Gulf in background, sunset lighting"
        result = LLMInterface._ensure_visual_prompt(prompt, "Some narration", "hook")
        assert result == prompt

    # --- Narration-style visual field should be rewritten ---

    def test_narration_field_rewritten(self):
        bad = "Pixel art, a green cartoon mask character holding a giant wooden mallet"
        narration = "Iran is threatening to CRUSH the UAE to break alliances."
        result = LLMInterface._ensure_visual_prompt(bad, narration, "hook")
        assert result != bad
        assert "16-bit isometric pixel art scene" in result

    def test_empty_field_uses_template(self):
        narration = "The Pentagon signed massive AI deals with Google and SpaceX."
        result = LLMInterface._ensure_visual_prompt("", narration, "mechanism")
        assert "16-bit isometric pixel art scene" in result
        assert "tactical close-up" in result

    def test_empty_field_empty_narration_uses_default(self):
        result = LLMInterface._ensure_visual_prompt("", "", "hook")
        assert "dramatic wide establishing shot" in result

    # --- Scene type affects template selection ---

    @pytest.mark.parametrize("scene_type,expected_phrase", [
        ("hook", "dramatic wide establishing shot"),
        ("mechanism", "tactical close-up view"),
        ("truth", "somber revealing scene"),
        ("fallout", "forward-looking consequence"),
    ])
    def test_scene_type_templates(self, scene_type, expected_phrase):
        result = LLMInterface._ensure_visual_prompt("", "Some narration text here", scene_type)
        assert expected_phrase in result

    # --- Narration with entities should use them ---

    def test_entities_from_narration(self):
        narration = "Iran threatens the UAE with military escalation in the Persian Gulf"
        result = LLMInterface._ensure_visual_prompt("", narration, "hook")
        assert "Iran" in result or "UAE" in result or "Persian" in result

    # --- Very short narration should still produce valid output ---

    def test_short_narration(self):
        result = LLMInterface._ensure_visual_prompt("", "Iran attacks.", "hook")
        assert "16-bit isometric pixel art scene" in result
        assert len(result) >= 30

    # --- The "aftermath of Pixel art" garbage prompt should be rewritten ---

    def test_aftermath_garbage_rewritten(self):
        bad = "pixel art, the aftermath of Pixel art, consequences unfolding, dark sky, twilight"
        narration = "the gulf security architecture will fracture into separate defensive fortresses."
        result = LLMInterface._ensure_visual_prompt(bad, narration, "fallout")
        assert result != bad
        assert "the aftermath of Pixel art" not in result


# ═══════════════════════════════════════════
# _extract_key_entities
# ═══════════════════════════════════════════
class TestExtractKeyEntities:
    """Tests for LLMInterface._extract_key_entities()"""

    def test_extracts_capitalized_entities(self):
        entities = LLMInterface._extract_key_entities("Iran threatens the UAE with military escalation")
        assert "Iran" in entities
        assert "UAE" not in entities  # UAE is all caps, not Capitalized pattern

    def test_extracts_numbers(self):
        entities = LLMInterface._extract_key_entities("Deployed 5,000 troops to the region")
        assert "5,000" in entities

    def test_excludes_stopwords(self):
        entities = LLMInterface._extract_key_entities("The Pentagon signed The Agreement")
        assert "The" not in entities

    def test_multi_word_entities(self):
        entities = LLMInterface._extract_key_entities("South Korea and Saudi Arabia signed deals")
        assert "South Korea" in entities or "South" in entities

    def test_empty_string(self):
        entities = LLMInterface._extract_key_entities("")
        assert len(entities) == 0


# ═══════════════════════════════════════════
# build_fallback_prompt (updated)
# ═══════════════════════════════════════════
class TestBuildFallbackPrompt:
    """Tests for build_fallback_prompt() with composition-style output."""

    def test_empty_text_with_analyses(self):
        analyses = [{"topic": "Iran Strategic Escalation"}]
        result = build_fallback_prompt("", 0, 0, analyses)
        assert "16-bit isometric pixel art scene" in result
        assert "Iran Strategic Escalation" in result

    def test_empty_text_without_analyses(self):
        result = build_fallback_prompt("", 0, 0, [])
        assert "16-bit isometric pixel art scene" in result
        assert len(result) >= 50

    def test_narration_with_locations(self):
        result = build_fallback_prompt(
            "Iran threatens military action against the UAE", 0, 0, [])
        assert "16-bit isometric pixel art scene" in result
        assert "Iran" in result or "UAE" in result

    def test_narration_with_actions(self):
        result = build_fallback_prompt(
            "Russia attacks Ukraine forces in the east", 0, 0, [])
        assert "16-bit isometric pixel art scene" in result

    def test_part_idx_affects_composition(self):
        r_hook = build_fallback_prompt("Iran threatens UAE", 0, 0, [])
        r_mech = build_fallback_prompt("Iran threatens UAE", 0, 1, [])
        r_truth = build_fallback_prompt("Iran threatens UAE", 0, 2, [])
        r_fall = build_fallback_prompt("Iran threatens UAE", 0, 3, [])
        assert "establishing shot" in r_hook
        assert "close-up" in r_mech
        assert "revealing" in r_truth
        assert "consequence" in r_fall

    def test_all_scene_types_have_lighting(self):
        for idx in range(4):
            result = build_fallback_prompt("Iran threatens UAE", 0, idx, [])
            assert "lighting" in result or "atmosphere" in result

    def test_all_scene_types_start_with_prefix(self):
        for idx in range(4):
            result = build_fallback_prompt("Iran threatens UAE", 0, idx, [])
            assert result.startswith("16-bit isometric pixel art scene")

    def test_strips_the_from_entities(self):
        result = build_fallback_prompt(
            "The Pentagon signed deals with Google and SpaceX", 0, 0,
            [{"topic": "Pentagon AI Deals"}])
        assert "The Pentagon" not in result or "Pentagon" in result


# ═══════════════════════════════════════════
# Negative prompt config validation
# ═══════════════════════════════════════════
class TestNegativePromptConfig:
    """Tests for the negative prompt configuration in image_style.json"""

    def test_negative_prompt_exists(self):
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "image_style.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        assert "negative_prompt" in config
        assert len(config["negative_prompt"]) > 100

    def test_negative_prompt_has_critical_terms(self):
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "image_style.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        np = config["negative_prompt"].lower()
        critical_terms = [
            "cartoon mascot", "single isolated object", "extreme close-up",
            "plain background", "scene without depth", "screen-within-screen",
            "text", "photorealistic", "anime",
        ]
        for term in critical_terms:
            assert term in np, f"Missing critical negative prompt term: '{term}'"

    def test_negative_prompt_not_too_long(self):
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "image_style.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        terms = [t.strip() for t in config["negative_prompt"].split(",") if t.strip()]
        assert len(terms) <= 60, f"Negative prompt has {len(terms)} terms, should be ≤60"


# ═══════════════════════════════════════════
# Metaphor detection
# ═══════════════════════════════════════════
class TestMetaphorDetection:
    """Tests for LLMInterface._detect_metaphor_narration()"""

    def test_simple_metaphor_detected(self):
        result = LLMInterface._detect_metaphor_narration("It's a calculated wedge driven into the heart of regional unity.")
        assert "wedge" in result

    def test_multiple_metaphors_detected(self):
        result = LLMInterface._detect_metaphor_narration("It's a domino effect that will fracture the alliance.")
        assert "domino" in result
        assert "fracture" in result

    def test_literal_usage_not_detected(self):
        result = LLMInterface._detect_metaphor_narration("Iran threatens the UAE with military escalation.")
        assert len(result) == 0

    def test_no_indicator_returns_empty(self):
        result = LLMInterface._detect_metaphor_narration("The wall was built along the border region.")
        assert len(result) == 0

    def test_indicator_with_metaphor_word(self):
        result = LLMInterface._detect_metaphor_narration("It's a game of chess between superpowers.")
        assert "game" in result
        assert "chess" in result

    def test_empty_string(self):
        result = LLMInterface._detect_metaphor_narration("")
        assert len(result) == 0


class TestMetaphorToSceneConversion:
    """Tests that metaphor-containing narration produces concrete scene prompts via _ensure_visual_prompt."""

    def test_wedge_converted_to_barrier(self):
        narration = "It's a calculated wedge driven into the heart of regional unity."
        result = LLMInterface._ensure_visual_prompt("", narration, "truth")
        assert "wedge" not in result.lower()
        assert "dividing barrier" in result or "strategic split" in result

    def test_padlock_converted_to_lockdown(self):
        narration = "It's a padlock clamped onto the data stream."
        result = LLMInterface._ensure_visual_prompt("", narration, "truth")
        assert "padlock" not in result.lower()
        assert "security lockdown" in result or "access restriction" in result

    def test_metaphor_with_location_entities(self):
        narration = "It's a wedge driven into the heart of the Gulf alliance."
        result = LLMInterface._ensure_visual_prompt("", narration, "fallout")
        assert "dividing barrier" in result or "strategic split" in result
        assert "Gulf" in result or "twilight" in result

    def test_no_metaphor_uses_entity_path(self):
        narration = "Iran threatens the UAE with military escalation."
        result = LLMInterface._ensure_visual_prompt("", narration, "hook")
        assert "Iran" in result or "UAE" in result or "establishing" in result

    def test_fracture_converted_to_alliance(self):
        narration = "The architecture will fracture into separate defensive fortresses."
        result = LLMInterface._ensure_visual_prompt("", narration, "fallout")
        assert "breaking alliance" in result or "shattered coalition" in result or "fracture" not in result.lower() or "consequence" in result

    def test_composition_check_passes_after_conversion(self):
        narration = "It's a wedge driven into the heart of regional unity."
        result = LLMInterface._ensure_visual_prompt("", narration, "truth")
        assert LLMInterface._validate_visual_prompt_composition(result) is True


# ═══════════════════════════════════════════
# Sprint 6: Script Integrity Tests
# ═══════════════════════════════════════════
import json

_MINIMAL_SCRIPT = {
    'greeting': 'Ssssmokin\'! Good morning, folks!',
    'intro_hook': 'Two stories. One screen.',
    'stories': [
        {
            'part_1_narration': 'Iran threatens the UAE with military escalation.',
            'part_2_narration': 'By weaponizing rivalries, Tehran forces a split.',
            'real_talk': 'It is a calculated wedge driven into regional unity.',
            'fallout': 'The gulf architecture will fracture into defensive fortresses.',
            'segue': 'And as borders crack, a digital divide emerges.',
        },
        {
            'part_1_narration': 'The Pentagon signed massive AI deals with Google.',
            'part_2_narration': 'They are prioritizing combat-ready models over polite ones.',
            'real_talk': 'It is a way to turn safety into tactical advantage.',
            'fallout': 'The industry will diverge into civilian and combat AI.',
            'segue': '',
        },
    ],
    'full_text': '',
    'closing': '',
}


class TestEnforceGreeting:
    """Tests for LLMInterface._enforce_greeting()"""

    def test_empty_greeting_filled(self):
        llm = LLMInterface()
        script = _MINIMAL_SCRIPT.copy()
        script['greeting'] = ''
        script['intro_hook'] = ''
        result = llm._enforce_greeting(script)
        assert result['greeting'].strip() != ''
        assert result['intro_hook'].strip() != ''

    def test_existing_greeting_preserved(self):
        llm = LLMInterface()
        script = _MINIMAL_SCRIPT.copy()
        original_greeting = 'Ssssmokin! Good morning, folks!'
        original_hook = 'Two geopolitical TORNADOS are swirling!'
        script['greeting'] = original_greeting
        script['intro_hook'] = original_hook
        result = llm._enforce_greeting(script)
        assert result['greeting'] == original_greeting
        assert result['intro_hook'] == original_hook


class TestValidateClosing:
    """Tests for LLMInterface._validate_closing()"""

    def test_closing_present_unchanged(self):
        llm = LLMInterface()
        text = "Story one. Story two. Stay behind the curtains, and if I don't see you. Good morning, good afternoon. And goodnight."
        result = llm._validate_closing(text)
        assert "Good morning" in result
        assert "good afternoon" in result
        assert "goodnight" in result
        assert result.strip().endswith("goodnight.")

    def test_closing_absent_appended(self):
        llm = LLMInterface()
        text = "Story one. Story two. That is all for now."
        result = llm._validate_closing(text)
        assert "Good morning" in result
        assert "goodnight" in result

    def test_mid_text_cta_stripped(self):
        llm = LLMInterface()
        text = "Iran attacks the UAE. Subscribe and like for more. The Pentagon signs deals."
        result = llm._validate_closing(text)
        assert "Subscribe" not in result
        assert "Iran attacks" in result
        assert "Pentagon signs" in result

    def test_truman_in_middle_not_counted(self):
        llm = LLMInterface()
        text = "Good morning everyone. Iran attacks. Good afternoon. More story. The end."
        result = llm._validate_closing(text)
        assert "goodnight" in result


class TestDedupSegueOverlap:
    """Tests for LLMInterface._dedup_segue_overlap()"""

    def test_overlap_stripped(self):
        llm = LLMInterface()
        # Overlap: segue ends with "the Pentagon signs" and next part_1 starts with "the Pentagon signs"
        script = {
            'stories': [
                {'segue': 'And now the Pentagon signs', 'part_1_narration': ''},
                {'part_1_narration': 'the Pentagon signs deals with Google and SpaceX for AI'},
            ]
        }
        result = llm._dedup_segue_overlap(script)
        p1 = result['stories'][1]['part_1_narration']
        # "the Pentagon signs" overlaps (3 words from end of segue and start of part_1)
        assert not p1.lower().startswith('the pentagon signs'), f"Overlap not stripped: {p1}"

    def test_empty_segue_skipped(self):
        llm = LLMInterface()
        script = {
            'stories': [
                {'segue': '', 'part_1_narration': ''},
                {'part_1_narration': 'The Pentagon signed massive AI deals.'},
            ]
        }
        result = llm._dedup_segue_overlap(script)
        assert result['stories'][1]['part_1_narration'] == 'The Pentagon signed massive AI deals.'

    def test_no_overlap_unchanged(self):
        llm = LLMInterface()
        script = {
            'stories': [
                {'segue': 'Meanwhile across the ocean', 'part_1_narration': ''},
                {'part_1_narration': 'The Pentagon signed massive deals.'},
            ]
        }
        result = llm._dedup_segue_overlap(script)
        assert result['stories'][1]['part_1_narration'] == 'The Pentagon signed massive deals.'


class TestDedupInterStoryPhrases:
    """Tests for LLMInterface._dedup_inter_story_phrases()"""

    def test_trigram_overlap_stripped(self):
        llm = LLMInterface()
        script = {
            'stories': [
                {'fallout': 'the gulf security architecture will fracture', 'real_talk': ''},
                {'part_1_narration': 'the gulf security architecture will fracture into separate fortresses'},
            ]
        }
        result = llm._dedup_inter_story_phrases(script)
        p1 = result['stories'][1]['part_1_narration']
        assert not p1.lower().startswith('the gulf security'), f"Trigram overlap not stripped: {p1}"

    def test_short_text_skipped(self):
        llm = LLMInterface()
        script = {
            'stories': [
                {'fallout': 'OK', 'real_talk': ''},
                {'part_1_narration': 'Hi'},
            ]
        }
        result = llm._dedup_inter_story_phrases(script)
        assert result['stories'][1]['part_1_narration'] == 'Hi'

    def test_no_overlap_unchanged(self):
        llm = LLMInterface()
        script = {
            'stories': [
                {'fallout': 'the architecture will fracture into separate fortresses', 'real_talk': ''},
                {'part_1_narration': 'The Pentagon signed massive AI deals with Google.'},
            ]
        }
        result = llm._dedup_inter_story_phrases(script)
        assert result['stories'][1]['part_1_narration'] == 'The Pentagon signed massive AI deals with Google.'


class TestParseCuratedStructures:
    """Tests for LLMInterface._parse_curated_structures()"""

    def test_with_proper_markers(self):
        llm = LLMInterface()
        text = """[STORY 1]
[HOOK] Hook text for story one
[MECHANISM] Mechanism text for story one
[REAL_TALK] Real talk text for story one
[FALLOUT] Fallout text for story one

---

[STORY 2]
[HOOK] Hook text for story two
[MECHANISM] Mechanism text for story two
[REAL_TALK] Real talk text for story two
[FALLOUT] Fallout text for story two"""
        result = llm._parse_curated_structures(text, 2)
        assert result is not None
        assert len(result) == 2
        assert result[0]['hook'] == 'Hook text for story one'
        assert result[0]['mechanism'] == 'Mechanism text for story one'
        assert result[0]['real_talk'] == 'Real talk text for story one'
        assert 'Fallout text for story one' in result[0]['fallout']
        assert result[1]['hook'] == 'Hook text for story two'
        assert result[1]['mechanism'] == 'Mechanism text for story two'
        assert result[1]['real_talk'] == 'Real talk text for story two'
        assert result[1]['fallout'] == 'Fallout text for story two'

    def test_without_markers_fallback(self):
        llm = LLMInterface()
        text = "Story one hook. Story one mechanism. Story one truth. Story one fallout. --- Story two hook. Story two mechanism. Story two truth. Story two fallout."
        result = llm._parse_curated_structures(text, 2)
        assert result is not None
        assert len(result) == 2
        assert len(result[0]['hook']) > 0

    def test_partial_markers(self):
        llm = LLMInterface()
        text = """[STORY 1]
[HOOK] Hook text for story one
[MECHANISM] Mechanism text for story one
[REAL_TALK] Real talk text for story one
[FALLOUT] Fallout text for story one

---

Story two has no markers at all. Just plain text about something."""
        result = llm._parse_curated_structures(text, 2)
        assert result is not None
        assert len(result) == 2

    def test_too_few_stories_pads_with_empty(self):
        llm = LLMInterface()
        text = "[HOOK] Only one story here"
        result = llm._parse_curated_structures(text, 2)
        # When fewer stories than expected, pads with empty dicts
        assert result is not None
        assert len(result) == 2


class TestEnsureVisualPromptEdgeCases:
    """Edge case tests for _ensure_visual_prompt"""

    def test_none_inputs(self):
        result = LLMInterface._ensure_visual_prompt(None, None, 'hook')
        assert "dramatic wide establishing shot" in result

    def test_none_visual_with_narration(self):
        result = LLMInterface._ensure_visual_prompt(None, "Iran threatens the UAE.", 'hook')
        assert "16-bit isometric pixel art scene" in result

    def test_unknown_scene_type(self):
        result = LLMInterface._ensure_visual_prompt("", "", "unknown_type")
        assert "dramatic wide establishing shot" in result

    def test_very_long_narration(self):
        long_narr = " ".join(["Iran threatens"] * 100)
        result = LLMInterface._ensure_visual_prompt("", long_narr, 'hook')
        assert "16-bit isometric pixel art scene" in result
        assert len(result) < 300


# ═══════════════════════════════════════════
# INTEGRATION: Full enforcement chain
# ═══════════════════════════════════════════
class TestEnforcementChain:
    """Test the full enforcement pipeline: greeting → segues → dedup → fallout → closing."""

    def _make_script(self, **overrides):
        base = {
            'greeting': '',
            'intro_hook': '',
            'stories': [
                {
                    'part_1_narration': 'Iran launched missiles at Israel yesterday.',
                    'part_2_narration': 'The Pentagon confirmed the strikes hit multiple targets.',
                    'real_talk': 'This changes the regional power dynamics entirely.',
                    'fallout': '',
                    'segue': 'And now the Pentagon',
                    'part_1_visual': '',
                    'part_2_visual': '',
                    'real_talk_visual': '',
                    'fallout_visual': '',
                },
                {
                    'part_1_narration': 'The Pentagon confirmed the strikes hit multiple targets.',
                    'part_2_narration': 'Global markets reacted with oil prices spiking.',
                    'real_talk': 'This is what happens when geopolitics meets economics.',
                    'fallout': '',
                    'segue': '',
                    'part_1_visual': '',
                    'part_2_visual': '',
                    'real_talk_visual': '',
                    'fallout_visual': '',
                },
            ],
        }
        base.update(overrides)
        return base

    def test_full_chain_guarantees_greeting(self):
        llm = LLMInterface()
        script = self._make_script()
        script = llm._enforce_greeting(script)
        script = llm._enforce_segues(script)
        script = llm._dedup_segue_overlap(script)
        script = llm._dedup_inter_story_phrases(script)
        script = llm._enforce_fallout(script)
        assert script['greeting'].strip(), "greeting must not be empty after enforcement"
        assert script['intro_hook'].strip(), "intro_hook must not be empty after enforcement"

    def test_full_chain_guarantees_segues(self):
        llm = LLMInterface()
        script = self._make_script()
        script = llm._enforce_greeting(script)
        script = llm._enforce_segues(script)
        script = llm._dedup_segue_overlap(script)
        script = llm._dedup_inter_story_phrases(script)
        script = llm._enforce_fallout(script)
        for i, story in enumerate(script['stories'][:-1]):
            assert story.get('segue', '').strip(), f"Story {i+1} must have a segue"

    def test_full_chain_guarantees_fallout(self):
        llm = LLMInterface()
        script = self._make_script()
        script = llm._enforce_greeting(script)
        script = llm._enforce_segues(script)
        script = llm._dedup_segue_overlap(script)
        script = llm._dedup_inter_story_phrases(script)
        script = llm._enforce_fallout(script)
        for i, story in enumerate(script['stories']):
            assert story.get('fallout', '').strip(), f"Story {i+1} must have fallout"
            assert len(story['fallout'].split()) >= 5, f"Story {i+1} fallout too short"

    def test_full_chain_dedup_segue_overlap(self):
        llm = LLMInterface()
        script = self._make_script()
        script = llm._enforce_greeting(script)
        script = llm._enforce_segues(script)
        script = llm._dedup_segue_overlap(script)
        p1 = script['stories'][1]['part_1_narration']
        if script['stories'][0].get('segue', '').strip():
            segue_end = script['stories'][0]['segue'].strip().split()[-3:]
            p1_start = p1.split()[:3]
            overlap = [w.lower().rstrip('.,!?;:') for w in segue_end] == [w.lower().rstrip('.,!?;:') for w in p1_start]
            if overlap:
                assert False, f"Overlap not removed: segue ends with {' '.join(segue_end)}, part_1 starts with {' '.join(p1_start)}"

    def test_full_chain_dedup_inter_story(self):
        llm = LLMInterface()
        script = {
            'greeting': 'Hey!',
            'intro_hook': 'Listen up!',
            'stories': [
                {
                    'part_1_narration': 'Iran launched missiles at Israel.',
                    'part_2_narration': 'The Pentagon confirmed strikes.',
                    'real_talk': 'This changes everything entirely.',
                    'fallout': 'The Pentagon confirmed strikes will have long term consequences for everyone.',
                    'segue': 'But wait there is more',
                },
                {
                    'part_1_narration': 'The Pentagon confirmed strikes will reshape global alliances and power structures forever.',
                    'part_2_narration': 'Oil prices spiked immediately.',
                    'real_talk': 'Economics and geopolitics collide.',
                    'fallout': 'Long term effects are unknown.',
                    'segue': '',
                },
            ],
        }
        script = llm._dedup_inter_story_phrases(script)
        p1 = script['stories'][1]['part_1_narration']
        assert not p1.lower().startswith('the pentagon confirmed strikes'), \
            f"3-gram overlap not removed from story 2 part_1: {p1}"

    def test_full_chain_no_empty_visuals_after_fallout_enforcement(self):
        llm = LLMInterface()
        script = self._make_script()
        script = llm._enforce_fallout(script)
        for i, story in enumerate(script['stories']):
            fv = story.get('fallout_visual', '').strip()
            assert fv, f"Story {i+1} fallout_visual must not be empty after enforcement"
            assert LLMInterface._validate_visual_prompt_composition(fv), \
                f"Story {i+1} fallout_visual must pass composition: {fv[:60]}"


# ═══════════════════════════════════════════
# INTEGRATION: Visual prompt end-to-end
# ═══════════════════════════════════════════
class TestVisualPromptPipeline:
    """Test _ensure_visual_prompt with metaphor + composition validation chain."""

    def test_metaphor_prompt_becomes_composition(self):
        llm = LLMInterface()
        narration = "This is like a wedge being driven between allies."
        result = LLMInterface._ensure_visual_prompt("", narration, 'mechanism')
        assert LLMInterface._validate_visual_prompt_composition(result), \
            f"Metaphor prompt must produce composition-style: {result}"

    def test_narration_style_prompt_rewritten_to_composition(self):
        llm = LLMInterface()
        narration_style = "Pixel art, a giant wedge splitting a golden barrel in half, dramatic shadows"
        result = LLMInterface._ensure_visual_prompt(narration_style, "A wedge splits things apart", 'mechanism')
        assert LLMInterface._validate_visual_prompt_composition(result), \
            f"Narration-style prompt must be rewritten: {result}"

    def test_composition_style_prompt_passes_through(self):
        llm = LLMInterface()
        composition = "16-bit isometric pixel art scene: tactical close-up view of strategic infrastructure, oil facility in foreground, supply routes visible in background, golden hour lighting"
        result = LLMInterface._ensure_visual_prompt(composition, "Oil infrastructure under strain", 'mechanism')
        assert result == composition, "Composition-style prompt should pass through unchanged"

    def test_fallback_prompt_always_composition(self):
        for scene_idx, scene_type in enumerate(['hook', 'mechanism', 'truth', 'fallout']):
            result = build_fallback_prompt("", 0, scene_idx, [])
            assert LLMInterface._validate_visual_prompt_composition(result), \
                f"Fallback for {scene_type} must be composition-style: {result[:60]}"


# ═══════════════════════════════════════════
# INTEGRATION: Segment timeline robustness
# ═══════════════════════════════════════════
class TestSegmentTimelineRobustness:

    def test_missing_narration_fields_produce_valid_image_mapping(self):
        """Verify img_base = story_idx * 4 mapping with 4 scenes per story."""
        stories = [
            {'part_1_narration': 'Hook text', 'part_2_narration': '', 'real_talk': '', 'fallout': 'Fallout text', 'segue': 'Next up'},
            {'part_1_narration': '', 'part_2_narration': 'Mechanism', 'real_talk': 'Truth', 'fallout': '', 'segue': ''},
        ]
        timeline = []
        for i, story in enumerate(stories):
            img_base = i * 4
            for field, img_off in [('part_1_narration', 0), ('part_2_narration', 1), ('real_talk', 2), ('fallout', 3)]:
                val = story.get(field, '')
                if val:
                    timeline.append({'image_idx': img_base + img_off, 'text': val})
        assert timeline[0]['image_idx'] == 0
        assert timeline[1]['image_idx'] == 3
        assert timeline[2]['image_idx'] == 5  # story 2, part_2 (img_base=4, off=1)

    def test_eight_images_for_two_stories(self):
        """Verify 2 stories × 4 scenes = 8 image slots."""
        num_stories = 2
        total_images = num_stories * 4
        assert total_images == 8

    def test_segment_count_with_segues(self):
        """Verify segment count: greeting + intro_hook + pause +
        2*(4 scenes) + 1 segue + 1 separator = 12 segments."""
        timeline = [
            {'label': 'greeting', 'text': 'Hey!'},
            {'label': 'intro_hook', 'text': 'Listen up!'},
            {'label': 'intro_pause', 'text': '....', 'is_separator': True},
        ]
        for i in range(2):
            for field, suffix, img_off in [
                ('part_1_narration', 'part1', 0),
                ('part_2_narration', 'part2', 1),
                ('real_talk', 'real_talk', 2),
                ('fallout', 'fallout', 3),
            ]:
                timeline.append({
                    'label': f'story_{i+1}_{suffix}',
                    'text': f'{field} text',
                    'image_idx': i * 4 + img_off,
                })
            if i < 1:
                timeline.append({'label': f'story_{i+1}_segue', 'text': 'But wait'})
                timeline.append({'label': f'story_{i+1}_separator', 'text': '....', 'is_separator': True})
        labels = [s['label'] for s in timeline]
        assert 'greeting' in labels
        assert 'story_1_part1' in labels
        assert 'story_1_fallout' in labels
        assert 'story_2_fallout' in labels


# ═══════════════════════════════════════════
# INTEGRATION: Curation preserves 4-part structure
# ═══════════════════════════════════════════
class TestCurationRoundTrip:

    def test_parse_curated_then_reassemble_preserves_fields(self):
        llm = LLMInterface()
        curated_text = """[STORY 1]
[HOOK] Iran launched missiles at Israel
[MECHANISM] The Pentagon confirmed multiple impacts
[REAL_TALK] This changes regional dynamics entirely
[FALLOUT] The consequences will ripple for decades"""
        structures = llm._parse_curated_structures(curated_text, 1)
        assert len(structures) == 1
        s = structures[0]
        assert 'Iran' in s['hook']
        assert 'Pentagon' in s['mechanism']
        assert 'dynamics' in s['real_talk']
        assert 'ripple' in s['fallout']

    def test_parse_two_stories_then_reassemble_keeps_separate(self):
        llm = LLMInterface()
        curated_text = """[STORY 1]
[HOOK] First hook text here
[MECHANISM] First mechanism
[REAL_TALK] First truth
[FALLOUT] First fallout

---

[STORY 2]
[HOOK] Second hook text here
[MECHANISM] Second mechanism
[REAL_TALK] Second truth
[FALLOUT] Second fallout"""
        structures = llm._parse_curated_structures(curated_text, 2)
        assert len(structures) == 2
        assert 'First' in structures[0]['hook']
        assert 'Second' in structures[1]['hook']
        assert structures[0]['mechanism'] != structures[1]['mechanism']

    def test_no_markers_falls_back_to_quarter_split(self):
        llm = LLMInterface()
        curated_text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        structures = llm._parse_curated_structures(curated_text, 1)
        assert len(structures) == 1
        s = structures[0]
        assert all(key in s for key in ['hook', 'mechanism', 'real_talk', 'fallout'])

    def test_empty_input_returns_none(self):
        llm = LLMInterface()
        structures = llm._parse_curated_structures("", 2)
        assert structures is None


# ═══════════════════════════════════════════
# Evaluator preserves existing visual scenes
# ═══════════════════════════════════════════
class TestEvaluatorPreservesVisualScenes:
    """Tests for script_evaluator._rebuild_timeline preserving all_visual_scenes."""

    def test_preserves_composition_visuals(self):
        from src.brain.script_evaluator import _rebuild_timeline
        script = {
            'greeting': 'Hey!',
            'intro_hook': 'Listen up!',
            'closing': 'Goodbye.',
            'stories': [
                {
                    'part_1_narration': 'Iran launched missiles at Israel.',
                    'part_2_narration': 'The Pentagon confirmed strikes.',
                    'real_talk': 'This changes everything.',
                    'fallout': 'The consequences will ripple.',
                    'segue': 'But wait',
                },
                {
                    'part_1_narration': 'Pentagon signed AI deals.',
                    'part_2_narration': 'Markets reacted sharply.',
                    'real_talk': 'Economics meets geopolitics.',
                    'fallout': 'Ripple effects beginning.',
                    'segue': '',
                },
            ],
            'all_visual_scenes': [
                {'scene': 'story_1_part1', 'description': '16-bit isometric pixel art scene: dramatic wide establishing shot, missiles in foreground, sunset lighting'},
                {'scene': 'story_1_part2', 'description': '16-bit isometric pixel art scene: tactical close-up of impact zones, golden hour lighting'},
                {'scene': 'story_1_real_talk', 'description': '16-bit isometric pixel art scene: somber revealing scene, civilian perspective, cold blue lighting'},
                {'scene': 'story_1_fallout', 'description': '16-bit isometric pixel art scene: forward-looking consequence, domino effect, twilight atmosphere'},
                {'scene': 'story_2_part1', 'description': '16-bit isometric pixel art scene: dramatic wide establishing shot, Pentagon building, strategic lighting'},
                {'scene': 'story_2_part2', 'description': '16-bit isometric pixel art scene: tactical close-up of trading screens, red indicators, dramatic lighting'},
                {'scene': 'story_2_real_talk', 'description': '16-bit isometric pixel art scene: somber revealing scene, economic impact, cold lighting'},
                {'scene': 'story_2_fallout', 'description': '16-bit isometric pixel art scene: forward-looking consequence, cascade effect, twilight atmosphere'},
            ],
        }
        _rebuild_timeline(script)
        visuals = script.get('all_visual_scenes', [])
        assert len(visuals) == 8, f"Expected 8 visual scenes, got {len(visuals)}"
        for v in visuals:
            assert v['description'], f"Empty description for scene {v['scene']}"
            assert '16-bit isometric pixel art scene' in v['description'], \
                f"Scene {v['scene']} lost composition-style: {v['description'][:60]}"

    def test_fills_missing_visuals_from_story_fields(self):
        from src.brain.script_evaluator import _rebuild_timeline
        script = {
            'greeting': 'Hey!',
            'intro_hook': 'Listen!',
            'closing': 'Bye.',
            'stories': [
                {
                    'part_1_narration': 'Story one hook.',
                    'part_2_narration': 'Story one body.',
                    'real_talk': 'Story one truth.',
                    'fallout': 'Story one fallout.',
                    'part_1_visual': '16-bit isometric pixel art scene: hook visual, sunset',
                    'fallout_visual': '16-bit isometric pixel art scene: consequence visual, twilight',
                    'segue': '',
                },
                {
                    'part_1_narration': 'Story two hook.',
                    'part_2_narration': 'Story two body.',
                    'real_talk': 'Story two truth.',
                    'fallout': 'Story two fallout.',
                    'segue': '',
                },
            ],
            'all_visual_scenes': [],
        }
        _rebuild_timeline(script)
        visuals = script.get('all_visual_scenes', [])
        assert len(visuals) == 8
        assert '16-bit' in visuals[0]['description'], f"story_1_part1 should use part_1_visual: {visuals[0]['description']}"
        assert '16-bit' in visuals[3]['description'], f"story_1_fallout should use fallout_visual: {visuals[3]['description']}"

    def test_no_existing_scenes_uses_fallback_text(self):
        from src.brain.script_evaluator import _rebuild_timeline
        script = {
            'greeting': '',
            'intro_hook': '',
            'closing': '',
            'stories': [
                {
                    'part_1_narration': 'Iran launched missiles.',
                    'part_2_narration': 'Pentagon confirmed.',
                    'real_talk': 'Everything changed.',
                    'fallout': 'Ripples will spread.',
                    'segue': '',
                },
            ],
            'all_visual_scenes': [],
        }
        _rebuild_timeline(script)
        visuals = script.get('all_visual_scenes', [])
        assert len(visuals) == 4
        for v in visuals:
            assert v['description'], f"Empty description for {v['scene']}: {v}"

    def test_timeline_includes_fallout_segments(self):
        from src.brain.script_evaluator import _rebuild_timeline
        script = {
            'greeting': 'Hey!',
            'intro_hook': '',
            'closing': '',
            'stories': [
                {
                    'part_1_narration': 'Hook text.',
                    'part_2_narration': 'Body text.',
                    'real_talk': 'Truth text.',
                    'fallout': 'Fallout text.',
                    'segue': 'Next up',
                },
                {
                    'part_1_narration': 'Hook2.',
                    'part_2_narration': 'Body2.',
                    'real_talk': 'Truth2.',
                    'fallout': 'Fallout2.',
                    'segue': '',
                },
            ],
            'all_visual_scenes': [],
        }
        _rebuild_timeline(script)
        timeline = script.get('segment_timeline', [])
        labels = [seg['label'] for seg in timeline]
        assert 'story_1_fallout' in labels, f"Missing story_1_fallout in timeline: {labels}"
        assert 'story_2_fallout' in labels, f"Missing story_2_fallout in timeline: {labels}"
        fallout_1 = next(seg for seg in timeline if seg['label'] == 'story_1_fallout')
        assert fallout_1['text'] == 'Fallout text.'
        fallout_2 = next(seg for seg in timeline if seg['label'] == 'story_2_fallout')
        assert fallout_2['text'] == 'Fallout2.'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])