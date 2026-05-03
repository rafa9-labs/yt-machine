"""
Unit tests for image pipeline pure functions:
- _sanitize_visual_prompt
- _truncate_prompt_for_resolution
- _progressive_content_scrub
- _CATEGORY_SAFE_PROMPTS

Run: python -m pytest tests/test_image_pipeline.py -v
"""
import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


def _import_function(module_path, func_name):
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


_sanitize_visual_prompt = _import_function('src.video.pixel_art_tool', '_sanitize_visual_prompt')
_truncate_prompt_for_resolution = _import_function('src.video.pixel_art_tool', '_truncate_prompt_for_resolution')
_progressive_content_scrub = _import_function('src.video.pixel_art_tool', '_progressive_content_scrub')
_CATEGORY_SAFE_PROMPTS = _import_function('src.video.pixel_art_tool', '_CATEGORY_SAFE_PROMPTS')


# ═══════════════════════════════════════════
# _sanitize_visual_prompt
# ═══════════════════════════════════════════
class TestSanitizeVisualPrompt:

    def test_gore_replaced(self):
        assert 'aftermath' in _sanitize_visual_prompt('gore on the battlefield')
        assert 'gore' not in _sanitize_visual_prompt('gore on the battlefield').lower()

    def test_bloody_replaced(self):
        assert 'impact scene' in _sanitize_visual_prompt('bloody aftermath')

    def test_bloodied_replaced(self):
        assert 'impact scene' in _sanitize_visual_prompt('bloodied soldiers')

    def test_country_names_preserved(self):
        prompt = '16-bit pixel art: Iran military forces near Persian Gulf'
        assert 'Iran' in _sanitize_visual_prompt(prompt)
        assert 'Persian' in _sanitize_visual_prompt(prompt)

    def test_equipment_names_preserved(self):
        prompt = '16-bit pixel art: radar equipment and military vehicles'
        assert 'radar' in _sanitize_visual_prompt(prompt)
        assert 'equipment' in _sanitize_visual_prompt(prompt)

    def test_multiple_gore_terms(self):
        prompt = 'gore and dead bodies after the massacre'
        result = _sanitize_visual_prompt(prompt)
        assert 'gore' not in result.lower()
        assert 'dead bodies' not in result.lower()
        assert 'massacre' not in result.lower()

    def test_empty_string(self):
        assert _sanitize_visual_prompt('') == ''

    def test_no_gore_unchanged(self):
        prompt = '16-bit isometric pixel art: military convoy at sunset'
        assert _sanitize_visual_prompt(prompt) == prompt

    def test_nsfw_replaced(self):
        assert 'scene' in _sanitize_visual_prompt('NSFW content')
        result = _sanitize_visual_prompt('nude figure')
        assert 'nude' not in result.lower()

    def test_case_insensitive(self):
        assert 'aftermath' in _sanitize_visual_prompt('GORE everywhere')
        assert 'impact scene' in _sanitize_visual_prompt('BLOODY battle')

    def test_mass_grave(self):
        result = _sanitize_visual_prompt('mass grave discovered')
        assert 'mass grave' not in result.lower()
        assert 'memorial' in result.lower()

    def test_body_bags(self):
        result = _sanitize_visual_prompt('body bags lined up')
        assert 'body bag' not in result.lower()
        assert 'aftermath' in result.lower()


# ═══════════════════════════════════════════
# _truncate_prompt_for_resolution
# ═══════════════════════════════════════════
class TestTruncatePromptForResolution:

    def test_short_prompt_unchanged(self):
        prompt = '16-bit pixel art: sunset over ocean.'
        assert _truncate_prompt_for_resolution(prompt, 272, 288) == prompt

    def test_long_prompt_truncated_small_resolution(self):
        sentences = [f'This is sentence number {i} with some extra words.' for i in range(10)]
        prompt = ' '.join(sentences)
        result = _truncate_prompt_for_resolution(prompt, 272, 288)
        result_sentences = [s for s in result.replace('. ', '.|').split('|') if s.strip()]
        assert len(result_sentences) <= 3

    def test_long_prompt_truncated_word_count(self):
        words = ' '.join([f'word{i}' for i in range(80)])
        prompt = f'First sentence. Second sentence. Third sentence. {words}.'
        result = _truncate_prompt_for_resolution(prompt, 272, 288)
        assert len(result.split()) <= 50

    def test_large_resolution_unchanged(self):
        prompt = ' '.join([f'Sentence {i} goes here with extra words.' for i in range(10)])
        assert _truncate_prompt_for_resolution(prompt, 1024, 1024) == prompt

    def test_boundary_resolution(self):
        prompt = 'Short prompt.'
        assert _truncate_prompt_for_resolution(prompt, 20, 20) == 'Short prompt.'

    def test_truncated_ends_with_punctuation(self):
        words = ' '.join([f'word{i}' for i in range(60)])
        prompt = f'Sentence one here. {words}.'
        result = _truncate_prompt_for_resolution(prompt, 272, 288)
        assert result.endswith(('.', '!', '?'))


# ═══════════════════════════════════════════
# _progressive_content_scrub
# ═══════════════════════════════════════════
class TestProgressiveContentScrub:

    def test_level1_removes_named_entities(self):
        prompt = 'Trump and Putin meet for diplomacy talks'
        result = _progressive_content_scrub(prompt, 'diplomacy', level=1)
        assert 'Trump' not in result
        assert 'Putin' not in result

    def test_level1_softens_military_terms(self):
        prompt = 'warship attacked the harbor with missiles'
        result = _progressive_content_scrub(prompt, 'warfare', level=1)
        assert 'warship' not in result
        assert 'missiles' not in result

    def test_level1_preserves_visual_terms(self):
        prompt = '16-bit pixel art: sunset landscape with ocean, dramatic lighting'
        result = _progressive_content_scrub(prompt, 'general', level=1)
        assert 'sunset' in result.lower()
        assert 'ocean' in result.lower()

    def test_level2_extracts_visual_keywords(self):
        prompt = 'Trump authorized an attack on the harbor at sunset with dramatic lighting'
        result = _progressive_content_scrub(prompt, 'warfare', level=2)
        assert 'sunset' in result.lower() or 'lighting' in result.lower() or '16-bit' in result.lower()

    def test_level2_no_visual_keywords_falls_to_category(self):
        prompt = 'Trump Putin diplomacy agreement signed'
        result = _progressive_content_scrub(prompt, 'diplomacy', level=2)
        assert 'diplomacy' in result.lower() or 'meeting' in result.lower()

    def test_level3_returns_category_safe(self):
        prompt = 'gore and bloody massacre everywhere'
        result = _progressive_content_scrub(prompt, 'warfare', level=3)
        assert result == _CATEGORY_SAFE_PROMPTS['warfare']

    def test_level3_unknown_category_fallback(self):
        prompt = 'something terrible'
        result = _progressive_content_scrub(prompt, 'nonexistent_category', level=3)
        assert result == _CATEGORY_SAFE_PROMPTS['general']

    def test_level3_all_categories_have_safe_prompts(self):
        expected = ['warfare', 'naval', 'aerial', 'arms_defense', 'markets',
                    'trade_sanctions', 'energy', 'commodities', 'diplomacy',
                    'political', 'espionage', 'protests', 'humanitarian',
                    'border', 'cyber', 'megaprojects', 'general']
        for cat in expected:
            assert cat in _CATEGORY_SAFE_PROMPTS, f'Missing category: {cat}'

    def test_level2_preserves_style_prefix(self):
        prompt = 'sunset ocean dramatic lighting harbor vessels'
        result = _progressive_content_scrub(prompt, 'naval', level=2)
        assert '16-bit' in result


# ═══════════════════════════════════════════
# _CATEGORY_SAFE_PROMPTS
# ═══════════════════════════════════════════
class TestCategorySafePrompts:

    def test_all_prompts_are_composition_style(self):
        for cat, prompt in _CATEGORY_SAFE_PROMPTS.items():
            assert prompt.startswith('16-bit isometric pixel art scene:'), \
                f'Category {cat} prompt does not start with expected prefix: {prompt[:40]}'

    def test_all_prompts_have_lighting(self):
        lighting_terms = ['lighting', 'atmosphere', 'sky', 'glow', 'hour']
        for cat, prompt in _CATEGORY_SAFE_PROMPTS.items():
            has_lighting = any(term in prompt.lower() for term in lighting_terms)
            assert has_lighting, f'Category {cat} prompt has no lighting term: {prompt}'

    def test_all_prompts_reasonable_length(self):
        for cat, prompt in _CATEGORY_SAFE_PROMPTS.items():
            word_count = len(prompt.split())
            assert 10 <= word_count <= 35, \
                f'Category {cat} prompt has {word_count} words (expected 10-35)'