"""
Unit tests for dedup logic — segue overlap, closing echo, curation fidelity.
No API calls, no network, fast.
Run: python -m pytest tests/test_dedup.py -v
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


def _make_llm():
    from src.brain.llm_interface import LLMInterface
    return LLMInterface()


# ══════════════════════════════════════════════════════════════
# 1. _dedup_segue_overlap — prefix overlap detection
# ══════════════════════════════════════════════════════════════
class TestDedupSegueOverlap:
    def setup_method(self):
        self.llm = _make_llm()

    def test_prefix_overlap_stripped(self):
        """AND before that even lands — ... appears in both segue and next part_1"""
        script = {
            'stories': [
                {
                    'segue': 'And before that even lands \u2014 Brent and Israel is already in motion.',
                    'part_1_narration': 'Some story 1 text',
                },
                {
                    'part_1_narration': 'And before that even lands \u2014 the forces behind Brent and Israel are already in motion.',
                    'part_2_narration': 'Goldman Sachs saw fees jump.',
                    'real_talk': 'trading gains are a mirage.',
                    'fallout': 'banks will lobby for regime change.',
                },
            ]
        }
        result = self.llm._dedup_segue_overlap(script)
        p1 = result['stories'][1]['part_1_narration']
        assert 'And before that even lands' not in p1, f"Prefix overlap NOT stripped: {p1}"
        assert 'forces behind' in p1, f"Expected 'forces behind' in: {p1}"

    def test_bridging_conjunction_3word_stripped(self):
        """3-word overlap starting with 'And' (bridging conjunction) IS stripped (threshold lowered to 3)"""
        script = {
            'stories': [
                {'segue': 'And now for something', 'part_1_narration': 'x'},
                {'part_1_narration': 'And now for the market is crashing hard.', 'part_2_narration': 'more text'},
            ]
        }
        result = self.llm._dedup_segue_overlap(script)
        p1 = result['stories'][1]['part_1_narration']
        assert 'And now for' not in p1, f"3-word overlap with bridging conjunction SHOULD be stripped: {p1}"
        assert 'the market' in p1, f"Expected 'the market' in: {p1}"

    def test_3word_non_bridging_not_stripped(self):
        """3-word overlap without a bridging conjunction should NOT trigger (threshold is 4)"""
        script = {
            'stories': [
                {'segue': 'The market is crashing today', 'part_1_narration': 'x'},
                {'part_1_narration': 'The market is Goldman Sachs is concerned about it', 'part_2_narration': 'more'},
            ]
        }
        result = self.llm._dedup_segue_overlap(script)
        p1 = result['stories'][1]['part_1_narration']
        assert 'The market is' in p1, f"3-word overlap without bridging conjunction should NOT be stripped: {p1}"

    def test_tail_overlap_still_works(self):
        """Original tail-vs-head overlap check still functions (2+ word match)"""
        script = {
            'stories': [
                {'segue': 'Meanwhile the market crash is accelerating fast', 'part_1_narration': 'x'},
                {'part_1_narration': 'accelerating fast Goldman Sachs is concerned about it', 'part_2_narration': 'more'},
            ]
        }
        result = self.llm._dedup_segue_overlap(script)
        p1 = result['stories'][1]['part_1_narration']
        assert not p1.startswith('accelerating fast'), f"Tail overlap NOT stripped: {p1}"

    def test_no_overlap_untouched(self):
        """No overlap at all — part_1 should be untouched"""
        script = {
            'stories': [
                {'segue': 'Meanwhile Saudi Arabia just cut off American airspace.', 'part_1_narration': 'x'},
                {'part_1_narration': 'Brent crude jumped eight percent after strikes in Iran.', 'part_2_narration': 'more'},
            ]
        }
        result = self.llm._dedup_segue_overlap(script)
        p1 = result['stories'][1]['part_1_narration']
        assert p1 == 'Brent crude jumped eight percent after strikes in Iran.'


# ══════════════════════════════════════════════════════════════
# 2. _reassemble_script — prefix overlap dedup at final assembly
# ══════════════════════════════════════════════════════════════
class TestReassembleScript:
    def setup_method(self):
        self.llm = _make_llm()

    def test_segue_prefix_overlap_stripped_in_reassemble(self):
        """The CRITICAL test: _reassemble_script strips prefix overlap between
        the preceding segue and the next story's body."""
        script = {
            'greeting': 'Baby you are not ready for this!',
            'intro_hook': '',
            'closing': 'Stay behind the curtains, and if I don\'t see you \u2014 good morning, good afternoon, and goodnight.',
            'stories': [
                {
                    'part_1_narration': 'India and Turkey rebooted their friendship.',
                    'part_2_narration': 'New routes avoid the stalled IMEC project.',
                    'real_talk': 'trade routes are just new ways to ignore old enemies.',
                    'fallout': 'Turkey becomes the ultimate gatekeeper.',
                    'segue': 'And before that even lands \u2014 Brent and Israel is already in motion.',
                },
                {
                    'part_1_narration': 'And before that even lands \u2014 the forces behind Brent and Israel are already in motion.',
                    'part_2_narration': 'Goldman Sachs saw fees jump forty-eight percent.',
                    'real_talk': 'trading gains are a deceptive veneer.',
                    'fallout': 'banks will lobby for regime change.',
                },
            ],
        }
        story_bodies = [
            'India and Turkey rebooted their friendship. New routes avoid the stalled IMEC project. trade routes are just new ways to ignore old enemies. Turkey becomes the ultimate gatekeeper.',
            'And before that even lands \u2014 the forces behind Brent and Israel are already in motion. Goldman Sachs saw fees jump forty-eight percent. trading gains are a deceptive veneer. banks will lobby for regime change.',
        ]
        result = self.llm._reassemble_script(script, story_bodies)
        # The "And before that even lands" should appear only ONCE (in the segue)
        count = result.lower().count('and before that even lands')
        assert count == 1, f"Expected 1 occurrence of 'And before that even lands', got {count} in: {result}"

    def test_no_overlap_reassemble_untouched(self):
        """No overlap — body should be kept as-is"""
        script = {
            'greeting': 'Baby you are not ready for this!',
            'intro_hook': '',
            'closing': 'Stay behind the curtains, and if I don\'t see you \u2014 good morning, good afternoon, and goodnight.',
            'stories': [
                {
                    'part_1_narration': 'India and Turkey rebooted their friendship.',
                    'part_2_narration': 'New routes avoid the stalled IMEC project.',
                    'real_talk': 'it\'s about logistics.',
                    'fallout': 'Turkey becomes the gatekeeper.',
                    'segue': 'Meanwhile Saudi Arabia just cut off American airspace.',
                },
                {
                    'part_1_narration': 'Brent crude jumped after strikes in Iran.',
                    'part_2_narration': 'Goldman Sachs saw fees jump.',
                    'real_talk': 'trading gains are a mirage.',
                    'fallout': 'banks will lobby for change.',
                },
            ],
        }
        story_bodies = [
            'India and Turkey rebooted their friendship. New routes avoid the stalled IMEC project. it\'s about logistics. Turkey becomes the gatekeeper.',
            'Brent crude jumped after strikes in Iran. Goldman Sachs saw fees jump. trading gains are a mirage. banks will lobby for change.',
        ]
        result = self.llm._reassemble_script(script, story_bodies)
        assert 'Brent crude jumped after strikes in Iran' in result


# ══════════════════════════════════════════════════════════════
# 3. Closing fallout echo dedup — prefix whitelist
# ══════════════════════════════════════════════════════════════
class TestClosingEcho:
    def setup_method(self):
        self.llm = _make_llm()

    def test_article_prefix_stripped(self):
        """"A structural recession..." echo where "A" is the prefix"""
        script = {
            'greeting': 'Baby you are not ready for this!',
            'intro_hook': '',
            'closing': 'A structural recession... Stay behind the curtains, and if I don\'t see you \u2014 good morning, good afternoon, and goodnight.',
            'stories': [
                {
                    'part_1_narration': 'Some story 1 text.',
                    'part_2_narration': 'More text.',
                    'real_talk': 'the truth is flat.',
                    'fallout': 'a fake market illusion will mask a deep structural recession.',
                    'segue': 'Meanwhile something else.',
                },
                {
                    'part_1_narration': 'Brent crude jumped.',
                    'part_2_narration': 'Goldman Sachs saw fees.',
                    'real_talk': 'trading gains are fake.',
                    'fallout': 'banks will lobby for regime change to unlock resources.',
                },
            ],
        }
        story_bodies = [
            'Some story 1 text. More text. the truth is flat. a fake market illusion will mask a deep structural recession.',
            'Brent crude jumped. Goldman Sachs saw fees. trading gains are fake. banks will lobby for regime change to unlock resources.',
        ]
        result = self.llm._reassemble_script(script, story_bodies)
        # "A structural recession" should NOT appear as a separate phrase before the trademark
        # The closing should start with "... Stay behind" or "Stay behind"
        closing_part = result[result.lower().find('stay behind'):]
        assert 'good morning' in closing_part.lower()

    def test_bare_echo_stripped(self):
        """"Naval expansion forced..." echo in closing is stripped (fallout itself remains in body)"""
        script = {
            'greeting': 'Hello!',
            'intro_hook': '',
            'closing': 'Naval expansion forced... Stay behind the curtains, and if I don\'t see you \u2014 good morning, good afternoon, and goodnight.',
            'stories': [
                {
                    'part_1_narration': 'a', 'part_2_narration': 'b',
                    'real_talk': 'c', 'fallout': 'd', 'segue': 'e',
                },
                {
                    'part_1_narration': 'a2', 'part_2_narration': 'b2',
                    'real_talk': 'c2',
                    'fallout': 'Naval expansion forced a rethink of strategy.',
                },
            ],
        }
        story_bodies = ['a b c d', 'a2 b2 c2 Naval expansion forced a rethink of strategy.']
        result = self.llm._reassemble_script(script, story_bodies)
        # The CLOSING should have "Naval expansion forced" stripped from its start
        # The closing appears at the end, just before/with "Stay behind the curtains"
        closing_idx = result.lower().rfind('stay behind the curtains')
        after_fallout = result[result.lower().rfind('naval expansion forced a rethink of strategy.'):]
        # The closing's "Naval expansion forced" echo should be stripped —
        # so "Naval expansion forced" should NOT appear right before "Stay behind"
        closing_section = result[closing_idx - 80:closing_idx]
        assert 'naval expansion forced...' not in closing_section.lower(), f"Echo NOT stripped from closing: {closing_section}"

    def test_bridge_preserved(self):
        """"And while..." bridge closing should be preserved"""
        script = {
            'greeting': 'Hello!',
            'intro_hook': '',
            'closing': 'And while Europe waits on shipments \u2014 the market is reacting to Brent crude. Stay behind the curtains, and if I don\'t see you \u2014 good morning, good afternoon, and goodnight.',
            'stories': [
                {
                    'part_1_narration': 'a', 'part_2_narration': 'b',
                    'real_talk': 'c', 'fallout': 'd', 'segue': 'e',
                },
                {
                    'part_1_narration': 'a2', 'part_2_narration': 'b2',
                    'real_talk': 'c2',
                    'fallout': 'Europe shipments are delayed and Brent crude surged.',
                },
            ],
        }
        story_bodies = ['a b c d', 'a2 b2 c2 Europe shipments are delayed and Brent crude surged.']
        result = self.llm._reassemble_script(script, story_bodies)
        # "And while Europe waits" should be preserved (it's a bridge)
        assert 'and while europe waits' in result.lower()


# ══════════════════════════════════════════════════════════════
# 4. _validate_curation_fidelity — hallucination stripping
# ══════════════════════════════════════════════════════════════
class TestCurationFidelity:
    def setup_method(self):
        self.llm = _make_llm()

    def test_hallucinated_sentence_stripped(self):
        """A sentence with <50% word overlap with original is removed"""
        curated = [
            'India and Turkey rebooted their friendship. They are ditching Kashmir grudges. Ssssmokin! Brent crude is surging.'
        ]
        original = [
            'India and Turkey rebooted their friendship. They are ditching Kashmir grudges. Trade routes are new ways to ignore enemies.'
        ]
        result = self.llm._validate_curation_fidelity(curated, original)
        # "Ssssmokin!" has almost zero overlap with original words
        assert 'ssssmokin' not in result[0].lower(), f"Hallucinated 'Ssssmokin!' not stripped: {result[0]}"
        # But the real sentences should remain
        assert 'India and Turkey' in result[0]

    def test_duplicate_adjacent_sentence_stripped(self):
        """Exact duplicate adjacent sentence is stripped"""
        curated = [
            'Trade routes are just new ways to ignore old enemies. trade routes are just new ways to ignore old enemies.'
        ]
        original = [
            'Trade routes are just new ways to ignore old enemies. Turkey becomes a gatekeeper.'
        ]
        result = self.llm._validate_curation_fidelity(curated, original)
        count = result[0].lower().count('trade routes are just new ways to ignore old enemies')
        assert count == 1, f"Duplicate sentence NOT stripped: {result[0]}"

    def test_valid_curation_untouched(self):
        """Curation with good fidelity should pass through"""
        curated = [
            'India and Turkey rebooted their friendship! They are ditching Kashmir grudges to move forward.'
        ]
        original = [
            'India and Turkey rebooted their friendship. They are ditching Kashmir grudges to move forward.'
        ]
        result = self.llm._validate_curation_fidelity(curated, original)
        assert 'India and Turkey' in result[0]
        assert 'Kashmir grudges' in result[0]

    def test_all_rejected_falls_back_to_original(self):
        """If ALL curated sentences are rejected, use original"""
        curated = ['Completely unrelated text about something else entirely.']
        original = ['India and Turkey rebooted their friendship. New trade routes are opening.']
        result = self.llm._validate_curation_fidelity(curated, original)
        assert 'India and Turkey' in result[0], f"Should fall back to original: {result[0]}"


# ══════════════════════════════════════════════════════════════
# 5. _rebuild_timeline — closing echo dedup in evaluator
# ══════════════════════════════════════════════════════════════
class TestRebuildTimeline:
    def test_closing_echo_stripped_in_timeline(self):
        """_rebuild_timeline should strip fallout echo from closing"""
        from src.brain.script_evaluator import _rebuild_timeline
        script = {
            'greeting': 'Baby you are not ready for this!',
            'intro_hook': '',
            'closing': 'A structural recession... Stay behind the curtains, and if I don\'t see you \u2014 good morning, good afternoon, and goodnight.',
            'stories': [
                {
                    'part_1_narration': 'India and Turkey just rebooted.',
                    'part_2_narration': 'New routes avoid the stalled IMEC.',
                    'real_talk': 'trade routes are just new ways to ignore old enemies.',
                    'fallout': 'Turkey becomes the gatekeeper for shipments.',
                    'segue': 'Meanwhile Brent crude surged.',
                },
                {
                    'part_1_narration': 'Brent crude jumped after strikes.',
                    'part_2_narration': 'Goldman Sachs saw fees jump.',
                    'real_talk': 'trading gains are a mirage.',
                    'fallout': 'a fake market illusion will mask a deep structural recession.',
                },
            ],
            'segment_timeline': [{'text': 'x', 'image_idx': 0, 'label': 'test'}],
            'all_visual_scenes': [],
        }
        _rebuild_timeline(script)
        closing_seg = None
        for seg in script['segment_timeline']:
            if seg.get('label') == 'closing':
                closing_seg = seg['text']
                break
        assert closing_seg is not None, "No closing segment in timeline"
        # "A structural recession" echo should be stripped
        # The closing should contain the trademark
        assert 'good morning' in closing_seg.lower() and 'goodnight' in closing_seg.lower()

    def test_segue_prefix_overlap_stripped_in_timeline(self):
        """_rebuild_timeline should strip prefix overlap between segue and next part_1"""
        from src.brain.script_evaluator import _rebuild_timeline
        script = {
            'greeting': 'Baby you are not ready for this!',
            'intro_hook': '',
            'closing': 'Stay behind the curtains, and if I don\'t see you \u2014 good morning, good afternoon, and goodnight.',
            'stories': [
                {
                    'part_1_narration': 'India and Turkey just rebooted.',
                    'part_2_narration': 'New routes avoid the stalled IMEC.',
                    'real_talk': 'trade routes are just new ways to ignore old enemies.',
                    'fallout': 'Turkey becomes the gatekeeper for shipments.',
                    'segue': 'And before that even lands \u2014 Brent and Israel is already in motion.',
                },
                {
                    'part_1_narration': 'And before that even lands \u2014 the forces behind Brent and Israel are already in motion.',
                    'part_2_narration': 'Goldman Sachs saw fees jump.',
                    'real_talk': 'trading gains are a deceptive veneer.',
                    'fallout': 'banks will lobby for regime change.',
                },
            ],
            'segment_timeline': [{'text': 'x', 'image_idx': 0, 'label': 'test'}],
            'all_visual_scenes': [],
        }
        _rebuild_timeline(script)
        p1_seg = None
        for seg in script['segment_timeline']:
            if seg.get('label') == 'story_2_part1':
                p1_seg = seg['text']
                break
        assert p1_seg is not None, "No story_2_part1 in timeline"
        assert 'And before that even lands' not in p1_seg, f"Prefix overlap NOT stripped in timeline: {p1_seg}"
        assert 'forces behind' in p1_seg, f"Content after overlap missing: {p1_seg}"


# ══════════════════════════════════════════════════════════════
# 6. _enforce_greeting — trademark enforcement
# ══════════════════════════════════════════════════════════════
class TestEnforceGreeting:
    def setup_method(self):
        self.llm = _make_llm()

    def test_trademark_greeting_forced(self):
        """Greeting must always be the trademark"""
        script = {
            'greeting': 'Ssssmokin!',
            'full_text': 'Ssssmokin! ... India and Turkey rebooted.',
            'stories': [
                {'part_1_narration': 'India and Turkey rebooted.'},
            ],
        }
        result = self.llm._enforce_greeting(script)
        assert result['greeting'] == 'Baby you are not ready for this!'

    def test_existing_trademark_preserved(self):
        """If greeting is already the trademark, leave it"""
        script = {
            'greeting': 'Baby you are not ready for this!',
            'full_text': 'Baby you are not ready for this! ... Content.',
            'stories': [
                {'part_1_narration': 'Content.'},
            ],
        }
        result = self.llm._enforce_greeting(script)
        assert result['greeting'] == 'Baby you are not ready for this!'


# ══════════════════════════════════════════════════════════════
# 7. Integration: full pipeline trace with the exact user scenario
# ══════════════════════════════════════════════════════════════
class TestIntegrationUserScenario:
    """Tests that reproduce the exact scenario from the user's output"""

    def setup_method(self):
        self.llm = _make_llm()

    def test_and_before_that_even_lands_not_duplicated(self):
        """The exact 'And before that even lands' overlap from the user's output"""
        script = {
            'greeting': 'Baby you are not ready for this!',
            'intro_hook': '',
            'closing': 'Unlocking resources... Stay behind the curtains, and if I don\'t see you \u2014 good morning, good afternoon, and goodnight.',
            'stories': [
                {
                    'part_1_narration': 'India and Turkey just started twelve rounds of talks after Operation Sindoor froze relations!',
                    'part_2_narration': 'New routes avoid the stalled IMEC project by using the Middle Corridor through Azerbaijan and Georgia.',
                    'real_talk': 'it\'s not about peace, it\'s about logistics.',
                    'fallout': 'india will subsidize these routes, turning turkey into a global gatekeeper.',
                    'segue': 'And before that even lands \u2014 Brent and Israel is already in motion.',
                },
                {
                    'part_1_narration': 'And before that even lands \u2014 the forces behind Brent and Israel are already in motion.',
                    'part_2_narration': 'Goldman Sachs saw fees jump forty-eight percent, but dealmaking crashed twenty-three percent!',
                    'real_talk': 'trading gains are a deceptive veneer masking a structural collapse.',
                    'fallout': 'banks will lobby for regime change to unlock thirty trillion dollars in resources.',
                },
            ],
        }
        # Step 1: enforcement dedup
        result = self.llm._dedup_segue_overlap(script)
        p1 = result['stories'][1]['part_1_narration']
        assert 'And before that even lands' not in p1, f"Enforcement dedup failed: {p1}"

        # Step 2: simulate curator overwriting part_1 (re-introducing overlap)
        result['stories'][1]['part_1_narration'] = 'And before that even lands \u2014 the forces behind Brent and Israel are already in motion.'

        # Step 3: reassemble — the FINAL dedup should catch it
        story_bodies = [
            'India and Turkey just started twelve rounds of talks after Operation Sindoor froze relations! New routes avoid the stalled IMEC project by using the Middle Corridor through Azerbaijan and Georgia. it\'s not about peace, it\'s about logistics. india will subsidize these routes, turning turkey into a global gatekeeper.',
            'And before that even lands \u2014 the forces behind Brent and Israel are already in motion. Goldman Sachs saw fees jump forty-eight percent, but dealmaking crashed twenty-three percent! trading gains are a deceptive veneer masking a structural collapse. banks will lobby for regime change to unlock thirty trillion dollars in resources.',
        ]
        full_text = self.llm._reassemble_script(result, story_bodies)
        count = full_text.lower().count('and before that even lands')
        assert count == 1, f"Expected 'And before that even lands' to appear exactly once in full_text, got {count}:\n{full_text}"

    def test_and_while_europe_waits_not_duplicated(self):
        """The earlier 'And while Europe waits on those shipments' overlap"""
        script = {
            'greeting': 'Baby you are not ready for this!',
            'intro_hook': '',
            'closing': '... Stay behind the curtains, and if I don\'t see you \u2014 good morning, good afternoon, and goodnight.',
            'stories': [
                {
                    'part_1_narration': 'India rebooted trade with Turkey.',
                    'part_2_narration': 'New corridors are opening.',
                    'real_talk': 'trade routes are just new ways to ignore old enemies.',
                    'fallout': 'Turkey becomes the ultimate gatekeeper for every shipment entering Europe.',
                    'segue': 'And while Europe waits on those shipments \u2014 Brent crude is already in motion.',
                },
                {
                    'part_1_narration': 'And while Europe waits on those shipments \u2014 the market is already reacting to a sudden surge in Brent crude.',
                    'part_2_narration': 'Goldman Sachs sees more fees, but it\'s a trap!',
                    'real_talk': 'the flashy trading numbers are hiding a rot in the core.',
                    'fallout': 'a fake market illusion will mask a deep structural recession.',
                },
            ],
        }
        # Enforce dedup
        result = self.llm._dedup_segue_overlap(script)
        p1 = result['stories'][1]['part_1_narration']
        assert 'And while Europe waits' not in p1, f"Prefix dedup failed: {p1}"

        # Simulate curator re-introducing it
        story_bodies = [
            'India rebooted trade with Turkey. New corridors are opening. trade routes are just new ways to ignore old enemies. Turkey becomes the ultimate gatekeeper for every shipment entering Europe.',
            'And while Europe waits on those shipments \u2014 the market is already reacting to a sudden surge in Brent crude. Goldman Sachs sees more fees, but it\'s a trap! the flashy trading numbers are hiding a rot in the core. a fake market illusion will mask a deep structural recession.',
        ]
        full_text = self.llm._reassemble_script(result, story_bodies)
        count = full_text.lower().count('and while europe waits on those shipments')
        assert count == 1, f"Expected 'And while Europe waits on those shipments' once, got {count}"


# ══════════════════════════════════════════════════════════════
# 5. _parse_curated_structures — marker gate validation
# ══════════════════════════════════════════════════════════════
class TestCuratedStructuresMarkerGate:
    def setup_method(self):
        self.llm = _make_llm()

    def test_no_markers_returns_none(self):
        """Unmarked curator output (no [HOOK]/[MECHANISM] etc.) must be rejected."""
        curated = (
            "Ethiopia is building a massive digital highway! "
            "Officials are pumping funds into local infrastructure.\n\n"
            "---\n\n"
            "A massive LNG tanker just sailed into a danger zone! "
            "QatarEnergy is testing global security."
        )
        result = self.llm._parse_curated_structures(curated, expected_count=2)
        assert result is None, f"Expected None for unmarked output, got {result}"

    def test_partial_markers_returns_none(self):
        """Output with only one [HOOK] marker for two stories must be rejected.
        Need at least expected_count * 2 markers (HOOK + one more per story)."""
        curated = (
            "[HOOK] Ethiopia is building a digital highway!\n"
            "[MECHANISM] Officials are pumping funds.\n\n"
            "---\n\n"
            "A massive LNG tanker just sailed into a danger zone! "
            "QatarEnergy is testing global security."
        )
        result = self.llm._parse_curated_structures(curated, expected_count=2)
        assert result is None, f"Expected None for 2 markers in 2 stories (need 4), got {result}"

    def test_full_markers_returns_structures(self):
        """Output with all 4 markers per story must parse correctly."""
        curated = (
            "[STORY 1]\n"
            "[HOOK] Ethiopia is building a massive digital highway!\n"
            "[MECHANISM] Officials are pumping funds into local infrastructure.\n"
            "[REAL_TALK] it's a digital fence to keep tech giants out.\n"
            "[FALLOUT] a new protectionist bloc will make it harder.\n\n"
            "---\n\n"
            "[STORY 2]\n"
            "[HOOK] A massive LNG tanker just sailed into a danger zone!\n"
            "[MECHANISM] QatarEnergy is testing global security.\n"
            "[REAL_TALK] doha is playing a high-stakes game of chicken.\n"
            "[FALLOUT] market risk premiums will drive over-ordering."
        )
        result = self.llm._parse_curated_structures(curated, expected_count=2)
        assert result is not None, "Expected parsed structures for properly marked output"
        assert len(result) == 2
        assert 'hook' in result[0]
        assert result[0]['hook'] == 'Ethiopia is building a massive digital highway!'
        assert result[1]['hook'] == 'A massive LNG tanker just sailed into a danger zone!'
        assert result[1]['fallout'] == 'market risk premiums will drive over-ordering.'

    def test_exactly_minimum_markers_passes(self):
        """2 stories with 4 markers (2 per story = minimum) should pass the gate."""
        curated = (
            "[STORY 1]\n"
            "[HOOK] Story one hook.\n"
            "[MECHANISM] Story one mechanism.\n"
            "[REAL_TALK] Story one real talk.\n"
            "[FALLOUT] Story one fallout.\n\n"
            "[STORY 2]\n"
            "[HOOK] Story two hook.\n"
            "[MECHANISM] Story two mechanism.\n"
            "[REAL_TALK] Story two real talk.\n"
            "[FALLOUT] Story two fallout."
        )
        result = self.llm._parse_curated_structures(curated, expected_count=2)
        assert result is not None, "Expected structures for minimum marker count"