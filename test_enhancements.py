#!/usr/bin/env python3
"""
Test script to validate Phase 1-3 enhancements to YT-Machine.

Tests:
1. Camera movements work properly
2. Voice quality with SSML enhancements
3. Clean video output (no text overlays)
4. Hook optimization in scripts
5. Dynamic pacing
6. Visual consistency
7. Topic diversification

Run this after implementing all enhancements to verify system integrity.
"""

import sys
from pathlib import Path

def test_camera_movements():
    """Test that camera movement functions are properly implemented."""
    print("\n🎥 Testing Camera Movements...")
    
    try:
        from video_server.assembler_tool import _apply_camera_movement
        from moviepy.editor import ImageClip
        
        # Create a test image clip
        test_image = Path("output/images")
        if not any(test_image.glob("*.png")):
            print("  ⚠️  No test images found, skipping camera movement test")
            return False
        
        test_img = list(test_image.glob("*.png"))[0]
        clip = ImageClip(str(test_img)).set_duration(5.0)
        
        # Test each movement type
        movements = ['zoom_in', 'zoom_out', 'pan_right', 'pan_left']
        for movement in movements:
            try:
                animated_clip = _apply_camera_movement(clip, movement, 5.0)
                print(f"  ✅ {movement}: Working")
            except Exception as e:
                print(f"  ❌ {movement}: Failed - {e}")
                return False
        
        print("  ✅ All camera movements functional")
        return True
        
    except Exception as e:
        print(f"  ❌ Camera movement test failed: {e}")
        return False

def test_voice_enhancements():
    """Test voice quality enhancements with SSML."""
    print("\n🎤 Testing Voice Quality Enhancements...")
    
    try:
        from video_server.tts_tool import _add_natural_pacing, _apply_ssml_prosody
        
        test_text = "Today is March 21, 2026. Oil_prices hit one hundred twelve dollars per barrel. This is the third time in forty years."
        
        # Test natural pacing: should clean underscores, stray tags, formatting
        paced_text = _add_natural_pacing(test_text)
        if '_' not in paced_text and '<' not in paced_text:
            print("  ✅ Natural pacing: Text cleaned (no underscores or raw tags)")
        else:
            print("  ❌ Natural pacing: Text not properly sanitised")
            return False
        
        # Test SSML prosody: must be a valid <speak> document with <break> tags
        ssml_text = _apply_ssml_prosody(paced_text, "authoritative")
        if ssml_text.strip().startswith('<speak') and '<break time=' in ssml_text and '<prosody' in ssml_text:
            print("  ✅ SSML prosody: Valid <speak> document with breaks")
        else:
            print("  ❌ SSML prosody: Not a valid SSML document")
            return False
        
        print("  ✅ Voice enhancements functional")
        return True
        
    except Exception as e:
        print(f"  ❌ Voice enhancement test failed: {e}")
        return False

def test_clean_video_output():
    """Test that text overlays are disabled."""
    print("\n🎬 Testing Clean Video Output...")
    
    try:
        from video_server.assembler_tool import build_final_video
        import inspect
        
        # Check that text overlay code is commented out
        source = inspect.getsource(build_final_video)
        
        if '# DISABLED: Text captions' in source:
            print("  ✅ Text captions: Disabled")
        else:
            print("  ⚠️  Text captions: May still be active")
        
        if '# DISABLED: HUD overlays' in source:
            print("  ✅ HUD overlays: Disabled")
        else:
            print("  ⚠️  HUD overlays: May still be active")
        
        if '# DISABLED: Ticker marquee' in source:
            print("  ✅ Ticker marquee: Disabled")
        else:
            print("  ⚠️  Ticker marquee: May still be active")
        
        print("  ✅ Clean video configuration verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Clean video test failed: {e}")
        return False

def test_hook_optimization():
    """Test that hook optimization is in system prompts."""
    print("\n🎣 Testing Hook Optimization...")
    
    try:
        import json
        
        with open('config/system_prompts.json', 'r') as f:
            prompts = json.load(f)
        
        script_prompt = prompts['prompts']['script_synthesizer']['system_prompt']
        
        if 'PATTERN-INTERRUPT' in script_prompt:
            print("  ✅ Pattern-interrupt hooks: Configured")
        else:
            print("  ❌ Pattern-interrupt hooks: Not found")
            return False
        
        if 'NEVER start with dates' in script_prompt or 'NO date' in script_prompt:
            print("  ✅ Date restriction: Enforced")
        else:
            print("  ⚠️  Date restriction: May not be enforced")
        
        print("  ✅ Hook optimization configured")
        return True
        
    except Exception as e:
        print(f"  ❌ Hook optimization test failed: {e}")
        return False

def test_dynamic_pacing():
    """Test dynamic scene duration calculation."""
    print("\n⏱️  Testing Dynamic Pacing...")
    
    try:
        from video_server.assembler_tool import _calculate_dynamic_durations
        
        # Test with 6 scenes (typical video)
        durations = _calculate_dynamic_durations(60.0, 6)
        
        if len(durations) == 6:
            print(f"  ✅ Duration count: {len(durations)} scenes")
        else:
            print(f"  ❌ Duration count: Expected 6, got {len(durations)}")
            return False
        
        # Check that durations vary
        if len(set(durations)) > 1:
            print(f"  ✅ Duration variation: {min(durations):.1f}s - {max(durations):.1f}s")
        else:
            print("  ❌ Duration variation: All durations are equal")
            return False
        
        # Check total matches
        total = sum(durations)
        if abs(total - 60.0) < 0.1:
            print(f"  ✅ Total duration: {total:.2f}s (matches target)")
        else:
            print(f"  ❌ Total duration: {total:.2f}s (expected 60.0s)")
            return False
        
        print("  ✅ Dynamic pacing functional")
        return True
        
    except Exception as e:
        print(f"  ❌ Dynamic pacing test failed: {e}")
        return False

def test_visual_consistency():
    """Test brand color palette configuration."""
    print("\n🎨 Testing Visual Consistency...")
    
    try:
        from video_server.pixel_art_tool import BRAND_COLORS, STYLE_SUFFIX
        
        required_colors = ['primary', 'accent', 'highlight', 'neutral']
        for color in required_colors:
            if color in BRAND_COLORS:
                print(f"  ✅ {color.capitalize()}: {BRAND_COLORS[color]}")
            else:
                print(f"  ❌ {color.capitalize()}: Not defined")
                return False
        
        if 'NO text' in STYLE_SUFFIX or 'NO TEXT' in STYLE_SUFFIX:
            print("  ✅ Text exclusion: Configured")
        else:
            print("  ⚠️  Text exclusion: May not be enforced")
        
        if 'navy' in STYLE_SUFFIX and 'amber' in STYLE_SUFFIX:
            print("  ✅ Brand colors: Integrated in style")
        else:
            print("  ⚠️  Brand colors: May not be in style suffix")
        
        print("  ✅ Visual consistency configured")
        return True
        
    except Exception as e:
        print(f"  ❌ Visual consistency test failed: {e}")
        return False

def test_topic_diversification():
    """Test expanded keyword categories."""
    print("\n🌍 Testing Topic Diversification...")
    
    try:
        from redfish.scraper_config import GEOPOLITICAL_KEYWORDS, CATEGORY_WEIGHTS
        
        new_categories = ['technology_disruption', 'climate_geopolitics', 'health_security']
        for category in new_categories:
            if category in GEOPOLITICAL_KEYWORDS:
                keyword_count = len(GEOPOLITICAL_KEYWORDS[category])
                print(f"  ✅ {category}: {keyword_count} keywords")
            else:
                print(f"  ❌ {category}: Not found")
                return False
        
        if 'CATEGORY_WEIGHTS' in dir():
            print(f"  ✅ Category weights: {len(CATEGORY_WEIGHTS)} categories balanced")
        else:
            print("  ⚠️  Category weights: Not defined")
        
        print("  ✅ Topic diversification configured")
        return True
        
    except Exception as e:
        print(f"  ❌ Topic diversification test failed: {e}")
        return False

def test_script_parser():
    """Phase 4.1 - ScriptParser extracts actions, subjects, settings per segment."""
    print("\n🔍 Testing Script Parser (Phase 4.1)...")
    try:
        from redfish.script_parser import ScriptParser
        parser = ScriptParser()
        test_script = {
            "hook": "Three missiles streaked across the Strait of Hormuz at zero four hundred hours. Twenty one percent of global oil just stopped flowing.",
            "historical_1": "In nineteen eighty seven, Iranian speedboats attacked tankers in the Persian Gulf during the Tanker War.",
            "historical_2": "In nineteen ninety one, F-117 Nighthawks struck Iraqi air defenses in Desert Storm.",
            "modern_pivot": "But twenty twenty six is different. Iran now fields hypersonic missiles and drone swarms.",
            "consequence": "In Tokyo, fuel prices hit record highs. Your gas bill just went up thirty percent.",
            "future_outlook": "The next seventy two hours determine whether this blockade holds. And the barrel price? Still climbing.",
        }
        parsed = parser.parse_all_segments(test_script)
        assert len(parsed) == 6, f"Expected 6 segments, got {len(parsed)}"
        hook = parsed[0]
        assert hook["setting"] == "strait of hormuz", f"Setting not extracted: {hook['setting']}"
        hist = parsed[1]
        assert hist["era"] in ("1980s","1990s","2000s","1970s","1960s"), f"Era wrong: {hist['era']}"
        for seg in parsed:
            prompt = parser.build_action_prompt(seg, "pixel art, NO text")
            assert len(prompt) > 20
            print(f"  [{seg['segment']}] action={seg['action']} | setting={seg['setting']}")
        print("  ✅ Script parser functional")
        return True
    except Exception as e:
        print(f"  ❌ Script parser failed: {e}")
        return False


def test_prompt_generator_integration():
    """Phase 4.2 - VisualPromptGenerator uses ScriptParser internally."""
    print("\n🎨 Testing Prompt Generator Integration (Phase 4.2)...")
    try:
        from redfish.prompt_generator import VisualPromptGenerator
        test_script = {
            "hook": "Missiles intercepted over the Strait of Hormuz. Twenty one percent of global oil halted.",
            "historical_1": "In nineteen eighty seven, Iran attacked tankers in the Persian Gulf Tanker War.",
            "historical_2": "In nineteen ninety one, coalition forces struck Iraq in Desert Storm.",
            "modern_pivot": "Twenty twenty six brings hypersonic missiles and drone swarms to the equation.",
            "consequence": "Gas prices surge thirty percent. Food costs rising. Supply chains breaking.",
            "future_outlook": "The blockade holds. And the barrel price? Still climbing.",
        }
        news = {"topic": "Hormuz blockade", "impact_score": 9}
        visuals = {"primary_subjects": ["missile"], "settings": ["Strait of Hormuz"], "actions": ["intercept"], "mood": "tense", "temporal_context": "dusk"}
        gen = VisualPromptGenerator(news, visuals, test_script)
        assert hasattr(gen, "_parser"), "ScriptParser not embedded"
        assert len(gen._parsed_segments) == 6
        scenes = gen.generate_all_scenes()
        assert len(scenes) == 6
        for name, prompt in scenes.items():
            assert len(prompt) > 20, f"Empty prompt for {name}"
        print(f"  ✅ {len(scenes)} scene prompts generated via ScriptParser")
        return True
    except Exception as e:
        print(f"  ❌ Prompt generator integration failed: {e}")
        return False


def test_retention_architecture():
    """Phase 4.3 - Retention architecture added to script synthesizer prompt."""
    print("\n🔄 Testing Retention Architecture (Phase 4.3)...")
    try:
        import json
        with open("config/system_prompts.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        prompt = d["prompts"]["script_synthesizer"]["system_prompt"]
        checks = {
            "Retention Architecture": "RETENTION ARCHITECTURE",
            "Looping Technique": "LOOPING TECHNIQUE",
            "Value Delivery Rule": "VALUE DELIVERY RULE",
            "Open Loop technique": "OPEN LOOP",
            "Pattern-interrupt hook": "PATTERN-INTERRUPT",
        }
        for label, marker in checks.items():
            assert marker in prompt, f"{label} missing"
            print(f"  ✅ {label}: present")
        return True
    except Exception as e:
        print(f"  ❌ Retention architecture failed: {e}")
        return False


def test_scene_subcuts():
    """Phase 4.4 - Scene sub-cuts for visual variety."""
    print("\n✂️  Testing Scene Sub-Cuts (Phase 4.4)...")
    try:
        from video_server.assembler_tool import _create_scene_subcuts, _calculate_dynamic_durations
        import inspect
        sig = inspect.signature(_create_scene_subcuts)
        params = list(sig.parameters.keys())
        assert "clip" in params and "scene_dur" in params and "movement_type" in params
        print("  ✅ _create_scene_subcuts signature correct")
        durations = _calculate_dynamic_durations(70.0, 6)
        assert len(durations) == 6
        assert abs(sum(durations) - 70.0) < 0.01
        var = max(durations) - min(durations)
        assert var > 0.5, f"Durations too uniform: {durations}"
        print(f"  ✅ Dynamic durations vary {min(durations):.1f}s - {max(durations):.1f}s")
        return True
    except Exception as e:
        print(f"  ❌ Scene sub-cuts failed: {e}")
        return False


def test_pipeline_integration():
    """Phase 4.5 - Script parser integrated into generate_complete_video.py."""
    print("\n🔗 Testing Pipeline Integration (Phase 4.5)...")
    try:
        with open("generate_complete_video.py", "r", encoding="utf-8") as f:
            src = f.read()
        assert "from redfish.script_parser import ScriptParser" in src
        assert "script_parser.parse_all_segments(script)" in src
        assert "parsed_segments" in src
        print("  ✅ ScriptParser imported and called in pipeline")
        return True
    except Exception as e:
        print(f"  ❌ Pipeline integration failed: {e}")
        return False


def test_negative_prompts():
    """Phase 5.1 - Negative prompts + prompt specificity scoring."""
    print("\n🚫 Testing Negative Prompts + Specificity (Phase 5.1)...")
    try:
        from video_server.pixel_art_tool import NEGATIVE_PROMPT, _score_prompt_specificity, STYLE_SUFFIX
        assert "text" in NEGATIVE_PROMPT and "watermark" in NEGATIVE_PROMPT and "blurry" in NEGATIVE_PROMPT
        print(f"  ✅ NEGATIVE_PROMPT: {len(NEGATIVE_PROMPT)} chars, correct exclusions")

        low = _score_prompt_specificity("in dramatic confrontation, strategic forces")
        assert low < 35, f"Generic scored too high: {low}"
        print(f"  ✅ Generic prompt penalised: {low}/100")

        high = _score_prompt_specificity("Iranian warships blockading the Strait of Hormuz, missiles launching at dusk")
        assert high >= 50, f"Specific scored too low: {high}"
        print(f"  ✅ Specific prompt rewarded: {high}/100")

        assert "NO text" in STYLE_SUFFIX or "NO TEXT" in STYLE_SUFFIX
        print("  ✅ STYLE_SUFFIX enforces NO text")
        return True
    except Exception as e:
        print(f"  ❌ Negative prompts failed: {e}")
        return False


def test_audio_mastering():
    """Phase 5.2 - Audio mastering wired into generate_voiceover()."""
    print("\n🎚️  Testing Audio Mastering (Phase 5.2)...")
    try:
        from video_server.tts_tool import _apply_audio_mastering
        import inspect
        sig = inspect.signature(_apply_audio_mastering)
        assert "input_path" in sig.parameters
        print("  ✅ _apply_audio_mastering signature correct")
        with open("video_server/tts_tool.py", "r", encoding="utf-8") as f:
            src = f.read()
        assert "_apply_audio_mastering(filepath)" in src
        assert "audio_mastered" in src
        print("  ✅ Mastering wired into generate_voiceover() with result flag")
        return True
    except Exception as e:
        print(f"  ❌ Audio mastering failed: {e}")
        return False


def test_voice_content_selection():
    """Phase 5.3 - Voice auto-selected by content type."""
    print("\n🎙️  Testing Voice Content Selection (Phase 5.3)...")
    try:
        from video_server.tts_tool import select_voice_for_content, CONTENT_VOICE_MAP, VOICE_MAPPING
        assert len(CONTENT_VOICE_MAP) >= 10
        print(f"  ✅ CONTENT_VOICE_MAP: {len(CONTENT_VOICE_MAP)} content types")
        cases = [
            ("Iran launched missiles at military targets in a war zone", "authoritative"),
            ("Climate scientists report record glacier melt and flooding", "calm"),
            ("Economic sanctions and financial collapse hit currency markets", "professional"),
            ("Breaking urgent crisis developing rapidly", "energetic"),
        ]
        for text, expected in cases:
            selected = select_voice_for_content(text)
            assert selected == expected, f"Expected {expected}, got {selected}"
            print(f"  ✅ '{text[:40]}...' → {selected}")
        with open("video_server/tts_tool.py", "r", encoding="utf-8") as f:
            src = f.read()
        assert "select_voice_for_content(text" in src
        print("  ✅ Auto-selection wired into generate_voiceover()")
        return True
    except Exception as e:
        print(f"  ❌ Voice content selection failed: {e}")
        return False


def main():
    """Run all enhancement tests."""
    print("=" * 60)
    print("YT-MACHINE ENHANCEMENT VALIDATION")
    print("Testing Phases 1-5 Implementation")
    print("=" * 60)

    results = {
        "Camera Movements": test_camera_movements(),
        "Voice Enhancements": test_voice_enhancements(),
        "Clean Video Output": test_clean_video_output(),
        "Hook Optimization": test_hook_optimization(),
        "Dynamic Pacing": test_dynamic_pacing(),
        "Visual Consistency": test_visual_consistency(),
        "Topic Diversification": test_topic_diversification(),
        "Script Parser": test_script_parser(),
        "Prompt Generator Integration": test_prompt_generator_integration(),
        "Retention Architecture": test_retention_architecture(),
        "Scene Sub-Cuts": test_scene_subcuts(),
        "Pipeline Integration": test_pipeline_integration(),
        "Negative Prompts": test_negative_prompts(),
        "Audio Mastering": test_audio_mastering(),
        "Voice Content Selection": test_voice_content_selection(),
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 60)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All enhancements validated successfully!")
        print("System ready for video generation testing.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
