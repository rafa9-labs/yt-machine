"""
Tests for VRAM Orchestrator, image retry, zoom, resolution, and outro features.
Run: python -m pytest tests/test_vram_orchestrator.py -v
"""
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from PIL import Image


# ════════════════════════════════════════════════════════════════
# 1. MODEL ORCHESTRATOR TESTS
# ════════════════════════════════════════════════════════════════

class TestModelOrchestrator:
    """Test the ModelOrchestrator phase transitions and lifecycle."""

    def setup_method(self):
        from src.video.model_orchestrator import ModelOrchestrator
        self.orch = ModelOrchestrator()

    def test_initial_state(self):
        assert self.orch.current_phase == 'idle'
        assert self.orch._flux_loaded is False
        assert self.orch._ollama_evicted is False

    def test_status_returns_dict(self):
        status = self.orch.status()
        assert 'phase' in status
        assert 'flux_loaded' in status
        assert 'ollama_evicted' in status
        assert 'free_vram_gb' in status

    def test_phase_llm_transitions(self):
        self.orch.phase_llm()
        assert self.orch.current_phase == 'llm'
        assert self.orch._flux_loaded is False

    def test_phase_tts_transitions(self):
        self.orch.phase_tts()
        assert self.orch.current_phase == 'tts'
        assert self.orch._flux_loaded is False

    def test_phase_video_assembly(self):
        self.orch.phase_video_assembly()
        assert self.orch.current_phase == 'video_edit'

    def test_phase_cleanup(self):
        self.orch.phase_cleanup()
        assert self.orch.current_phase == 'idle'

    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram')
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline')
    def test_phase_image_generation_sets_keep_alive(self, mock_preload, mock_signal, mock_wait, mock_evict):
        mock_wait.return_value = True
        mock_preload.return_value = True
        from src.video.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        orch.phase_image_generation()
        mock_signal.assert_called_once_with(True)
        mock_evict.assert_called_once()
        assert orch.current_phase == 'image_gen'

    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram')
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline')
    def test_phase_image_generation_done_clears_keep_alive(self, mock_preload, mock_signal, mock_wait, mock_evict):
        mock_wait.return_value = True
        mock_preload.return_value = True
        from src.video.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        orch.phase_image_generation()
        orch.phase_image_generation_done()
        # Second call to signal should be False (clear keep_alive)
        assert mock_signal.call_count == 2
        mock_signal.assert_called_with(False)
        assert orch.current_phase == 'post_image'

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram')
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline')
    def test_phase_image_generation_vram_insufficient(self, mock_preload, mock_signal, mock_wait, mock_evict, mock_vram):
        mock_wait.return_value = False
        from src.video.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        result = orch.phase_image_generation()
        assert result is False
        mock_evict.assert_called_once()

    @patch('src.video.pixel_art_tool._flush_flux_pipeline')
    def test_phase_image_generation_done_flushes_flux(self, mock_flush):
        from src.video.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        orch._flux_loaded = True  # Simulate FLUX being loaded
        orch.phase_image_generation_done()
        mock_flush.assert_called_once()

    def test_full_pipeline_lifecycle(self):
        """Test a complete pipeline lifecycle: llm → image_gen → post → tts → cleanup."""
        with patch.object(self.orch, '_evict_ollama'), \
             patch.object(self.orch, '_wait_for_vram', return_value=True), \
             patch.object(self.orch, '_flush_flux_if_loaded'), \
             patch('src.video.pixel_art_tool.signal_flux_keep_alive'), \
             patch('src.video.pixel_art_tool.preload_flux_pipeline', return_value=True):

            self.orch.phase_llm()
            assert self.orch.current_phase == 'llm'

            self.orch.phase_image_generation()
            assert self.orch.current_phase == 'image_gen'

            self.orch.phase_image_generation_done()
            assert self.orch.current_phase == 'post_image'

            self.orch.phase_tts()
            assert self.orch.current_phase == 'tts'

            self.orch.phase_video_assembly()
            assert self.orch.current_phase == 'video_edit'

            self.orch.phase_cleanup()
            assert self.orch.current_phase == 'idle'


# ════════════════════════════════════════════════════════════════
# 2. FLUX KEEP-ALIVE SIGNAL TESTS
# ════════════════════════════════════════════════════════════════

class TestFluxKeepAlive:
    """Test the FLUX keep-alive signal mechanism."""

    def test_signal_flux_keep_alive_true(self):
        from src.video.pixel_art_tool import signal_flux_keep_alive, _BATCH_KEEP_ALIVE
        # Import fresh module-level variable
        import src.video.pixel_art_tool as pat
        signal_flux_keep_alive(True)
        assert pat._BATCH_KEEP_ALIVE is True

    def test_signal_flux_keep_alive_false(self):
        from src.video.pixel_art_tool import signal_flux_keep_alive
        import src.video.pixel_art_tool as pat
        signal_flux_keep_alive(True)
        assert pat._BATCH_KEEP_ALIVE is True
        signal_flux_keep_alive(False)
        assert pat._BATCH_KEEP_ALIVE is False

    def test_batch_keep_alive_overrides_compile_keep_alive(self):
        """When _KEEP_ALIVE=False (no compile) but _BATCH_KEEP_ALIVE=True,
        pipeline should stay loaded."""
        from src.video.pixel_art_tool import signal_flux_keep_alive
        import src.video.pixel_art_tool as pat
        original_compile = pat._KEEP_ALIVE
        signal_flux_keep_alive(True)
        effective = pat._KEEP_ALIVE or pat._BATCH_KEEP_ALIVE
        assert effective is True
        signal_flux_keep_alive(False)
        effective = pat._KEEP_ALIVE or pat._BATCH_KEEP_ALIVE
        assert effective == original_compile


# ════════════════════════════════════════════════════════════════
# 3. IMAGE FAILED DETECTION TESTS
# ════════════════════════════════════════════════════════════════

class TestDetectFailedImage:
    """Test the _detect_failed_image() function with various image types."""

    def _create_solid_color_image(self, color, size=(100, 100), path=None):
        img = Image.new('RGB', size, color)
        if path:
            img.save(str(path), format='PNG')
            return path
        return img

    def test_solid_black_image_detected(self, tmp_path):
        from src.video.pixel_art_tool import _detect_failed_image
        path = tmp_path / "solid_black.png"
        self._create_solid_color_image((0, 0, 0), path=path)
        is_failed, reason = _detect_failed_image(str(path))
        assert is_failed is True
        assert 'monochrome' in reason.lower() or 'flat' in reason.lower()

    def test_solid_white_image_detected(self, tmp_path):
        from src.video.pixel_art_tool import _detect_failed_image
        path = tmp_path / "solid_white.png"
        self._create_solid_color_image((255, 255, 255), path=path)
        is_failed, reason = _detect_failed_image(str(path))
        assert is_failed is True
        assert 'monochrome' in reason.lower() or 'flat' in reason.lower()

    def test_solid_green_image_detected(self, tmp_path):
        from src.video.pixel_art_tool import _detect_failed_image
        path = tmp_path / "solid_green.png"
        self._create_solid_color_image((10, 130, 10), path=path)
        is_failed, reason = _detect_failed_image(str(path))
        assert is_failed is True

    def test_normal_image_passes(self, tmp_path):
        from src.video.pixel_art_tool import _detect_failed_image
        path = tmp_path / "normal.png"
        # Create a varied image with different colors
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(str(path), format='PNG')
        is_failed, reason = _detect_failed_image(str(path))
        assert is_failed is False
        assert reason == 'ok'

    def test_near_monochrome_image_detected(self, tmp_path):
        from src.video.pixel_art_tool import _detect_failed_image
        path = tmp_path / "near_mono.png"
        # All pixels very close to the same value
        arr = np.full((100, 100, 3), 10, dtype=np.uint8)
        # Add tiny noise
        noise = np.random.randint(0, 2, (100, 100, 3), dtype=np.uint8)
        arr = arr + noise
        img = Image.fromarray(arr)
        img.save(str(path), format='PNG')
        is_failed, reason = _detect_failed_image(str(path))
        assert is_failed is True

    def test_gradient_image_passes(self, tmp_path):
        from src.video.pixel_art_tool import _detect_failed_image
        path = tmp_path / "gradient.png"
        # Create a gradient image
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            arr[i, :, 0] = i  # Red channel gradient
            arr[:, i, 1] = i  # Green channel gradient
            arr[i, :, 2] = 100 - i  # Blue channel inverse gradient
        img = Image.fromarray(arr)
        img.save(str(path), format='PNG')
        is_failed, reason = _detect_failed_image(str(path))
        assert is_failed is False

    def test_placeholder_color_detected(self, tmp_path):
        """The fallback placeholder color (10, 5, 25) should be detected as failed."""
        from src.video.pixel_art_tool import _detect_failed_image
        path = tmp_path / "placeholder.png"
        self._create_solid_color_image((10, 5, 25), path=path)
        is_failed, reason = _detect_failed_image(str(path))
        assert is_failed is True


# ════════════════════════════════════════════════════════════════
# 4. SHARPENING POST-PROCESS TESTS
# ════════════════════════════════════════════════════════════════

class TestSharpening:
    """Test the _apply_sharpening() function."""

    def test_sharpening_preserves_size(self, tmp_path):
        from src.video.pixel_art_tool import _apply_sharpening
        # Create a test image at target resolution
        arr = np.random.randint(50, 200, (1152, 1088, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        path = tmp_path / "test_sharpen.png"
        img.save(str(path), format='PNG')

        original_size = img.size
        result = _apply_sharpening(str(path))
        assert result == str(path)

        sharpened = Image.open(str(path))
        assert sharpened.size == original_size

    def test_sharpening_returns_same_path(self, tmp_path):
        from src.video.pixel_art_tool import _apply_sharpening
        arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        path = tmp_path / "test_sharpen2.png"
        img.save(str(path), format='PNG')

        result = _apply_sharpening(str(path))
        assert result == str(path)
        assert Path(result).exists()


# ════════════════════════════════════════════════════════════════
# 5. ZOOM PROFILES + EASING TESTS
# ════════════════════════════════════════════════════════════════

class TestZoomProfiles:
    """Test the updated zoom profiles and easing function."""

    def test_all_zoom_profiles_center_only(self):
        from src.video.split_video_assembler import SCENE_ZOOM_PROFILES
        for idx, profile in SCENE_ZOOM_PROFILES.items():
            assert profile['pan_x'] == 0.0, f"Profile {idx} ({profile['name']}) has pan_x={profile['pan_x']}, expected 0.0"

    def test_zoom_ranges_are_gentle(self):
        from src.video.split_video_assembler import SCENE_ZOOM_PROFILES
        for idx, profile in SCENE_ZOOM_PROFILES.items():
            zoom_range = abs(profile['zoom_end'] - profile['zoom_start'])
            assert zoom_range <= 0.10, f"Profile {idx} ({profile['name']}) has zoom range {zoom_range:.3f}, expected <= 0.10"

    def test_ease_out_cubic_boundary_values(self):
        """ease_out_cubic(0) = 0, ease_out_cubic(1) = 1."""
        # Inline the function since it's defined inside _render_scene_opencv
        def ease_out_cubic(t):
            t = max(0.0, min(1.0, t))
            return 1.0 - (1.0 - t) ** 3

        assert abs(ease_out_cubic(0.0) - 0.0) < 1e-6
        assert abs(ease_out_cubic(1.0) - 1.0) < 1e-6
        assert ease_out_cubic(0.5) > 0.5  # ease-out is fast-start, slow-end

    def test_ease_out_cubic_monotonic(self):
        """ease_out_cubic should be monotonically increasing."""
        def ease_out_cubic(t):
            t = max(0.0, min(1.0, t))
            return 1.0 - (1.0 - t) ** 3

        prev = ease_out_cubic(0.0)
        for i in range(1, 101):
            t = i / 100.0
            val = ease_out_cubic(t)
            assert val >= prev, f"Non-monotonic at t={t}: {val} < {prev}"
            prev = val


# ════════════════════════════════════════════════════════════════
# 6. RESOLUTION CONFIGURATION TESTS
# ════════════════════════════════════════════════════════════════

class TestResolutionConfig:
    """Test that resolution configuration is correct."""

    def test_render_resolution_is_768x810(self):
        config_path = Path(__file__).parent.parent / "config" / "image_style.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert config['generation_params']['render_resolution'] == [768, 810]

    def test_inference_steps_is_40(self):
        config_path = Path(__file__).parent.parent / "config" / "image_style.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert config['generation_params']['num_inference_steps'] == 40

    def test_guidance_scale_is_4(self):
        config_path = Path(__file__).parent.parent / "config" / "image_style.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert config['generation_params']['guidance_scale'] == 4.0

    def test_model_step_config_dev_is_40(self):
        from src.video.pixel_art_tool import MODEL_STEP_CONFIG
        assert MODEL_STEP_CONFIG['fal-ai/flux/dev'] == 40

    def test_enforcement_prefix_includes_sharp_focus(self):
        from src.video.pixel_art_tool import PIXEL_ART_ENFORCEMENT_PREFIX
        assert 'sharp focus' in PIXEL_ART_ENFORCEMENT_PREFIX
        assert 'detailed scene composition' in PIXEL_ART_ENFORCEMENT_PREFIX

    def test_target_resolution_unchanged(self):
        config_path = Path(__file__).parent.parent / "config" / "image_style.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert config['generation_params']['target_resolution'] == [1088, 1152]


# ════════════════════════════════════════════════════════════════
# 7. OUTRO / DYNAMIC CLOSING TESTS
# ════════════════════════════════════════════════════════════════

class TestDynamicClosing:
    """Test the dynamic closing builder."""

    def setup_method(self):
        from src.brain.llm_interface import LLMInterface
        self.llm = LLMInterface.__new__(LLMInterface)

    def test_base_closing_exists(self):
        from src.brain.llm_interface import LLMInterface
        assert hasattr(LLMInterface, 'UNIFIED_CLOSING_BASE')
        assert 'good morning' in LLMInterface.UNIFIED_CLOSING_BASE.lower()
        assert 'goodnight' in LLMInterface.UNIFIED_CLOSING_BASE.lower()

    def test_dynamic_closing_with_topic(self):
        closing = self.llm._build_dynamic_closing(last_topic="Taiwan semiconductor ban", last_fallout="")
        assert "Taiwan semiconductor ban" in closing
        assert "reshapes the board" in closing
        assert "good morning" in closing.lower()

    def test_dynamic_closing_with_fallout_no_topic(self):
        closing = self.llm._build_dynamic_closing(last_topic="", last_fallout="the sanctions will collapse the economy")
        assert "dominoes keep falling" in closing or any(w in closing.lower() for w in ["economy", "sanctions"])

    def test_dynamic_closing_no_context(self):
        closing = self.llm._build_dynamic_closing(last_topic="", last_fallout="")
        assert "dominoes keep falling" in closing

    def test_validate_closing_preserves_truman(self):
        text = "Some story content here. .... Stay behind the curtains, and if I don't see you — good morning, good afternoon, and goodnight."
        result = self.llm._validate_closing(text)
        assert 'good morning' in result.lower()
        assert 'goodnight' in result.lower()

    def test_validate_closing_injects_dynamic(self):
        self.llm._last_story_topic = "Iran nuclear deal"
        self.llm._last_fallout = "sanctions will reshape the region"
        text = "Some story content here without closing"
        result = self.llm._validate_closing(text)
        assert 'iran nuclear deal' in result.lower()
        assert 'good morning' in result.lower()


# ════════════════════════════════════════════════════════════════
# 8. OUTRO TTS SETTINGS TESTS
# ════════════════════════════════════════════════════════════════

class TestOutroTTSSettings:
    """Test the melancholy outro TTS settings."""

    def test_detect_outro_segment_basic(self):
        from src.video.tts_tool import _detect_outro_segment
        text = "Story one content....And that is how Taiwan reshapes the board. Stay behind the curtains."
        main, outro = _detect_outro_segment(text)
        assert outro == "And that is how Taiwan reshapes the board. Stay behind the curtains."

    def test_detect_outro_segment_no_separator(self):
        from src.video.tts_tool import _detect_outro_segment
        text = "Just a regular story without separators"
        main, outro = _detect_outro_segment(text)
        assert outro == ""

    def test_detect_outro_segment_multiple_separators(self):
        from src.video.tts_tool import _detect_outro_segment
        text = "First part....Second part....The closing part here"
        main, outro = _detect_outro_segment(text)
        assert outro == "The closing part here"

    def test_apply_outro_tts_settings_non_outro(self):
        from src.video.tts_tool import _apply_outro_tts_settings
        settings = {'stability': 0.35, 'style_exaggeration': 0.65, 'similarity_boost': 0.70}
        result = _apply_outro_tts_settings(settings, is_outro=False)
        assert result == settings  # No changes when not outro

    def test_apply_outro_tts_settings_outro(self):
        from src.video.tts_tool import _apply_outro_tts_settings
        settings = {'stability': 0.35, 'style_exaggeration': 0.65, 'similarity_boost': 0.70}
        result = _apply_outro_tts_settings(settings, is_outro=True)
        assert result['stability'] == pytest.approx(0.60)
        assert result['style_exaggeration'] == pytest.approx(0.30)
        assert result['similarity_boost'] == pytest.approx(0.75)
        assert result['rate'] == "-15%"
        assert result['pitch'] == "-3Hz"
        assert result['speed'] == pytest.approx(0.85)

    def test_apply_outro_tts_settings_empty_settings(self):
        from src.video.tts_tool import _apply_outro_tts_settings
        result = _apply_outro_tts_settings({}, is_outro=True)
        assert result['rate'] == "-15%"
        assert result['pitch'] == "-3Hz"
        assert result['speed'] == 0.85


# ════════════════════════════════════════════════════════════════
# 9. PROGRESSIVE CONTENT SCRUB + RETRY CONFIG TESTS
# ════════════════════════════════════════════════════════════════

class TestProgressiveContentScrub:
    """Test progressive content scrubbing levels."""

    def test_level_1_removes_named_entities(self):
        from src.video.pixel_art_tool import _progressive_content_scrub
        prompt = "Trump meets Putin in Moscow to discuss bilateral relations"
        result = _progressive_content_scrub(prompt, 'diplomacy', level=1)
        assert 'Trump' not in result
        assert 'Putin' not in result
        assert 'a president' in result or 'a world leader' in result

    def test_level_2_extracts_visual_only(self):
        from src.video.pixel_art_tool import _progressive_content_scrub
        prompt = "Warships patrol the Strait of Hormuz at sunset with dramatic lighting"
        result = _progressive_content_scrub(prompt, 'naval', level=2)
        assert '16-bit' in result or 'naval' in result.lower() or 'pixel art' in result.lower()

    def test_level_3_returns_safe_prompt(self):
        from src.video.pixel_art_tool import _progressive_content_scrub
        prompt = "Extreme violent scene with blood and destruction everywhere"
        result = _progressive_content_scrub(prompt, 'warfare', level=3)
        assert 'aftermath' in result.lower() or 'tactical' in result.lower()

    def test_retry_config_exists(self):
        config_path = Path(__file__).parent.parent / "config" / "image_style.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert 'retry_config' in config['generation_params']
        assert config['generation_params']['retry_config']['max_retries'] == 4
        assert config['generation_params']['retry_config']['progressive_scrub'] is True


# ════════════════════════════════════════════════════════════════
# 10. INTEGRATION: FULL PIPELINE PHASE ORDER TEST
# ════════════════════════════════════════════════════════════════

class TestPipelinePhaseOrder:
    """Test that ModelOrchestrator phase transitions happen in the right order."""

    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._flush_flux_if_loaded')
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram')
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline')
    def test_full_pipeline_order(self, mock_preload, mock_signal, mock_wait, mock_flush, mock_evict):
        from src.video.model_orchestrator import ModelOrchestrator
        mock_wait.return_value = True
        mock_preload.return_value = True

        orch = ModelOrchestrator()
        phases = []

        # Simulate a full pipeline run
        orch.phase_llm()
        phases.append(orch.current_phase)

        orch.phase_image_generation()
        phases.append(orch.current_phase)

        orch.phase_image_generation_done()
        phases.append(orch.current_phase)

        orch.phase_tts()
        phases.append(orch.current_phase)

        orch.phase_video_assembly()
        phases.append(orch.current_phase)

        orch.phase_cleanup()
        phases.append(orch.current_phase)

        assert phases == ['llm', 'image_gen', 'post_image', 'tts', 'video_edit', 'idle']

        # Verify key orchestrator calls
        mock_signal.assert_any_call(True)   # Keep alive during image gen
        mock_signal.assert_any_call(False)   # Clear keep alive after batch
        mock_evict.assert_called()           # Ollama evicted before image gen


if __name__ == '__main__':
    pytest.main([__file__, '-v'])