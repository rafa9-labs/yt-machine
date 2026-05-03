"""
VRAM Budget & Model Switching Validation Tests.

Validates that the ModelOrchestrator enforces the 24GB VRAM budget
including inference overhead, and that model transitions never exceed GPU capacity.

Run: python -m pytest tests/test_vram_budget.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
from src.video.model_orchestrator import ModelOrchestrator, GPU_TOTAL_GB, GPU_BASE_OVERHEAD_GB


# ════════════════════════════════════════════════════════════════
# 1. VRAM BUDGET VALIDATION — Phase-by-phase accounting
# ════════════════════════════════════════════════════════════════

class TestVRAMBudget:
    """Validate that every pipeline phase fits within 24GB."""

    def test_vram_estimates_exist(self):
        est = ModelOrchestrator.VRAM_ESTIMATES
        assert 'ollama' in est
        assert 'flux_dev_8bit' in est
        assert 'flux_dev_bf16' in est
        assert 'kokoro' in est

    def test_inference_overhead_exists(self):
        inf = ModelOrchestrator.INFERENCE_OVERHEAD
        assert 'flux_dev_8bit' in inf
        assert 'ollama' in inf
        assert 'kokoro' in inf

    def test_inference_overhead_is_significant(self):
        """FLUX inference overhead should be >0 — this was the bug."""
        assert ModelOrchestrator.INFERENCE_OVERHEAD['flux_dev_8bit'] > 0, \
            "FLUX inference overhead must be nonzero — intermediate tensors consume VRAM"

    def test_flux_weights_and_inference_both_tracked(self):
        est = ModelOrchestrator.VRAM_ESTIMATES
        inf = ModelOrchestrator.INFERENCE_OVERHEAD
        assert est['flux_dev_8bit'] == 12.5, "8-bit FLUX weights should be 12.5GB, not 14GB"
        assert inf['flux_dev_8bit'] == pytest.approx(5.5), "8-bit FLUX inference overhead at 768x810"

    def test_ollama_fits_in_24gb(self):
        total = ModelOrchestrator.VRAM_ESTIMATES['ollama'] + GPU_BASE_OVERHEAD_GB
        assert total <= GPU_TOTAL_GB

    def test_flux_8bit_weights_fit_in_24gb(self):
        total = ModelOrchestrator.VRAM_ESTIMATES['flux_dev_8bit'] + GPU_BASE_OVERHEAD_GB
        assert total <= GPU_TOTAL_GB

    def test_flux_8bit_peak_fits_in_24gb(self):
        est = ModelOrchestrator.VRAM_ESTIMATES
        inf = ModelOrchestrator.INFERENCE_OVERHEAD
        total = GPU_BASE_OVERHEAD_GB + est['flux_dev_8bit'] + inf['flux_dev_8bit']
        assert total <= GPU_TOTAL_GB, f"FLUX peak ({total}GB) must fit in {GPU_TOTAL_GB}GB"

    def test_flux_bf16_exceeds_24gb(self):
        total = ModelOrchestrator.VRAM_ESTIMATES['flux_dev_bf16'] + GPU_BASE_OVERHEAD_GB
        assert total > GPU_TOTAL_GB

    def test_ollama_plus_flux_weights_exceeds_24gb(self):
        est = ModelOrchestrator.VRAM_ESTIMATES
        total = est['ollama'] + est['flux_dev_8bit'] + GPU_BASE_OVERHEAD_GB
        assert total > GPU_TOTAL_GB, "Ollama+FLUX weights alone must exceed 24GB"

    def test_ollama_plus_flux_peak_exceeds_24gb(self):
        est = ModelOrchestrator.VRAM_ESTIMATES
        inf = ModelOrchestrator.INFERENCE_OVERHEAD
        total = GPU_BASE_OVERHEAD_GB + est['ollama'] + est['flux_dev_8bit'] + inf['flux_dev_8bit']
        assert total > GPU_TOTAL_GB, f"Ollama+FLUX peak ({total}GB) must exceed 24GB"


# ════════════════════════════════════════════════════════════════
# 2. PEAK VRAM CALCULATION — Weights + inference overhead
# ════════════════════════════════════════════════════════════════

class TestPeakVRAM:
    """Test peak VRAM calculation including inference overhead."""

    def setup_method(self):
        self.orch = ModelOrchestrator()

    def test_peak_vram_for_ollama(self):
        peak = self.orch.peak_vram_for_model('ollama')
        expected = GPU_BASE_OVERHEAD_GB + ModelOrchestrator.VRAM_ESTIMATES['ollama'] + ModelOrchestrator.INFERENCE_OVERHEAD['ollama']
        assert peak == pytest.approx(expected)

    def test_peak_vram_for_flux(self):
        peak = self.orch.peak_vram_for_model('flux_dev_8bit')
        expected = GPU_BASE_OVERHEAD_GB + 12.5 + 5.5
        assert peak == pytest.approx(expected)

    def test_peak_vram_for_kokoro(self):
        peak = self.orch.peak_vram_for_model('kokoro')
        expected = GPU_BASE_OVERHEAD_GB + 0.2 + 0.1
        assert peak == pytest.approx(expected)

    def test_idle_vram_is_base_overhead(self):
        assert self.orch.vram_used_gb() == pytest.approx(GPU_BASE_OVERHEAD_GB)

    def test_loaded_model_without_inference(self):
        self.orch._register_model('flux_dev_8bit')
        expected = GPU_BASE_OVERHEAD_GB + 12.5
        assert self.orch.vram_used_gb() == pytest.approx(expected)

    def test_loaded_model_with_inference(self):
        self.orch._register_model('flux_dev_8bit')
        self.orch.begin_inference('flux_dev_8bit')
        expected = GPU_BASE_OVERHEAD_GB + 12.5 + 5.5
        assert self.orch.vram_used_gb() == pytest.approx(expected)

    def test_inference_end_releases_overhead(self):
        self.orch._register_model('flux_dev_8bit')
        self.orch.begin_inference('flux_dev_8bit')
        peak = self.orch.vram_used_gb()
        self.orch.end_inference()
        after = self.orch.vram_used_gb()
        assert after < peak
        assert after == pytest.approx(GPU_BASE_OVERHEAD_GB + 12.5)


# ════════════════════════════════════════════════════════════════
# 3. BUDGET VALIDATION METHOD — Formal checker
# ════════════════════════════════════════════════════════════════

class TestVRAMBudgetValidation:
    """Test the formal budget validation method."""

    def test_validate_returns_dict(self):
        result = ModelOrchestrator.validate_vram_budget()
        assert 'valid' in result
        assert 'phases' in result
        assert 'violations' in result
        assert 'headroom' in result

    def test_all_stable_phases_fit_in_24gb(self):
        result = ModelOrchestrator.validate_vram_budget(total_gpu_gb=24.0)
        for phase_name, budget in result['phases'].items():
            if not budget['brief_overlap']:
                assert budget['peak_gb'] <= 24.0, f"{phase_name} peak {budget['peak_gb']}GB > 24GB"

    def test_image_gen_inference_peak_explicitly_tracked(self):
        result = ModelOrchestrator.validate_vram_budget(total_gpu_gb=24.0)
        assert 'image_gen (inference peak)' in result['phases']
        peak = result['phases']['image_gen (inference peak)']['peak_gb']
        weights = result['phases']['image_gen (idle)']['weights_gb']
        assert peak > weights, "Inference peak should be higher than idle weights"

    def test_ollama_plus_flux_transition_is_flagged(self):
        result = ModelOrchestrator.validate_vram_budget(total_gpu_gb=24.0)
        transition_key = 'llm→image_gen (Ollama→FLUX transition)'
        assert transition_key in result['phases']
        assert result['phases'][transition_key]['brief_overlap'] is True
        transition_violations = [v for v in result['violations'] if v['brief_overlap']]
        assert len(transition_violations) >= 1

    def test_valid_despite_transition_overlap(self):
        result = ModelOrchestrator.validate_vram_budget(total_gpu_gb=24.0)
        assert result['valid'] is True

    def test_each_stable_phase_has_positive_headroom(self):
        result = ModelOrchestrator.validate_vram_budget(total_gpu_gb=24.0)
        for phase_name, headroom in result['headroom'].items():
            if '→' not in phase_name:
                assert headroom > 0, f"{phase_name} has 0 or negative headroom"

    def test_image_gen_inference_peak_headroom(self):
        result = ModelOrchestrator.validate_vram_budget(total_gpu_gb=24.0)
        est = ModelOrchestrator.VRAM_ESTIMATES
        inf = ModelOrchestrator.INFERENCE_OVERHEAD
        expected_peak = GPU_BASE_OVERHEAD_GB + est['flux_dev_8bit'] + inf['flux_dev_8bit']
        expected_headroom = round(24.0 - expected_peak, 2)
        assert result['headroom']['image_gen (inference peak)'] == expected_headroom

    def test_budget_on_12gb_gpu_flagged(self):
        result = ModelOrchestrator.validate_vram_budget(total_gpu_gb=12.0)
        non_transition = [v for v in result['violations'] if not v['brief_overlap']]
        assert len(non_transition) >= 2

    def test_phases_include_both_weights_and_peak(self):
        result = ModelOrchestrator.validate_vram_budget(total_gpu_gb=24.0)
        for phase_name, budget in result['phases'].items():
            assert 'weights_gb' in budget
            assert 'peak_gb' in budget
            assert budget['peak_gb'] >= budget['weights_gb']


# ════════════════════════════════════════════════════════════════
# 4. MODEL SWITCHING — Occupancy tracking through pipeline phases
# ════════════════════════════════════════════════════════════════

class TestModelSwitching:
    """Test that model transitions track VRAM occupancy correctly."""

    def setup_method(self):
        self.orch = ModelOrchestrator()

    def test_initial_state_empty(self):
        assert self.orch.vram_used_gb() == pytest.approx(GPU_BASE_OVERHEAD_GB)
        assert self.orch.loaded_models == {}
        assert self.orch._inference_active is None

    def test_phase_llm_loads_ollama(self):
        with patch.object(self.orch, '_evict_ollama'), \
             patch.object(self.orch, '_flush_flux_if_loaded'), \
             patch.object(self.orch, '_get_free_vram', return_value=22.0):
            self.orch.phase_llm()
        assert 'ollama' in self.orch.loaded_models
        assert self.orch.vram_used_gb() == pytest.approx(GPU_BASE_OVERHEAD_GB + ModelOrchestrator.VRAM_ESTIMATES['ollama'])

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram', return_value=True)
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline', return_value=True)
    def test_phase_image_gen_swaps_ollama_for_flux(self, mock_preload, mock_signal, mock_wait, mock_vram):
        with patch.object(self.orch, '_flush_flux_if_loaded'):
            self.orch.phase_llm()
        assert 'ollama' in self.orch.loaded_models

        self.orch.phase_image_generation()
        assert 'ollama' not in self.orch.loaded_models
        assert 'flux_dev_8bit' in self.orch.loaded_models

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram', return_value=True)
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline', return_value=True)
    def test_phase_tts_swaps_flux_for_kokoro(self, mock_preload, mock_signal, mock_wait, mock_vram):
        self.orch.phase_image_generation()
        with patch.object(self.orch, '_flush_flux_if_loaded') as mock_flush:
            def flush_side():
                self.orch._unregister_model('flux_dev_8bit')
                self.orch._flux_loaded = False
            mock_flush.side_effect = flush_side
            self.orch.phase_tts()
        assert 'kokoro' in self.orch.loaded_models

    def test_phase_cleanup_clears_all(self):
        self.orch._loaded_models = {'ollama': 14.0, 'kokoro': 0.2}
        with patch.object(self.orch, '_evict_ollama'), \
             patch.object(self.orch, '_flush_flux_if_loaded'):
            self.orch.phase_cleanup()
        assert self.orch.loaded_models == {}
        assert self.orch.vram_used_gb() == pytest.approx(GPU_BASE_OVERHEAD_GB)

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram', return_value=True)
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline', return_value=True)
    def test_vram_never_exceeds_24gb_during_full_pipeline(self, mock_preload, mock_signal, mock_wait, mock_vram):
        phases = []

        phases.append(('idle', self.orch.vram_used_gb()))

        with patch.object(self.orch, '_flush_flux_if_loaded'):
            self.orch.phase_llm()
        phases.append(('llm', self.orch.vram_used_gb()))

        self.orch.phase_image_generation()
        phases.append(('image_gen idle', self.orch.vram_used_gb()))

        self.orch.begin_inference('flux_dev_8bit')
        phases.append(('image_gen peak', self.orch.vram_used_gb()))
        self.orch.end_inference()

        self.orch.phase_image_generation_done()
        phases.append(('post_image', self.orch.vram_used_gb()))

        with patch.object(self.orch, '_flush_flux_if_loaded'):
            self.orch.phase_tts()
        phases.append(('tts', self.orch.vram_used_gb()))

        self.orch.phase_video_assembly()
        phases.append(('video_edit', self.orch.vram_used_gb()))

        with patch.object(self.orch, '_evict_ollama'), \
             patch.object(self.orch, '_flush_flux_if_loaded'):
            self.orch.phase_cleanup()
        phases.append(('cleanup', self.orch.vram_used_gb()))

        for phase, vram in phases:
            assert vram <= GPU_TOTAL_GB, f"Phase {phase} uses {vram:.1f}GB > {GPU_TOTAL_GB}GB"

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram', return_value=True)
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline', return_value=True)
    def test_ollama_and_flux_never_coexist(self, mock_preload, mock_signal, mock_wait, mock_vram):
        with patch.object(self.orch, '_flush_flux_if_loaded'):
            self.orch.phase_llm()
        assert not ('ollama' in self.orch.loaded_models and 'flux_dev_8bit' in self.orch.loaded_models)

        self.orch.phase_image_generation()
        assert not ('ollama' in self.orch.loaded_models and 'flux_dev_8bit' in self.orch.loaded_models)


# ════════════════════════════════════════════════════════════════
# 5. CAN_LOAD_MODEL & CAN_RUN_INFERENCE — Pre-flight checks
# ════════════════════════════════════════════════════════════════

class TestPreFlightChecks:
    """Test pre-flight VRAM checks for loading and inference."""

    def setup_method(self):
        self.orch = ModelOrchestrator()

    def test_can_load_ollama_on_empty_gpu(self):
        assert self.orch.can_load_model('ollama') is True

    def test_can_load_flux_on_empty_gpu(self):
        assert self.orch.can_load_model('flux_dev_8bit') is True

    def test_cannot_load_flux_with_ollama_loaded(self):
        self.orch._register_model('ollama')
        assert not self.orch.can_load_model('flux_dev_8bit')

    def test_cannot_load_ollama_with_flux_loaded(self):
        self.orch._register_model('flux_dev_8bit')
        assert not self.orch.can_load_model('ollama')

    def test_can_load_kokoro_alongside_ollama(self):
        self.orch._register_model('ollama')
        assert self.orch.can_load_model('kokoro')

    def test_can_load_kokoro_alongside_flux(self):
        self.orch._register_model('flux_dev_8bit')
        assert self.orch.can_load_model('kokoro')

    def test_can_run_inference_on_empty_gpu(self):
        self.orch._register_model('flux_dev_8bit')
        assert self.orch.can_run_inference('flux_dev_8bit') is True

    def test_cannot_run_inference_if_model_not_loaded(self):
        # Model not registered — can_run_inference should return False
        assert 'flux_dev_8bit' not in self.orch.loaded_models
        result = self.orch.can_run_inference('flux_dev_8bit')
        assert result is False

    def test_begin_inference_returns_false_if_not_loaded(self):
        assert self.orch.begin_inference('flux_dev_8bit') is False

    def test_begin_inference_tracks_overhead(self):
        self.orch._register_model('flux_dev_8bit')
        assert self.orch.begin_inference('flux_dev_8bit') is True
        assert self.orch._inference_active == 'flux_dev_8bit'
        expected = GPU_BASE_OVERHEAD_GB + 12.5 + 5.5
        assert self.orch.vram_used_gb() == pytest.approx(expected)

    def test_end_inference_releases_overhead(self):
        self.orch._register_model('flux_dev_8bit')
        self.orch.begin_inference('flux_dev_8bit')
        self.orch.end_inference()
        assert self.orch._inference_active is None
        assert self.orch.vram_used_gb() == pytest.approx(GPU_BASE_OVERHEAD_GB + 12.5)

    def test_inference_cycle_weight_to_peak_to_idle(self):
        """Simulates the actual FLUX generation cycle."""
        self.orch._register_model('flux_dev_8bit')
        idle_vram = self.orch.vram_used_gb()  # 1.5 + 12.5 = 14.0

        self.orch.begin_inference('flux_dev_8bit')
        peak_vram = self.orch.vram_used_gb()  # 14.0 + 5.5 = 19.5

        self.orch.end_inference()
        back_to_idle = self.orch.vram_used_gb()  # 14.0

        assert peak_vram > idle_vram
        assert back_to_idle == pytest.approx(idle_vram)
        assert peak_vram <= GPU_TOTAL_GB, f"Peak {peak_vram}GB exceeds {GPU_TOTAL_GB}GB"

    def test_unknown_model_always_fits(self):
        assert self.orch.can_load_model('unknown_model') is True

    def test_can_load_model_with_inference_on_empty_gpu(self):
        assert self.orch.can_load_model_with_inference('flux_dev_8bit') is True

    def test_cannot_load_with_inference_when_ollama_present(self):
        self.orch._register_model('ollama')
        assert not self.orch.can_load_model_with_inference('flux_dev_8bit')

    def test_headroom_calculation_with_inference(self):
        self.orch._register_model('flux_dev_8bit')
        self.orch.begin_inference('flux_dev_8bit')
        expected_avail = GPU_TOTAL_GB - GPU_BASE_OVERHEAD_GB - 12.5 - 5.5
        assert self.orch.vram_available_gb() == pytest.approx(expected_avail)


# ════════════════════════════════════════════════════════════════
# 6. STATUS REPORTING
# ════════════════════════════════════════════════════════════════

class TestStatusReporting:
    def test_status_includes_inference_active(self):
        orch = ModelOrchestrator()
        status = orch.status()
        assert 'inference_active' in status
        assert status['inference_active'] is None

    def test_status_shows_during_inference(self):
        orch = ModelOrchestrator()
        orch._register_model('flux_dev_8bit')
        orch.begin_inference('flux_dev_8bit')
        status = orch.status()
        assert status['inference_active'] == 'flux_dev_8bit'
        assert status['vram_used_gb'] == pytest.approx(GPU_BASE_OVERHEAD_GB + 12.5 + 5.5)


# ════════════════════════════════════════════════════════════════
# 7. EDGE CASES
# ════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_double_register_no_double_count(self):
        orch = ModelOrchestrator()
        orch._register_model('ollama')
        first = orch.vram_used_gb()
        orch._register_model('ollama')
        assert orch.vram_used_gb() == first

    def test_unregister_nonexistent_safe(self):
        orch = ModelOrchestrator()
        orch._unregister_model('nonexistent')
        assert orch.vram_used_gb() == pytest.approx(GPU_BASE_OVERHEAD_GB)

    def test_cleanup_after_partial_pipeline(self):
        orch = ModelOrchestrator()
        orch._register_model('ollama')
        orch._register_model('kokoro')
        with patch.object(orch, '_evict_ollama'), \
             patch.object(orch, '_flush_flux_if_loaded'):
            orch.phase_cleanup()
        assert orch.loaded_models == {}
        assert orch._inference_active is None

    def test_clear_all_resets_inference(self):
        orch = ModelOrchestrator()
        orch._register_model('flux_dev_8bit')
        orch.begin_inference('flux_dev_8bit')
        orch._clear_all_models()
        assert orch._inference_active is None
        assert orch.loaded_models == {}

    def test_inference_without_prior_load_fails(self):
        orch = ModelOrchestrator()
        assert orch.begin_inference('flux_dev_8bit') is False

    def test_end_inference_when_none_active(self):
        orch = ModelOrchestrator()
        orch.end_inference()
        assert orch._inference_active is None


# ════════════════════════════════════════════════════════════════
# 8. PRE-PIPELINE GPU SWEEP — VRAM guard + memory fraction cap
# ════════════════════════════════════════════════════════════════

class TestPrePipeline:
    """Test phase_pre_pipeline() — the hard guard that prevents OOM crashes."""

    def setup_method(self):
        self.orch = ModelOrchestrator()

    @patch('src.video.model_orchestrator.ModelOrchestrator._set_memory_fraction', return_value=True)
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    def test_pre_pipeline_succeeds_with_enough_vram(self, mock_vram, mock_evict, mock_frac):
        result = self.orch.phase_pre_pipeline()
        assert result is True
        assert self.orch.current_phase == 'ready'

    @patch('src.video.model_orchestrator.ModelOrchestrator._set_memory_fraction', return_value=True)
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=5.0)
    def test_pre_pipeline_fails_with_insufficient_vram(self, mock_vram, mock_evict, mock_frac):
        # Override timeout to 3s so test doesn't hang
        result = self.orch.phase_pre_pipeline(timeout=3)
        assert result is False

    @patch('src.video.model_orchestrator.ModelOrchestrator._set_memory_fraction', return_value=True)
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram')
    def test_pre_pipeline_waits_then_succeeds(self, mock_vram, mock_evict, mock_frac):
        mock_vram.side_effect = [5.0, 5.0, 5.0, 22.0]
        result = self.orch.phase_pre_pipeline(timeout=15)
        assert result is True
        assert self.orch.current_phase == 'ready'

    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    def test_pre_pipeline_does_NOT_set_memory_fraction(self, mock_vram, mock_evict):
        """Memory fraction cap is deferred to phase_image_generation, not set in pre_pipeline."""
        with patch.object(self.orch, '_set_memory_fraction') as mock_frac:
            mock_frac.return_value = True
            self.orch.phase_pre_pipeline()
            mock_frac.assert_not_called()

    @patch('src.video.model_orchestrator.ModelOrchestrator._set_memory_fraction', return_value=True)
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    def test_pre_pipeline_succeeds_when_vram_sufficient(self, mock_vram, mock_evict, mock_frac):
        result = self.orch.phase_pre_pipeline()
        assert result is True

    @patch('src.video.model_orchestrator.ModelOrchestrator._set_memory_fraction', return_value=True)
    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    def test_pre_pipeline_calls_evict_ollama(self, mock_vram, mock_frac):
        with patch.object(self.orch, '_evict_ollama') as mock_evict:
            self.orch.phase_pre_pipeline()
            mock_evict.assert_called_once()

    def test_set_memory_fraction_can_be_reset(self):
        """Memory fraction can be set, reset, and set again (lifecycle: image_gen only)."""
        orch = ModelOrchestrator()
        # Initially not set
        assert orch._memory_fraction_set is False
        # Set it
        with patch('torch.cuda.set_per_process_memory_fraction'):
            with patch('torch.cuda.is_available', return_value=True):
                orch._set_memory_fraction()
        assert orch._memory_fraction_set is True
        # Reset it
        with patch('torch.cuda.set_per_process_memory_fraction'):
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.empty_cache'):
                    with patch('gc.collect'):
                        orch._reset_memory_fraction()
        assert orch._memory_fraction_set is False
        # Can set again
        with patch('torch.cuda.set_per_process_memory_fraction'):
            with patch('torch.cuda.is_available', return_value=True):
                orch._set_memory_fraction()
        assert orch._memory_fraction_set is True

    def test_pre_pipeline_status_includes_memory_cap(self):
        orch = ModelOrchestrator()
        status = orch.status()
        assert 'memory_fraction_set' in status
        assert 'memory_fraction_cap_gb' in status


# ════════════════════════════════════════════════════════════════
# 9. MEMORY FRACTION CAP — Hard CUDA limit
# ════════════════════════════════════════════════════════════════

class TestMemoryFractionCap:
    """Test that memory fraction is properly calculated."""

    def test_memory_fraction_cap_on_24gb(self):
        orch = ModelOrchestrator()
        cap = 24.0 * 0.85
        assert cap == pytest.approx(20.4)

    def test_memory_fraction_cap_custom_gpu(self):
        orch = ModelOrchestrator(gpu_total_gb=16.0, memory_fraction=0.85)
        assert orch._gpu_total_gb == 16.0
        assert orch._memory_fraction == 0.85
        cap = 16.0 * 0.85
        assert cap == pytest.approx(13.6)

    def test_validate_vram_budget_includes_memory_cap(self):
        result = ModelOrchestrator.validate_vram_budget(total_gpu_gb=24.0)
        assert 'memory_fraction_cap_gb' in result
        assert result['memory_fraction_cap_gb'] == pytest.approx(20.4)

    def test_peak_flux_within_memory_fraction_cap(self):
        est = ModelOrchestrator.VRAM_ESTIMATES
        inf = ModelOrchestrator.INFERENCE_OVERHEAD
        peak_flux = GPU_BASE_OVERHEAD_GB + est['flux_dev_8bit'] + inf['flux_dev_8bit']
        cap = 24.0 * 0.85  # 20.4 GB
        assert peak_flux <= cap, f"FLUX peak {peak_flux}GB exceeds memory fraction cap {cap}GB"

    def test_peak_ollama_within_memory_fraction_cap(self):
        est = ModelOrchestrator.VRAM_ESTIMATES
        inf = ModelOrchestrator.INFERENCE_OVERHEAD
        peak_ollama = GPU_BASE_OVERHEAD_GB + est['ollama'] + inf['ollama']
        cap = 24.0 * 0.85  # 20.4 GB
        assert peak_ollama <= cap, f"Ollama peak {peak_ollama}GB exceeds memory fraction cap {cap}GB"

    def test_pytorch_env_var_set(self):
        import os
        assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'max_split_size_mb:128'


# ════════════════════════════════════════════════════════════════
# 10. MEMORY FRACTION LIFECYCLE — set in image_gen, reset after
# ════════════════════════════════════════════════════════════════

class TestMemoryFractionLifecycle:
    """Test that the CUDA memory fraction cap is set ONLY during image generation.

    The cap must NOT be set during the LLM phase (Ollama needs full GPU).
    The cap must be set during image generation (FLUX needs protection from OOM).
    The cap must be released after image generation (Kokoro needs full GPU).
    """

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    def test_pre_pipeline_does_NOT_set_memory_fraction(self, mock_evict, mock_vram):
        """phase_pre_pipeline must NOT set the memory fraction cap."""
        orch = ModelOrchestrator()
        with patch.object(orch, '_set_memory_fraction') as mock_set:
            mock_set.return_value = True
            orch.phase_pre_pipeline()
            mock_set.assert_not_called()
        assert orch._memory_fraction_set is False

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._set_memory_fraction')
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram', return_value=True)
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline', return_value=True)
    def test_phase_image_generation_sets_memory_fraction(self, mock_preload, mock_keepalive, mock_wait, mock_set, mock_evict, mock_vram):
        """phase_image_generation must set the memory fraction cap."""
        mock_set.return_value = True
        orch = ModelOrchestrator()
        result = orch.phase_image_generation()
        mock_set.assert_called_once()

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram', return_value=True)
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline', return_value=True)
    def test_image_generation_done_resets_memory_fraction(self, mock_preload, mock_keepalive, mock_wait, mock_evict, mock_vram):
        """phase_image_generation_done must reset the memory fraction cap to 100%."""
        orch = ModelOrchestrator()
        orch._memory_fraction_set = True  # Simulate cap being set during image_gen
        with patch.object(orch, '_reset_memory_fraction') as mock_reset:
            orch.phase_image_generation_done()
            mock_reset.assert_called_once()

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram', return_value=True)
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline', return_value=True)
    def test_phase_cleanup_resets_memory_fraction(self, mock_preload, mock_keepalive, mock_wait, mock_evict, mock_vram):
        """phase_cleanup must reset the memory fraction cap."""
        orch = ModelOrchestrator()
        orch._memory_fraction_set = True
        with patch.object(orch, '_reset_memory_fraction') as mock_reset:
            orch.phase_cleanup()
            mock_reset.assert_called_once()

    def test_reset_memory_fraction_clears_flag(self):
        """_reset_memory_fraction must set _memory_fraction_set to False."""
        orch = ModelOrchestrator()
        orch._memory_fraction_set = True
        with patch('torch.cuda.set_per_process_memory_fraction') as mock_set:
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.empty_cache'):
                    with patch('gc.collect'):
                        orch._reset_memory_fraction()
        assert orch._memory_fraction_set is False

    def test_reset_memory_fraction_noop_if_not_set(self):
        """_reset_memory_fraction should be a no-op if no cap was set."""
        orch = ModelOrchestrator()
        orch._memory_fraction_set = False
        with patch('torch.cuda.set_per_process_memory_fraction') as mock_set:
            with patch('torch.cuda.is_available', return_value=True):
                orch._reset_memory_fraction()
        mock_set.assert_not_called()
        assert orch._memory_fraction_set is False

    def test_full_lifecycle_no_cap_during_llm(self):
        """Memory fraction must NOT be set during the LLM phase."""
        orch = ModelOrchestrator()
        assert orch._memory_fraction_set is False

        # Pre-pipeline (should NOT set cap)
        with patch.object(orch, '_evict_ollama'):
            with patch.object(orch, '_get_free_vram', return_value=22.0):
                with patch.object(orch, '_set_memory_fraction') as mock_set:
                    mock_set.return_value = True
                    orch.phase_pre_pipeline()
                    mock_set.assert_not_called()

        # LLM phase (should NOT set cap)
        with patch.object(orch, '_flush_flux_if_loaded'):
            orch.phase_llm()
        assert orch._memory_fraction_set is False

    @patch('src.video.model_orchestrator.ModelOrchestrator._get_free_vram', return_value=22.0)
    @patch('src.video.model_orchestrator.ModelOrchestrator._set_memory_fraction')
    @patch('src.video.model_orchestrator.ModelOrchestrator._evict_ollama')
    @patch('src.video.model_orchestrator.ModelOrchestrator._wait_for_vram', return_value=True)
    @patch('src.video.pixel_art_tool.signal_flux_keep_alive')
    @patch('src.video.pixel_art_tool.preload_flux_pipeline', return_value=True)
    def test_set_called_in_image_gen_reset_in_done(self, mock_preload, mock_keepalive, mock_wait, mock_evict, mock_set, mock_vram):
        """Cap is set at image_gen start and released at image_gen_done."""
        mock_set.return_value = True
        orch = ModelOrchestrator()

        # image_gen sets the cap
        orch.phase_image_generation()
        mock_set.assert_called_once()

        # image_gen_done releases the cap
        with patch.object(orch, '_reset_memory_fraction') as mock_reset:
            orch.phase_image_generation_done()
            mock_reset.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])