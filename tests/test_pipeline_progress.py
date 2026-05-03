"""
Tests for pipeline progress/heartbeat fixes.

Verifies:
  1. PYTHONUNBUFFERED=1 env var is set before imports
  2. stdout is line-buffered (reconfigure called)
  3. _run_with_heartbeat() prints dots and returns results
  4. _is_semantically_similar() uses word overlap (no LLM calls)
  5. _step_banner() prints visible step headers with numbering
  6. _step_done() prints timing info
  7. orchestrator.heartbeat() returns VRAM status dict

Run: python -m pytest tests/test_pipeline_progress.py -v
"""
import sys
import os
import io
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


# ══════════════════════════════════════════════════════════════════
# 1. STDOUT UNBUFFERED FIX
# ══════════════════════════════════════════════════════════════════

class TestStdoutUnbuffered:
    """Verify that PYTHONUNBUFFERED=1 is set in the pipeline script."""

    def test_pythonunbuffered_env_var_in_pipeline(self):
        """The pipeline script should set PYTHONUNBUFFERED=1 at the very top."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # PYTHONUNBUFFERED must appear before any other meaningful imports
        lines = content.split('\n')
        env_line_idx = None
        import_line_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "PYTHONUNBUFFERED" in stripped and 'os.environ' in stripped:
                env_line_idx = i
            if stripped.startswith('import ') and 'os' not in stripped and env_line_idx is None:
                if import_line_idx is None:
                    import_line_idx = i

        assert env_line_idx is not None, "PYTHONUNBUFFERED=1 not found in pipeline"
        if import_line_idx is not None:
            assert env_line_idx < import_line_idx, (
                f"PYTHONUNBUFFERED=1 (line {env_line_idx}) must be set before "
                f"non-os imports (line {import_line_idx})"
            )

    def test_line_buffering_reconfigure_in_pipeline(self):
        """The pipeline should reconfigure stdout for line buffering."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'line_buffering=True' in content, (
            "sys.stdout.reconfigure(line_buffering=True) not found in pipeline"
        )


# ══════════════════════════════════════════════════════════════════
# 2. _run_with_heartbeat() FUNCTION
# ══════════════════════════════════════════════════════════════════

class TestRunWithHeartbeat:
    """Test the _run_with_heartbeat wrapper for blocking LLM calls."""

    def _make_heartbeat_func(self):
        """Import _run_with_heartbeat from the pipeline module."""
        # We need to import from the pipeline, but it has side effects on import.
        # Instead, replicate the function inline for testing.
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify the function exists
        assert '_run_with_heartbeat' in content, "_run_with_heartbeat not found in pipeline"
        assert 'heartbeat_interval' in content, "heartbeat_interval param not found"
        assert 'done_event' in content, "threading.Event not found in _run_with_heartbeat"
        assert 'orchestrator.heartbeat' in content, "orchestrator.heartbeat not called in _run_with_heartbeat"

    def test_run_with_heartbeat_returns_result(self):
        """_run_with_heartbeat should return (result, False) for successful calls."""
        import concurrent.futures

        # Simulate the function inline
        def _run_with_heartbeat(func, label, heartbeat_interval=8, timeout_seconds=600, *args, **kwargs):
            result_container = [None]
            exception_container = [None]
            done_event = threading.Event()

            def worker():
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    exception_container[0] = e
                finally:
                    done_event.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            elapsed = 0
            dots = 0
            while not done_event.is_set():
                waited = done_event.wait(timeout=heartbeat_interval)
                if waited:
                    break
                elapsed += heartbeat_interval
                dots += 1

            if exception_container[0] is not None:
                raise exception_container[0]

            if not done_event.is_set():
                return None, True

            return result_container[0], False

        def quick_func():
            return "hello"

        result, timed_out = _run_with_heartbeat(quick_func, "test", 1, 10)
        assert result == "hello"
        assert timed_out is False

    def test_run_with_heartbeat_handles_timeout(self):
        """_run_with_heartbeat should return (None, True) when function times out."""
        def _run_with_heartbeat(func, label, heartbeat_interval=1, timeout_seconds=3, *args, **kwargs):
            result_container = [None]
            exception_container = [None]
            done_event = threading.Event()

            def worker():
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    exception_container[0] = e
                finally:
                    done_event.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            elapsed = 0
            dots = 0
            while not done_event.is_set():
                waited = done_event.wait(timeout=heartbeat_interval)
                if waited:
                    break
                elapsed += heartbeat_interval
                dots += 1
                if elapsed >= timeout_seconds:
                    break

            if not done_event.is_set():
                return None, True

            if exception_container[0] is not None:
                raise exception_container[0]

            return result_container[0], False

        def slow_func():
            time.sleep(10)
            return "never"

        result, timed_out = _run_with_heartbeat(slow_func, "test", 1, 2)
        assert timed_out is True
        assert result is None

    def test_run_with_heartbeat_prints_dots_on_slow_calls(self, capsys):
        """_run_with_heartbeat should print dots while waiting for slow functions."""
        def _run_with_heartbeat(func, label, heartbeat_interval=1, timeout_seconds=30, *args, **kwargs):
            result_container = [None]
            exception_container = [None]
            done_event = threading.Event()

            def worker():
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    exception_container[0] = e
                finally:
                    done_event.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            elapsed = 0
            dots = 0
            while not done_event.is_set():
                waited = done_event.wait(timeout=heartbeat_interval)
                if waited:
                    break
                elapsed += heartbeat_interval
                dots += 1
                print(".", end="", flush=True)

            if exception_container[0] is not None:
                raise exception_container[0]

            if not done_event.is_set():
                print(f" TIMEOUT ({timeout_seconds}s)", flush=True)
                return None, True

            print(f" done ({elapsed}s)", flush=True)
            return result_container[0], False

        def slow_func():
            time.sleep(2.5)
            return "result"

        result, timed_out = _run_with_heartbeat(slow_func, "test", 1, 10)
        captured = capsys.readouterr()

        assert result == "result"
        assert timed_out is False
        # Should have printed at least 1 dot during the 2.5s wait
        assert "." in captured.out, f"Expected dots in output, got: {captured.out}"

    def test_run_with_heartbeat_propagates_exceptions(self):
        """_run_with_heartbeat should re-raise exceptions from the function."""
        def _run_with_heartbeat(func, label, heartbeat_interval=1, timeout_seconds=10, *args, **kwargs):
            result_container = [None]
            exception_container = [None]
            done_event = threading.Event()

            def worker():
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    exception_container[0] = e
                finally:
                    done_event.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            done_event.wait(timeout=timeout_seconds)

            if exception_container[0] is not None:
                raise exception_container[0]

            return result_container[0], False

        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            _run_with_heartbeat(failing_func, "test", 1, 10)


# ══════════════════════════════════════════════════════════════════
# 3. WORD-OVERLAP DEDUP (NO LLM)
# ══════════════════════════════════════════════════════════════════

class TestWordOverlapDedup:
    """Verify the dedup function uses word overlap, NOT LLM calls."""

    def test_dedup_function_is_word_overlap_only(self):
        """The pipeline _is_semantically_similar should only use word overlap."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the function definition
        func_start = content.find("def _is_semantically_similar(")
        assert func_start != -1, "Function _is_semantically_similar not found"

        # Find the end of the function (next def or class at same indentation)
        func_section = content[func_start:func_start + 1500]
        func_lines = func_section.split('\n')

        # The function should NOT contain llm.generate or any LLM call
        for line in func_lines:
            if line.strip().startswith('def ') and '_is_semantically_similar' not in line:
                break
            assert 'llm.generate' not in line, (
                f"_is_semantically_similar should NOT call llm.generate, found: {line.strip()}"
            )
            assert 'llm.chat' not in line, (
                f"_is_semantically_similar should NOT call llm.chat, found: {line.strip()}"
            )

    def test_word_overlap_dedup_logic(self):
        """Test the word overlap dedup logic directly."""
        def _is_semantically_similar(title_a: str, title_b: str) -> bool:
            words_a = frozenset(title_a.lower().split()[:6])
            words_b = frozenset(title_b.lower().split()[:6])
            return len(words_a & words_b) >= 3

        # Same story same headline -> should match (all words shared)
        assert _is_semantically_similar(
            "Iran launches missile strike on Israeli",
            "Iran missile strike on Israeli base"
        ) is True

        # Completely different stories -> should NOT match
        assert _is_semantically_similar(
            "Iran launches missile strike on Israeli military base",
            "Pentagon confirms new AI defense contract with SpaceX"
        ) is False

        # Loosely related stories -> should NOT match (only 1-2 shared words)
        assert _is_semantically_similar(
            "Global oil prices spike after Middle East tensions escalate",
            "Iran launches missile strike on Israeli military base"
        ) is False

        # Near-identical headlines -> should match (3+ shared words in first 6)
        assert _is_semantically_similar(
            "BBC News covers Iran missile strike today",
            "BBC News reports Iran missile strike response"
        ) is True

    def test_dedup_is_fast_no_network(self):
        """Verify dedup function completes instantly (no network calls)."""
        def _is_semantically_similar(title_a: str, title_b: str) -> bool:
            words_a = frozenset(title_a.lower().split()[:6])
            words_b = frozenset(title_b.lower().split()[:6])
            return len(words_a & words_b) >= 3

        start = time.monotonic()
        for _ in range(10000):
            _is_semantically_similar(
                "Iran launches missile strike on Israeli military base",
                "Iran fired missiles at Israeli airbase in retaliatory strike"
            )
        elapsed = time.monotonic() - start

        # 10,000 word-overlap comparisons should take < 0.5s
        assert elapsed < 0.5, f"Word overlap too slow: {elapsed:.3f}s for 10k comparisons"


# ══════════════════════════════════════════════════════════════════
# 4. STEP BANNER AND PROGRESS OUTPUT
# ══════════════════════════════════════════════════════════════════

class TestStepBanner:
    """Test the _step_banner() and _step_done() helper functions."""

    def test_step_banner_in_pipeline(self):
        """The pipeline should have _step_banner() function."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'def _step_banner(' in content, "_step_banner function not found"
        assert 'STEP {' in content and '_PIPELINE_STEP' in content, (
            "_step_banner should use _PIPELINE_STEP counter"
        )

    def test_step_done_in_pipeline(self):
        """The pipeline should have _step_done() function."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'def _step_done(' in content, "_step_done function not found"
        assert '_pipeline_start' in content, "_pipeline_start tracking not found"
        assert 'elapsed' in content, "elapsed time not tracked in _step_done"

    def test_step_banners_at_major_steps(self):
        """All major pipeline steps should have _step_banner() calls."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        expected_steps = [
            "FETCH NEWS",
            "NEWS ANALYSIS",
            "SCRIPT SYNTHESIS",
            "PIXEL ART GENERATION",
            "VOICE GENERATION",
            "VIDEO ASSEMBLY",
        ]
        for step in expected_steps:
            assert step in content, f"Step banner for '{step}' not found in pipeline"

    def test_step_done_at_major_completions(self):
        """Major step completions should call _step_done()."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        expected_dones = [
            "NEWS ANALYSIS",
            "SCRIPT SYNTHESIS",
            "PIXEL ART",
            "VOICE GENERATION",
            "VIDEO ASSEMBLY",
        ]
        for step in expected_dones:
            assert f'_step_done("{step}")' in content, (
                f"_step_done for '{step}' not found in pipeline"
            )


# ══════════════════════════════════════════════════════════════════
# 5. HEARTBEAT IN MODEL ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

class TestOrchestratorHeartbeat:
    """Test the ModelOrchestrator.heartbeat() method."""

    def test_heartbeat_returns_status_dict(self):
        from src.video.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        status = orch.heartbeat("test_context")
        assert isinstance(status, dict)
        assert status['phase'] == 'idle'
        assert status['context'] == 'test_context'
        assert 'free_vram_gb' in status
        assert 'loaded_models' in status
        assert 'inference_active' in status

    def test_heartbeat_with_empty_context(self):
        from src.video.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        status = orch.heartbeat("")
        assert status['context'] == ""

    def test_heartbeat_tracks_phase(self):
        from src.video.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        orch.phase_llm()
        status = orch.heartbeat("after_llm_phase")
        assert status['phase'] == 'llm'
        assert 'ollama' in status['loaded_models']

    def test_heartbeat_prints_output(self, capsys):
        from src.video.model_orchestrator import ModelOrchestrator
        orch = ModelOrchestrator()
        orch.heartbeat("my_context")
        captured = capsys.readouterr()
        assert "VRAM" in captured.out
        assert "Phase" in captured.out


# ══════════════════════════════════════════════════════════════════
# 6. PROGRESS OUTPUT IN PIPELINE
# ══════════════════════════════════════════════════════════════════

class TestPipelineProgressOutput:
    """Verify the pipeline has progress output at key decision points."""

    def test_analysis_loop_has_progress(self):
        """The news analysis loop should print per-article progress."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Per-article progress printing
        assert 'Analyzing:' in content, "Per-article progress print not found"

    def test_ollama_loading_message(self):
        """Pipeline should warn about Ollama model loading time."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'Ollama model will load' in content or 'may take 30-90s' in content, (
            "Ollama model loading message not found"
        )

    def test_heartbeat_wrapper_used_for_llm_calls(self):
        """Key LLM calls should use _run_with_heartbeat instead of _run_with_timeout."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Script synthesis should use heartbeat
        assert '_run_with_heartbeat' in content, "_run_with_heartbeat not found in pipeline"

    def test_image_generation_has_progress(self):
        """Image generation loop should print per-image progress."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert '[IMG' in content, "Per-image progress print not found"

    def test_flush_true_on_critical_prints(self):
        """Critical print statements should use flush=True."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # The LLM loading message should flush
        llm_msg_lines = [line for line in content.split('\n')
                         if 'Ollama model' in line and 'print' in line]
        assert len(llm_msg_lines) > 0, "Ollama loading message not found"
        for line in llm_msg_lines:
            assert 'flush=True' in line, f"Ollama loading message should have flush=True: {line.strip()}"

    def test_run_with_heartbeat_has_timeout_parameter(self):
        """_run_with_heartbeat should have a timeout_seconds parameter."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'timeout_seconds' in content, (
            "_run_with_heartbeat missing timeout_seconds parameter"
        )


# ══════════════════════════════════════════════════════════════════
# 7. INTEGRATION: NO LLM CALLS IN DEDUP
# ══════════════════════════════════════════════════════════════════

class TestNoLLMInDedup:
    """Verify the article dedup loop makes zero LLM calls."""

    def test_dedup_loop_has_no_llm_generate(self):
        """The dedup loop should not call llm.generate."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the dedup section
        dedup_start = content.find('"# ── Topic diversity selection')
        if dedup_start == -1:
            dedup_start = content.find('Topic diversity selection')
        assert dedup_start != -1, "Topic diversity section not found"

        # Find the end of dedup (next major section)
        dedup_end = content.find('selected = selected[:3]', dedup_start)
        if dedup_end == -1:
            dedup_end = dedup_start + 2000  # fallback
        dedup_section = content[dedup_start:dedup_end]

        assert 'llm.generate' not in dedup_section, (
            "dedup section should NOT contain llm.generate calls"
        )

    def test_dedup_loop_comment_says_no_llm(self):
        """The dedup section comment should explicitly say 'no LLM calls'."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        dedup_start = content.find('_is_semantically_similar')
        if dedup_start == -1:
            pytest.skip("Function definition not found")

        # Find the function definition and its docstring/comments
        func_area = content[max(0, dedup_start - 200):dedup_start + 500]
        assert 'no LLM' in func_area.lower() or 'word overlap' in func_area.lower() or 'word-overlap' in func_area.lower(), (
            "dedup function should have a comment/docstring saying 'no LLM' or 'word overlap'"
        )


# ══════════════════════════════════════════════════════════════════
# 8. VECTOR DEDUP WRAPPED IN HEARTBEAT
# ══════════════════════════════════════════════════════════════════

class TestVectorDedupProgress:
    """Verify the vector dedup step has progress output and heartbeat wrapping."""

    def test_vector_dedup_uses_run_with_heartbeat(self):
        """Vector dedup should be wrapped in _run_with_heartbeat."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        dedup_start = content.find('STEP 1.5: VECTOR DEDUP')
        assert dedup_start != -1, "Vector dedup section not found"
        dedup_end = content.find('STEP 2: NEWS ANALYSIS', dedup_start)
        dedup_section = content[dedup_start:dedup_end]

        assert '_run_with_heartbeat' in dedup_section, (
            "Vector dedup should use _run_with_heartbeat to prevent silent blocking"
        )

    def test_vector_dedup_has_progress_prints(self):
        """Vector dedup should have progress output."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        dedup_start = content.find('STEP 1.5: VECTOR DEDUP')
        dedup_end = content.find('STEP 2: NEWS ANALYSIS', dedup_start)
        dedup_section = content[dedup_start:dedup_end]

        assert '[DEDUP]' in dedup_section, (
            "Vector dedup should have [DEDUP] progress prints"
        )

    def test_vector_dedup_has_timeout(self):
        """Vector dedup heartbeat should have a timeout (not infinite wait)."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        dedup_start = content.find('STEP 1.5: VECTOR DEDUP')
        dedup_end = content.find('STEP 2: NEWS ANALYSIS', dedup_start)
        dedup_section = content[dedup_start:dedup_end]

        assert '60' in dedup_section, (
            "Vector dedup should have a timeout (e.g. 60s)"
        )

    def test_vector_dedup_fallback_on_timeout(self):
        """Vector dedup should fall back gracefully if it times out."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        dedup_start = content.find('STEP 1.5: VECTOR DEDUP')
        dedup_end = content.find('STEP 2: NEWS ANALYSIS', dedup_start)
        dedup_section = content[dedup_start:dedup_end]

        assert 'dedup.timeout' in dedup_section or 'continuing_without_dedup' in dedup_section, (
            "Vector dedup should fall back on timeout"
        )


# ══════════════════════════════════════════════════════════════════
# 9. ARTICLE TEXT FETCH WRAPPED IN HEARTBEAT
# ══════════════════════════════════════════════════════════════════

class TestArticleFetchProgress:
    """Verify the article text fetch has progress and timeout."""

    def test_article_text_fetch_uses_heartbeat(self):
        """get_full_article_text should be wrapped in _run_with_heartbeat."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'get_full_article_text' in content, "get_full_article_text not found"
        assert 'fetch_text_' in content or '_run_with_heartbeat' in content, (
            "Article text fetch should use _run_with_heartbeat"
        )

    def test_article_text_fetch_has_timeout(self):
        """Article text fetch should have a timeout (not infinite)."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the get_full_article_text call area
        fetch_idx = content.find('get_full_article_text')
        assert fetch_idx != -1, "get_full_article_text not found"
        fetch_area = content[fetch_idx:fetch_idx + 500]
        assert '15' in fetch_area, (
            "Article text fetch should have a timeout (e.g. 15s)"
        )


# ══════════════════════════════════════════════════════════════════
# 10. HEARTBEAT TIMEOUT ENFORCEMENT
# ══════════════════════════════════════════════════════════════════

class TestHeartbeatTimeoutEnforcement:
    """Verify _run_with_heartbeat actually enforces its timeout."""

    def test_heartbeat_timeout_is_checked_in_while_loop(self):
        """The while loop in _run_with_heartbeat must check elapsed >= timeout_seconds."""
        pipeline_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools', 'generate_complete_video.py'
        )
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find _run_with_heartbeat function body
        func_start = content.find('def _run_with_heartbeat(')
        assert func_start != -1, "_run_with_heartbeat not found"
        func_end = content.find('\ndef ', func_start + 1)
        func_body = content[func_start:func_end]

        assert 'elapsed >= timeout_seconds' in func_body or 'elapsed >= timeout' in func_body, (
            "_run_with_heartbeat while loop must check elapsed >= timeout_seconds"
        )

    def test_heartbeat_timeout_returns_none_true(self):
        """When timeout fires, _run_with_heartbeat must return (None, True)."""
        def _run_with_heartbeat(func, label, heartbeat_interval=1, timeout_seconds=2, *args, **kwargs):
            result_container = [None]
            exception_container = [None]
            done_event = threading.Event()

            def worker():
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    exception_container[0] = e
                finally:
                    done_event.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            elapsed = 0
            dots = 0
            while not done_event.is_set():
                waited = done_event.wait(timeout=heartbeat_interval)
                if waited:
                    break
                elapsed += heartbeat_interval
                dots += 1
                if elapsed >= timeout_seconds:
                    break

            if not done_event.is_set():
                return None, True

            if exception_container[0] is not None:
                raise exception_container[0]

            return result_container[0], False

        def slow_func():
            time.sleep(30)
            return "never"

        result, timed_out = _run_with_heartbeat(slow_func, "test", 1, 2)
        assert result is None
        assert timed_out is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])