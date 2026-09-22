"""Guard against stdlib/structlog logging API confusion.

The codebase uses two logging styles and they are not interchangeable:

  structlog  (src/brain/log.py)  log.info("event.name", key=value, ...)
  stdlib     (logging.getLogger) log.info("message %s", value)

Passing structlog-style keyword arguments to a stdlib logger raises
``TypeError: Logger._log() got an unexpected keyword argument`` **at call
time**, not at import. So a broken call is invisible until the exact branch
executes — which is how eleven of them reached a merged PR and stayed hidden
until a stub-driven test happened to hit the over-budget path.

Found this way:
  - 10 calls added in PR #16 on the word-budget enforcement path
  - 1 pre-existing on the curator fallback path (`curation.no_markers`)
  - 1 pre-existing on the assembly path (`assembly.ass_empty`)

The last two were latent crashes: that branch had simply never run.

``extra=`` is the one legitimate keyword — stdlib logging supports it for
structured payloads. Everything else is rejected.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
LOG_METHODS = ("info", "warning", "error", "debug", "critical", "exception")
ALLOWED_KEYWORDS = {"extra"}


def _stdlib_logger_modules():
    """Modules that log through stdlib ``logging``, not structlog."""
    found = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "logging.getLogger" not in text:
            continue
        if "structlog" in text:
            continue
        found.append(path)
    return found


def _offending_calls(path: Path):
    """Return (lineno, method, kwargs) for stdlib calls with bad keywords."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "log":
            continue
        if func.attr not in LOG_METHODS or not node.keywords:
            continue
        bad = {k.arg for k in node.keywords} - ALLOWED_KEYWORDS
        if bad:
            offenders.append((node.lineno, func.attr, sorted(bad)))
    return offenders


class TestStdlibLoggerCalls:
    def test_no_stdlib_logger_receives_structlog_style_kwargs(self):
        """The defect this guards: TypeError at call time on a live branch.

        Eleven such calls existed across ``llm_interface`` and
        ``split_video_assembler``; two of them had never executed, so the bug
        was undetectable by the test suite until a branch was exercised.
        """
        offenders = []
        for path in _stdlib_logger_modules():
            for lineno, method, kwargs in _offending_calls(path):
                offenders.append(
                    f"{path.relative_to(SRC.parent)}:{lineno} "
                    f"log.{method}(...) kwargs={kwargs}"
                )
        assert not offenders, (
            "stdlib logging does not accept structlog-style keyword arguments; "
            "use %-formatting or pass extra={...}:\n  " + "\n  ".join(offenders)
        )

    def test_detector_actually_detects(self, tmp_path):
        """Negative control: the AST check must fail on a known-bad call."""
        bad = tmp_path / "sample.py"
        bad.write_text(
            "import logging\n"
            "log = logging.getLogger(__name__)\n"
            "log.warning('event', words=1)\n",
            encoding="utf-8",
        )
        assert _offending_calls(bad), "detector missed an obviously bad call"

    def test_detector_allows_extra_and_percent_formatting(self, tmp_path):
        """`extra=` is stdlib-correct and must not be flagged."""
        good = tmp_path / "sample.py"
        good.write_text(
            "import logging\n"
            "log = logging.getLogger(__name__)\n"
            "log.info('event: %d', 1)\n"
            "log.warning('event', extra={'k': 1})\n",
            encoding="utf-8",
        )
        assert _offending_calls(good) == []


class TestTheSpecificPathsThatWereBroken:
    """The three paths that carried broken calls must run without raising."""

    def test_word_budget_logs_use_percent_formatting(self):
        source = (SRC / "brain" / "llm_interface.py").read_text(encoding="utf-8")
        # The validate.* calls were the PR #16 regression; they now use %-style
        # or no arguments at all.
        assert "log.info(\"validate." not in source, (
            "word-budget logging reverted to structlog-style event names on a "
            "stdlib logger"
        )

    def test_curation_fallback_log_is_safe(self):
        source = (SRC / "brain" / "llm_interface.py").read_text(encoding="utf-8")
        assert "log.warning(\"curation.no_markers\"" not in source, (
            "the latent curator-fallback crash is back"
        )

    def test_assembly_empty_ass_log_is_safe(self):
        source = (SRC / "video" / "split_video_assembler.py").read_text(
            encoding="utf-8"
        )
        assert 'log.warning("assembly.ass_empty", reason=' not in source, (
            "the latent assembly crash is back"
        )
