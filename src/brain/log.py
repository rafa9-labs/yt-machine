"""
═══════════════════════════════════════════════════════════════════════════════
brain/log.py — Phase 8: Structured Logging Configuration (structlog)
═══════════════════════════════════════════════════════════════════════════════

WHY STRUCTLOG? (Educational — read this carefully)
─────────────────────────────────────────────────
Your old pipeline used print() with emoji prefixes:
    print("✅ Async scraper found 47 articles")
    print("❌ Script synthesis failed: timeout")

Problems with print():
    1. No log levels — can't filter INFO from ERROR
    2. No structure — can't grep by project_id or step_name
    3. No timestamps — don't know WHEN something happened
    4. No file output — _TeeWriter hacks sys.stdout (breaks libraries)
    5. No machine parsing — can't feed logs to Elasticsearch, Loki, etc.

structlog gives you ALL of this:

    # Development (human-readable, colored):
    [2026-04-15 14:05:00] INFO  pipeline.step_complete step=news_fetch duration_s=2.3 articles=47

    # Production (JSON — machine-parseable):
    {"event": "pipeline.step_complete", "step": "news_fetch", "duration_s": 2.3,
     "articles": 47, "level": "info", "timestamp": "2026-04-15T14:05:00Z"}

HOW IT WORKS:
    1. configure() sets up processors (like a pipeline for log events)
    2. get_logger("module_name") returns a bound logger
    3. log.info("event_name", key=value) emits structured data
    4. Processors transform the data into console text or JSON

SAFETY:
    If structlog fails to import → falls back to standard logging.
    Your pipeline NEVER crashes because of a logging issue.

USAGE:
    from src.brain.log import get_logger
    log = get_logger("pipeline")
    log.info("step.start", step="news_fetch")
    log.info("step.complete", step="news_fetch", duration_s=2.3, articles=47)
    log.error("step.failed", step="news_fetch", error=str(e))

    # Bind context (all future logs include this):
    log = log.bind(project_id=1776133524, pipeline_version="v2")
    log.info("step.complete", step="script_synthesis")  # includes project_id

ENV TOGGLES:
    LOG_FORMAT=console|json   (default: console)
    LOG_LEVEL=DEBUG|INFO|WARNING|ERROR   (default: INFO)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# ── Try to import structlog; fall back to standard logging if unavailable ──
# WHY TRY/EXCEPT? The safety principle from Phases 1-7: if ANY new dependency
# fails, the pipeline must still work. structlog is nice-to-have, not required.
_TRY_STRUCTLOG = True
try:
    import structlog
    from structlog.stdlib import ProcessorFormatter
    from structlog.dev import ConsoleRenderer
except ImportError:
    _TRY_STRUCTLOG = False


# ══════════════════════════════════════════════════════════════════════════
# PROCESSORS — Functions that transform log events in the pipeline
# ══════════════════════════════════════════════════════════════════════════
# WHY PROCESSORS? structlog uses a processor chain — each function receives
# the log event dict, can modify it, and passes it to the next processor.
# This is more flexible than logging.Formatter because you can:
#   - Add context (project_id, step_name) to every event
#   - Filter events by level or content
#   - Transform output format without changing calling code

def _add_timestamp(logger, method_name, event_dict):
    """Add ISO-8601 timestamp to every log event.
    
    WHY? Without this, you'd have to manually include the time in every
    print() call. structlog does it automatically.
    """
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def _add_log_level(logger, method_name, event_dict):
    """Add log level (info, warning, error, debug) to every event.
    
    WHY? The method_name (e.g., "info") becomes a field you can filter on:
        grep '"level": "error"' pipeline.log
    """
    if method_name == "warn":
        # stdlib uses "warning", structlog uses "warn" — normalize
        event_dict["level"] = "warning"
    else:
        event_dict["level"] = method_name
    return event_dict


def _add_caller_info(logger, method_name, event_dict):
    """Add the module that logged the event.
    
    WHY? When you have 10 modules logging, you need to know WHICH module
    produced a given log line.
    """
    # structlog stdlib already adds this via add_log_level
    record = event_dict.get("_record")
    if record:
        event_dict.setdefault("module", record.name)
    return event_dict


def _filter_redacted(logger, method_name, event_dict):
    """Redact sensitive fields from log output.
    
    WHY? You don't want API keys or tokens leaking into log files.
    Even though you shouldn't log them, this is a safety net.
    """
    sensitive_keys = {"api_key", "token", "password", "secret", "authorization"}
    for key in list(event_dict.keys()):
        if key.lower() in sensitive_keys:
            event_dict[key] = "[REDACTED]"
    return event_dict


# ══════════════════════════════════════════════════════════════════════════
# LOG FILE SETUP — File handler that writes to output/logs/
# ══════════════════════════════════════════════════════════════════════════

def _get_log_file_path() -> Path:
    """Create and return the path for today's log file.
    
    WHY ONE FILE PER DAY (not per run)?
    The old _TeeWriter created a new file per run, which is fine for debugging
    but terrible for production monitoring. One file per day means:
      - Easy to find logs for a specific day
      - logrotate can clean up old files
      - grep can search across all runs that day
    """
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    return log_dir / f"pipeline_{date_str}.log"


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Called once at startup
# ══════════════════════════════════════════════════════════════════════════

_configured = False


def configure():
    """Configure structlog for the pipeline. Call once at startup.
    
    WHAT THIS DOES:
    1. Sets up standard logging to capture LangChain/FastAPI/Uvicorn logs
    2. Configures structlog processors (timestamp, level, caller info)
    3. Chooses renderer: console (dev) or JSON (production)
    4. Adds file handler for persistent log storage
    
    WHY configure() INSTEAD OF module-level setup?
    Module-level code runs on import, before .env is loaded. This function
    is called explicitly after load_dotenv(), so it can read LOG_FORMAT.
    """
    global _configured
    if _configured:
        return
    _configured = True
    
    log_format = os.getenv("LOG_FORMAT", "console").lower()
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    if not _TRY_STRUCTLOG:
        # ── Fallback: standard logging ──
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(str(_get_log_file_path()), encoding='utf-8'),
            ],
        )
        return
    
    # ── Standard logging setup (captures LangChain, FastAPI, Uvicorn, etc.) ──
    # WHY? These libraries use standard logging. structlog needs to intercept
    # their log records via ProcessorFormatter.
    
    # Console handler — human-readable output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # File handler — persistent JSON logs
    file_handler = logging.FileHandler(
        str(_get_log_file_path()), encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    
    # Choose renderer based on LOG_FORMAT env var
    if log_format == "json":
        # JSON renderer — for production, Docker, Elasticsearch, Loki
        # WHY JSON? Machine-parseable. You can pipe to jq:
        #   tail -f output/logs/pipeline_20260415.log | jq '.level == "error"'
        from structlog.processors import JSONRenderer
        console_renderer = ConsoleRenderer()  # Console stays human-readable
        file_renderer = JSONRenderer()        # File gets JSON
    else:
        # Console renderer — for development (colored, aligned)
        # WHY CONSOLE? During development you want readable output, not JSON.
        # structlog's ConsoleRenderer adds colors and alignment.
        console_renderer = ConsoleRenderer()
        file_renderer = ConsoleRenderer()     # File also human-readable in dev
    
    # Configure standard logging handlers with simple formatters
    # WHY simple formatters? structlog 25.x changed ProcessorFormatter's API.
    # Using standard logging.Formatter is compatible across ALL structlog versions.
    # Standard logging (LangChain, FastAPI, Uvicorn) → formatted normally.
    # structlog loggers → processed through structlog's own pipeline.
    
    console_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    console_handler.setFormatter(console_fmt)
    file_handler.setFormatter(file_fmt)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # ── Configure structlog itself ──
    # WHY these processors? They run in order for each log.info()/error() call:
    #   1. add_log_level → attaches "info"/"error"/"warning" to the event dict
    #   2. PositionalArgumentsFormatter → handles positional args (like stdlib)
    #   3. StackInfoRenderer → renders stack traces as strings
    #   4. format_exc_info → converts exc_info tuples to strings
    #   5. ConsoleRenderer/JSONRenderer → final output formatting
    final_renderer = file_renderer if log_format == "json" else console_renderer
    
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            _add_timestamp,
            _filter_redacted,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            final_renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "pipeline"):
    """Get a structured logger instance.
    
    USAGE:
        log = get_logger("pipeline")
        log.info("step.start", step="news_fetch")
        log.info("step.complete", step="news_fetch", duration_s=2.3)
        log.error("step.failed", step="news_fetch", error="connection timeout")
        
        # Bind context (all future logs include this):
        log = log.bind(project_id=1776133524)
        log.info("step.complete", step="script_synthesis")
        # → includes project_id=1776133524 automatically
    
    WHY get_logger() INSTEAD OF structlog.get_logger()?
    This function handles the fallback case where structlog is not installed.
    If structlog fails to import, you get a standard logging.Logger that
    accepts the same keyword arguments (they're just ignored).
    """
    configure()  # Ensure configuration is applied
    
    if _TRY_STRUCTLOG:
        return structlog.get_logger(name)
    else:
        # ── Fallback: return standard logger that accepts **kwargs ──
        # WHY this wrapper? structlog accepts log.info("msg", key=value).
        # Standard logging does NOT accept kwargs — they'd crash.
        # This wrapper silently ignores the kwargs so your code works either way.
        return _StdlibFallback(name)


class _StdlibFallback:
    """Fallback logger that mimics structlog's interface using standard logging.
    
    WHY? If structlog is not installed, code like:
        log.info("step.complete", step="news_fetch", duration_s=2.3)
    would crash with standard logging because it doesn't accept **kwargs.
    
    This wrapper formats kwargs into the message string and delegates to
    the standard logging module. You lose structured output, but the pipeline
    still runs.
    """
    
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._bindings = {}
    
    def bind(self, **kwargs):
        """Create a child logger with additional context."""
        child = _StdlibFallback(self._logger.name)
        child._bindings = {**self._bindings, **kwargs}
        return child
    
    def _format_msg(self, event: str, **kwargs) -> str:
        """Combine event + kwargs into a readable string."""
        all_ctx = {**self._bindings, **kwargs}
        if all_ctx:
            pairs = " ".join(f"{k}={v}" for k, v in all_ctx.items())
            return f"{event}  {pairs}"
        return event
    
    def debug(self, event: str, **kwargs):
        self._logger.debug(self._format_msg(event, **kwargs))
    
    def info(self, event: str, **kwargs):
        self._logger.info(self._format_msg(event, **kwargs))
    
    def warning(self, event: str, **kwargs):
        self._logger.warning(self._format_msg(event, **kwargs))
    
    def error(self, event: str, **kwargs):
        self._logger.error(self._format_msg(event, **kwargs))
    
    def critical(self, event: str, **kwargs):
        self._logger.critical(self._format_msg(event, **kwargs))
