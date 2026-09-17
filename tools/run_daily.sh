#!/bin/bash
# Geopolitical Sentinel — Daily automation script (macOS / launchd)
# Run via launchd, cron, or manually: bash tools/run_daily.sh
#
# The launchd agent installed by `python src/automate.py --install-schedule`
# invokes this script. pmset wakes the Mac ~10 minutes earlier.
#
# SAFETY: only one pipeline run may be active at a time. The pipeline takes
# an inter-process lock, and we pre-check it here so a scheduled run backs
# off cleanly instead of fighting a manual run for 20+ GB of unified memory.

set -uo pipefail

# launchd starts children with a minimal PATH. ffmpeg, llama-server,
# mlx_lm.server and ollama all live in /opt/homebrew/bin.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# Platforms to publish to. Drop "tiktok" here until the Content Posting API
# app is approved — an empty TikTok credential set fails that platform only,
# but there is no reason to wait on it during every run.
PUBLISH_PLATFORMS="${PUBLISH_PLATFORMS:-youtube,tiktok}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

echo "$LOG_PREFIX Starting daily pipeline (publish=$PUBLISH_PLATFORMS)"

# Activate virtual environment
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    echo "$LOG_PREFIX No virtualenv found (.venv/venv) — cannot run the pipeline."
    exit 1
fi

# Fail loudly rather than falling back to a system Python that lacks the
# pipeline's dependencies (langchain, moviepy, kokoro, ...).
if ! python -c "import dotenv, requests" 2>/dev/null; then
    echo "$LOG_PREFIX Active python ($(command -v python)) is missing dependencies."
    echo "$LOG_PREFIX Install with: uv pip install --python .venv/bin/python -r requirements-macos.txt"
    exit 1
fi

# Keep the Mac awake for the duration of the run. An idle sleep mid-pipeline
# would suspend a ~80 minute generation and can leave llama-server holding
# unified memory across the sleep boundary. -s prevents system idle sleep;
# it is released automatically when this script exits.
CAFFEINATE_PID=""
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -s -i -w $$ &
    CAFFEINATE_PID=$!
    echo "$LOG_PREFIX caffeinate active (pid $CAFFEINATE_PID)"
fi

cleanup() {
    if [ -n "$CAFFEINATE_PID" ] && kill -0 "$CAFFEINATE_PID" 2>/dev/null; then
        kill "$CAFFEINATE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Back off cleanly if another run is active. The check delegates to
# PipelineLock so a STALE lock (crashed run, stale pid) is treated as free —
# a bare `[ -f "$LOCK" ]` test would wedge the daily job forever.
if ! python -c "
import sys
sys.path.insert(0, '.')
from src.models.runtime import PipelineLock
lock = PipelineLock()
if not lock.path.exists():
    sys.exit(0)
owner = lock._read_owner() or {}
sys.exit(1 if not lock._is_stale(owner) else 0)
" 2>/dev/null; then
    echo "$LOG_PREFIX Another pipeline run is active — skipping this run."
    exit 0
fi

# Run the pipeline: generate video, then publish to the configured platforms.
# Publishing happens inside automate.py so a publish failure does not mask a
# successful generation (the video is already on disk either way).
python src/automate.py --publish "$PUBLISH_PLATFORMS"
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "$LOG_PREFIX Pipeline completed successfully."
else
    echo "$LOG_PREFIX Pipeline failed with exit code $EXIT_CODE."
fi

exit "$EXIT_CODE"
