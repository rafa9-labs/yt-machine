#!/bin/bash
# Geopolitical Sentinel — Daily automation script
# Run via Windows Task Scheduler or manually in WSL
# Usage: bash run_daily.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "[$(date)] Starting daily pipeline..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the full automation: generate + publish
python src/automate.py --publish youtube,tiktok

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Pipeline completed successfully."
else
    echo "[$(date)] Pipeline failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE
