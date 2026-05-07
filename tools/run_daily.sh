#!/bin/bash
# Geopolitical Sentinel — Daily automation script
# Run via Windows Task Scheduler (wsl -e bash /home/USER/yt-machine/tools/run_daily.sh)
# Or manually: bash tools/run_daily.sh

set -e

PROJECT_DIR="$HOME/yt-machine"
cd "$PROJECT_DIR"

echo "[$(date)] Starting daily pipeline..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the full automation: generate + publish to YouTube + TikTok
python src/automate.py --publish youtube,tiktok

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Pipeline completed successfully."
else
    echo "[$(date)] Pipeline failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE