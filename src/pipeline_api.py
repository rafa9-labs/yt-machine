"""
Geopolitical Sentinel — Pipeline API Server
Thin Flask API wrapper around generate_complete_video.py and publish_video.py.
Exposes endpoints for n8n/Docker automation.

Usage:
    python pipeline_api.py                # Start on port 8000
    python pipeline_api.py --port 8080    # Custom port

Endpoints:
    POST /generate     → Generate a new video
    POST /publish      → Publish latest video to platforms
    GET  /status       → Check pipeline status
    GET  /latest       → Get latest video info
"""

import os
import sys
import json
import argparse
import threading
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger('pipeline-api')

try:
    from flask import Flask, request, jsonify
except ImportError:
    logger.error("Flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__)

# Track generation state
_generation_status = {
    "status": "idle",
    "last_run": None,
    "last_result": None,
    "current_job": None,
}
_generation_lock = threading.Lock()


@app.route('/status', methods=['GET'])
def status():
    """Check pipeline status."""
    return jsonify({
        "pipeline": "geopolitical-sentinel",
        "status": _generation_status["status"],
        "last_run": _generation_status["last_run"],
        "current_job": _generation_status["current_job"],
    })


@app.route('/latest', methods=['GET'])
def latest():
    """Get latest video info."""
    from publish_video import find_latest_video, load_video_metadata
    
    try:
        video_info = find_latest_video()
        metadata = load_video_metadata(video_info)
        return jsonify({
            "status": "found",
            "video_path": video_info["video_path"],
            "project_dir": video_info["project_dir"],
            "metadata": metadata,
        })
    except FileNotFoundError as e:
        return jsonify({"status": "no_video", "error": str(e)}), 404


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a new video."""
    with _generation_lock:
        if _generation_status["status"] == "running":
            return jsonify({
                "status": "already_running",
                "message": "Video generation already in progress",
                "current_job": _generation_status["current_job"],
            }), 409
        
        _generation_status["status"] = "running"
        _generation_status["current_job"] = f"video_{int(datetime.now().timestamp())}"
    
    # Run generation in background thread
    job_id = _generation_status["current_job"]
    thread = threading.Thread(target=_run_generation, args=(job_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "started",
        "job_id": job_id,
        "message": "Video generation started. Check /status for progress.",
    })


def _run_generation(job_id: str):
    """Background thread: run the full video pipeline."""
    global _generation_status
    logger.info(f"🎬 Starting video generation: {job_id}")
    
    try:
        # Import and run the main pipeline
        import subprocess
        result = subprocess.run(
            [sys.executable, "generate_complete_video.py"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(Path(__file__).parent.parent),
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Video generation complete: {job_id}")
            _generation_status.update({
                "status": "completed",
                "last_run": datetime.now().isoformat(),
                "last_result": {"returncode": 0, "stdout": result.stdout[-500:]},
                "current_job": None,
            })
        else:
            logger.error(f"❌ Video generation failed: {result.stderr[:500]}")
            _generation_status.update({
                "status": "failed",
                "last_run": datetime.now().isoformat(),
                "last_result": {"returncode": result.returncode, "stderr": result.stderr[-500:]},
                "current_job": None,
            })
    
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Video generation timed out: {job_id}")
        _generation_status.update({
            "status": "timeout",
            "last_run": datetime.now().isoformat(),
            "current_job": None,
        })
    
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        _generation_status.update({
            "status": "error",
            "last_run": datetime.now().isoformat(),
            "last_result": {"error": str(e)},
            "current_job": None,
        })


@app.route('/publish', methods=['POST'])
def publish():
    """Publish the latest video to platforms."""
    data = request.get_json() or {}
    platforms = data.get("platforms", None)
    video_path = data.get("video_path", None)
    
    from publish_video import publish_video as do_publish
    
    try:
        results = do_publish(
            video_path=video_path,
            platforms=platforms,
            dry_run=False,
        )
        
        all_ok = all(r.get("status") != "error" for r in results)
        return jsonify({
            "status": "published" if all_ok else "partial",
            "results": results,
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check for Docker."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geopolitical Sentinel — Pipeline API")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    args = parser.parse_args()
    
    logger.info(f"🚀 Pipeline API starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)