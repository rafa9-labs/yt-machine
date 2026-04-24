"""
FastAPI Server — Phase 4: Flask → FastAPI Migration
=====================================================

WHY FASTAPI? (Educational — read this carefully)
─────────────────────────────────────────────────
Your old pipeline_api.py used Flask. Here's why FastAPI is better:

1. ASYNC BY DEFAULT
   Flask is synchronous — each request blocks a thread. If 3 n8n webhooks
   fire simultaneously, Flask needs 3 threads. FastAPI uses async/await,
   handling thousands of concurrent requests on a single thread.

   OLD (Flask — synchronous, blocks):
       @app.route('/generate', methods=['POST'])
       def generate():                      # ← blocks until subprocess finishes
           subprocess.run(...)              # ← thread is stuck waiting

   NEW (FastAPI — async, non-blocking):
       @app.post('/generate')
       async def generate(req: GenerateRequest):  # ← yields to event loop
           await run_in_threadpool(...)            # ← other requests can run

2. PYDANTIC REQUEST VALIDATION
   Flask: data = request.get_json() or {}    ← no validation, typos silently ignored
   FastAPI: req: GenerateRequest             ← auto-validated against your Pydantic model

   If n8n sends {"platforms": ["fake_platform"]}, FastAPI returns:
   HTTP 422 Unprocessable Entity
   {"detail": [{"loc": ["body", "platforms", 0], "msg": "...not a valid enumeration member"}]}

3. AUTO-GENERATED DOCS
   Visit http://localhost:8000/docs → interactive Swagger UI
   Visit http://localhost:8000/redoc → ReDoc documentation
   No extra code needed — FastAPI generates these from your Pydantic models.

4. BACKGROUND TASKS
   Flask: threading.Thread(target=_run_generation).start()  ← manual, fragile
   FastAPI: background_tasks.add_task(_run_generation)      ← built-in, managed

5. UVICORN (production server)
   Flask uses Werkzeug — a development server, NOT meant for production.
   Uvicorn is an ASGI server built on uvloop (fast async event loop).

Usage:
    python server.py                     # Start on port 8000
    uvicorn server:app --reload          # Development with auto-reload
    uvicorn server:app --workers 4       # Production with 4 workers
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv()

# ── Phase 8: Structured logging via brain/log.py ──
# WHY? Replaces logging.basicConfig() with structlog configuration.
# If structlog is installed → JSON-capable, leveled, timestamped logs.
# If not → falls back to standard logging transparently.
from src.brain.log import get_logger
logger = get_logger("pipeline-api")

# ── Import our Pydantic models from Phase 1 ──
from src.models.schemas import (
    GenerateRequest,
    GenerateResponse,
    VideoStatus,
    Platform,
)


# ── Application State ─────────────────────────────────────────────────────
# WHY A STATE DICT? FastAPI's lifespan system (below) initializes resources
# once at startup and cleans up at shutdown. This dict holds the state.
# In production, you'd use a real database — but for a single-instance pipeline,
# in-memory state works fine.
_generation_status = {
    "status": "idle",
    "last_run": None,
    "last_result": None,
    "current_job": None,
}


# ── LIFESPAN (startup/shutdown) ───────────────────────────────────────────
# WHY LIFESPAN? Flask had no concept of "startup code." You just put
# initialization at module level and hoped it worked. FastAPI's lifespan
# runs code BEFORE the server accepts requests and AFTER it shuts down.
#
# EXAMPLE USES:
#   - Initialize database connection pool on startup
#   - Pull Ollama models if not cached
#   - Close database connections on shutdown
#
# @asynccontextmanager is Python's async context manager. It yields control
# to the server, then runs cleanup when the server shuts down.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize resources. Shutdown: clean up."""
    logger.info("🚀 Pipeline API starting up...")
    logger.info("📋 Auto-docs available at http://localhost:8000/docs")

    # ── Startup tasks go here ──
    # Example: init_db() to create PostgreSQL tables
    # Example: pull Ollama model if not cached

    yield  # ← Server runs here, handling requests

    # ── Shutdown tasks go here ──
    logger.info("🛑 Pipeline API shutting down...")


# ── CREATE FASTAPI APP ────────────────────────────────────────────────────
# WHY title/description? These appear in the auto-generated docs at /docs.
# n8n can discover available endpoints just by visiting this URL.
app = FastAPI(
    title="Geopolitical Sentinel — Pipeline API",
    description=(
        "Automated geopolitical video generation pipeline.\n\n"
        "## Workflow\n"
        "1. **POST /generate** — Start video generation\n"
        "2. **GET /status** — Check generation progress\n"
        "3. **GET /latest** — Get latest video info\n"
        "4. **POST /publish** — Publish to platforms\n"
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── RESPONSE MODELS (for auto-docs) ──────────────────────────────────────
# WHY SEPARATE RESPONSE MODELS? FastAPI uses these to generate accurate
# Swagger documentation. Without them, /docs would show "returns any JSON."

class StatusResponse(BaseModel):
    """GET /status response model."""
    pipeline: str = Field(description="Pipeline name")
    status: str = Field(description="Current status: idle | running | completed | failed")
    last_run: Optional[str] = None
    current_job: Optional[str] = None


class LatestResponse(BaseModel):
    """GET /latest response model."""
    status: str = Field(description="found | no_video")
    video_path: Optional[str] = None
    project_dir: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


class PublishRequest(BaseModel):
    """POST /publish request model — replaces raw request.get_json()."""
    platforms: Optional[List[str]] = Field(
        default=None,
        description="Platforms to publish to. None = all configured platforms."
    )
    video_path: Optional[str] = Field(
        default=None,
        description="Specific video to publish. None = latest."
    )


class PublishResponse(BaseModel):
    """POST /publish response model."""
    status: str = Field(description="published | partial | error")
    results: Optional[List[dict]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """GET /health response — for Docker health checks."""
    status: str = "healthy"


# ── BACKGROUND TASK FUNCTION ──────────────────────────────────────────────
# WHY run_in_threadpool? subprocess.run() is a BLOCKING call — it waits for
# the subprocess to finish. In an async server, blocking the event loop
# prevents ALL other requests from being handled.
#
# FastAPI's run_in_threadpool() runs the blocking function in a separate
# thread, yielding control back to the event loop. Other requests can still
# be handled while the video generates.

def _send_to_telegram(job_id: str, stdout: str) -> dict:
    """
    Find the latest generated video and send it to Telegram.
    Called automatically after successful video generation.
    If TELEGRAM_BOT_TOKEN is not configured, skips silently.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        logger.info("📱 Telegram not configured (TELEGRAM_BOT_TOKEN/CHAT_ID) — skipping auto-send")
        return {"skipped": True, "reason": "not configured"}

    try:
        from tools.telegram_sender import send_video_to_telegram

        # Find the video file for this job
        project_dir = Path(__file__).parent / "output" / "projects" / job_id
        if not project_dir.exists():
            # Fallback: find latest video
            from publish_video import find_latest_video
            latest = find_latest_video()
            video_path = latest.get("video_path")
        else:
            # Look for the video in the job's project directory
            video_files = list(project_dir.glob(f"{job_id}.mp4"))
            # Exclude .remux.mp4 files
            video_files = [v for v in video_files if ".remux" not in v.name]
            if not video_files:
                return {"skipped": True, "reason": f"No video found in {project_dir}"}
            video_path = str(video_files[0])

        logger.info(f"📱 Sending video to Telegram: {video_path}")

        result = send_video_to_telegram(
            video_path=video_path,
            caption=f"📹 Masker Daily News — {datetime.now().strftime('%b %d, %Y')}",
        )

        if result.get("success"):
            logger.info(f"📱 ✅ Video sent to Telegram (msg_id={result.get('message_id')})")
        else:
            logger.warning(f"📱 ❌ Telegram send failed: {result.get('error')}")

        return result

    except Exception as e:
        logger.warning(f"📱 Telegram auto-send error: {e}")
        return {"error": str(e)}


async def _run_generation(job_id: str):
    """Run the unified video generation pipeline in a background thread.
    
    The v1/v2 pipelines have been merged into generate_complete_video.py.
    All features (async scraper, LangChain, pgvector, Telegram) are built-in
    with graceful fallbacks. No pipeline version toggle needed.
    """
    global _generation_status
    
    pipeline_script = "generate_complete_video.py"
    
    logger.info(f"🎬 Starting video generation: {job_id} (pipeline=unified)")

    def _blocking_run():
        """This runs in a thread pool — blocking is OK here."""
        return subprocess.run(
            [sys.executable, pipeline_script],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(Path(__file__).parent),
        )

    try:
        # ── run_in_threadpool runs the blocking function off the event loop ──
        import asyncio
        result = await asyncio.get_event_loop().run_in_executor(None, _blocking_run)

        if result.returncode == 0:
            logger.info(f"✅ Video generation complete: {job_id}")

            # ── Auto-send to Telegram ──
            telegram_result = _send_to_telegram(job_id, result.stdout)

            _generation_status.update({
                "status": "completed",
                "last_run": datetime.now().isoformat(),
                "last_result": {
                    "returncode": 0,
                    "stdout": result.stdout[-500:],
                    "telegram": telegram_result,
                },
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


# ── ENDPOINTS ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """
    Health check endpoint for Docker HEALTHCHECK and load balancers.

    WHY IS THIS IMPORTANT? Docker's HEALTHCHECK (in Dockerfile) calls this
    every 30 seconds. If it fails 3 times, Docker marks the container unhealthy
    and can auto-restart it. Without this, a crashed server looks "running" to Docker.
    """
    return HealthResponse(status="healthy")


@app.get("/status", response_model=StatusResponse, tags=["pipeline"])
async def status():
    """
    Check the current pipeline status.

    Returns whether a generation is running, idle, or recently completed.
    n8n can poll this endpoint to decide whether to trigger a new generation.
    """
    return StatusResponse(
        pipeline="geopolitical-sentinel",
        status=_generation_status["status"],
        last_run=_generation_status["last_run"],
        current_job=_generation_status["current_job"],
    )


@app.get("/latest", response_model=LatestResponse, tags=["pipeline"])
async def latest():
    """
    Get info about the latest generated video.

    Returns video path, project directory, and metadata (topic, script, etc.)
    """
    try:
        from publish_video import find_latest_video, load_video_metadata
        video_info = find_latest_video()
        metadata = load_video_metadata(video_info)
        return LatestResponse(
            status="found",
            video_path=video_info.get("video_path"),
            project_dir=video_info.get("project_dir"),
            metadata=metadata,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/generate", response_model=GenerateResponse, tags=["pipeline"])
async def generate(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start video generation in the background.

    WHY BackgroundTasks? Flask needed manual threading:
        thread = threading.Thread(target=_run_generation)
        thread.daemon = True
        thread.start()

    FastAPI's BackgroundTasks is built-in:
        background_tasks.add_task(_run_generation, job_id)

    The task runs after the response is sent. FastAPI manages the lifecycle.

    WHY REQUEST VALIDATION? The `request: GenerateRequest` parameter tells
    FastAPI to parse and validate the JSON body. Invalid data → 422 error.
    Try sending {"max_word_count": 99999} — you'll get a clear validation error.
    """
    # ── Guard: prevent concurrent generations ──
    if _generation_status["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Video generation already in progress",
                "current_job": _generation_status["current_job"],
            },
        )

    # ── Create job ──
    job_id = f"video_{int(datetime.now().timestamp())}"
    _generation_status.update({
        "status": "running",
        "current_job": job_id,
    })

    # ── Schedule background generation ──
    background_tasks.add_task(_run_generation, job_id)

    return GenerateResponse(
        project_id=job_id,
        status=VideoStatus.SCRAPED,
        message="Video generation started. Check /status for progress.",
    )


@app.post("/publish", response_model=PublishResponse, tags=["pipeline"])
async def publish(request: PublishRequest):
    """
    Publish the latest (or specified) video to platforms.

    WHY A REQUEST MODEL? Flask used:
        data = request.get_json() or {}
        platforms = data.get("platforms", None)    ← no validation

    FastAPI uses:
        request: PublishRequest                        ← validated
        request.platforms                              ← typed as Optional[List[str]]

    If someone sends {"platforms": 123}, FastAPI returns 422 automatically.
    """
    try:
        from publish_video import publish_video as do_publish

        results = do_publish(
            video_path=request.video_path,
            platforms=request.platforms,
            dry_run=False,
        )

        all_ok = all(r.get("status") != "error" for r in results)
        return PublishResponse(
            status="published" if all_ok else "partial",
            results=results,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENTRY POINT ───────────────────────────────────────────────────────────
# WHY uvicorn.run()? Flask used app.run() which starts Werkzeug (dev server).
# uvicorn is a production-grade ASGI server. The `if __name__` guard ensures
# this only runs when you execute `python server.py` directly, not when
# imported by uvicorn (which imports the `app` object directly).
if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Geopolitical Sentinel — Pipeline API")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    args = parser.parse_args()

    logger.info(f"🚀 FastAPI server starting on {args.host}:{args.port}")
    uvicorn.run(
        "server:app",              # ← module:variable format (enables --reload)
        host=args.host,
        port=args.port,
        reload=args.reload,
    )