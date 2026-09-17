# ── yt-machine API image ────────────────────────────────────────────────
# Build context is the REPOSITORY ROOT (see docker-compose.yml):
#   docker compose -f infra/docker-compose.yml build
#
# Layer strategy:
#   1. System deps (ffmpeg, fonts, Playwright Chromium libs)
#   2. Python deps (infra/requirements.docker.txt — no CUDA/torch stack)
#   3. Playwright browser binary
#   4. Application code

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System dependencies ───────────────────────────────────────────────
# ffmpeg         — audio/video processing (moviepy, TTS mastering)
# fonts-dejavu   — subtitle rendering fallback (see subtitle_renderer.py)
# curl           — container HEALTHCHECK
# libnss3 etc.   — Playwright Chromium runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu \
    fonts-liberation \
    curl \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────
# Kept separate from app code so code edits don't invalidate the pip layer.
COPY infra/requirements.docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Playwright browser binary (cached in its own layer) ───────────────
RUN playwright install chromium

# ── YouTube publishing clients ────────────────────────────────────────
RUN pip install --no-cache-dir google-auth-oauthlib google-api-python-client

# ── Application code ──────────────────────────────────────────────────
COPY . .

RUN mkdir -p output/projects output/images output/videos output/logs \
    output/audio output/publish_logs credentials

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# uvicorn (not `python server.py`) for clean SIGTERM handling.
# src.server:app — server.py lives in src/, and PYTHONPATH=/app makes the
# absolute `src.*` imports work.
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
