# ── Phase 6: Production Dockerfile ──────────────────────────────────────
# WHY MULTI-STAGE? Not used here (single stage is fine for this project),
# but the principles are:
# 1. Slim base image → smaller attack surface, faster pulls
# 2. System deps installed in one layer → cached if unchanged
# 3. Python deps in another layer → cached if requirements.txt unchanged
# 4. App code last → code changes don't invalidate dependency cache

FROM python:3.12-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────
# WHY THESE PACKAGES?
# - ffmpeg: audio/video processing (moviepy, edge-tts output)
# - fonts-dejavu/fonts-liberation: subtitle rendering
# - curl: Docker health check (HEALTHCHECK below uses it)
# - libnss3/libnspr4/libatk1.0...: Playwright Chromium dependencies
#   (without these, `playwright install chromium` fails silently)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu \
    fonts-liberation \
    curl \
    # ── Playwright Chromium system dependencies ──
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
# WHY COPY requirements.txt SEPARATELY? Docker layers are cached.
# If only app code changes, this layer is reused (no re-pip-install).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Playwright browser binary ─────────────────────────────────────────
# WHY SEPARATE FROM pip INSTALL? Playwright's browser is a ~200MB download.
# Installing it in its own layer means it's cached even if Python deps change.
# --with-deps installs OS-level Chromium dependencies (redundant with above,
# but ensures Playwright has everything it needs).
RUN playwright install chromium

# ── Google API clients (for YouTube publishing) ───────────────────────
RUN pip install --no-cache-dir google-auth-oauthlib google-api-python-client

# ── Copy application code ─────────────────────────────────────────────
COPY . .

# Create output directories
RUN mkdir -p output/projects output/images output/logs output/publish_logs credentials

# ── Expose API port ───────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────
# WHY --start-period=60s? Video generation deps (Ollama, models) take time
# to initialize. Without this grace period, Docker might kill a healthy
# container that's still warming up.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start FastAPI server with Uvicorn ─────────────────────────────────
# WHY uvicorn INSTEAD OF `python server.py`?
# Direct uvicorn is the recommended production deployment. It avoids the
# overhead of going through Python's __main__ and gives cleaner signal
# handling in containers (SIGTERM for graceful shutdown).
#
# WHY --host 0.0.0.0? Inside Docker, binding to localhost (127.0.0.1) only
# accepts connections FROM the container itself. 0.0.0.0 means "accept
# connections on all network interfaces" — required for port mapping.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
