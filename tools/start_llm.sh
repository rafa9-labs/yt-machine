#!/usr/bin/env bash
# ── Start the local LLM stack for yt-machine ────────────────────────────
# 1. mlx_lm.server  — serves Qwen3.8-27B OptiQ (MLX, Apple Silicon GPU)
# 2. ollama_mlx_bridge.py — translates Ollama API → OpenAI API on :11434
#
# Usage:
#   tools/start_llm.sh start     # start both (loads model, ~1-2 min)
#   tools/start_llm.sh stop      # stop both
#   tools/start_llm.sh status    # show state
#
# Env overrides:
#   MLX_MODEL   (default mlx-community/Qwen3.8-27B-OptiQ-4bit)
#   MLX_PORT    (default 8080)
#   BRIDGE_PORT (default 11434)
#   BIND_HOST   (default 127.0.0.1; use 0.0.0.0 to reach it from Docker)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/output/logs"
mkdir -p "$LOG_DIR"

MLX_MODEL="${MLX_MODEL:-mlx-community/Qwen3.8-27B-OptiQ-4bit}"
MLX_PORT="${MLX_PORT:-8080}"
BRIDGE_PORT="${BRIDGE_PORT:-11434}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
THINKING="${MLX_ENABLE_THINKING:-0}"

MLX_PID_FILE="$LOG_DIR/mlx_server.pid"
BRIDGE_PID_FILE="$LOG_DIR/ollama_bridge.pid"

MLX_BIN="$(command -v mlx_lm.server || true)"
PY3="$(command -v python3 || true)"

is_up() { curl -sf "http://127.0.0.1:$1$2" >/dev/null 2>&1; }

start() {
    if [ -z "$MLX_BIN" ]; then
        echo "mlx_lm.server not found. Install with: brew install mlx-lm" >&2
        exit 1
    fi

    if is_up "$MLX_PORT" "/v1/models"; then
        echo "[llm] mlx_lm.server already running on :$MLX_PORT"
    else
        echo "[llm] starting mlx_lm.server ($MLX_MODEL) on :$MLX_PORT ..."
        CHAT_ARGS=""
        if [ "$THINKING" != "1" ]; then
            CHAT_ARGS='--chat-template-args {"enable_thinking":false}'
        fi
        # shellcheck disable=SC2086
        nohup "$MLX_BIN" --model "$MLX_MODEL" --host 127.0.0.1 --port "$MLX_PORT" \
            $CHAT_ARGS >> "$LOG_DIR/mlx_server.log" 2>&1 &
        echo $! > "$MLX_PID_FILE"

        echo -n "[llm] loading model (first run can take 1-3 min)"
        for _ in $(seq 1 360); do
            if is_up "$MLX_PORT" "/v1/models"; then
                echo " ready."
                break
            fi
            echo -n "."
            sleep 1
        done
        if ! is_up "$MLX_PORT" "/v1/models"; then
            echo " FAILED (see $LOG_DIR/mlx_server.log)" >&2
            exit 1
        fi
    fi

    if is_up "$BRIDGE_PORT" "/api/tags"; then
        echo "[llm] ollama bridge already running on :$BRIDGE_PORT"
    else
        echo "[llm] starting ollama bridge on :$BRIDGE_PORT ..."
        MLX_MODEL="$MLX_MODEL" MLX_BASE_URL="http://127.0.0.1:$MLX_PORT" \
            nohup "$PY3" "$ROOT/tools/ollama_mlx_bridge.py" --host "$BIND_HOST" --port "$BRIDGE_PORT" \
            >> "$LOG_DIR/ollama_bridge.log" 2>&1 &
        echo $! > "$BRIDGE_PID_FILE"

        for _ in $(seq 1 30); do
            if is_up "$BRIDGE_PORT" "/api/tags"; then
                echo "[llm] bridge ready."
                break
            fi
            sleep 1
        done
    fi

    echo "[llm] Ollama-compatible endpoint: http://localhost:$BRIDGE_PORT"
}

stop_one() {
    local pid_file="$1" name="$2"
    if [ -f "$pid_file" ]; then
        local pid
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[llm] stopping $name (pid $pid)"
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

status() {
    if is_up "$MLX_PORT" "/v1/models"; then
        echo "[llm] mlx_lm.server: UP   (:$MLX_PORT)"
    else
        echo "[llm] mlx_lm.server: down (:$MLX_PORT)"
    fi
    if is_up "$BRIDGE_PORT" "/api/tags"; then
        echo "[llm] ollama bridge: UP   (:$BRIDGE_PORT)"
    else
        echo "[llm] ollama bridge: down (:$BRIDGE_PORT)"
    fi
}

case "${1:-start}" in
    start) start ;;
    stop)
        stop_one "$BRIDGE_PID_FILE" "ollama bridge"
        stop_one "$MLX_PID_FILE" "mlx_lm.server"
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
    status) status ;;
    *)
        echo "usage: $0 {start|stop|restart|status}" >&2
        exit 1
        ;;
esac
