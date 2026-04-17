"""
Telegram Video Sender — Send generated videos to a Telegram chat.
Uses the Bot API directly via requests (no extra dependency needed).

Setup:
  1. Create a bot via @BotFather → get TELEGRAM_BOT_TOKEN
  2. Send /start to your bot → get TELEGRAM_CHAT_ID (use /getUpdates API)
  3. Add both to .env

Usage (standalone):
  python -m tools.telegram_sender output/projects/video_XXX/video_XXX.mp4

Usage (importable):
  from tools.telegram_sender import send_video_to_telegram
  send_video_to_telegram("path/to/video.mp4", caption="Daily news!")
"""

import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Telegram Bot API limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB for sendVideo


def send_video_to_telegram(
    video_path: str,
    caption: str = None,
    bot_token: str = None,
    chat_id: str = None,
) -> dict:
    """
    Send a video file to a Telegram chat via Bot API.

    Args:
        video_path: Path to the MP4 file
        caption: Optional caption (max 1024 chars)
        bot_token: Telegram bot token (defaults to env var)
        chat_id: Telegram chat ID (defaults to env var)

    Returns:
        dict with success status and message info or error
    """
    token = bot_token or TELEGRAM_BOT_TOKEN
    chat = chat_id or TELEGRAM_CHAT_ID

    if not token:
        return {"success": False, "error": "TELEGRAM_BOT_TOKEN not set. Add it to .env"}
    if not chat:
        return {"success": False, "error": "TELEGRAM_CHAT_ID not set. Add it to .env"}

    video = Path(video_path)
    if not video.exists():
        return {"success": False, "error": f"Video not found: {video_path}"}

    file_size = video.stat().st_size
    if file_size > MAX_FILE_SIZE:
        return {
            "success": False,
            "error": f"Video too large: {file_size / (1024*1024):.1f}MB (max 50MB)"
        }

    # Default caption: filename
    if not caption:
        caption = f"📹 {video.stem}"

    # Truncate caption to Telegram limit
    if len(caption) > 1024:
        caption = caption[:1021] + "..."

    url = f"https://api.telegram.org/bot{token}/sendVideo"

    print(f"  [TELEGRAM] Sending video: {video.name} ({file_size / (1024*1024):.1f}MB)")

    try:
        with open(video, "rb") as f:
            files = {"video": (video.name, f, "video/mp4")}
            data = {
                "chat_id": chat,
                "caption": caption,
                "parse_mode": "HTML",
                "supports_streaming": "true",  # Allow inline streaming in Telegram
            }
            response = requests.post(url, files=files, data=data, timeout=120)

        result = response.json()

        if response.status_code == 200 and result.get("ok"):
            msg = result.get("result", {})
            print(f"  [TELEGRAM] ✅ Video sent successfully (msg_id={msg.get('message_id')})")
            return {
                "success": True,
                "message_id": msg.get("message_id"),
                "chat_id": msg.get("chat", {}).get("id"),
            }
        else:
            error_desc = result.get("description", "Unknown error")
            print(f"  [TELEGRAM] ❌ API error: {error_desc}")
            return {"success": False, "error": error_desc}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out (120s) — video may be too large"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def send_message(
    text: str,
    bot_token: str = None,
    chat_id: str = None,
) -> dict:
    """
    Send a text message to a Telegram chat.
    Useful for status notifications (generation started, failed, etc.)
    """
    token = bot_token or TELEGRAM_BOT_TOKEN
    chat = chat_id or TELEGRAM_CHAT_ID

    if not token or not chat:
        return {"success": False, "error": "Telegram credentials not configured"}

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        response = requests.post(url, json={
            "chat_id": chat,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=30)

        result = response.json()
        if response.status_code == 200 and result.get("ok"):
            return {"success": True, "message_id": result["result"]["message_id"]}
        else:
            return {"success": False, "error": result.get("description", "Unknown error")}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_chat_id(bot_token: str = None) -> None:
    """
    Helper: Get your chat ID by reading recent bot messages.
    Run this, then send /start to your bot, and it will print your chat ID.
    """
    token = bot_token or TELEGRAM_BOT_TOKEN
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in .env first")
        return

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    response = requests.get(url, timeout=10)
    result = response.json()

    if not result.get("ok"):
        print(f"API error: {result.get('description')}")
        return

    updates = result.get("result", [])
    if not updates:
        print("No messages found. Send /start to your bot first, then run this again.")
        return

    for update in updates[-5:]:
        msg = update.get("message", {})
        chat = msg.get("chat", {})
        if chat.get("type") == "private":
            print(f"Chat ID: {chat['id']} (user: {chat.get('first_name', 'unknown')})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tools.telegram_sender <video_path> [caption]")
        print("       python -m tools.telegram_sender --get-chat-id")
        sys.exit(1)

    if sys.argv[1] == "--get-chat-id":
        get_chat_id()
        sys.exit(0)

    video_path = sys.argv[1]
    caption = sys.argv[2] if len(sys.argv) > 2 else None

    result = send_video_to_telegram(video_path, caption=caption)
    if result["success"]:
        print(f"✅ Sent!")
    else:
        print(f"❌ Failed: {result['error']}")
        sys.exit(1)