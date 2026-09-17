#!/usr/bin/env python3
"""
TikTok OAuth — authorization + token refresh for unattended posting.
====================================================================

WHY THIS EXISTS
───────────────
TikTok's Content Posting API issues access tokens that expire in ~24 hours.
A daily 06:00 launchd job therefore cannot rely on a static
TIKTOK_ACCESS_TOKEN past the first day: every unattended run would 401.

This module keeps a refresh token in the environment/token file and exchanges
it for a fresh access token before publishing.

    tools/tiktok_auth.py --authorize   # one-time: print the consent URL, then
                                       # exchange the returned code
    tools/tiktok_auth.py --refresh     # exchange the refresh token now
    tools/tiktok_auth.py --check       # show token state without changing it

TOKEN STORAGE
─────────────
Tokens are written to credentials/tiktok_token.json (already gitignored).
`publish_tiktok` prefers that file over the static .env values, because a
file can be updated by the refresh flow while .env cannot be at runtime.

SCOPES
──────
    video.publish   — Direct Post to the user's own account
    video.upload    — draft/inbox upload (used when Direct Post is not granted)

An unaudited TikTok app can normally only post with
privacy_level=SELF_ONLY. Publish errors mentioning privacy are an app-review
issue, not a bug — see SETUP.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a hard dep elsewhere
    load_dotenv = None

if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")

AUTHORIZE_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
DEFAULT_TOKEN_FILE = "credentials/tiktok_token.json"
SCOPES = "video.publish,video.upload"


def token_path() -> Path:
    raw = os.getenv("TIKTOK_TOKEN_FILE", DEFAULT_TOKEN_FILE)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_token() -> dict:
    path = token_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_token(data: dict) -> None:
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # 0600: the file holds a bearer token.
    path.chmod(0o600)


def _post_form(url: str, data: dict, timeout: int = 30) -> dict:
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Cache-Control": "no-cache",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_access_token() -> str:
    """Return a usable access token: cached file first, env fallback.

    Called by publish_tiktok on every publish. Token refresh is the caller's
    responsibility (refresh_access_token) so a network hiccup does not turn
    into a silent credential change mid-upload.
    """
    data = load_token()
    token = data.get("access_token")
    if not token:
        return os.getenv("TIKTOK_ACCESS_TOKEN", "").strip()
    return token


def token_is_expired(skew_seconds: int = 300) -> bool:
    data = load_token()
    expires_at = data.get("expires_at")
    if not expires_at:
        return False
    try:
        deadline = datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    return datetime.now() >= deadline - timedelta(seconds=skew_seconds)


def refresh_access_token() -> dict:
    """Exchange the stored refresh token for a new access token.

    TikTok rotates the refresh token on every exchange, so the new pair must
    be persisted immediately or the next refresh fails.
    """
    data = load_token()
    refresh_token = data.get("refresh_token") or os.getenv("TIKTOK_REFRESH_TOKEN", "")
    client_key = os.getenv("TIKTOK_CLIENT_KEY", "").strip()
    client_secret = os.getenv("TIKTOK_CLIENT_SECRET", "").strip()

    if not all([refresh_token, client_key, client_secret]):
        return {
            "success": False,
            "error": (
                "Missing refresh credentials. Need TIKTOK_CLIENT_KEY, "
                "TIKTOK_CLIENT_SECRET and a stored refresh token "
                "(run --authorize once)."
            ),
        }

    try:
        payload = _post_form(TOKEN_URL, {
            "client_key": client_key,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        })
    except Exception as exc:
        return {"success": False, "error": f"Refresh request failed: {exc}"}

    if "access_token" not in payload:
        return {
            "success": False,
            "error": payload.get("error_description")
            or payload.get("error")
            or str(payload)[:200],
        }

    expires_in = int(payload.get("expires_in", 86400))
    stored = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", refresh_token),
        "expires_at": (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
        "scope": payload.get("scope", SCOPES),
        "refreshed_at": datetime.now().isoformat(),
    }
    save_token(stored)
    return {
        "success": True,
        "expires_at": stored["expires_at"],
        "scope": stored["scope"],
    }


def ensure_fresh_token() -> dict:
    """Refresh only when the cached token is expired/absent (used by publisher)."""
    if not load_token() and not os.getenv("TIKTOK_ACCESS_TOKEN", "").strip():
        return {"success": False, "error": "No TikTok token available"}
    if token_is_expired():
        return refresh_access_token()
    return {"success": True, "refreshed": False}


def build_authorize_url() -> str:
    client_key = os.getenv("TIKTOK_CLIENT_KEY", "").strip()
    redirect_uri = os.getenv("TIKTOK_REDIRECT_URI", "").strip()
    if not client_key or not redirect_uri:
        raise SystemExit(
            "Set TIKTOK_CLIENT_KEY and TIKTOK_REDIRECT_URI in .env first.\n"
            "The redirect URI must match the one registered in the TikTok app."
        )
    params = {
        "client_key": client_key,
        "scope": SCOPES,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "state": os.urandom(8).hex(),
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str) -> dict:
    client_key = os.getenv("TIKTOK_CLIENT_KEY", "").strip()
    client_secret = os.getenv("TIKTOK_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("TIKTOK_REDIRECT_URI", "").strip()

    payload = _post_form(TOKEN_URL, {
        "client_key": client_key,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    })
    if "access_token" not in payload:
        return {
            "success": False,
            "error": payload.get("error_description")
            or payload.get("error")
            or str(payload)[:200],
        }

    expires_in = int(payload.get("expires_in", 86400))
    stored = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", ""),
        "expires_at": (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
        "scope": payload.get("scope", SCOPES),
        "authorized_at": datetime.now().isoformat(),
    }
    save_token(stored)
    return {"success": True, "expires_at": stored["expires_at"]}


def show_status() -> bool:
    data = load_token()
    if not data:
        print(f"No token file at {token_path()}")
        env_token = os.getenv("TIKTOK_ACCESS_TOKEN", "").strip()
        print(f"Static TIKTOK_ACCESS_TOKEN in .env: {'set' if env_token else 'not set'}")
        return False

    print(f"Token file     : {token_path()}")
    print(f"Expires at     : {data.get('expires_at', 'unknown')}")
    print(f"Expired (5m skew): {token_is_expired()}")
    print(f"Refresh token  : {'present' if data.get('refresh_token') else 'MISSING'}")
    print(f"Scope          : {data.get('scope', 'unknown')}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="TikTok OAuth helper")
    parser.add_argument("--authorize", action="store_true",
                        help="Print the consent URL to obtain an auth code")
    parser.add_argument("--code", type=str,
                        help="Authorization code to exchange (with --authorize)")
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh the stored access token now")
    parser.add_argument("--check", action="store_true",
                        help="Show token state without changing it")
    args = parser.parse_args()

    if args.refresh:
        result = refresh_access_token()
        print(json.dumps(result, indent=2))
        return 0 if result.get("success") else 1

    if args.authorize:
        if args.code:
            result = exchange_code(args.code)
            print(json.dumps(result, indent=2))
            return 0 if result.get("success") else 1
        print("Open this URL, approve access, then copy the 'code' query param:\n")
        print(build_authorize_url())
        print("\nThen run:")
        print("  python tools/tiktok_auth.py --authorize --code <CODE>")
        return 0

    if args.check or len(sys.argv) == 1:
        show_status()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
