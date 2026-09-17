#!/usr/bin/env python3
"""
YouTube OAuth — one-time interactive consent for unattended uploads.
=====================================================================

WHY THIS EXISTS
───────────────
`publish_video.py --dry-run` returns before the OAuth block (the dry-run
check is above the credential flow), so it can NOT produce a token. Without
a cached token the first *real* upload tries to open a browser — which fails
under launchd, where there is no GUI session attached to the job.

This script performs the consent once, while you are sitting at the Mac:

    .venv/bin/python tools/youtube_auth.py            # consent + verify
    .venv/bin/python tools/youtube_auth.py --check    # verify cached token

WHAT IT CREATES
───────────────
    credentials/youtube_token.json   (already gitignored)

The token carries a refresh_token, so the publisher refreshes it silently on
every later run. Re-run this script only if you revoke access or delete the
token file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

DEFAULT_CLIENT_SECRETS = "credentials/youtube_client_secrets.json"
DEFAULT_TOKEN = "credentials/youtube_token.json"


def _paths() -> tuple:
    secrets_env = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", DEFAULT_CLIENT_SECRETS)
    token_env = os.getenv("YOUTUBE_CREDENTIALS_FILE", DEFAULT_TOKEN)
    # Relative paths in .env are relative to the repo root, not the cwd.
    secrets = Path(secrets_env)
    token = Path(token_env)
    if not secrets.is_absolute():
        secrets = PROJECT_ROOT / secrets
    if not token.is_absolute():
        token = PROJECT_ROOT / token
    return secrets, token


def check_token() -> bool:
    """Report whether the cached token exists, parses, and refreshes."""
    _, token_path = _paths()
    if not token_path.exists():
        print(f"❌ No cached token at {token_path}")
        print("   Run: .venv/bin/python tools/youtube_auth.py")
        return False

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        print("❌ Google libraries missing. Run:")
        print("   uv pip install --python .venv/bin/python google-auth-oauthlib google-api-python-client")
        return False

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    print(f"✅ Token found: {token_path}")
    print(f"   valid={creds.valid} expired={creds.expired} "
          f"refresh_token={'yes' if creds.refresh_token else 'NO'}")

    if not creds.refresh_token:
        print("⚠️  No refresh_token — unattended runs will fail. Re-run this script.")
        return False

    if creds.expired or not creds.valid:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
            print("   refreshed and re-saved")
        except Exception as exc:
            print(f"❌ Refresh failed: {exc}")
            print("   Re-run this script to consent again.")
            return False

    return True


def run_consent() -> bool:
    """Open a browser for OAuth consent and cache the resulting token."""
    try:
        import google_auth_oauthlib.flow
    except ImportError:
        print("❌ Google libraries missing. Run:")
        print("   uv pip install --python .venv/bin/python google-auth-oauthlib google-api-python-client")
        return False

    secrets_path, token_path = _paths()

    if not secrets_path.exists():
        print(f"❌ Client secrets not found: {secrets_path}")
        print()
        print("Create them once in Google Cloud Console:")
        print("  1. console.cloud.google.com → new project")
        print("  2. APIs & Services → Library → enable 'YouTube Data API v3'")
        print("  3. APIs & Services → Credentials → Create credentials")
        print("     → OAuth client ID → Application type: Desktop app")
        print("  4. Download the JSON and save it as:")
        print(f"     {secrets_path}")
        return False

    print(f"Client secrets : {secrets_path}")
    print(f"Token target   : {token_path}")
    print()
    print("A browser window will open. Sign in with the Google account that")
    print("owns the YouTube channel and approve the upload permission.")
    print()

    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        str(secrets_path), SCOPES
    )
    # port=0 picks a free loopback port; access_type=offline is what yields a
    # refresh_token, and prompt=consent forces one even on a repeat consent.
    creds = flow.run_local_server(
        port=0, access_type="offline", prompt="consent"
    )

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    print()
    print(f"✅ Token cached at {token_path}")

    if not creds.refresh_token:
        print("⚠️  No refresh_token returned — unattended runs may fail.")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YouTube OAuth consent helper for unattended uploads"
    )
    parser.add_argument("--check", action="store_true",
                        help="Only verify the cached token, do not open a browser")
    args = parser.parse_args()

    if args.check:
        return 0 if check_token() else 1

    if check_token():
        return 0

    print()
    return 0 if run_consent() else 1


if __name__ == "__main__":
    sys.exit(main())
