"""
Geopolitical Sentinel — Video Publisher
Uploads completed videos to YouTube, TikTok, and Instagram Reels.

Usage:
    python publish_video.py                          # Publish latest video to all platforms
    python publish_video.py --video PATH             # Publish specific video
    python publish_video.py --platform youtube        # Publish only to YouTube
    python publish_video.py --platform youtube,tiktok # Publish to YouTube + TikTok
    python publish_video.py --dry-run                 # Preview what would be published

Environment variables (set in .env):
    # YouTube
    YOUTUBE_CLIENT_SECRETS_FILE   - Path to OAuth2 credentials JSON
    YOUTUBE_CREDENTIALS_FILE      - Path to cached OAuth token (auto-created)
    
    # TikTok
    TIKTOK_CLIENT_KEY             - TikTok developer app client key
    TIKTOK_CLIENT_SECRET          - TikTok developer app client secret
    TIKTOK_ACCESS_TOKEN           - TikTok content posting access token
    
    # Instagram
    INSTAGRAM_ACCESS_TOKEN        - Meta long-lived access token
    INSTAGRAM_BUSINESS_ACCOUNT_ID - Instagram business account ID
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load the repo-root .env explicitly: load_dotenv() alone depends on the cwd,
# which differs between a manual run (repo root) and launchd (WorkingDirectory).
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger('publisher')

# Privacy for the upload. Keep "private" until the first upload has been
# verified end to end, then set YOUTUBE_PRIVACY=public (or "unlisted").
YOUTUBE_PRIVACY = os.getenv("YOUTUBE_PRIVACY", "public").strip().lower()

# ── Video Discovery ──────────────────────────────────────────────────────────

def _select_deliverable(video_files) -> "Path | None":
    """Pick the file to upload from a project's MP4s.

    A project can hold more than one MP4: the lossless master
    (`video_<id>.master.mp4`) is the archival artifact, and the size-bounded
    delivery copy (`video_<id>.mp4`) is what providers should receive. The
    delivery copy is always the one to publish — the master exceeds
    Telegram's 50 MB cap and is read fully into memory by the TikTok and
    Instagram uploaders.

    Fails closed: if a master is the only candidate (delivery failed, or
    YT_DELIVERY_ENABLED was turned off) it is still returned, because
    publishing something is better than silently publishing nothing — the
    provider will reject an oversize file with a clear error.
    """
    candidates = [f for f in video_files if "TEMP" not in f.name]
    if not candidates:
        return None

    deliveries = [
        f for f in candidates
        if not f.name.endswith(".master.mp4")
    ]
    if deliveries:
        # Prefer the canonical video_<id>.mp4 over any other artifact.
        canonical = [f for f in deliveries if f.name.endswith(".mp4")
                     and ".master." not in f.name]
        chosen = sorted(canonical or deliveries)
        return chosen[0]

    masters = [f for f in candidates if f.name.endswith(".master.mp4")]
    return sorted(masters)[0] if masters else None


def find_latest_video() -> dict:
    """Find the most recent completed video in output/projects/."""
    projects_dir = Path("output/projects")
    if not projects_dir.exists():
        raise FileNotFoundError("No output/projects directory found")
    
    latest = None
    latest_time = 0
    
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        
        manifest_path = project_dir / "manifest.json"
        main_video = _select_deliverable(list(project_dir.glob("*.mp4")))
        
        if not main_video:
            continue
        
        mtime = main_video.stat().st_mtime
        if mtime > latest_time:
            latest_time = mtime
            latest = {
                "video_path": str(main_video),
                "manifest_path": str(manifest_path) if manifest_path.exists() else None,
                "project_dir": str(project_dir),
            }
    
    if not latest:
        raise FileNotFoundError("No completed videos found in output/projects/")
    
    return latest


def load_video_metadata(video_info: dict) -> dict:
    """Load metadata: first try platform_metadata.json, then manifest.json."""
    metadata = {
        "title": "Geopolitical Sentinel — Daily Intelligence Briefing",
        "description": (
            "Geopolitical analysis with pixel art visuals. "
            "The angle mainstream media buries — every event connects to a pattern.\n\n"
            "#geopolitics #news #analysis #pixelart #intelligence"
        ),
        "tags": ["geopolitics", "news", "analysis", "pixel art", "intelligence", "military"],
    }

    # Step 1: Try platform_metadata.json (generated by PlatformMetadataGenerator)
    project_dir = video_info.get("project_dir", "")
    if project_dir:
        pm_path = Path(project_dir) / "platform_metadata.json"
        if pm_path.exists():
            try:
                with open(pm_path, 'r', encoding='utf-8') as f:
                    pm = json.load(f)
                # Use YouTube-specific metadata if available
                yt = pm.get('youtube', {})
                if yt.get('title'):
                    metadata["title"] = yt['title'][:100]
                if yt.get('description'):
                    metadata["description"] = yt['description'][:5000]
                if yt.get('tags'):
                    metadata["tags"] = yt['tags'][:20]
                # Also merge common hashtags
                common = pm.get('common_hashtags', [])
                if common:
                    for tag in common[:5]:
                        if tag not in metadata["tags"]:
                            metadata["tags"].append(tag)
                logger.info(f"Loaded platform metadata from {pm_path}")
            except Exception as e:
                logger.warning(f"Could not parse platform_metadata.json: {e}")

    # Step 2: Fall back to manifest.json
    manifest_path = video_info.get("manifest_path")
    if manifest_path and Path(manifest_path).exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            # Only override title if we don't already have a good one
            if metadata["title"] == "Geopolitical Sentinel — Daily Intelligence Briefing":
                if "title" in manifest and manifest["title"]:
                    metadata["title"] = str(manifest["title"])[:100]
                elif "hook" in manifest and manifest["hook"]:
                    metadata["title"] = str(manifest["hook"])[:100]
                elif "script" in manifest:
                    script_data = manifest["script"]
                    if isinstance(script_data, dict):
                        script_text = script_data.get("full_text", "") or script_data.get("raw_text", "") or json.dumps(script_data)
                    else:
                        script_text = str(script_data)
                    for line in script_text.split("\n"):
                        line = line.strip()
                        if len(line) > 20 and len(line) < 100:
                            metadata["title"] = line[:100]
                            break

            # Extract trending terms for tags
            if "trending_terms" in manifest:
                trending = manifest["trending_terms"][:5]
                metadata["tags"].extend(trending)

            # Extract category
            if "category" in manifest:
                category = manifest["category"]
                metadata["tags"].append(category)
                metadata["description"] = f"[{category.upper()}] {metadata['description']}"

            # Build rich description
            if "stories" in manifest:
                stories = manifest["stories"]
                story_lines = [f"• {s.get('headline', 'Story update')}" for s in stories[:3]]
                metadata["description"] += "\n\nStories covered:\n" + "\n".join(story_lines)

        except Exception as e:
            logger.warning(f"Could not parse manifest: {e}")

    return metadata


# ── YouTube Publisher ────────────────────────────────────────────────────────

def publish_youtube(video_path: str, metadata: dict, dry_run: bool = False) -> dict:
    """
    Upload video to YouTube as a Short.
    
    Requires: YOUTUBE_CLIENT_SECRETS_FILE (OAuth2 credentials JSON from Google Cloud Console)
    First run opens browser for OAuth consent — token cached for subsequent runs.
    """
    logger.info(f"📺 YouTube: Preparing upload for {video_path}")
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would upload to YouTube Shorts")
        logger.info(f"  Title: {metadata['title']}")
        logger.info(f"  Tags: {metadata['tags'][:5]}")
        return {"platform": "youtube", "status": "dry_run", "title": metadata["title"]}
    
    try:
        import google_auth_oauthlib.flow
        import googleapiclient.discovery
        import googleapiclient.http
    except ImportError:
        logger.error("  ❌ Missing Google API libraries. Run: pip install google-auth-oauthlib google-api-python-client")
        return {"platform": "youtube", "status": "error", "error": "Missing dependencies"}
    
    client_secrets = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "credentials/youtube_client_secrets.json")
    credentials_file = os.getenv("YOUTUBE_CREDENTIALS_FILE", "credentials/youtube_token.json")
    
    if not Path(client_secrets).exists():
        logger.error(f"  ❌ YouTube client secrets not found: {client_secrets}")
        logger.error(f"  → Download from Google Cloud Console → APIs & Services → Credentials")
        return {"platform": "youtube", "status": "error", "error": "Missing client secrets"}
    
    # OAuth2 flow
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    
    creds = None
    if Path(credentials_file).exists():
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(credentials_file, scopes)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                client_secrets, scopes
            )
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        Path(credentials_file).parent.mkdir(parents=True, exist_ok=True)
        with open(credentials_file, 'w') as f:
            f.write(creds.to_json())
    
    # Build YouTube service
    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=creds)
    
    # Prepare upload
    privacy = YOUTUBE_PRIVACY if YOUTUBE_PRIVACY in ("public", "unlisted", "private") else "public"
    if privacy != "public":
        logger.info(f"  ℹ️ Uploading with privacy={privacy} (YOUTUBE_PRIVACY)")

    body = {
        "snippet": {
            "title": metadata["title"][:100],
            "description": metadata["description"][:5000],
            "tags": metadata["tags"][:20],
            "categoryId": "25",  # News & Politics
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
            "embeddable": True,
        },
    }
    
    # Upload
    media = googleapiclient.http.MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
    )
    
    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )
    
    logger.info("  ⏳ Uploading to YouTube...")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.info(f"  📤 Upload progress: {int(status.progress() * 100)}%")
    
    video_id = response.get("id")
    video_url = f"https://youtube.com/shorts/{video_id}"
    logger.info(f"  ✅ Published to YouTube ({privacy}): {video_url}")

    return {
        "platform": "youtube",
        "status": "published",
        "video_id": video_id,
        "url": video_url,
        "privacy": privacy,
    }


# ── TikTok Publisher ─────────────────────────────────────────────────────────

def publish_tiktok(video_path: str, metadata: dict, dry_run: bool = False) -> dict:
    """
    Upload video to TikTok via Content Posting API.

    Credentials: TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET and an access token.
    Access tokens expire in ~24h, so the token is read from
    credentials/tiktok_token.json when present and refreshed automatically
    (see tools/tiktok_auth.py). A static TIKTOK_ACCESS_TOKEN in .env still
    works for testing, but will fail on the second day.
    """
    logger.info(f"🎵 TikTok: Preparing upload for {video_path}")
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would upload to TikTok")
        logger.info(f"  Title: {metadata['title']}")
        return {"platform": "tiktok", "status": "dry_run", "title": metadata["title"]}
    
    client_key = os.getenv("TIKTOK_CLIENT_KEY")
    client_secret = os.getenv("TIKTOK_CLIENT_SECRET")

    # Refresh the OAuth token first — a launchd run happens hours after the
    # previous one, so the cached token is usually past its 24h lifetime.
    access_token = ""
    try:
        from tools.tiktok_auth import ensure_fresh_token, get_access_token

        refreshed = ensure_fresh_token()
        if not refreshed.get("success"):
            logger.warning(f"  ⚠️ TikTok token refresh skipped: {refreshed.get('error')}")
        access_token = get_access_token()
    except Exception as exc:
        logger.warning(f"  ⚠️ TikTok token helper unavailable ({exc}); using .env token")
        access_token = os.getenv("TIKTOK_ACCESS_TOKEN", "").strip()

    if not access_token:
        access_token = os.getenv("TIKTOK_ACCESS_TOKEN", "").strip()

    if not all([client_key, client_secret, access_token]):
        logger.error("  ❌ TikTok credentials not configured in .env")
        logger.error("  → Set TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET, TIKTOK_ACCESS_TOKEN")
        logger.error("  → Or run once: python tools/tiktok_auth.py --authorize")
        return {"platform": "tiktok", "status": "error", "error": "Missing credentials"}
    
    try:
        import requests
    except ImportError:
        logger.error("  ❌ Missing requests library")
        return {"platform": "tiktok", "status": "error", "error": "Missing requests"}
    
    # TikTok Content Posting API — Direct Post method
    # Step 1: Initialize upload
    video_size = os.path.getsize(video_path)
    
    init_url = "https://open.tiktokapis.com/v2/post/publish/video/init/"
    init_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    init_body = {
        "post_info": {
            "title": metadata["title"][:150],
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
            "video_cover_timestamp_ms": 1000,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": video_size,
            "chunk_size": video_size,  # Single chunk upload
            "total_chunk_count": 1,
        },
    }
    
    init_response = requests.post(init_url, headers=init_headers, json=init_body)
    
    if init_response.status_code != 200:
        logger.error(f"  ❌ TikTok init failed: {init_response.text[:200]}")
        return {"platform": "tiktok", "status": "error", "error": init_response.text[:200]}
    
    init_data = init_response.json()["data"]
    publish_id = init_data["publish_id"]
    upload_url = init_data["upload_url"]
    
    # Step 2: Upload video content
    logger.info("  ⏳ Uploading to TikTok...")
    with open(video_path, "rb") as f:
        video_content = f.read()
    
    upload_headers = {
        "Content-Type": "video/mp4",
        "Content-Length": str(video_size),
    }
    upload_response = requests.put(upload_url, headers=upload_headers, data=video_content)
    
    if upload_response.status_code not in (200, 201):
        logger.error(f"  ❌ TikTok upload failed: {upload_response.text[:200]}")
        return {"platform": "tiktok", "status": "error", "error": f"Upload failed: {upload_response.status_code}"}
    
    # Step 3: Check status
    import time
    status_url = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
    for attempt in range(10):
        time.sleep(5)
        status_response = requests.post(
            status_url,
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json={"publish_id": publish_id},
        )
        if status_response.status_code == 200:
            status_data = status_response.json()["data"]
            if status_data["status"] == "PUBLISH_COMPLETE":
                tiktok_url = f"https://tiktok.com/@geopoliticalsentinel/video/{publish_id}"
                logger.info(f"  ✅ Published to TikTok: {tiktok_url}")
                return {
                    "platform": "tiktok",
                    "status": "published",
                    "publish_id": publish_id,
                    "url": tiktok_url,
                }
            elif status_data["status"] == "PROCESSING_UPLOAD":
                logger.info(f"  ⏳ TikTok processing... ({attempt + 1}/10)")
                continue
            elif status_data["status"] == "SEND_TO_USER_INBOX":
                logger.info(f"  ⚠️ TikTok sent to inbox for review")
                return {"platform": "tiktok", "status": "inbox", "publish_id": publish_id}
            else:
                logger.error(f"  ❌ TikTok status: {status_data}")
                return {"platform": "tiktok", "status": "error", "error": str(status_data)}
    
    return {"platform": "tiktok", "status": "timeout", "publish_id": publish_id}


# ── Instagram Publisher ──────────────────────────────────────────────────────

def publish_instagram(video_path: str, metadata: dict, dry_run: bool = False) -> dict:
    """
    Upload video to Instagram as a Reel via Graph API.
    
    Requires: INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID
    """
    logger.info(f"📸 Instagram: Preparing upload for {video_path}")
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would upload to Instagram Reels")
        logger.info(f"  Title: {metadata['title']}")
        return {"platform": "instagram", "status": "dry_run", "title": metadata["title"]}
    
    access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
    account_id = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    
    if not all([access_token, account_id]):
        logger.error("  ❌ Instagram credentials not configured in .env")
        logger.error("  → Set INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID")
        return {"platform": "instagram", "status": "error", "error": "Missing credentials"}
    
    try:
        import requests
    except ImportError:
        return {"platform": "instagram", "status": "error", "error": "Missing requests"}
    
    # Instagram Reels API — Step 1: Create container
    # Note: Video must be publicly accessible URL OR uploaded via resumable upload
    # For local files, we use the resumable upload approach
    
    # Step 1: Create upload session
    create_url = f"https://graph.facebook.com/v19.0/{account_id}/media"
    
    # Build caption from metadata
    caption = f"{metadata['title']}\n\n{metadata['description'][:2200]}"
    # Add hashtags
    hashtag_str = " ".join(f"#{t.replace(' ', '')}" for t in metadata["tags"][:8])
    caption = f"{caption}\n\n{hashtag_str}"
    
    create_params = {
        "media_type": "REELS",
        "access_token": access_token,
        "caption": caption[:2200],  # Instagram limit
    }
    
    # Step 1a: Initiate resumable upload
    import requests as req
    
    # For local files, we need to use the resumable upload API
    upload_url_resp = req.post(
        f"https://graph.facebook.com/v19.0/{account_id}/media",
        params={
            "media_type": "REELS",
            "access_token": access_token,
            "upload_type": "resumable",
        }
    )
    
    if upload_url_resp.status_code != 200:
        # Fallback: try with video URL (if hosted)
        logger.error(f"  ❌ Instagram upload init failed: {upload_url_resp.text[:200]}")
        logger.error("  → Instagram requires video to be at a public URL or use app review-approved API")
        return {"platform": "instagram", "status": "error", "error": upload_url_resp.text[:200]}
    
    upload_data = upload_url_resp.json()
    container_id = upload_data.get("id")
    upload_session_url = upload_data.get("upload_url")
    
    if not upload_session_url:
        logger.error("  ❌ No upload URL returned from Instagram")
        return {"platform": "instagram", "status": "error", "error": "No upload URL"}
    
    # Step 2: Upload video binary
    logger.info("  ⏳ Uploading to Instagram...")
    video_size = os.path.getsize(video_path)
    
    with open(video_path, "rb") as f:
        video_content = f.read()
    
    upload_resp = req.post(
        upload_session_url,
        headers={
            "Authorization": f"OAuth {access_token}",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(video_size),
            "Offset": "0",
        },
        data=video_content,
    )
    
    if upload_resp.status_code not in (200, 201):
        logger.error(f"  ❌ Instagram video upload failed: {upload_resp.text[:200]}")
        return {"platform": "instagram", "status": "error", "error": f"Upload failed: {upload_resp.status_code}"}
    
    # Step 3: Wait for processing
    import time
    for attempt in range(15):
        time.sleep(10)
        status_resp = req.get(
            f"https://graph.facebook.com/v19.0/{container_id}",
            params={
                "fields": "status_code",
                "access_token": access_token,
            }
        )
        status = status_resp.json().get("status_code", "UNKNOWN")
        
        if status == "FINISHED":
            # Step 4: Publish the container
            publish_resp = req.post(
                f"https://graph.facebook.com/v19.0/{account_id}/media_publish",
                params={
                    "creation_id": container_id,
                    "access_token": access_token,
                }
            )
            
            if publish_resp.status_code == 200:
                media_id = publish_resp.json().get("id")
                ig_url = f"https://instagram.com/reel/{media_id}"
                logger.info(f"  ✅ Published to Instagram: {ig_url}")
                return {
                    "platform": "instagram",
                    "status": "published",
                    "media_id": media_id,
                    "url": ig_url,
                }
            else:
                logger.error(f"  ❌ Instagram publish failed: {publish_resp.text[:200]}")
                return {"platform": "instagram", "status": "error", "error": publish_resp.text[:200]}
        
        elif status == "ERROR":
            logger.error(f"  ❌ Instagram processing error")
            return {"platform": "instagram", "status": "error", "error": "Processing failed"}
        
        logger.info(f"  ⏳ Instagram processing... ({attempt + 1}/15)")
    
    return {"platform": "instagram", "status": "timeout", "container_id": container_id}


# ── Publish ledger (idempotency) ─────────────────────────────────────────────

LEDGER_PATH = Path("output/publish_logs/.published.json")

# Errors that no amount of retrying will fix: missing credentials or files.
# Everything else (network blips, 5xx, rate limits) is retried.
_NON_RETRYABLE_MARKERS = (
    "missing client secrets",
    "missing credentials",
    "credentials not configured",
    "video not found",
    "video too large",
)


def _video_fingerprint(video_path: str) -> str:
    """Stable identity for a rendered video: path + size + mtime.

    Not a content hash — hashing a 20 MB MP4 on every run is wasted I/O, and
    any re-encode changes size/mtime anyway.
    """
    p = Path(video_path).resolve()
    stat = p.stat()
    return f"{p}|{stat.st_size}|{int(stat.st_mtime)}"


def _load_ledger() -> dict:
    if not LEDGER_PATH.exists():
        return {}
    try:
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_ledger(ledger: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def _is_retryable(result: dict) -> bool:
    """False for configuration errors that will fail identically every time."""
    error = str(result.get("error", "")).lower()
    return not any(marker in error for marker in _NON_RETRYABLE_MARKERS)


# ── Main Publisher ───────────────────────────────────────────────────────────

def publish_video(
    video_path: str = None,
    platforms: list = None,
    dry_run: bool = False,
    force: bool = False,
) -> list:
    """
    Publish a video to specified platforms.

    Args:
        video_path: Path to video file. If None, finds latest in output/projects/
        platforms: List of platform names. If None, publishes to all configured.
        dry_run: If True, preview without actually publishing.
        force: Re-publish even if this exact video was already published.

    Returns:
        List of result dicts from each platform.

    IDEMPOTENCY: a video that already succeeded on a platform is skipped, so a
    duplicate scheduled run (or a manual run after the cron) cannot double-post
    the same file. Use force=True to override.
    """
    all_platforms = ["youtube", "tiktok", "instagram"]
    
    if not platforms:
        platforms = all_platforms
    
    # Find video
    if video_path:
        if not Path(video_path).exists():
            logger.error(f"Video not found: {video_path}")
            return []
        # Derive project_dir from the video's parent. Without this the rich
        # platform_metadata.json is never read for explicit paths and the
        # upload falls back to the generic placeholder title.
        video_file = Path(video_path).resolve()
        project_dir = video_file.parent
        manifest = project_dir / "manifest.json"
        video_info = {
            "video_path": str(video_file),
            "manifest_path": str(manifest) if manifest.exists() else None,
            "project_dir": str(project_dir),
        }
    else:
        video_info = find_latest_video()
    
    video_path = video_info["video_path"]
    logger.info(f"🎬 Video: {video_path}")
    
    # Load metadata
    metadata = load_video_metadata(video_info)
    logger.info(f"📝 Title: {metadata['title']}")
    
    # Publish to each platform
    results = []
    publishers = {
        "youtube": publish_youtube,
        "tiktok": publish_tiktok,
        "instagram": publish_instagram,
    }

    ledger = _load_ledger()
    fingerprint = _video_fingerprint(video_path)
    already = ledger.get(fingerprint, {})

    for platform in platforms:
        if platform not in publishers:
            logger.warning(f"Unknown platform: {platform}")
            continue

        # Idempotency: a duplicate scheduled run must not double-post.
        if not force and platform in already and not dry_run:
            prev = already[platform]
            logger.info(f"\n{'='*50}")
            logger.info(f"⏭️  {platform.upper()} already published for this video — skipping")
            logger.info(f"   {prev.get('url', '(no url recorded)')}")
            results.append({
                "platform": platform,
                "status": "already_published",
                "url": prev.get("url"),
                "video_id": prev.get("video_id"),
            })
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"📡 Publishing to {platform.upper()}")
        logger.info(f"{'='*50}")

        result = None
        for attempt in range(1, 4):
            try:
                result = publishers[platform](video_path, metadata, dry_run)
                if result.get("status") in ("published", "dry_run", "inbox"):
                    break

                # Configuration errors cannot be fixed by waiting: a missing
                # token or client-secret file will fail identically on every
                # attempt, and the backoff only delays the rest of the run.
                if not _is_retryable(result):
                    logger.error(f"  ❌ {platform}: not retrying — {result.get('error')}")
                    break

                if attempt < 3:
                    wait = 30 * attempt
                    logger.warning(f"  ⚠️ {platform} attempt {attempt}/3 failed, retrying in {wait}s...")
                    time.sleep(wait)
            except Exception as e:
                logger.error(f"  ❌ {platform} attempt {attempt}/3 exception: {e}")
                if attempt < 3:
                    wait = 30 * attempt
                    logger.warning(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    result = {"platform": platform, "status": "error", "error": str(e)}

        if result is None:
            result = {"platform": platform, "status": "error", "error": "All 3 attempts failed"}
        results.append(result)

        # Record only real successes — a failed platform must be retried on
        # the next run, not silently marked as done.
        if result.get("status") == "published" and not dry_run:
            already[platform] = {
                "url": result.get("url"),
                "video_id": result.get("video_id") or result.get("publish_id"),
                "published_at": datetime.now().isoformat(),
            }
            ledger[fingerprint] = already
            _save_ledger(ledger)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 PUBLISH SUMMARY")
    logger.info(f"{'='*50}")
    for r in results:
        status_icon = {
            "published": "✅", "dry_run": "🔍", "error": "❌",
            "timeout": "⏳", "inbox": "📥", "already_published": "⏭️",
        }.get(r["status"], "?")
        logger.info(f"  {status_icon} {r['platform']}: {r['status']}")
        if "url" in r:
            logger.info(f"     → {r['url']}")
        if "error" in r:
            logger.info(f"     → {r['error'][:100]}")
    
    # Save publish log
    log_dir = Path("output/publish_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"publish_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump({
            "video": video_path,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "metadata": metadata,
        }, f, indent=2)
    logger.info(f"\n📄 Publish log saved: {log_file}")
    
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geopolitical Sentinel — Video Publisher")
    parser.add_argument("--video", type=str, help="Path to specific video file")
    parser.add_argument("--platform", type=str, help="Platform(s): youtube,tiktok,instagram")
    parser.add_argument("--dry-run", action="store_true", help="Preview without publishing")
    parser.add_argument("--force", action="store_true",
                        help="Re-publish even if this video already succeeded")
    
    args = parser.parse_args()
    
    platforms = None
    if args.platform:
        platforms = [p.strip() for p in args.platform.split(",")]
    
    results = publish_video(
        video_path=args.video,
        platforms=platforms,
        dry_run=args.dry_run,
        force=args.force,
    )
    
    # Exit with error if any platform failed
    if any(r["status"] == "error" for r in results):
        sys.exit(1)