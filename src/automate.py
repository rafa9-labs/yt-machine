"""
Geopolitical Sentinel — Master Automation Script
Chains: Wake-on-LAN → wait for PC → run pipeline → Telegram notification → publish.

Usage:
    python automate.py                          # Full auto: wake PC, generate, notify
    python automate.py --generate               # Just generate + notify (PC already on)
    python automate.py --publish                # Generate + publish to all platforms
    python automate.py --publish youtube,tiktok # Generate + publish to specific platforms
    python automate.py --wake-only              # Just send WOL, don't generate
    python automate.py --schedule "08:00"       # Schedule for specific time
    python automate.py --install-schedule       # Install as Windows scheduled task

Environment:
    WOL_MAC           — Target PC MAC address (for WOL)
    WOL_BROADCAST     — Broadcast address (default: 255.255.255.255)
    WOL_PORT          — WOL UDP port (default: 9)
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID — For status notifications
"""

import os
import sys
import json
import time
import socket
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PROJECT_ROOT = Path(__file__).parent.resolve()
PIPELINE_SCRIPT = PROJECT_ROOT / "generate_complete_video.py"
LOG_DIR = PROJECT_ROOT / "output" / "logs"

WOL_WAIT_SECONDS = int(os.getenv("WOL_WAIT_SECONDS", "120"))
WOL_PING_HOST = os.getenv("WOL_PING_HOST", "localhost")
WOL_PING_PORT = int(os.getenv("WOL_PING_PORT", "11434"))
PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT", "900"))


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, level: str = "INFO"):
    line = f"{_timestamp()} [{level}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"automate_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def send_telegram_notification(text: str) -> bool:
    try:
        from tools.telegram_sender import send_message
        result = send_message(text)
        return result.get("success", False)
    except Exception as e:
        _log(f"Telegram notification failed: {e}", "WARN")
        return False


def wake_pc() -> bool:
    _log("Sending Wake-on-LAN packet...")
    from tools.wake_pc import send_wol
    result = send_wol()
    if not result["success"]:
        _log(f"WOL failed: {result['error']}", "ERROR")
        return False
    _log(f"WOL sent to {result['mac']} ({result['packets_sent']} packets)")
    return True


def wait_for_pc(host: str = None, port: int = None, timeout: int = None) -> bool:
    host = host or WOL_PING_HOST
    port = port or WOL_PING_PORT
    timeout = timeout or WOL_WAIT_SECONDS

    _log(f"Waiting for {host}:{port} to come online (timeout: {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((host, port))
            sock.close()
            elapsed = int(time.time() - start)
            _log(f"PC is online! ({elapsed}s)")
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            elapsed = int(time.time() - start)
            if elapsed % 15 == 0:
                _log(f"Still waiting... ({elapsed}/{timeout}s)")
            time.sleep(3)

    _log(f"PC did not come online within {timeout}s", "ERROR")
    return False


def is_pc_online(host: str = None, port: int = None) -> bool:
    host = host or WOL_PING_HOST
    port = port or WOL_PING_PORT
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect((host, port))
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def run_pipeline(
    skip_images: bool = False,
    no_telegram: bool = False,
    resume: str = None,
) -> dict:
    cmd = [sys.executable, str(PIPELINE_SCRIPT)]
    if skip_images:
        cmd.append("--skip-images")
    if no_telegram:
        cmd.append("--no-telegram")
    if resume:
        cmd.extend(["--resume", resume])

    _log(f"Starting pipeline: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PIPELINE_TIMEOUT,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace",
        )
        elapsed = int(time.time() - start_time)

        if result.returncode == 0:
            _log(f"Pipeline completed successfully in {elapsed}s")
            return {
                "success": True,
                "elapsed_seconds": elapsed,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1000:] if result.stdout else "",
            }
        else:
            _log(f"Pipeline failed (exit {result.returncode}) after {elapsed}s", "ERROR")
            _log(f"stderr: {result.stderr[-500:]}" if result.stderr else "No stderr", "ERROR")
            return {
                "success": False,
                "elapsed_seconds": elapsed,
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            }

    except subprocess.TimeoutExpired:
        elapsed = int(time.time() - start_time)
        _log(f"Pipeline timed out after {elapsed}s (limit: {PIPELINE_TIMEOUT}s)", "ERROR")
        return {"success": False, "elapsed_seconds": elapsed, "error": "timeout"}
    except Exception as e:
        _log(f"Pipeline exception: {e}", "ERROR")
        return {"success": False, "error": str(e)}


def find_latest_video() -> str | None:
    projects_dir = PROJECT_ROOT / "output" / "projects"
    if not projects_dir.exists():
        return None

    latest = None
    latest_mtime = 0

    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        for mp4 in project_dir.glob("*.mp4"):
            if "TEMP" in mp4.name:
                continue
            mtime = mp4.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = str(mp4)

    return latest


def publish_video(video_path: str = None, platforms: list = None) -> list:
    _log(f"Publishing video to {', '.join(platforms or ['all platforms'])}...")
    try:
        from publish_video import publish_video as do_publish
        results = do_publish(video_path=video_path, platforms=platforms)
        for r in results:
            status = r.get("status", "unknown")
            platform = r.get("platform", "?")
            icon = "OK" if status == "published" else "FAIL"
            _log(f"  [{icon}] {platform}: {status}")
            if r.get("url"):
                _log(f"       -> {r['url']}")
        return results
    except Exception as e:
        _log(f"Publish failed: {e}", "ERROR")
        return [{"status": "error", "error": str(e)}]


def schedule_task(run_time: str = None) -> bool:
    task_name = "GeopoliticalSentinel_DailyVideo"
    script_path = str(PROJECT_ROOT / "automate.py")

    cmd = [
        "schtasks", "/Create",
        "/TN", task_name,
        "/TR", f'"{sys.executable}" "{script_path}" --generate',
        "/SC", "DAILY",
        "/F",
    ]

    if run_time:
        cmd.extend(["/ST", run_time.replace(":", "")])
    else:
        cmd.extend(["/ST", "0800"])

    _log(f"Creating scheduled task: {task_name}")
    _log(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            _log(f"Scheduled task created: {task_name}")
            return True
        else:
            _log(f"schtasks failed: {result.stderr}", "ERROR")
            return False
    except Exception as e:
        _log(f"Failed to create scheduled task: {e}", "ERROR")
        return False


def remove_scheduled_task() -> bool:
    task_name = "GeopoliticalSentinel_DailyVideo"
    _log(f"Removing scheduled task: {task_name}")
    try:
        result = subprocess.run(
            ["schtasks", "/Delete", "/TN", task_name, "/F"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            _log("Scheduled task removed")
            return True
        else:
            _log(f"schtasks delete failed: {result.stderr}", "ERROR")
            return False
    except Exception as e:
        _log(f"Failed to remove scheduled task: {e}", "ERROR")
        return False


def run_full_automation(
    wake: bool = True,
    publish: bool = False,
    platforms: list = None,
    skip_images: bool = False,
) -> dict:
    _log("=" * 60)
    _log("GEOPOLITICAL SENTINEL — AUTOMATION START")
    _log("=" * 60)

    report = {
        "started_at": datetime.now().isoformat(),
        "wake": None,
        "pipeline": None,
        "publish": None,
        "telegram_notifications": [],
    }

    # Phase 1: Wake PC (if requested and needed)
    if wake:
        if is_pc_online():
            _log("PC already online, skipping WOL")
            report["wake"] = {"status": "already_online"}
        else:
            wol_ok = wake_pc()
            if wol_ok:
                pc_online = wait_for_pc()
                report["wake"] = {"status": "online" if pc_online else "timeout"}
                if not pc_online:
                    send_telegram_notification("Pipeline FAILED: PC did not wake up")
                    report["telegram_notifications"].append("wol_timeout_alert")
                    _log("Aborting: PC not online", "ERROR")
                    return report
            else:
                report["wake"] = {"status": "wol_failed"}
                send_telegram_notification("Pipeline FAILED: Could not send WOL packet")
                report["telegram_notifications"].append("wol_failed_alert")
                return report
    else:
        report["wake"] = {"status": "skipped"}

    # Phase 2: Notify generation starting
    send_telegram_notification("Video generation started...")
    report["telegram_notifications"].append("started")

    # Phase 3: Run pipeline
    pipeline_result = run_pipeline(skip_images=skip_images)
    report["pipeline"] = pipeline_result

    if pipeline_result["success"]:
        video_path = find_latest_video()
        elapsed = pipeline_result.get("elapsed_seconds", 0)
        _log(f"Video ready: {video_path} ({elapsed}s)")

        send_telegram_notification(
            f"Video generated successfully in {elapsed}s! "
            f"Delivery via pipeline Telegram integration."
        )
        report["telegram_notifications"].append("success")

        # Phase 4: Publish (optional)
        if publish and video_path:
            pub_results = publish_video(video_path, platforms)
            report["publish"] = pub_results

            published = [r for r in pub_results if r.get("status") == "published"]
            if published:
                urls = "\n".join(f"  - {r['platform']}: {r.get('url', 'N/A')}" for r in published)
                send_telegram_notification(f"Published to {len(published)} platform(s):\n{urls}")
                report["telegram_notifications"].append("published")
    else:
        error_msg = pipeline_result.get("stderr_tail", pipeline_result.get("error", "Unknown"))
        send_telegram_notification(f"Pipeline FAILED: {error_msg[:200]}")
        report["telegram_notifications"].append("failure_alert")

    # Final report
    report["completed_at"] = datetime.now().isoformat()
    _log("=" * 60)
    _log("AUTOMATION COMPLETE")
    _log(f"Pipeline: {'SUCCESS' if pipeline_result['success'] else 'FAILED'}")
    _log("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Geopolitical Sentinel — Master Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automate.py                            # Wake PC, generate, Telegram notify
  python automate.py --generate                 # Generate only (PC already on)
  python automate.py --publish                  # Generate + publish to all platforms
  python automate.py --publish youtube          # Generate + publish to YouTube only
  python automate.py --wake-only                # Just send WOL packet
  python automate.py --schedule "08:00"         # Install daily task at 8 AM
  python automate.py --remove-schedule          # Remove scheduled task
        """,
    )
    parser.add_argument("--generate", action="store_true",
                        help="Just generate video (skip WOL)")
    parser.add_argument("--publish", nargs="?", const="all", default=None,
                        help="Generate + publish. Optionally specify platforms (youtube,tiktok,instagram)")
    parser.add_argument("--skip-images", action="store_true",
                        help="Use placeholder images (faster testing)")
    parser.add_argument("--wake-only", action="store_true",
                        help="Only send WOL packet, don't generate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume pipeline from project folder")
    parser.add_argument("--schedule", type=str, metavar="HH:MM",
                        help="Install as Windows scheduled task at given time")
    parser.add_argument("--install-schedule", action="store_true",
                        help="Install as Windows scheduled task (default 08:00)")
    parser.add_argument("--remove-schedule", action="store_true",
                        help="Remove the Windows scheduled task")
    parser.add_argument("--no-notify", action="store_true",
                        help="Skip Telegram status notifications")

    args = parser.parse_args()

    if args.wake_only:
        ok = wake_pc()
        sys.exit(0 if ok else 1)

    if args.remove_schedule:
        ok = remove_scheduled_task()
        sys.exit(0 if ok else 1)

    if args.install_schedule or args.schedule:
        run_time = args.schedule or "08:00"
        ok = schedule_task(run_time)
        sys.exit(0 if ok else 1)

    do_wake = not args.generate and args.publish is None
    do_publish = args.publish is not None
    platforms = None
    if args.publish and args.publish != "all":
        platforms = [p.strip() for p in args.publish.split(",")]

    if args.generate and not do_publish:
        _log("Generate-only mode (no WOL, no publish)")
        result = run_pipeline(skip_images=args.skip_images)
        if result["success"]:
            send_telegram_notification("Video generated successfully!")
        else:
            send_telegram_notification(f"Pipeline FAILED: {result.get('stderr_tail', 'Unknown')[:200]}")
        sys.exit(0 if result["success"] else 1)

    report = run_full_automation(
        wake=do_wake,
        publish=do_publish,
        platforms=platforms,
        skip_images=args.skip_images,
    )

    pipeline_ok = report.get("pipeline", {}).get("success", False)
    sys.exit(0 if pipeline_ok else 1)


if __name__ == "__main__":
    main()
