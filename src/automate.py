"""
Geopolitical Sentinel — Master Automation Script
Chains: (optional Wake-on-LAN) → run pipeline → Telegram notification → publish.

Usage:
    python automate.py                          # Full auto: generate, notify
    python automate.py --generate               # Just generate + notify
    python automate.py --publish                # Generate + publish to all platforms
    python automate.py --publish youtube,tiktok # Generate + publish to specific platforms
    python automate.py --install-schedule "06:00"  # Install launchd agent (macOS)
    python automate.py --install-wake           # Schedule a daily pmset wake (needs sudo)
    python automate.py --show-schedule          # Inspect installed schedule
    python automate.py --remove-schedule        # Remove the launchd agent

Environment:
    WOL_MAC           — Target PC MAC address (optional; only for remote-WOL setups)
    WOL_BROADCAST     — Broadcast address (default: 255.255.255.255)
    WOL_PORT          — WOL UDP port (default: 9)
    RUN_TIME          — Daily run time for launchd (default: 06:00)
    WAKE_TIME         — Daily pmset wake time (default: 05:50:00)
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

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")

# `from tools.*` and `from publish_video import ...` both need explicit roots:
# this file lives in src/, so neither the repo root nor src/ is on sys.path
# when launched as `python src/automate.py`.
for _root in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _root not in sys.path:
        sys.path.insert(0, _root)

PIPELINE_SCRIPT = PROJECT_ROOT / "tools" / "generate_complete_video.py"
RUN_DAILY_SCRIPT = PROJECT_ROOT / "tools" / "run_daily.sh"
LOG_DIR = PROJECT_ROOT / "output" / "logs"

WOL_WAIT_SECONDS = int(os.getenv("WOL_WAIT_SECONDS", "120"))
WOL_PING_HOST = os.getenv("WOL_PING_HOST", "localhost")
WOL_PING_PORT = int(os.getenv("WOL_PING_PORT", "11434"))
# A full run is dominated by eight image generations plus LLM/TTS/assembly.
# The old 900s default killed healthy runs mid-way; keep the automation and
# API ceilings aligned for the same pipeline.
PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT", "14400"))

# Empty on a standalone Mac: there is no remote PC to wake, so the wake phase
# is skipped instead of aborting the run. Set this only for remote-WOL setups.
WOL_MAC = os.getenv("WOL_MAC", "").strip()

# macOS launchd agent identity for the daily run.
LAUNCHD_LABEL = os.getenv("LAUNCHD_LABEL", "com.rafa9labs.ytmachine")
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
# Wake schedule — pmset fires this before the launchd job so model loading
# has network/disk warmed and the job does not start from a cold standby.
WAKE_TIME = os.getenv("WAKE_TIME", "05:50:00")
RUN_TIME = os.getenv("RUN_TIME", "06:00")

# launchd gives children a minimal PATH; ffmpeg/llama-server/mlx_lm.server/
# ollama all live in /opt/homebrew/bin on Apple Silicon.
LAUNCHD_PATH = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, level: str = "INFO"):
    line = f"{_timestamp()} [{level}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"automate_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Telegram is optional. When --no-notify is passed (or no credentials are
# configured) notifications are suppressed instead of failing per call.
NOTIFY_ENABLED = True


def send_telegram_notification(text: str) -> bool:
    if not NOTIFY_ENABLED:
        return False
    try:
        from tools.telegram_sender import send_message
        result = send_message(text)
        if not result.get("success"):
            _log(f"Telegram notification skipped: {result.get('error')}", "WARN")
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


def _pipeline_busy() -> bool:
    """True when another process holds the pipeline lock.

    The lock file is written by tools/generate_complete_video.py. We check
    it here so a scheduled run can exit cleanly instead of racing a manual
    run and then dying with a confusing exit code.
    """
    try:
        from src.models.runtime import PipelineLock
        lock = PipelineLock()
        if not lock.path.exists():
            return False
        owner = lock._read_owner() or {}
        if lock._is_stale(owner):
            return False
        return True
    except Exception:
        return False


def run_pipeline(
    skip_images: bool = False,
    no_telegram: bool = False,
    resume: str = None,
) -> dict:
    if _pipeline_busy():
        _log("Pipeline lock is held by another run — refusing to start a second job", "WARN")
        return {"success": False, "error": "Another pipeline run is already active"}

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
        # start_new_session=True puts the pipeline and its model server into
        # one process group. On timeout we kill the whole group — otherwise
        # a runaway llama-server (20+ GB) survives the timeout and the next
        # run OOMs on a 32 GiB machine.
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=PIPELINE_TIMEOUT)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            stdout, stderr = "", ""
            elapsed = int(time.time() - start_time)
            _log(f"Pipeline timed out after {elapsed}s (limit: {PIPELINE_TIMEOUT}s)", "ERROR")
            return {"success": False, "elapsed_seconds": elapsed, "error": "timeout"}

        elapsed = int(time.time() - start_time)

        if process.returncode == 0:
            _log(f"Pipeline completed successfully in {elapsed}s")
            return {
                "success": True,
                "elapsed_seconds": elapsed,
                "returncode": process.returncode,
                "stdout_tail": stdout[-1000:] if stdout else "",
            }

        _log(f"Pipeline failed (exit {process.returncode}) after {elapsed}s", "ERROR")
        _log(f"stderr: {stderr[-500:]}" if stderr else "No stderr", "ERROR")
        return {
            "success": False,
            "elapsed_seconds": elapsed,
            "returncode": process.returncode,
            "stderr_tail": stderr[-500:] if stderr else "",
        }

    except Exception as e:
        _log(f"Pipeline exception: {e}", "ERROR")
        return {"success": False, "error": str(e)}


def _signal_group_or_process(pid: int, sig: int) -> None:
    """Signal a child's process group — never our own.

    `os.killpg(os.getpgid(pid), ...)` is safe only when the target group
    differs from ours. If it ever resolves to our group, the call would
    kill this process mid-shutdown with no error. Compare first, refuse on
    a match, and fall back to the single pid.
    """
    try:
        target_pgid = os.getpgid(pid)
    except ProcessLookupError:
        return

    if target_pgid == os.getpgrp():
        _log(f"Refusing killpg — group {target_pgid} is our own; signalling pid only", "WARN")
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        return

    try:
        os.killpg(target_pgid, sig)
    except Exception:
        pass


def _kill_process_group(process: subprocess.Popen) -> None:
    """Terminate a pipeline process group, escalating to SIGKILL."""
    import signal as _signal

    _signal_group_or_process(process.pid, _signal.SIGTERM)

    try:
        process.wait(timeout=20)
        return
    except Exception:
        pass

    _signal_group_or_process(process.pid, _signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except Exception:
        pass


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
        icons = {
            "published": "OK",
            "already_published": "SKIP",
            "dry_run": "DRY",
            "inbox": "INBOX",
        }
        for r in results:
            status = r.get("status", "unknown")
            platform = r.get("platform", "?")
            icon = icons.get(status, "FAIL")
            _log(f"  [{icon}] {platform}: {status}")
            if r.get("url"):
                _log(f"       -> {r['url']}")
        return results
    except Exception as e:
        _log(f"Publish failed: {e}", "ERROR")
        return [{"status": "error", "error": str(e)}]


def _plist_xml(run_time: str) -> str:
    """Render the launchd agent that runs the daily pipeline.

    StartCalendarInterval (not StartInterval) is what lets launchd run the
    job on a wall-clock schedule after the pmset wake, instead of counting
    from the moment the agent was loaded.
    """
    hour, minute = _parse_hhmm(run_time)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>{RUN_DAILY_SCRIPT}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{PROJECT_ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{LAUNCHD_PATH}</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{LOG_DIR / "launchd.out.log"}</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR / "launchd.err.log"}</string>
    <key>RunAtLoad</key>
    <false/>
    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
"""


def _parse_hhmm(run_time: str) -> tuple:
    try:
        parts = run_time.strip().split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
        return hour, minute
    except (ValueError, IndexError):
        _log(f"Invalid time '{run_time}', falling back to {RUN_TIME}", "WARN")
        return _parse_hhmm(RUN_TIME)


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *args], capture_output=True, text=True, timeout=30
    )


def schedule_task(run_time: str = None) -> bool:
    """Install/replace the launchd agent that runs the pipeline daily.

    WHY launchd, not cron: cron cannot run a job that was missed while the
    Mac slept, and it has no wake primitive. launchd reads
    StartCalendarInterval and fires the job when the machine is up; pmset
    supplies the actual wake (see install_wake_schedule).
    """
    run_time = run_time or RUN_TIME

    # Never overwrite a plist without unloading first — launchd keeps the
    # old schedule in memory and the bootout/bootstrap below is what makes
    # the edit take effect.
    remove_scheduled_task(quiet=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)

    hour, minute = _parse_hhmm(run_time)
    LAUNCHD_PLIST.write_text(_plist_xml(run_time), encoding="utf-8")

    uid = os.getuid()
    result = _launchctl("bootstrap", f"gui/{uid}", str(LAUNCHD_PLIST))
    if result.returncode != 0:
        # bootstrap fails when the label is already loaded under a different
        # path; one retry after a bootout clears that state.
        _launchctl("bootout", f"gui/{uid}/{LAUNCHD_LABEL}")
        result = _launchctl("bootstrap", f"gui/{uid}", str(LAUNCHD_PLIST))

    if result.returncode != 0:
        _log(f"launchctl bootstrap failed: {result.stderr.strip()}", "ERROR")
        return False

    _launchctl("enable", f"gui/{uid}/{LAUNCHD_LABEL}")
    _log(f"launchd agent installed: {LAUNCHD_PLIST}")
    _log(f"Daily run scheduled for {hour:02d}:{minute:02d}")
    _log("Next: run 'python src/automate.py --install-wake' so the Mac wakes before the run")
    return True


def remove_scheduled_task(quiet: bool = False) -> bool:
    uid = os.getuid()
    _launchctl("bootout", f"gui/{uid}/{LAUNCHD_LABEL}")

    if LAUNCHD_PLIST.exists():
        LAUNCHD_PLIST.unlink()
        if not quiet:
            _log(f"Removed launchd agent: {LAUNCHD_PLIST}")
        return True

    if not quiet:
        _log("No launchd agent was installed")
    return True


def install_wake_schedule(wake_time: str = None) -> bool:
    """Register a recurring pmset wake before the daily run.

    WHY THIS NEEDS sudo: pmset schedules are stored in the SMC/PMU and can
    only be written as root. Run is surfaced to the user instead of a failed
    subprocess, because a password prompt in launchd context would hang.
    """
    import sys as _sys

    if _sys.platform != "darwin":
        _log("Wake scheduling is macOS-only (pmset)", "ERROR")
        return False

    wake_time = wake_time or WAKE_TIME
    _parse_hhmm(wake_time)  # validate & normalise

    cmd = ["pmset", "repeat", "wakeorpoweron", "MTWRFSU", wake_time]
    _log(f"Wake schedule command: sudo {' '.join(cmd)}")

    result = subprocess.run(
        ["sudo", "-n", *cmd], capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        _log(f"Mac will wake daily at {wake_time}")
        return True

    if "password" in (result.stderr or "").lower():
        _log("sudo needs a password — run this once in a terminal:", "WARN")
        _log(f"  sudo {' '.join(cmd)}")
        return False

    _log(f"pmset failed: {result.stderr.strip()}", "ERROR")
    return False


def remove_wake_schedule() -> bool:
    """Cancel the recurring wake schedule (idempotent)."""
    result = subprocess.run(
        ["sudo", "-n", "pmset", "repeat", "cancel"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        _log("Wake schedule cancelled")
        return True
    if "password" in (result.stderr or "").lower():
        _log("Run once manually: sudo pmset repeat cancel", "WARN")
        return False
    _log(f"pmset cancel failed: {result.stderr.strip()}", "ERROR")
    return False


def configure_power(idle_sleep_min: int = None) -> bool:
    """Set the AC idle-sleep timer so the Mac sleeps after the run.

    This Mac currently ships with AC `sleep 0` (never idle-sleep), which would
    leave it awake all day after the 05:50 wake. A short AC idle timer closes
    the loop: wake → run → publish → idle → sleep.

    Battery sleep is deliberately left alone — on battery the default is
    already short, and shortening it further can interrupt manual work.
    """
    idle_sleep_min = idle_sleep_min or int(os.getenv("IDLE_SLEEP_MIN", "20"))
    cmds = [
        ["pmset", "-c", "sleep", str(idle_sleep_min)],
        ["pmset", "-c", "displaysleep", str(min(idle_sleep_min, 10))],
    ]

    for cmd in cmds:
        full = ["sudo", "-n", *cmd]
        result = subprocess.run(full, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            if "password" in (result.stderr or "").lower():
                _log("sudo needs a password — run these once in a terminal:", "WARN")
                for c in cmds:
                    _log(f"  sudo {' '.join(c)}")
                return False
            _log(f"pmset failed: {' '.join(cmd)}: {result.stderr.strip()}", "ERROR")
            return False

    _log(f"AC idle sleep set to {idle_sleep_min}m "
         f"(display sleep {min(idle_sleep_min, 10)}m)")
    _log("Mac will sleep after the pipeline finishes and the idle timer elapses")
    return True


def show_power() -> None:
    """Print the power settings that govern the daily wake/sleep cycle."""
    result = subprocess.run(
        ["pmset", "-g", "custom"], capture_output=True, text=True, timeout=10
    )
    section = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("AC Power"):
            section = "AC"
            print("AC Power:")
        elif stripped.startswith("Battery Power"):
            section = "Battery"
            print("Battery Power:")
        elif section and ("sleep" in stripped or "womp" in stripped
                          or "powernap" in stripped or "standby" in stripped):
            print(f"  {stripped}")



def show_schedule() -> None:
    """Print the current launchd agent state and pmset wake schedule."""
    print(f"launchd label : {LAUNCHD_LABEL}")
    print(f"plist         : {LAUNCHD_PLIST} "
          f"({'present' if LAUNCHD_PLIST.exists() else 'missing'})")

    listed = _launchctl("list", LAUNCHD_LABEL)
    if listed.returncode == 0:
        # `launchctl list <label>` reports state; the last exit code only
        # appears in `launchctl print gui/<uid>/<label>`.
        exit_status = "n/a"
        printed = _launchctl("print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}")
        if printed.returncode == 0:
            for line in printed.stdout.splitlines():
                if "last exit code" in line.lower():
                    exit_status = line.split("=")[-1].strip()
                    break
        print(f"launchd state : loaded (last exit code {exit_status})")
    else:
        print("launchd state : not loaded")

    sched = subprocess.run(
        ["pmset", "-g", "sched"], capture_output=True, text=True, timeout=10
    )
    wake_lines = [l for l in sched.stdout.splitlines() if "wake" in l.lower()]
    print("pmset wake    : " + ("\n                ".join(wake_lines) or "none scheduled"))


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

    # Phase 1: Wake check (only meaningful for remote-WOL setups — on this Mac
    # the machine is already awake because pmset woke it before launchd ran).
    if wake and not WOL_MAC:
        _log("No WOL_MAC configured — running locally without a wake step")
        report["wake"] = {"status": "local"}
    elif wake:
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
  python automate.py                              # Full auto: generate + notify
  python automate.py --publish youtube,tiktok     # Generate + publish (daily job)
  python automate.py --generate                   # Generate only, no publish
  python automate.py --install-schedule "06:00"   # Install launchd agent (macOS)
  python automate.py --install-wake               # Schedule a pmset wake
  python automate.py --show-schedule              # Inspect installed schedule
  python automate.py --remove-schedule            # Remove launchd agent
        """,
    )
    parser.add_argument("--generate", action="store_true",
                        help="Just generate video (skip wake + publish)")
    parser.add_argument("--publish", nargs="?", const="all", default=None,
                        help="Generate + publish. Optionally specify platforms (youtube,tiktok,instagram)")
    parser.add_argument("--skip-images", action="store_true",
                        help="Use placeholder images (faster testing)")
    parser.add_argument("--wake-only", action="store_true",
                        help="Only send the Wake-on-LAN packet, don't generate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume pipeline from project folder")
    parser.add_argument("--schedule", type=str, metavar="HH:MM",
                        help="Install the launchd agent at the given daily time")
    parser.add_argument("--install-schedule", nargs="?", const=RUN_TIME, default=None,
                        metavar="HH:MM",
                        help=f"Install the launchd agent (default {RUN_TIME})")
    parser.add_argument("--remove-schedule", action="store_true",
                        help="Remove the launchd agent")
    parser.add_argument("--install-wake", nargs="?", const=WAKE_TIME, default=None,
                        metavar="HH:MM:SS",
                        help=f"Schedule a daily pmset wake (default {WAKE_TIME}; needs sudo)")
    parser.add_argument("--remove-wake", action="store_true",
                        help="Cancel the recurring pmset wake schedule")
    parser.add_argument("--configure-power", nargs="?", const=20, default=None,
                        type=int, metavar="MINUTES",
                        help="Set the AC idle-sleep timer (default 20m; needs sudo)")
    parser.add_argument("--show-power", action="store_true",
                        help="Show AC/battery sleep settings")
    parser.add_argument("--show-schedule", action="store_true",
                        help="Show launchd agent state and pmset wake schedule")
    parser.add_argument("--no-notify", action="store_true",
                        help="Skip Telegram status notifications")

    args = parser.parse_args()

    global NOTIFY_ENABLED
    if args.no_notify:
        NOTIFY_ENABLED = False
        _log("Telegram notifications disabled (--no-notify)")

    if args.wake_only:
        ok = wake_pc()
        sys.exit(0 if ok else 1)

    if args.show_schedule:
        show_schedule()
        sys.exit(0)

    if args.show_power:
        show_power()
        sys.exit(0)

    if args.configure_power is not None:
        ok = configure_power(args.configure_power)
        sys.exit(0 if ok else 1)

    if args.remove_wake:
        ok = remove_wake_schedule()
        sys.exit(0 if ok else 1)

    if args.install_wake is not None:
        ok = install_wake_schedule(args.install_wake)
        sys.exit(0 if ok else 1)

    if args.remove_schedule:
        ok = remove_scheduled_task()
        sys.exit(0 if ok else 1)

    if args.install_schedule is not None:
        ok = schedule_task(args.install_schedule)
        sys.exit(0 if ok else 1)

    if args.schedule:
        ok = schedule_task(args.schedule)
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
