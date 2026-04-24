"""
Rerun: Re-generate TTS + video from an existing project.
Updates the closing text to the new Masker outro, reuses everything else.
"""
import sys
import os
import json
import time
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, ".")
from video_server.tts_tool import generate_voiceover
from video_server.split_video_assembler import build_split_video

SOURCE_PROJECT = sys.argv[1] if len(sys.argv) > 1 else "output/projects/video_1776830311"
source = Path(SOURCE_PROJECT)
if not source.exists():
    print(f"ERROR: {source} not found")
    sys.exit(1)

NEW_CLOSING = "Stay behind the curtain. Subscribe, like. And if I don't see you. Good morning. Good afternoon. And good night."

print(f"[RERUN] Source project: {source}")

# Load original script
with open(source / "script_segments.json", "r", encoding="utf-8") as f:
    script = json.load(f)

full_text = script.get("full_text", "")
if not full_text:
    print("[RERUN] ERROR: No full_text in script")
    sys.exit(1)

# Strip old closing and inject new one
stripped = re.sub(
    r'(\.{3,4}|\.*)\s*(Stay behind|Subscribe|These were|And these were|And if I don\'t see you|good MORNING|good morning).*$',
    '', full_text, flags=re.IGNORECASE | re.DOTALL
).rstrip()

# Also strip any trailing subscribe patterns
stripped = re.sub(
    r'\s*(?:Subscribe|subscribe)[,.\s]*(?:like|and\s+like|hit\s+like)?[,.\s]*(?:share|comment|follow)?\s*.*$',
    '', stripped, flags=re.IGNORECASE | re.DOTALL
).rstrip()

new_full_text = stripped + " .... " + NEW_CLOSING
print(f"[RERUN] Original: {len(full_text.split())} words")
print(f"[RERUN] Updated:  {len(new_full_text.split())} words")
print(f"[RERUN] New closing: {NEW_CLOSING}")

# Copy images
source_images = source / "images"
existing_images = sorted(
    list(source_images.glob("*.png")) + list(source_images.glob("*.jpg")),
    key=lambda p: p.name
)
if len(existing_images) < 6:
    print(f"ERROR: Only {len(existing_images)} images found, need 6")
    sys.exit(1)
print(f"[RERUN] Found {len(existing_images)} images")

# Create new project
timestamp = int(time.time())
new_folder = Path(f"output/projects/video_{timestamp}")
new_folder.mkdir(parents=True, exist_ok=True)
(new_folder / "images").mkdir(exist_ok=True)
print(f"[RERUN] New project: {new_folder}")

generated_images = []
for img in existing_images[:6]:
    dst = new_folder / "images" / f"reused_{img.name}"
    shutil.copy2(str(img), str(dst))
    generated_images.append(str(dst))

# Update script with new closing
script["full_text"] = new_full_text

# Update segment_timeline closing
if "segment_timeline" in script:
    for seg in script["segment_timeline"]:
        if seg.get("label") == "closing":
            seg["text"] = NEW_CLOSING
            print(f"[RERUN] Updated closing segment: {NEW_CLOSING[:50]}...")

# Save updated script
script_file = new_folder / "script_segments.json"
script_file.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
script_txt = new_folder / "script.txt"
script_txt.write_text(new_full_text, encoding="utf-8")
script_curated = new_folder / "script_curated.txt"
script_curated.write_text(new_full_text, encoding="utf-8")

# TTS
print("[RERUN] Generating voiceover with updated closing...")
tts_result = generate_voiceover(new_full_text, "authoritative")
if not tts_result or not tts_result.get("success"):
    print("[RERUN] ERROR: TTS failed")
    sys.exit(1)

voice_file = tts_result["path"]
word_timestamps = tts_result.get("word_timestamps", [])
voice_duration = tts_result.get("estimated_duration_seconds", 0)
print(f"[RERUN] Voice: {voice_duration}s, {len(word_timestamps)} word timestamps")

# Copy voiceover
voice_dest = new_folder / "voiceover.mp3"
shutil.copy2(voice_file, str(voice_dest))

# Get hook text
hook_text = ""
checkpoint_file = source / "checkpoint.json"
if checkpoint_file.exists():
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    for t in ckpt.get("analysis_topics", []):
        if t:
            hook_text = t
            break

# Video assembly
print("[RERUN] Assembling video...")
video_output = str(new_folder / f"video_{timestamp}.mp4")

assembly_result = build_split_video(
    audio_path=str(voice_dest),
    image_paths=generated_images[:6],
    output_path=video_output,
    script_text=new_full_text,
    word_timestamps=word_timestamps,
    hook_text=hook_text,
    scene_timestamps=None,
)

if assembly_result and assembly_result.get("success"):
    final_path = assembly_result.get("path", video_output)
    size_mb = Path(final_path).stat().st_size / 1024 / 1024
    print(f"\n[RERUN] DONE! Video: {final_path} ({size_mb:.1f}MB)")
    print(f"[RERUN] Duration: {voice_duration}s")
else:
    error = assembly_result.get("error", "unknown") if assembly_result else "no result"
    print(f"[RERUN] Assembly failed: {error}")
