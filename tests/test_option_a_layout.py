"""
Test Option A Layout: Rebuild video using existing project assets.
No fal credits needed — reuses images and re-generates TTS (free with Kokoro/edge-tts).
Tests the new 60/40 split layout (1152px image / 768px avatar).
"""
import sys
import json
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Use the latest project with existing assets
PROJECT = Path("output/projects/video_1775240601")
IMAGES_DIR = PROJECT / "images"
AUDIO_PATH = PROJECT / "voiceover.mp3"
SCRIPT_PATH = PROJECT / "script.txt"
SCRIPT_SEGMENTS = PROJECT / "script_segments.json"

print("🎬 TEST VIDEO — Option A Layout (60/40 split)")
print("=" * 60)

# 1. Load existing assets
print("\n📁 Loading existing assets...")
script_text = SCRIPT_PATH.read_text(encoding='utf-8').strip()
print(f"  Script: {len(script_text)} chars, {len(script_text.split())} words")

# Find PNG images in images subdirectory
image_files = sorted(IMAGES_DIR.glob("*.png"))
print(f"  Images: {len(image_files)} found")
for img in image_files:
    print(f"    - {img.name}")

if not image_files:
    print("❌ No images found!")
    sys.exit(1)

if not AUDIO_PATH.exists():
    print("❌ No voiceover.mp3 found!")
    sys.exit(1)
print(f"  Audio: {AUDIO_PATH.name}")

# Load hook text from segments
hook_text = ""
if SCRIPT_SEGMENTS.exists():
    segments = json.loads(SCRIPT_SEGMENTS.read_text(encoding='utf-8'))
    hook_raw = segments.get('greeting', segments.get('hook', ''))
    if isinstance(hook_raw, dict):
        hook_text = hook_raw.get('text', hook_raw.get('narration', str(hook_raw)))
    elif isinstance(hook_raw, str):
        hook_text = hook_raw
print(f"  Hook: {hook_text[:80]}...")

# 2. Re-generate TTS to get word timestamps (FREE)
print("\n🎤 Re-generating TTS for word timestamps (free)...")
from video_server.tts_tool import generate_voiceover

tts_result = generate_voiceover(script_text, "authoritative")
if not tts_result.get('success'):
    print(f"❌ TTS failed: {tts_result.get('error')}")
    sys.exit(1)

word_timestamps = tts_result.get('word_timestamps', [])
print(f"  ✅ Word timestamps: {len(word_timestamps)} words")
if word_timestamps:
    print(f"  First word: {word_timestamps[0]}")
    print(f"  Last word: {word_timestamps[-1]}")

# Use the new TTS audio (better sync with timestamps)
new_audio = Path(tts_result['path'])
print(f"  Audio path: {new_audio}")

# 3. Build video with Option A layout
print("\n🎬 Building video with Option A layout (60/40)...")
from video_server.split_video_assembler import build_split_video

output_path = str(PROJECT / "video_option_a_test.mp4")
image_paths = [str(p) for p in image_files]

result = build_split_video(
    audio_path=str(new_audio),
    image_paths=image_paths,
    output_path=output_path,
    script_text=script_text,
    word_timestamps=word_timestamps,
    hook_text=hook_text,
)

if result.get('success'):
    print(f"\n✅ VIDEO CREATED: {result.get('path')}")
    print(f"  Duration: {result.get('duration_seconds')}s")
    print(f"  Size: {result.get('file_size_mb')}MB")
    print(f"  Resolution: {result.get('resolution')}")
    print(f"  Scenes: {result.get('scenes')}")
    print(f"  Subtitles: {result.get('subtitles')} clips")
    print(f"  Effects: {', '.join(result.get('effects_applied', []))}")
else:
    print(f"\n❌ Video build failed: {result.get('error')}")