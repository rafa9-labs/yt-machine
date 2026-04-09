"""
Test script for image generation + TTS + assembler improvements.
Uses existing images (free) to verify the new pipeline.

Tests:
1. TTS: am_adam voice with comedic pauses
2. Assembler: Native-size image handling
3. Full video output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from video_server.tts_tool import _add_natural_pacing, generate_voiceover
from video_server.split_video_assembler import build_split_video

# Test script with punchlines for comedic timing
TEST_SCRIPT = (
    'Here is the situation. Russia has been playing chess while everyone else plays checkers! '
    'Their latest move? A "strategic withdrawal" that just happens to reposition forces near NATO borders. '
    'Coincidence? I think not. The Pentagon is reportedly "deeply concerned" -- which in diplomat speak '
    'means someone just spilled coffee on the war plans. Meanwhile, China is watching from the sidelines '
    'thinking: "This is fine. Everything is fine." One shot before Easter, and the whole geopolitical '
    'board could flip! Stay tuned.'
)

def main():
    print("=" * 60)
    print("  TESTING IMPROVEMENTS: Voice + Image + Comedic Timing")
    print("=" * 60)
    
    # ── Step 1: Test comedic pause injection ──
    print("\n[1/4] Testing comedic pause injection...")
    paced = _add_natural_pacing(TEST_SCRIPT)
    print(f"  ORIGINAL: {TEST_SCRIPT[:100]}...")
    print(f"  PACED:    {paced[:120]}...")
    pause_count = paced.count('...')
    print(f"  ✓ Injected {pause_count} comedic pauses")
    
    # ── Step 2: Generate TTS with am_adam voice ──
    print("\n[2/4] Generating TTS with am_adam voice...")
    tts_result = generate_voiceover(TEST_SCRIPT, voice_tone="authoritative")
    
    if not tts_result.get('success'):
        print(f"  ✗ TTS failed: {tts_result.get('error')}")
        return
    
    audio_path = tts_result['path']
    duration = tts_result.get('estimated_duration_seconds', 0)
    engine = tts_result.get('engine', 'unknown')
    voice = tts_result.get('voice', 'unknown')
    word_ts = tts_result.get('word_timestamps', [])
    
    print(f"  ✓ Voice: {voice} (engine: {engine})")
    print(f"  ✓ Duration: {duration:.1f}s")
    print(f"  ✓ Word timestamps: {len(word_ts)} words")
    
    # ── Step 3: Find existing images ──
    print("\n[3/4] Looking for existing scene images...")
    img_dir = Path("output/images")
    images = sorted(img_dir.glob("pixel_art_*.png"))[:5] if img_dir.exists() else []
    
    if not images:
        # Fallback: look in any output subfolder
        output_dir = Path("output")
        images = sorted(output_dir.rglob("*.png"))[:5]
    
    if not images:
        print("  ✗ No images found! Run generate_complete_video.py first.")
        return
    
    image_paths = [str(img) for img in images[:5]]
    print(f"  ✓ Found {len(image_paths)} images")
    for p in image_paths:
        print(f"    - {Path(p).name}")
    
    # ── Step 4: Build video ──
    print("\n[4/4] Building split video...")
    
    timestamp = int(os.path.getmtime(__file__))
    output_dir = Path(f"output/projects/video_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "test_improvements.mp4")
    
    result = build_split_video(
        audio_path=audio_path,
        image_paths=image_paths,
        output_path=output_path,
        script_text=paced,  # Use paced text for subtitle alignment
        word_timestamps=word_ts,
        hook_text="GEOPOLITICAL CHESS ♟️",
    )
    
    if result.get('success'):
        size_mb = result.get('file_size_mb', 0)
        print(f"\n{'='*60}")
        print(f"  ✓ VIDEO CREATED: {result['path']}")
        print(f"  ✓ Size: {size_mb} MB | Duration: {result['duration_seconds']}s")
        print(f"  ✓ Scenes: {result['scenes']} | Subtitles: {result['subtitles']}")
        print(f"{'='*60}")
        
        # Open the video
        os.startfile(result['path'])
    else:
        print(f"  ✗ Video build failed: {result.get('error')}")


if __name__ == "__main__":
    main()