"""Rebuild video from existing project assets (latest project)."""
import json, os, sys, glob
from pathlib import Path

# Find latest project
projects = glob.glob("output/projects/video_*")
projects.sort(key=os.path.getmtime, reverse=True)
project_dir = projects[0].replace("\\", "/")
print(f"Using latest project: {project_dir}")

manifest_path = f"{project_dir}/manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    m = json.load(f)

assets = m.get("assets", {})
script = m.get("script", {})

# Build image paths from assets
image_names = assets.get("images", [])
image_paths = [os.path.join(project_dir, "images", name) for name in image_names]

voiceover = os.path.join(project_dir, assets.get("voiceover", "voiceover.mp3"))
hook_text = ""
analyses = m.get("analyses", [])
if analyses:
    # Use last (most important) story's topic
    hook_text = analyses[-1].get("topic", "Geopolitical Update")

# Load curated script
curated_path = os.path.join(project_dir, "script_curated.txt")
if os.path.exists(curated_path):
    with open(curated_path, "r", encoding="utf-8") as f:
        script_text = f.read()
else:
    script_text = script.get("full_text", "")

# Load or generate word timestamps for subtitle sync
word_timestamps = []

# 1. Check manifest for saved TTS word timestamps
manifest_wt = m.get("tts", {}).get("word_timestamps", [])
if manifest_wt:
    word_timestamps = manifest_wt
    print(f"Loaded {len(word_timestamps)} word timestamps from manifest")

# 2. Generate from audio + script using calibrated estimation
if not word_timestamps and script_text and os.path.exists(voiceover):
    print("Generating word timestamps from audio file...")
    from src.video.tts_tool import _get_faster_whisper_timestamps
    word_timestamps = _get_faster_whisper_timestamps(voiceover, script_text)
    if word_timestamps:
        print(f"Generated {len(word_timestamps)} word timestamps")
        # Save to manifest for future rebuilds
        if "tts" not in m:
            m["tts"] = {}
        m["tts"]["word_timestamps"] = word_timestamps
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        print("Saved word timestamps to manifest.json")

# Build scene timestamps from timeline (with fuzzy matching + pre-roll offset)
segment_timeline = script.get("segment_timeline", [])
scene_timestamps = []

# Get total audio duration
from moviepy.editor import AudioFileClip
audio_clip = AudioFileClip(voiceover)
total_dur = audio_clip.duration
audio_clip.close()
print(f"Audio duration: {total_dur:.1f}s")

if segment_timeline and word_timestamps:
    num_images = len(image_paths)
    image_times = [{"start": None, "end": None} for _ in range(num_images)]
    
    # Strip dead segments (greeting/intro_hook removed from TTS)
    stripped_greeting = script.get("greeting", "")
    stripped_intro_hook = script.get("intro_hook", "")
    cleaned_timeline = []
    for seg in segment_timeline:
        seg_text = seg.get("text", "").strip()
        seg_label = seg.get("label", "")
        if seg_label == "intro" and stripped_greeting and stripped_greeting[:20].lower() in seg_text.lower():
            print(f"  Stripped dead intro segment: \"{seg_text[:50]}...\"")
            continue
        if seg_label == "intro_pause" and stripped_greeting:
            print(f"  Stripped dead intro_pause segment")
            continue
        cleaned_timeline.append(seg)
    segment_timeline = cleaned_timeline
    print(f"  Cleaned timeline: {len(segment_timeline)} active segments")
    
    # Fuzzy matching function (same as generate_complete_video.py)
    def _fuzzy_find_segment(seg_text, word_timestamps):
        seg_words = seg_text.lower().split()
        seg_clean = [w.strip(".,!?;:'\"()-") for w in seg_words if len(w.strip(".,!?;:'\"()-")) > 1]
        if len(seg_clean) < 2:
            return None, None
        wt_clean = [wt.get("word", "").lower().strip(".,!?;:'\"()-") for wt in word_timestamps]
        best_match = None
        best_match_len = 0
        search_words = seg_clean[:8]
        for start_i in range(len(wt_clean)):
            match_count = 0
            seg_i = 0
            wt_i = start_i
            skips = 0
            while seg_i < len(search_words) and wt_i < len(wt_clean) and skips <= 3:
                if wt_clean[wt_i] == search_words[seg_i] or (
                    len(wt_clean[wt_i]) > 2 and len(search_words[seg_i]) > 2 and
                    (wt_clean[wt_i].startswith(search_words[seg_i][:4]) or
                     search_words[seg_i].startswith(wt_clean[wt_i][:4]))
                ):
                    match_count += 1
                    seg_i += 1
                    wt_i += 1
                    skips = 0
                else:
                    skips += 1
                    wt_i += 1
            if match_count > best_match_len and match_count >= 2:
                best_match_len = match_count
                end_wt_i = min(start_i + len(seg_clean) + 2, len(word_timestamps) - 1)
                best_match = (
                    word_timestamps[start_i].get("start", 0),
                    word_timestamps[end_wt_i].get("end", word_timestamps[start_i].get("start", 0) + 5)
                )
        return best_match if best_match else (None, None)
    
    # Match each segment to word timestamps
    for seg in segment_timeline:
        img_idx = seg.get("image_idx", 0)
        if img_idx >= num_images:
            continue
        seg_text = seg.get("text", "").strip()
        if not seg_text or seg_text in ("....", "...", ".."):
            continue
        seg_start, seg_end = _fuzzy_find_segment(seg_text, word_timestamps)
        if seg_start is not None:
            if image_times[img_idx]["start"] is None or seg_start < image_times[img_idx]["start"]:
                image_times[img_idx]["start"] = seg_start
            if seg_end is not None and (image_times[img_idx]["end"] is None or seg_end > image_times[img_idx]["end"]):
                image_times[img_idx]["end"] = seg_end
    
    # Fill gaps
    for i, it in enumerate(image_times):
        if it["start"] is None:
            it["start"] = (total_dur / num_images) * i
        if it["end"] is None:
            if i + 1 < num_images and image_times[i + 1]["start"] is not None:
                it["end"] = image_times[i + 1]["start"]
            else:
                it["end"] = (total_dur / num_images) * (i + 1)
    
    # Ensure first starts at 0, last ends at total
    if image_times:
        image_times[-1]["end"] = max(image_times[-1]["end"], total_dur)
        image_times[0]["start"] = 0
    
    # Bridge gaps
    for i in range(len(image_times) - 1):
        gap = image_times[i + 1]["start"] - image_times[i]["end"]
        if gap > 0.1:
            split_point = image_times[i]["end"] + gap * 0.7
            image_times[i]["end"] = split_point
            image_times[i + 1]["start"] = split_point
    
    # Pre-roll offset: images appear 1s before their narration
    PREROLL_OFFSET = 0.3
    for i in range(1, len(image_times)):
        new_start = max(0, image_times[i]["start"] - PREROLL_OFFSET)
        if new_start < image_times[i - 1]["start"]:
            new_start = image_times[i - 1]["start"]
        image_times[i]["start"] = new_start
    
    scene_timestamps = image_times
    print(f"  Built {len(scene_timestamps)} scene timestamps (fuzzy + {PREROLL_OFFSET}s pre-roll)")
    for i, ts in enumerate(scene_timestamps):
        dur = ts["end"] - ts["start"]
        print(f"    Image {i}: {ts['start']:.2f}s -> {ts['end']:.2f}s ({dur:.2f}s)")

elif segment_timeline:
    # Fallback: proportional word-count (legacy)
    total_words = max(len(script_text.split()), 1)
    img_segments = {}
    elapsed = 0.0
    for seg in segment_timeline:
        idx = seg.get("image_idx", 0)
        seg_text = seg.get("text", "")
        seg_words = len(seg_text.split())
        seg_dur = (seg_words / total_words) * total_dur
        if seg.get("is_separator"):
            seg_dur = min(seg_dur, 0.5)
        if idx not in img_segments:
            img_segments[idx] = {"start": elapsed, "end": elapsed + seg_dur}
        else:
            img_segments[idx]["end"] = elapsed + seg_dur
        elapsed += seg_dur
    for i in range(len(image_paths)):
        if i in img_segments:
            s = img_segments[i]
            scene_timestamps.append({"start": s["start"], "end": min(s["end"], total_dur)})
        else:
            scene_timestamps.append({"start": 0, "end": total_dur})
    print(f"  Built {len(scene_timestamps)} scene timestamps (proportional fallback)")

print(f"Images: {len(image_paths)}")
print(f"Audio: {voiceover}")
print(f"Hook: {hook_text}")
print(f"Words: {len(script_text.split())}")
print(f"Word timestamps: {len(word_timestamps)}")
print(f"Scene timestamps: {len(scene_timestamps)}")

# Unload ollama to free RAM
import subprocess
try:
    subprocess.run(["ollama", "stop", "gemma4:26b"], timeout=30, capture_output=True)
    print("Unloaded Gemma 4 from RAM")
except:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.video.split_video_assembler import build_split_video

project_id = os.path.basename(project_dir).replace("video_", "")
result = build_split_video(
    audio_path=voiceover,
    image_paths=image_paths,
    output_path=os.path.join(project_dir, f"video_{project_id}.mp4"),
    script_text=script_text,
    word_timestamps=word_timestamps,
    hook_text=hook_text,
    scene_timestamps=scene_timestamps if scene_timestamps else None,
)

if result.get("success"):
    print(f"\n✅ VIDEO CREATED: {result['path']}")
    print(f"   Size: {result.get('file_size_mb', 0):.1f}MB")
else:
    print(f"\n❌ FAILED: {result.get('error')}")