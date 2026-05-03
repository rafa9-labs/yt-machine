# Final 4 Points — Implementation Plan

## Point 1: Image Generation Retry System (Auto-regeneration on Failure)

### Files to modify:
- `src/video/pixel_art_tool.py`
- `tools/generate_complete_video.py`
- `config/image_style.json`

### Changes in `src/video/pixel_art_tool.py`:

**A) Add `_detect_failed_image()` function** (insert after `_upscale_pixel_art()`, around line 422):

```python
def _detect_failed_image(image_path: str) -> Tuple[bool, str]:
    """
    Detect visually failed images: solid-color, near-monochrome, or extremely low variance.
    Returns (is_failed: bool, reason: str).
    is_failed=True means the image is likely a failed generation (not usable).
    """
    try:
        import numpy as np
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)

        mean_val = arr.mean()
        std_val = arr.std()

        # Check 1: Near-monochrome
        if std_val < 8.0:
            return True, f"near_monochrome (std={std_val:.1f})"

        # Check 2: Per-channel flatness
        r_std = arr[:, :, 0].std()
        g_std = arr[:, :, 1].std()
        b_std = arr[:, :, 2].std()
        if r_std < 5.0 and g_std < 5.0 and b_std < 5.0:
            return True, f"flat_color (r_std={r_std:.1f}, g_std={g_std:.1f}, b_std={b_std:.1f})"

        # Check 3: Uniform color across corners
        h, w = arr.shape[:2]
        corners = [
            arr[:h//8, :w//8],
            arr[:h//8, -w//8:],
            arr[-h//8:, :w//8],
            arr[-h//8:, -w//8:],
        ]
        corner_means = [c.mean() for c in corners]
        corner_spread = max(corner_means) - min(corner_means)
        if corner_spread < 3.0 and std_val < 15.0:
            return True, f"uniform_color (corner_spread={corner_spread:.1f}, std={std_val:.1f})"

        # Check 4: Low edge detail
        edge_row_diff = np.abs(np.diff(arr[h//2, :, 0], axis=0)).mean()
        edge_col_diff = np.abs(np.diff(arr[:, w//2, 0], axis=0)).mean()
        if edge_row_diff < 3.0 and edge_col_diff < 3.0 and std_val < 25.0:
            return True, f"low_detail (edge_row={edge_row_diff:.1f}, edge_col={edge_col_diff:.1f}, std={std_val:.1f})"

        return False, "ok"
    except Exception as e:
        print(f"  [IMG] Warning: Failed image detection error: {e}")
        return False, f"detection_error ({e})"
```

**B) Add `_apply_sharpening()` function** (insert after `_detect_failed_image()`):

```python
def _apply_sharpening(input_path: str) -> str:
    """
    Apply unsharp mask sharpening to an upscaled pixel-art image.
    """
    try:
        from PIL import ImageFilter
        img = Image.open(input_path)
        sharpened = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        sharpened.save(input_path, format='PNG')
        print(f"  [IMG] Applied unsharp mask sharpening")
        return input_path
    except Exception as e:
        print(f"  [IMG] Warning: Sharpening failed: {e}")
        return input_path
```

**C) Integrate `_detect_failed_image()` into `generate_pixel_art()`**:

After the image is saved and upscaled (both local FLUX and fal.ai code paths), add:
```python
# Check if image looks like a failed generation (solid color, monochrome, etc.)
is_failed, fail_reason = _detect_failed_image(str(output_path))
if is_failed:
    print(f"  [IMG] Detected failed image: {fail_reason}")
    return {
        "success": True,  # True so caller can decide to retry
        "filename": filename,
        "path": str(output_path),
        "prompt_used": enforced_prompt,
        "visual_type": visual_type,
        "source": "failed_detection",
        "note": f"Auto-detected failed generation: {fail_reason}",
        "detected_failure": fail_reason,
        "output_directory": str(OUTPUT_DIR),
    }
```

**D) In the content-policy retry block** (around line 1831-1898), change each scrub level to try `fal-ai/flux/dev` first before falling to `fal-ai/flux/schnell`:

Instead of:
```python
retry_result = fal_client.run("fal-ai/flux/schnell", arguments=retry_args)
```

Use:
```python
for retry_model in [FAL_MODEL, "fal-ai/flux/schnell"]:
    try:
        retry_args["num_inference_steps"] = MODEL_STEP_CONFIG.get(retry_model, 28)
        retry_result = fal_client.run(retry_model, arguments=retry_args)
        retry_image_url = retry_result["images"][0]["url"]
        # ... save and upscale ...
        break
    except Exception as model_err:
        print(f"  [IMG] Retry model {retry_model} failed at scrub level {scrub_level}: {str(model_err)[:80]}")
        continue
```

**E) In `tools/generate_complete_video.py`** (lines 1027-1076), enhance the retry loop:

- Increase max attempts from 3 to 4
- Add `scrub_level` tracking that increases on detection of failed images or content policy blocks
- When `generate_pixel_art()` returns `detected_failure`, increment scrub level and regenerate with progressively scrubbed prompt
- On 4th attempt, use `_CATEGORY_SAFE_PROMPTS` dictionary from pixel_art_tool as the prompt

### Changes in `config/image_style.json`:

Add `retry_config` section inside `generation_params`:
```json
"retry_config": {
    "max_retries": 4,
    "scrub_on_content_policy": true,
    "detect_solid_color": true,
    "solid_color_std_threshold": 8.0,
    "flat_color_channel_threshold": 5.0,
    "progressive_scrub": true
}
```

---

## Point 2: Zoom from Center + Smoother Zoom

### File: `src/video/split_video_assembler.py`

**A) Update `SCENE_ZOOM_PROFILES`** (line 60-65):

```python
# BEFORE:
SCENE_ZOOM_PROFILES = {
    0: {'name': 'HOOK',      'zoom_start': 1.20, 'zoom_end': 1.05, 'pan_x': 0.0},
    1: {'name': 'MECHANISM', 'zoom_start': 1.15, 'zoom_end': 1.0,  'pan_x': 0.0},
    2: {'name': 'TRUTH',     'zoom_start': 1.0,  'zoom_end': 1.12, 'pan_x': 0.0},
    3: {'name': 'FALLOUT',   'zoom_start': 1.10, 'zoom_end': 1.0,  'pan_x': -0.3},
}

# AFTER:
SCENE_ZOOM_PROFILES = {
    0: {'name': 'HOOK',      'zoom_start': 1.08, 'zoom_end': 1.02, 'pan_x': 0.0},
    1: {'name': 'MECHANISM', 'zoom_start': 1.06, 'zoom_end': 1.01, 'pan_x': 0.0},
    2: {'name': 'TRUTH',     'zoom_start': 1.01, 'zoom_end': 1.07, 'pan_x': 0.0},
    3: {'name': 'FALLOUT',   'zoom_start': 1.05, 'zoom_end': 1.01, 'pan_x': 0.0},
}
```

All pan_x values are 0.0 (center-only zoom). Zoom ranges are gentle (1.01-1.08).

**B) Replace smootherstep with ease-out cubic** in `_render_scene_opencv()` (around line 223-225):

```python
# BEFORE:
def smootherstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6 - 15) + 10)

# AFTER:
def ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3
```

And update line 235:
```python
# BEFORE: eased = smootherstep(progress)
# AFTER:  eased = ease_out_cubic(progress)
```

**C) Also update in `_render_scene_ffmpeg_fallback()`** — find smootherstep usage and replace with ease_out_cubic equivalent for ffmpeg expression:
```python
# Before: zoom = zoom_start + (zoom_end - zoom_start) * (on-1)/(total_frames-1)
# After (ease-out cubic): zoom = zoom_start + (zoom_end - zoom_start) * (1 - pow(1 - (on-1)/(total_frames-1), 3))
```

---

## Point 3: Higher Image Resolution (768x810, 1.4x upscale)

### File: `config/image_style.json`

**A) Update `generation_params`**:

```json
// BEFORE:
"render_resolution": [272, 288],
"target_resolution": [1088, 1152],
"num_inference_steps": 28,
"guidance_scale": 3.5,

// AFTER:
"render_resolution": [768, 810],
"target_resolution": [1088, 1152],
"num_inference_steps": 40,
"guidance_scale": 4.0,
```

### File: `src/video/pixel_art_tool.py`

**A) Update `PIXEL_ART_ENFORCEMENT_PREFIX`** (line 34-38):

```python
# BEFORE:
PIXEL_ART_ENFORCEMENT_PREFIX = (
    "Clean 32-bit pixel art, high contrast news graphic, isometric style, "
    "uniform grid-aligned pixels, no anti-aliasing, no color bleeding, "
    "no soft gradients"
)

# AFTER:
PIXEL_ART_ENFORCEMENT_PREFIX = (
    "Clean 32-bit pixel art, high contrast news graphic, isometric style, "
    "uniform grid-aligned pixels, no anti-aliasing, no color bleeding, "
    "no soft gradients, sharp focus, detailed scene composition"
)
```

**B) Update `_upscale_pixel_art()`** to call `_apply_sharpening()` after save:

After `upscaled.save(input_path, format='PNG')` (line 419), add:
```python
try:
    _apply_sharpening(input_path)
except Exception as sharpen_err:
    print(f"  [IMG] Warning: Post-sharpening failed: {sharpen_err}")
```

**C) Update `MODEL_STEP_CONFIG`** (line 71-76):

```python
# BEFORE:
"fal-ai/flux/dev": 28,

# AFTER:
"fal-ai/flux/dev": 40,
```

**D) Update local FLUX default steps** (around line 1676):

```python
# BEFORE:
local_steps = int(os.environ.get("LOCAL_FLUX_STEPS", "20"))

# AFTER:
local_steps = int(os.environ.get("LOCAL_FLUX_STEPS", "40"))
```

---

## Point 4: Better Outro + Seamless Transition

### File: `src/brain/llm_interface.py`

**A) Replace static `UNIFIED_CLOSING` with dynamic closing builder** (line 893-894):

```python
# BEFORE:
UNIFIED_CLOSING = "Stay behind the curtains, and if I don't see you. Good morning, good afternoon. And goodnight."

# AFTER:
UNIFIED_CLOSING_BASE = "Stay behind the curtains, and if I don't see you — good morning, good afternoon, and goodnight."

def _build_dynamic_closing(self, last_fallout: str = "", last_topic: str = "") -> str:
    """
    Build a dynamic closing that references the last story's topic,
    creating a seamless bridge from the final fallout to the sign-off.
    The bridge drops the manic Mask persona and transitions into melancholy.
    """
    import re

    if last_topic:
        topic_clean = last_topic.strip().rstrip('.')
        bridge = f"And that is how {topic_clean} reshapes the board. "
    elif last_fallout:
        words = re.findall(r'\b\w+\b', last_fallout.lower())
        key_nouns = [w for w in words[-6:] if len(w) > 3 and w not in
                      {'that', 'this', 'with', 'from', 'they', 'their',
                       'have', 'been', 'will', 'would', 'could', 'what',
                       'when', 'where', 'which', 'there', 'these', 'those'}]
        if key_nouns:
            bridge = f"And just like that, {key_nouns[-1]} rewrites the rules. "
        else:
            bridge = "And just like that, the dominoes keep falling. "
    else:
        bridge = "And just like that, the dominoes keep falling. "

    return bridge + self.UNIFIED_CLOSING_BASE
```

**B) Update all references to `UNIFIED_CLOSING`** in llm_interface.py:

Search for `self.UNIFIED_CLOSING` and replace with dynamic closing calls. Key locations:
- Line ~1537: `closing = self.UNIFIED_CLOSING` → `closing = self._build_dynamic_closing(last_fallout, last_topic)`
- Line ~1064: `result = stripped + ' .... ' + self.UNIFIED_CLOSING` → `result = stripped + ' .... ' + self._build_dynamic_closing(last_fallout, last_topic)`
- Line ~865: `_validate_closing()` — update to handle dynamic closing

**C) Update `_validate_closing()`** (line 1017+):

Change the Truman detection to use a broader window:
```python
# BEFORE:
tail = text_lower[-150:] if len(text_lower) > 150 else text_lower

# AFTER:
tail = text_lower[-300:] if len(text_lower) > 300 else text_lower
```

Update the closing injection to pass the stored last story context:
```python
if hasattr(self, '_last_story_topic') and self._last_story_topic:
    closing = self._build_dynamic_closing(
        last_fallout=getattr(self, '_last_fallout', ''),
        last_topic=self._last_story_topic
    )
else:
    closing = self.UNIFIED_CLOSING_BASE

result = stripped + ' .... ' + closing
```

**D) Update `synthesize_multi_news_script()`** — before setting closing (around line 1537):

```python
stories = script.get('stories', [])
last_story = stories[-1] if stories else {}
last_topic = last_story.get('topic', '') or (news_analyses[-1].get('topic', '') if news_analyses else '')
last_fallout = last_story.get('fallout', '')
self._last_story_topic = last_topic
self._last_fallout = last_fallout
closing = self._build_dynamic_closing(last_fallout=last_fallout, last_topic=last_topic)
```

### File: `config/system_prompts.json`

**A) Add closing/bridge instructions to `multi_news_synthesizer` system prompt**:

Add these lines to the CRITICAL RULES section:
```
- CLOSING BRIDGE RULE: The closing MUST have TWO parts: (1) a bridge sentence that connects the last story's fallout to a reflective sign-off, referencing the last story's topic directly. Then (2) drop into melancholy — no caps, no exclamations, just quiet reflection. The bridge creates seamless continuity. Example: "And that is how [topic] reshapes the board... Stay behind the curtains, and if I don't see you — good morning, good afternoon, and goodnight."
- The closing voice SLOWS DOWN and SOFTENS. No manic energy. The Mask drops the act here — it is the Truman Show moment. Melancholy, honest, reflective.
```

### File: `src/video/tts_tool.py`

**A) Add outro detection function**:

```python
import re

def _detect_outro_segment(text: str) -> Tuple[str, str]:
    """
    Split full script text into main content and outro segment.
    The outro is identified by the last '....' separator and trailing text.
    Returns (main_text, outro_text).
    """
    parts = re.split(r'\.{4,}', text)
    if len(parts) >= 2:
        main = '....'.join(parts[:-1])
        outro = parts[-1].strip()
        return main, outro
    return text, ""


def _apply_outro_tts_settings(voice_settings: dict, is_outro: bool = False) -> dict:
    """
    Modify TTS voice settings for outro segments.
    Creates a melancholy, natural spoken tone: slower, softer, more stable.
    """
    if not is_outro:
        return voice_settings

    settings = voice_settings.copy()

    # ElevenLabs: higher stability, lower style for melancholy
    if 'stability' in settings:
        settings['stability'] = min(1.0, settings.get('stability', 0.35) + 0.25)
    if 'style_exaggeration' in settings:
        settings['style_exaggeration'] = max(0.0, settings.get('style_exaggeration', 0.65) - 0.35)
    if 'similarity_boost' in settings:
        settings['similarity_boost'] = min(1.0, settings.get('similarity_boost', 0.70) + 0.05)

    # Edge TTS: slower rate, lower pitch
    settings['rate'] = "-15%"
    settings['pitch'] = "-3Hz"

    # Kokoro: slower speed
    settings['speed'] = 0.85

    return settings
```

**B) Add outro reverb post-processing function**:

```python
def _apply_outro_reverb(audio_path: str) -> str:
    """
    Apply subtle reverb tail to outro audio for melancholy atmosphere.
    ffmpeg aecho: 0.8 gain, 0.88 feedback, 60ms delay, 0.4 decay.
    Also applies a 1-second fade-out at the end.
    """
    ffmpeg_exe = _find_ffmpeg()
    if not ffmpeg_exe:
        return audio_path

    output_path = audio_path.rsplit('.', 1)[0] + '_outro_reverb.' + audio_path.rsplit('.', 1)[1]
    try:
        cmd = [
            ffmpeg_exe, '-y',
            '-i', audio_path,
            '-af', 'aecho=0.8:0.88:60:0.4,afade=t=out:st=0:d=1',
            '-ar', '44100',
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
        if result.returncode == 0 and Path(output_path).exists():
            Path(audio_path).unlink(missing_ok=True)
            return output_path
    except Exception as e:
        print(f"  [TTS] Warning: Outro reverb failed: {e}")
    return audio_path
```

**C) Integrate outro detection into the TTS pipeline**:

In the main TTS generation flow (where chunks are assembled and sent to the TTS engine), detect the outro segment and apply different voice settings:

1. After splitting text by `....`, identify the last chunk as outro
2. For the outro chunk, call `_apply_outro_tts_settings(settings, is_outro=True)`
3. After generating outro audio, call `_apply_outro_reverb()` on the outro audio file
4. Merge back with the main audio during assembly

---

## Implementation Order

1. **Point 3** (Resolution) — Simplest config + small code changes
2. **Point 2** (Zoom) — Small code change, immediate visual impact
3. **Point 1** (Retry system) — Most complex, needs testing
4. **Point 4** (Outro) — Multi-file, needs careful TTS integration

Each point can be implemented and tested independently before moving to the next.