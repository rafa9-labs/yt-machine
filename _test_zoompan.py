import time, tempfile
from pathlib import Path
from video_server.split_video_assembler import _render_scene_ffmpeg

images = list(Path('output/projects').rglob('*.png'))[:6]
print(f"Testing with {len(images)} images")

for idx, img in enumerate(images):
    tmp = tempfile.NamedTemporaryFile(suffix=f"_scene{idx}.mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    start = time.time()
    ok = _render_scene_ffmpeg(str(img), duration=8.0, scene_idx=idx, output_path=tmp_path)
    elapsed = time.time() - start

    size = Path(tmp_path).stat().st_size if Path(tmp_path).exists() else 0
    effect = "zoom-out" if idx % 2 == 0 else "pan"
    print(f"  Scene {idx} ({effect}): ok={ok}, {elapsed:.1f}s, {size/1024:.0f}KB")

    if ok and size > 0:
        Path(tmp_path).unlink()

total_time = 6 * 0.9
print(f"\nEstimated total for 6 scenes: ~{total_time:.0f}s (was 15-20 min)")
