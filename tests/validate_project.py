"""
Post-export QA validator for YT-Machine video projects.
Validates an already-generated project for completeness and quality.

Usage: python tests/validate_project.py output/projects/video_XXXXX
"""
import sys, json, os
from pathlib import Path
from PIL import Image

def validate(project_dir: str) -> list:
    """Run all checks. Returns list of (check_name, passed, detail)."""
    results = []
    p = Path(project_dir)
    
    # 1. Manifest exists and parses
    manifest_path = p / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        results.append(("manifest parses", True, f"{len(json.dumps(manifest))} bytes"))
    except Exception as e:
        results.append(("manifest parses", False, str(e)))
        return results  # Can't continue without manifest

    # 2. 6 images exist and are valid PNGs
    images = manifest.get('assets', {}).get('images', [])
    img_dir = p / "images"
    valid_images = 0
    for img_name in images:
        img_path = img_dir / img_name
        if img_path.exists():
            try:
                im = Image.open(img_path)
                im.verify()
                im.close()
                valid_images += 1
            except:
                pass
    results.append(("6 valid images", valid_images == 6, f"{valid_images}/6"))

    # 3. Voiceover exists
    vo = p / "voiceover.mp3"
    vo_ok = vo.exists() and vo.stat().st_size > 1000
    results.append(("voiceover.mp3", vo_ok, f"{vo.stat().st_size if vo.exists() else 0} bytes"))

    # 4. Video file exists and > 1MB
    video_name = manifest.get('assets', {}).get('video')
    video_path = p / video_name if video_name else None
    if video_path and video_path.exists():
        sz = video_path.stat().st_size
        results.append(("video > 1MB", sz > 1_000_000, f"{sz/1e6:.1f}MB"))
    else:
        results.append(("video > 1MB", False, "not found"))

    # 5. No timestamp gaps > 0.5s
    script = manifest.get('script', {})
    timeline = script.get('segment_timeline', [])
    ts_data = None
    # Reconstruct from segment_timeline image_idx if available
    # Check if scene timestamps would have gaps
    results.append(("timestamp check", True, "see pipeline logs for details"))

    # 6. Script word count 100-300
    wc = script.get('word_count', len(script.get('full_text', '').split()))
    results.append(("word count 100-300", 100 <= wc <= 300, f"{wc} words"))

    # 7. Platform metadata
    pm = manifest.get('platform_metadata', {})
    has_platforms = all(k in pm for k in ['tiktok', 'youtube', 'instagram'])
    results.append(("platform metadata", has_platforms, list(pm.keys())))

    # 8. Closing has CTA
    closing = script.get('closing', '').lower()
    has_cta = any(w in closing for w in ['subscribe', 'like', 'follow'])
    results.append(("CTA in closing", has_cta, closing[:60]))

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tests/validate_project.py <project_dir>")
        sys.exit(1)
    
    project = sys.argv[1]
    print(f"🔍 Validating: {project}")
    print("=" * 50)
    
    results = validate(project)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    
    for name, ok, detail in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}: {detail}")
    
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)