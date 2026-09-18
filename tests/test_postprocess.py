"""Tests for deterministic pixel-grid post-processing."""

import numpy as np
import pytest
from PIL import Image

from src.video.postprocess import count_colors, process_pixel_art, process_pixel_art_file
from src.video.split_video_assembler import (
    _find_ffmpeg,
    _prepare_pixel_scene_frame,
    _render_scene_opencv,
)


def _source_image():
    image = Image.new("RGB", (768, 768))
    pixels = image.load()
    palette = [(i * 29 % 256, i * 53 % 256, i * 83 % 256) for i in range(64)]
    for y in range(768):
        for x in range(768):
            pixels[x, y] = palette[((x // 3) + (y // 5)) % len(palette)]
    return image


def test_process_pixel_art_preserves_dimensions_and_palette_bound():
    processed = process_pixel_art(_source_image())
    assert processed.size == (768, 768)
    assert count_colors(processed) <= 32


def test_nearest_assembler_transform_preserves_postprocessed_palette():
    import cv2

    processed = process_pixel_art(_source_image())
    bgr = cv2.cvtColor(np.array(processed), cv2.COLOR_RGB2BGR)
    frame = _prepare_pixel_scene_frame(bgr)
    assert frame.shape[:2] == (1152, 1080)
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) <= 32


def test_process_pixel_art_file_writes_metadata(tmp_path):
    source_path = tmp_path / "raw.png"
    output_path = tmp_path / "processed.png"
    _source_image().save(source_path)
    result = process_pixel_art_file(
        source_path,
        output_path,
        {"postprocess": {
            "logical_size": [192, 192],
            "output_size": [768, 768],
            "colors": 32,
        }},
    )
    assert output_path.exists()
    assert result["source_size"] == [768, 768]
    assert result["colors_written"] <= 32


def test_encoded_scene_clip_preserves_palette(tmp_path):
    """The scene codec must not undo the PNG palette guarantee."""
    if not _find_ffmpeg():
        pytest.skip("ffmpeg is required for encoded-scene validation")

    source_path = tmp_path / "scene.png"
    video_path = tmp_path / "scene.mp4"
    process_pixel_art(_source_image()).save(source_path)

    assert _render_scene_opencv(str(source_path), 1.0, 0, str(video_path))

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    assert ok
    assert (width, height) == (1080, 1152)
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) <= 32
