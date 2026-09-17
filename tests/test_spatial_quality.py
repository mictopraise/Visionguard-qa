from pathlib import Path

import cv2
import numpy as np

from app.vision.spatial_quality import (
    analyze_spatial_quality,
    measure_blockiness_frame,
    measure_blur_frame,
)


def _bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_blur_measurement_drops_after_gaussian_blur() -> None:
    rng = np.random.default_rng(7)
    textured = rng.integers(0, 256, size=(180, 320), dtype=np.uint8)
    blurred = cv2.GaussianBlur(textured, (21, 21), 7.0)

    sharp_score = measure_blur_frame(_bgr(textured))
    blur_score = measure_blur_frame(_bgr(blurred))

    assert sharp_score > blur_score * 5.0


def test_blockiness_measurement_detects_eight_pixel_boundaries() -> None:
    clean = np.full((128, 128), 120, dtype=np.uint8)
    blocked = clean.copy()
    for y in range(0, 128, 8):
        for x in range(0, 128, 8):
            blocked[y:y + 8, x:x + 8] = 60 if ((x // 8) + (y // 8)) % 2 == 0 else 180

    clean_score = measure_blockiness_frame(_bgr(clean))
    blocked_score = measure_blockiness_frame(_bgr(blocked))

    assert blocked_score["boundary_strength"] > clean_score["boundary_strength"] + 20.0
    assert blocked_score["ratio"] > 2.0


def test_video_spatial_analysis_surfaces_blocky_candidate(tmp_path: Path) -> None:
    output = tmp_path / "blocky.avi"
    width = height = 128
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (width, height))
    assert writer.isOpened()

    gray = np.zeros((height, width), dtype=np.uint8)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            gray[y:y + 8, x:x + 8] = 50 if ((x // 8) + (y // 8)) % 2 == 0 else 200
    frame = _bgr(gray)
    for _ in range(24):
        writer.write(frame)
    writer.release()

    result = analyze_spatial_quality(output, sample_interval_seconds=0.25)

    assert result["sample_count"] >= 5
    assert any(item["artifact"] == "blocky" for item in result["candidate_artifacts"])
    assert result["calibration_status"] == "provisional"
