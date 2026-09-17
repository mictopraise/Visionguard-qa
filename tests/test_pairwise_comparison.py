from pathlib import Path
import shutil

import cv2
import numpy as np

from benchmark.create_synthetic import create_moving_square_video
from app.services.first_pass import _final_verdict
from app.vision.pairwise_comparison import compare_aligned_videos


def _inject_brightness_block(source: Path, output: Path, start: int, end: int) -> None:
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if start <= frame_index <= end:
            frame = np.full_like(frame, 255)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()


def test_identical_pair_has_no_comparative_issues(tmp_path: Path) -> None:
    a = create_moving_square_video(tmp_path / "a.avi", seconds=3.0)
    b = tmp_path / "b.avi"
    shutil.copyfile(a, b)

    result = compare_aligned_videos(a, b)

    assert result["sample_count"] > 5
    assert abs(result["alignment"]["offset_seconds"]) < 0.25
    assert result["issues"] == []


def test_sustained_visual_degradation_is_found(tmp_path: Path) -> None:
    a = create_moving_square_video(tmp_path / "a.avi", seconds=4.0)
    b = tmp_path / "b.avi"
    _inject_brightness_block(a, b, start=36, end=60)

    result = compare_aligned_videos(a, b, sample_interval_seconds=0.10)

    assert result["sample_count"] > 10
    assert any(issue["type"] in {"pairwise_brightness_mismatch", "pairwise_visual_mismatch"} for issue in result["issues"])


def test_high_confidence_b_degradation_recommends_regeneration() -> None:
    clean = {"issues": []}
    comparative = {
        "issues": [{
            "type": "pairwise_visual_mismatch",
            "start_time": 2.0,
            "end_time": 2.6,
            "confidence": 0.9,
            "affected_video": "B",
            "details": {},
        }]
    }

    verdict = _final_verdict(clean, clean, comparative)

    assert verdict["status"] == "FAIL"
    assert verdict["action"] == "REGENERATE"
