import cv2
import numpy as np

from app.vision.spatial_quality import (
    measure_oversharpening_frame,
    measure_oversmoothing_frame,
)


def _textured_frame() -> np.ndarray:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    for y in range(0, 180, 12):
        for x in range(0, 320, 12):
            value = 230 if ((x // 12) + (y // 12)) % 2 == 0 else 35
            frame[y:y + 12, x:x + 12] = value
    cv2.putText(
        frame,
        "DETAIL",
        (70, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (128, 128, 128),
        2,
        cv2.LINE_AA,
    )
    return frame


def test_aggressive_sharpening_increases_edge_residual_signal() -> None:
    original = _textured_frame()
    blurred = cv2.GaussianBlur(original, (0, 0), 1.1)
    sharpened = cv2.addWeighted(original, 2.6, blurred, -1.6, 0)

    baseline = measure_oversharpening_frame(original)
    candidate = measure_oversharpening_frame(sharpened)

    assert candidate["edge_residual_mean"] > baseline["edge_residual_mean"]
    assert candidate["residual_p95"] > baseline["residual_p95"]


def test_smoothing_reduces_fine_texture_ratio() -> None:
    original = _textured_frame()
    smoothed = cv2.GaussianBlur(original, (0, 0), 3.0)

    baseline = measure_oversmoothing_frame(original)
    candidate = measure_oversmoothing_frame(smoothed)

    assert candidate["fine_to_coarse_ratio"] < baseline["fine_to_coarse_ratio"]
    assert candidate["fine_energy"] < baseline["fine_energy"]
    assert candidate["coarse_energy"] > 0.0
