import cv2
import numpy as np

from app.vision.spatial_quality import measure_color_shift_frame


def _neutral_color_chart() -> np.ndarray:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    colors = [
        (128, 128, 128),
        (70, 70, 70),
        (190, 190, 190),
        (120, 160, 200),
        (200, 150, 110),
        (110, 180, 130),
    ]
    width = frame.shape[1] // len(colors)
    for i, color in enumerate(colors):
        frame[:, i * width:(i + 1) * width] = color
    return frame


def _apply_lab_shift(frame: np.ndarray, da: int, db: int) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab[..., 1] = np.clip(lab[..., 1] + da, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] + db, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def test_controlled_chroma_shift_increases_lab_bias() -> None:
    original = _neutral_color_chart()
    shifted = _apply_lab_shift(original, da=28, db=-24)

    baseline = measure_color_shift_frame(original)
    candidate = measure_color_shift_frame(shifted)

    assert candidate["chroma_bias"] > baseline["chroma_bias"] + 10.0
    assert abs(candidate["mean_a"] - baseline["mean_a"]) > 10.0
    assert abs(candidate["mean_b"] - baseline["mean_b"]) > 10.0
