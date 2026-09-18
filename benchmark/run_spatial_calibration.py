from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.vision.spatial_quality import (
    measure_blur_frame,
    measure_blockiness_frame,
    measure_color_shift_frame,
    measure_oversharpening_frame,
    measure_oversmoothing_frame,
)

RESULT_DIR = Path("benchmark/results")
REPORT_PATH = RESULT_DIR / "spatial_calibration.json"


def _base_frame(seed: int = 0, width: int = 320, height: int = 180) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    wave = 40.0 * np.sin(x / 11.0) + 28.0 * np.cos(y / 9.0)
    texture = rng.normal(0.0, 14.0, size=(height, width))
    gray = np.clip(128.0 + wave + texture, 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame, (35, 35), (130, 130), (70, 150, 220), -1)
    cv2.circle(frame, (235, 88), 45, (190, 95, 80), -1)
    cv2.putText(frame, "VG", (130, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 230), 2, cv2.LINE_AA)
    return frame


def _blur(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(frame, (0, 0), 3.0)


def _blocky(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(2, w // 8), max(2, h // 8)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _oversharpen(frame: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
    return cv2.addWeighted(frame, 2.8, blur, -1.8, 0)


def _oversmooth(frame: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(frame, 15, 110, 110)


def _color_shift(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab[..., 1] = np.clip(lab[..., 1] + 30, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] - 26, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _signals(frame: np.ndarray) -> dict[str, float]:
    block = measure_blockiness_frame(frame)
    sharpen = measure_oversharpening_frame(frame)
    smooth = measure_oversmoothing_frame(frame)
    color = measure_color_shift_frame(frame)
    return {
        "blur": measure_blur_frame(frame),
        "blocky": float(block["ratio"]),
        "oversharpening": float(sharpen["edge_residual_mean"]),
        "oversmooth": float(smooth["fine_to_coarse_ratio"]),
        "color_shift": float(color["chroma_bias"]),
    }


def _summarize(clean: list[float], degraded: list[float], direction: str) -> dict:
    clean_med = float(np.median(clean))
    degraded_med = float(np.median(degraded))
    if direction == "lower":
        separated = degraded_med < clean_med
        ratio = clean_med / max(degraded_med, 1e-6)
    else:
        separated = degraded_med > clean_med
        ratio = degraded_med / max(clean_med, 1e-6)
    return {
        "clean_median": clean_med,
        "degraded_median": degraded_med,
        "direction": direction,
        "separated": bool(separated),
        "separation_ratio": float(ratio),
    }


def run() -> dict:
    transforms = {
        "blur": (_blur, "lower"),
        "blocky": (_blocky, "higher"),
        "oversharpening": (_oversharpen, "higher"),
        "oversmooth": (_oversmooth, "lower"),
        "color_shift": (_color_shift, "higher"),
    }

    report: dict[str, dict] = {}
    for artifact, (transform, direction) in transforms.items():
        clean_values: list[float] = []
        degraded_values: list[float] = []
        for seed in range(12):
            clean = _base_frame(seed)
            degraded = transform(clean)
            clean_values.append(_signals(clean)[artifact])
            degraded_values.append(_signals(degraded)[artifact])
        report[artifact] = _summarize(clean_values, degraded_values, direction)

    report["summary"] = {
        "artifact_count": 5,
        "separated_count": sum(1 for key, item in report.items() if key != "summary" and item["separated"]),
        "all_directionally_separated": all(
            item["separated"] for key, item in report.items() if key != "summary"
        ),
        "sample_pairs_per_artifact": 12,
        "calibration_scope": "synthetic_directional",
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
