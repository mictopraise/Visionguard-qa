from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionSample:
    frame_index: int
    time_seconds: float
    magnitude: float


@dataclass(frozen=True)
class MotionAnomaly:
    frame_index: int
    time_seconds: float
    magnitude: float
    baseline: float
    threshold: float
    spike_ratio: float


def _motion_score(previous_gray: np.ndarray, current_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.percentile(magnitude, 90))


def analyze_motion(video_path: Path, resize_width: int = 320) -> tuple[list[MotionSample], list[MotionAnomaly]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata: {video_path}")

    resize_height = max(2, int(round(height * resize_width / width)))

    ok, previous = cap.read()
    if not ok:
        cap.release()
        return [], []

    previous_gray = cv2.resize(
        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
        (resize_width, resize_height),
        interpolation=cv2.INTER_AREA,
    )

    samples: list[MotionSample] = []
    frame_index = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (resize_width, resize_height),
            interpolation=cv2.INTER_AREA,
        )
        score = _motion_score(previous_gray, current_gray)
        samples.append(
            MotionSample(
                frame_index=frame_index,
                time_seconds=frame_index / fps,
                magnitude=score,
            )
        )
        previous_gray = current_gray
        frame_index += 1

    cap.release()

    if len(samples) < 5:
        return samples, []

    values = np.array([sample.magnitude for sample in samples], dtype=np.float32)
    baseline = float(np.median(values))
    mad = float(np.median(np.abs(values - baseline)))
    robust_sigma = 1.4826 * mad
    threshold = baseline + max(1.0, 6.0 * robust_sigma)

    anomalies = [
        MotionAnomaly(
            frame_index=sample.frame_index,
            time_seconds=sample.time_seconds,
            magnitude=sample.magnitude,
            baseline=baseline,
            threshold=threshold,
            spike_ratio=(sample.magnitude / max(baseline, 1e-6)),
        )
        for sample in samples
        if sample.magnitude > threshold
    ]

    return samples, anomalies
