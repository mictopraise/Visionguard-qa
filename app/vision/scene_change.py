from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneChange:
    frame: int
    time: float
    score: float


def detect_scene_changes(
    video_path: Path,
    *,
    threshold: float = 0.55,
    min_gap_frames: int = 8,
) -> list[SceneChange]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS for video: {video_path}")

    ok, prev = cap.read()
    if not ok:
        cap.release()
        return []

    prev_hsv = cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)
    prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(prev_hist, prev_hist)

    changes: list[SceneChange] = []
    frame_index = 1
    last_change = -10_000

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        correlation = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
        score = float(np.clip(1.0 - correlation, 0.0, 2.0))

        if score >= threshold and frame_index - last_change >= min_gap_frames:
            changes.append(SceneChange(frame=frame_index, time=frame_index / fps, score=score))
            last_change = frame_index

        prev_hist = hist
        frame_index += 1

    cap.release()
    return changes
