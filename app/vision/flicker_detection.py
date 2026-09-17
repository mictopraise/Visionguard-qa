from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FlickerWindow:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    score: float


def detect_flicker_windows(
    video_path: Path,
    *,
    z_threshold: float = 4.0,
    min_events: int = 3,
    merge_gap_frames: int = 3,
) -> list[FlickerWindow]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS for video: {video_path}")

    brightness: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(np.mean(gray)))
    cap.release()

    if len(brightness) < 4:
        return []

    diffs = np.abs(np.diff(np.asarray(brightness, dtype=np.float32)))
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    scale = max(1e-6, 1.4826 * mad)
    zscores = (diffs - median) / scale
    event_frames = [int(i + 1) for i, z in enumerate(zscores) if z >= z_threshold]

    if len(event_frames) < min_events:
        return []

    groups: list[list[int]] = [[event_frames[0]]]
    for frame in event_frames[1:]:
        if frame - groups[-1][-1] <= merge_gap_frames:
            groups[-1].append(frame)
        else:
            groups.append([frame])

    windows: list[FlickerWindow] = []
    for group in groups:
        if len(group) < min_events:
            continue
        start = group[0]
        end = group[-1]
        score = float(max(zscores[max(0, start - 1): min(len(zscores), end)]))
        windows.append(
            FlickerWindow(
                start_frame=start,
                end_frame=end,
                start_time=start / fps,
                end_time=end / fps,
                score=score,
            )
        )
    return windows
