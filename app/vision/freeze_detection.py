from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass(frozen=True)
class FreezeWindow:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    confidence: float


def detect_freeze_windows(
    video_path: Path,
    similarity_threshold: float = 0.997,
    min_duration_seconds: float = 0.4,
) -> list[FreezeWindow]:
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

    prev = cv2.resize(
        cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
        (160, 90),
        interpolation=cv2.INTER_AREA,
    )

    min_frames = max(2, int(round(min_duration_seconds * fps)))
    windows: list[FreezeWindow] = []
    run_start: int | None = None
    run_scores: list[float] = []
    frame_index = 1

    def close_run(run_end: int) -> None:
        nonlocal run_start, run_scores
        if run_start is None:
            return

        repeated_frames = run_end - run_start + 1
        if repeated_frames >= min_frames:
            start_time = run_start / fps
            end_time = run_end / fps
            windows.append(
                FreezeWindow(
                    start_frame=run_start,
                    end_frame=run_end,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=max(0.0, end_time - start_time),
                    confidence=float(np.mean(run_scores)) if run_scores else 0.0,
                )
            )

        run_start = None
        run_scores = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (160, 90),
            interpolation=cv2.INTER_AREA,
        )
        similarity = max(
            0.0,
            1.0 - float(np.mean(cv2.absdiff(prev, gray))) / 255.0,
        )

        if similarity >= similarity_threshold:
            if run_start is None:
                run_start = frame_index - 1
                run_scores = []
            run_scores.append(similarity)
        else:
            close_run(frame_index - 1)

        prev = gray
        frame_index += 1

    close_run(frame_index - 1)
    cap.release()
    return windows
