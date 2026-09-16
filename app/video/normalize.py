from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2

@dataclass(frozen=True)
class NormalizationPlan:
    target_fps: float
    target_width: int
    target_height: int

def choose_normalization_plan(fps_a: float, width_a: int, height_a: int, fps_b: float, width_b: int, height_b: int, max_fps: float = 30.0) -> NormalizationPlan:
    valid_fps = [v for v in (fps_a, fps_b) if v > 0]
    target_fps = min(valid_fps) if valid_fps else 24.0
    target_fps = min(target_fps, max_fps)
    target_width = max(2, min(width_a, width_b))
    target_height = max(2, min(height_a, height_b))
    target_width -= target_width % 2
    target_height -= target_height % 2
    return NormalizationPlan(target_fps, target_width, target_height)

def normalize_video(source: Path, destination: Path, plan: NormalizationPlan) -> Path:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {source}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS for video: {source}")
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), plan.target_fps, (plan.target_width, plan.target_height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Unable to create normalized video: {destination}")
    frame_interval = 1.0 / plan.target_fps
    next_output_time = 0.0
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_index / source_fps
        if timestamp + 1e-9 >= next_output_time:
            resized = cv2.resize(frame, (plan.target_width, plan.target_height), interpolation=cv2.INTER_AREA)
            writer.write(resized)
            next_output_time += frame_interval
        frame_index += 1
    cap.release()
    writer.release()
    return destination
