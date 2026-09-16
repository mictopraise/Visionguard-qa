from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

@dataclass(frozen=True)
class AlignmentResult:
    offset_frames: int
    offset_seconds: float
    score: float
    compared_frames: int

def _read_gray_samples(path: Path, sample_step: int = 3, max_samples: int = 120) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    samples = []
    frame_index = 0
    while len(samples) < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append(cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA))
        frame_index += 1
    cap.release()
    return samples

def estimate_temporal_offset(video_a: Path, video_b: Path, fps: float, max_offset_seconds: float = 2.0, sample_step: int = 3) -> AlignmentResult:
    a = _read_gray_samples(video_a, sample_step)
    b = _read_gray_samples(video_b, sample_step)
    if not a or not b:
        raise ValueError("Insufficient frames for alignment")
    max_offset_samples = max(1, int((max_offset_seconds * fps) / sample_step))
    best_score, best_offset, best_count = -1.0, 0, 0
    for offset in range(-max_offset_samples, max_offset_samples + 1):
        start_a = max(0, -offset)
        start_b = max(0, offset)
        overlap = min(len(a) - start_a, len(b) - start_b)
        if overlap < 3:
            continue
        sims = []
        for i in range(overlap):
            diff = cv2.absdiff(a[start_a + i], b[start_b + i])
            sims.append(max(0.0, 1.0 - float(np.mean(diff)) / 255.0))
        score = float(np.mean(sims))
        if score > best_score:
            best_score, best_offset, best_count = score, offset, overlap
    offset_frames = best_offset * sample_step
    return AlignmentResult(offset_frames, offset_frames / fps, best_score, best_count)
