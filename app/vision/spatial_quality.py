from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def measure_blur_frame(frame: np.ndarray) -> float:
    """Return Laplacian variance; lower values indicate less high-frequency detail."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 640:
        scale = 640.0 / gray.shape[1]
        gray = cv2.resize(gray, (640, max(2, int(round(gray.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_blockiness_frame(frame: np.ndarray, block_size: int = 8) -> dict:
    """Estimate 8-pixel block-boundary discontinuity relative to ordinary gradients."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.shape[1] > 640:
        scale = 640.0 / gray.shape[1]
        gray = cv2.resize(gray, (640, max(2, int(round(gray.shape[0] * scale)))), interpolation=cv2.INTER_AREA)

    vertical = np.abs(np.diff(gray, axis=1))
    horizontal = np.abs(np.diff(gray, axis=0))

    v_idx = np.arange(1, gray.shape[1])
    h_idx = np.arange(1, gray.shape[0])
    v_boundary = (v_idx % block_size) == 0
    h_boundary = (h_idx % block_size) == 0

    boundary_values: list[np.ndarray] = []
    interior_values: list[np.ndarray] = []
    if np.any(v_boundary):
        boundary_values.append(vertical[:, v_boundary])
        interior_values.append(vertical[:, ~v_boundary])
    if np.any(h_boundary):
        boundary_values.append(horizontal[h_boundary, :])
        interior_values.append(horizontal[~h_boundary, :])

    if not boundary_values or not interior_values:
        return {"ratio": 1.0, "boundary_strength": 0.0, "interior_strength": 0.0}

    boundary_strength = float(np.mean(np.concatenate([v.ravel() for v in boundary_values])))
    interior_strength = float(np.mean(np.concatenate([v.ravel() for v in interior_values])))
    ratio = boundary_strength / max(interior_strength, 1e-6)
    return {
        "ratio": ratio,
        "boundary_strength": boundary_strength,
        "interior_strength": interior_strength,
    }


def analyze_spatial_quality(
    video_path: Path,
    *,
    sample_interval_seconds: float = 0.5,
    blur_candidate_threshold: float = 45.0,
    blockiness_ratio_threshold: float = 1.45,
    blockiness_strength_threshold: float = 6.0,
) -> dict:
    """Sample a video for provisional blur/blockiness evidence.

    Candidate thresholds are intentionally provisional. They surface measurable evidence
    for calibration and do not directly assign MOS or final Sage artifact labels.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata: {video_path}")

    duration = frame_count / fps
    times: list[float] = []
    blur_values: list[float] = []
    block_ratios: list[float] = []
    boundary_strengths: list[float] = []

    t = 0.0
    while t <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        times.append(t)
        blur_values.append(measure_blur_frame(frame))
        block = measure_blockiness_frame(frame)
        block_ratios.append(float(block["ratio"]))
        boundary_strengths.append(float(block["boundary_strength"]))
        t += sample_interval_seconds
    cap.release()

    if not times:
        return {"sample_count": 0, "candidate_artifacts": [], "metrics": {}}

    blur_arr = np.asarray(blur_values, dtype=np.float32)
    block_arr = np.asarray(block_ratios, dtype=np.float32)
    strength_arr = np.asarray(boundary_strengths, dtype=np.float32)

    blur_median = float(np.median(blur_arr))
    blur_low_fraction = float(np.mean(blur_arr < blur_candidate_threshold))
    block_median = float(np.median(block_arr))
    block_high_fraction = float(np.mean((block_arr >= blockiness_ratio_threshold) & (strength_arr >= blockiness_strength_threshold)))

    candidates: list[dict] = []
    if blur_low_fraction >= 0.50:
        candidates.append({
            "artifact": "blurring",
            "status": "candidate",
            "confidence": min(0.95, 0.45 + 0.5 * blur_low_fraction),
            "reason": "Low Laplacian detail energy persisted across at least half of sampled frames; calibration/context is still required to distinguish artifact blur from intentional blur.",
            "metrics": {
                "median_laplacian_variance": blur_median,
                "low_detail_fraction": blur_low_fraction,
                "provisional_threshold": blur_candidate_threshold,
            },
        })

    if block_high_fraction >= 0.35:
        candidates.append({
            "artifact": "blocky",
            "status": "candidate",
            "confidence": min(0.95, 0.45 + 0.5 * block_high_fraction),
            "reason": "8-pixel boundary discontinuities were repeatedly stronger than ordinary image gradients; benchmark calibration is still required before assigning a final artifact label.",
            "metrics": {
                "median_boundary_ratio": block_median,
                "high_blockiness_fraction": block_high_fraction,
                "ratio_threshold": blockiness_ratio_threshold,
                "boundary_strength_threshold": blockiness_strength_threshold,
            },
        })

    return {
        "sample_count": len(times),
        "candidate_artifacts": candidates,
        "metrics": {
            "median_laplacian_variance": blur_median,
            "blur_low_detail_fraction": blur_low_fraction,
            "median_blockiness_ratio": block_median,
            "blockiness_high_fraction": block_high_fraction,
        },
        "calibration_status": "provisional",
    }
