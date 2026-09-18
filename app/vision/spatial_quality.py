from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _resize_gray(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > max_width:
        scale = max_width / gray.shape[1]
        gray = cv2.resize(
            gray,
            (max_width, max(2, int(round(gray.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def measure_blur_frame(frame: np.ndarray) -> float:
    """Return Laplacian variance; lower values indicate less high-frequency detail."""
    gray = _resize_gray(frame)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_blockiness_frame(frame: np.ndarray, block_size: int = 8) -> dict:
    """Estimate 8-pixel block-boundary discontinuity relative to ordinary gradients."""
    gray = _resize_gray(frame).astype(np.float32)

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


def measure_oversharpening_frame(frame: np.ndarray) -> dict:
    """Measure edge overshoot/high-frequency excess associated with aggressive sharpening.

    This is intentionally a low-level signal, not a semantic artifact decision.
    """
    gray = _resize_gray(frame).astype(np.float32)
    base = cv2.GaussianBlur(gray, (0, 0), 1.0)
    residual = gray - base

    residual_abs = np.abs(residual)
    edge_mask = np.abs(cv2.Laplacian(base, cv2.CV_32F)) >= 8.0
    if np.any(edge_mask):
        edge_residual = float(np.mean(residual_abs[edge_mask]))
        extreme_fraction = float(np.mean(residual_abs[edge_mask] >= 12.0))
    else:
        edge_residual = 0.0
        extreme_fraction = 0.0

    p95 = float(np.percentile(residual_abs, 95))
    return {
        "edge_residual_mean": edge_residual,
        "residual_p95": p95,
        "extreme_edge_fraction": extreme_fraction,
    }


def measure_oversmoothing_frame(frame: np.ndarray) -> dict:
    """Measure loss of fine texture while retaining coarse structure.

    Low fine-to-coarse detail ratio is evidence consistent with oversmoothing, but
    intentional low-texture content must be handled during calibration/context review.
    """
    gray = _resize_gray(frame).astype(np.float32)
    fine = gray - cv2.GaussianBlur(gray, (0, 0), 0.8)
    coarse = gray - cv2.GaussianBlur(gray, (0, 0), 2.4)

    fine_energy = float(np.mean(np.abs(fine)))
    coarse_energy = float(np.mean(np.abs(coarse)))
    detail_ratio = fine_energy / max(coarse_energy, 1e-6)

    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    textured_fraction = float(np.mean(gradient >= 12.0))
    return {
        "fine_energy": fine_energy,
        "coarse_energy": coarse_energy,
        "fine_to_coarse_ratio": detail_ratio,
        "textured_fraction": textured_fraction,
    }


def analyze_spatial_quality(
    video_path: Path,
    *,
    sample_interval_seconds: float = 0.5,
    blur_candidate_threshold: float = 45.0,
    blockiness_ratio_threshold: float = 1.45,
    blockiness_strength_threshold: float = 6.0,
    oversharpen_edge_threshold: float = 6.5,
    oversharpen_extreme_fraction_threshold: float = 0.08,
    oversmooth_ratio_threshold: float = 0.32,
    oversmooth_texture_fraction_threshold: float = 0.18,
) -> dict:
    """Sample a video for provisional spatial-quality evidence.

    Thresholds remain provisional. Measurements surface calibration candidates and do
    not directly assign MOS or final Sage artifact labels.
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
    sharpen_edge_values: list[float] = []
    sharpen_extreme_fractions: list[float] = []
    smooth_ratios: list[float] = []
    texture_fractions: list[float] = []

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

        sharpen = measure_oversharpening_frame(frame)
        sharpen_edge_values.append(float(sharpen["edge_residual_mean"]))
        sharpen_extreme_fractions.append(float(sharpen["extreme_edge_fraction"]))

        smooth = measure_oversmoothing_frame(frame)
        smooth_ratios.append(float(smooth["fine_to_coarse_ratio"]))
        texture_fractions.append(float(smooth["textured_fraction"]))

        t += sample_interval_seconds
    cap.release()

    if not times:
        return {"sample_count": 0, "candidate_artifacts": [], "metrics": {}}

    blur_arr = np.asarray(blur_values, dtype=np.float32)
    block_arr = np.asarray(block_ratios, dtype=np.float32)
    strength_arr = np.asarray(boundary_strengths, dtype=np.float32)
    sharpen_edge_arr = np.asarray(sharpen_edge_values, dtype=np.float32)
    sharpen_extreme_arr = np.asarray(sharpen_extreme_fractions, dtype=np.float32)
    smooth_ratio_arr = np.asarray(smooth_ratios, dtype=np.float32)
    texture_arr = np.asarray(texture_fractions, dtype=np.float32)

    blur_median = float(np.median(blur_arr))
    blur_low_fraction = float(np.mean(blur_arr < blur_candidate_threshold))
    block_median = float(np.median(block_arr))
    block_high_fraction = float(
        np.mean(
            (block_arr >= blockiness_ratio_threshold)
            & (strength_arr >= blockiness_strength_threshold)
        )
    )

    sharpen_median = float(np.median(sharpen_edge_arr))
    sharpen_extreme_median = float(np.median(sharpen_extreme_arr))
    sharpen_high_fraction = float(
        np.mean(
            (sharpen_edge_arr >= oversharpen_edge_threshold)
            & (sharpen_extreme_arr >= oversharpen_extreme_fraction_threshold)
        )
    )

    smooth_ratio_median = float(np.median(smooth_ratio_arr))
    texture_median = float(np.median(texture_arr))
    smooth_low_fraction = float(
        np.mean(
            (smooth_ratio_arr <= oversmooth_ratio_threshold)
            & (texture_arr <= oversmooth_texture_fraction_threshold)
        )
    )

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

    if sharpen_high_fraction >= 0.35:
        candidates.append({
            "artifact": "oversharpening",
            "status": "candidate",
            "confidence": min(0.95, 0.45 + 0.5 * sharpen_high_fraction),
            "reason": "High-frequency edge residuals and extreme edge overshoot persisted across sampled frames; this is provisional evidence for oversharpening/ringing.",
            "metrics": {
                "median_edge_residual": sharpen_median,
                "median_extreme_edge_fraction": sharpen_extreme_median,
                "high_oversharpen_fraction": sharpen_high_fraction,
                "edge_threshold": oversharpen_edge_threshold,
                "extreme_fraction_threshold": oversharpen_extreme_fraction_threshold,
            },
        })

    if smooth_low_fraction >= 0.35:
        candidates.append({
            "artifact": "oversmooth",
            "status": "candidate",
            "confidence": min(0.95, 0.45 + 0.5 * smooth_low_fraction),
            "reason": "Fine texture energy remained weak relative to coarse structure across sampled frames; this is provisional evidence for oversmoothing/detail loss.",
            "metrics": {
                "median_fine_to_coarse_ratio": smooth_ratio_median,
                "median_textured_fraction": texture_median,
                "low_texture_fraction": smooth_low_fraction,
                "ratio_threshold": oversmooth_ratio_threshold,
                "texture_fraction_threshold": oversmooth_texture_fraction_threshold,
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
            "median_oversharpen_edge_residual": sharpen_median,
            "oversharpen_high_fraction": sharpen_high_fraction,
            "median_fine_to_coarse_ratio": smooth_ratio_median,
            "oversmooth_low_texture_fraction": smooth_low_fraction,
        },
        "calibration_status": "provisional",
    }
