from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.video.alignment import estimate_temporal_offset


@dataclass(frozen=True)
class PairwiseSample:
    time_a: float
    time_b: float
    visual_difference: float
    brightness_delta: float
    motion_a: float
    motion_b: float
    motion_ratio: float


@dataclass(frozen=True)
class PairwiseIssue:
    type: str
    start_time: float
    end_time: float
    confidence: float
    affected_video: str
    details: dict


def _fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS for video: {path}")
    return fps


def _gray(frame: np.ndarray, size: tuple[int, int] = (192, 108)) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)


def _motion(prev: np.ndarray | None, current: np.ndarray) -> float:
    if prev is None:
        return 0.0
    diff = cv2.absdiff(prev, current)
    return float(np.mean(diff))


def compare_aligned_videos(
    video_a: Path,
    video_b: Path,
    *,
    sample_interval_seconds: float = 0.20,
    max_offset_seconds: float = 2.0,
) -> dict:
    fps_a = _fps(video_a)
    fps_b = _fps(video_b)
    alignment_fps = min(fps_a, fps_b)
    alignment = estimate_temporal_offset(video_a, video_b, alignment_fps, max_offset_seconds=max_offset_seconds)

    cap_a = cv2.VideoCapture(str(video_a))
    cap_b = cv2.VideoCapture(str(video_b))
    if not cap_a.isOpened() or not cap_b.isOpened():
        cap_a.release(); cap_b.release()
        raise ValueError("Unable to open one or both videos")

    duration_a = float(cap_a.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / fps_a
    duration_b = float(cap_b.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / fps_b

    offset = alignment.offset_seconds
    start_a = max(0.0, -offset)
    start_b = max(0.0, offset)
    overlap = min(duration_a - start_a, duration_b - start_b)

    samples: list[PairwiseSample] = []
    prev_a = None
    prev_b = None
    t = 0.0
    while t <= max(0.0, overlap):
        ta = start_a + t
        tb = start_b + t
        cap_a.set(cv2.CAP_PROP_POS_MSEC, ta * 1000.0)
        cap_b.set(cv2.CAP_PROP_POS_MSEC, tb * 1000.0)
        oka, fa = cap_a.read()
        okb, fb = cap_b.read()
        if not oka or not okb:
            break

        ga = _gray(fa)
        gb = _gray(fb)
        visual_difference = float(np.mean(cv2.absdiff(ga, gb))) / 255.0
        brightness_delta = abs(float(np.mean(ga)) - float(np.mean(gb))) / 255.0
        ma = _motion(prev_a, ga)
        mb = _motion(prev_b, gb)
        motion_ratio = (max(ma, mb) + 1.0) / (min(ma, mb) + 1.0)
        samples.append(PairwiseSample(ta, tb, visual_difference, brightness_delta, ma, mb, motion_ratio))
        prev_a, prev_b = ga, gb
        t += sample_interval_seconds

    cap_a.release(); cap_b.release()

    if len(samples) < 5:
        return {
            "alignment": alignment.__dict__,
            "sample_count": len(samples),
            "issues": [],
            "summary": "Insufficient aligned samples for comparative QA.",
        }

    visual_values = np.array([s.visual_difference for s in samples], dtype=np.float32)
    visual_baseline = float(np.median(visual_values))
    visual_mad = float(np.median(np.abs(visual_values - visual_baseline)))
    visual_threshold = visual_baseline + max(0.08, 5.0 * 1.4826 * visual_mad)

    candidates: list[dict] = []
    for s in samples:
        if s.motion_ratio >= 3.0 and max(s.motion_a, s.motion_b) >= 8.0:
            affected = "A" if s.motion_a > s.motion_b else "B"
            candidates.append({
                "type": "pairwise_motion_mismatch",
                "time": s.time_a,
                "confidence": min(1.0, (s.motion_ratio - 1.0) / 4.0),
                "affected_video": affected,
                "details": {"motion_a": s.motion_a, "motion_b": s.motion_b, "motion_ratio": s.motion_ratio},
            })
        elif s.brightness_delta >= 0.18 and s.visual_difference >= visual_threshold:
            candidates.append({
                "type": "pairwise_brightness_mismatch",
                "time": s.time_a,
                "confidence": min(1.0, s.brightness_delta / 0.35),
                "affected_video": "B",
                "details": {"brightness_delta": s.brightness_delta, "visual_difference": s.visual_difference},
            })
        elif s.visual_difference >= visual_threshold * 1.35:
            candidates.append({
                "type": "pairwise_visual_mismatch",
                "time": s.time_a,
                "confidence": min(1.0, s.visual_difference / max(visual_threshold * 1.8, 1e-6)),
                "affected_video": "B",
                "details": {"visual_difference": s.visual_difference, "threshold": visual_threshold},
            })

    grouped: list[PairwiseIssue] = []
    if candidates:
        current = dict(candidates[0])
        current["start"] = current["time"]
        current["end"] = current["time"]
        current["count"] = 1
        for item in candidates[1:]:
            same = item["type"] == current["type"] and item["affected_video"] == current["affected_video"]
            nearby = item["time"] <= current["end"] + sample_interval_seconds * 1.6
            if same and nearby:
                current["end"] = item["time"]
                current["count"] += 1
                current["confidence"] = max(current["confidence"], item["confidence"])
            else:
                if current["count"] >= 2:
                    grouped.append(PairwiseIssue(current["type"], current["start"], current["end"], current["confidence"], current["affected_video"], {**current["details"], "sample_count": current["count"]}))
                current = dict(item)
                current["start"] = item["time"]
                current["end"] = item["time"]
                current["count"] = 1
        if current["count"] >= 2:
            grouped.append(PairwiseIssue(current["type"], current["start"], current["end"], current["confidence"], current["affected_video"], {**current["details"], "sample_count": current["count"]}))

    return {
        "alignment": alignment.__dict__,
        "sample_count": len(samples),
        "visual_baseline": visual_baseline,
        "visual_threshold": visual_threshold,
        "issues": [issue.__dict__ for issue in grouped],
        "summary": f"Compared {len(samples)} aligned samples with estimated offset {alignment.offset_seconds:.3f}s.",
    }
