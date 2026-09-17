from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _targeted_motion_metrics(
    video_path: Path,
    *,
    start_time: float,
    end_time: float,
    padding_seconds: float = 0.7,
    resize_width: int = 192,
) -> dict:
    """Re-analyze a narrow window around a suspected motion defect.

    The first pass identifies candidate spikes. This second pass compares optical-flow
    magnitude inside the candidate window against nearby motion before/after it. A large
    local contrast is stronger evidence of a true temporal discontinuity than raw spike
    count alone.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"inside_median": 0.0, "outside_median": 0.0, "contrast_ratio": 0.0, "inside_samples": 0}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        return {"inside_median": 0.0, "outside_median": 0.0, "contrast_ratio": 0.0, "inside_samples": 0}

    resize_height = max(2, int(round(height * resize_width / width)))
    window_start = max(0.0, start_time - padding_seconds)
    window_end = max(end_time, start_time) + padding_seconds
    cap.set(cv2.CAP_PROP_POS_MSEC, window_start * 1000.0)

    ok, previous = cap.read()
    if not ok:
        cap.release()
        return {"inside_median": 0.0, "outside_median": 0.0, "contrast_ratio": 0.0, "inside_samples": 0}

    previous_gray = cv2.resize(
        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
        (resize_width, resize_height),
        interpolation=cv2.INTER_AREA,
    )

    inside: list[float] = []
    outside: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current_time = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if current_time > window_end:
            break

        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (resize_width, resize_height),
            interpolation=cv2.INTER_AREA,
        )
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
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
        score = float(np.percentile(magnitude, 90))

        if start_time <= current_time <= max(end_time, start_time + (1.0 / fps)):
            inside.append(score)
        else:
            outside.append(score)

        previous_gray = gray

    cap.release()

    inside_median = float(np.median(inside)) if inside else 0.0
    outside_median = float(np.median(outside)) if outside else 0.0
    contrast_ratio = inside_median / max(outside_median, 0.05)
    return {
        "inside_median": inside_median,
        "outside_median": outside_median,
        "contrast_ratio": contrast_ratio,
        "inside_samples": len(inside),
        "outside_samples": len(outside),
    }


def confirm_issue(
    issue: dict,
    *,
    targeted_metrics: dict | None = None,
    near_scene_change: bool = False,
) -> dict:
    """Decide whether a grouped first-pass issue is actionable after confirmation."""
    issue_type = issue["type"]
    start = float(issue.get("start_time", 0.0))
    end = float(issue.get("end_time", start))
    duration = max(0.0, end - start)
    confidence = float(issue.get("confidence", 0.0))
    details = dict(issue.get("details", {}))
    raw_event_count = int(details.get("raw_event_count", 1))

    if issue_type == "scene_change":
        return {
            "status": "context",
            "confirmed": False,
            "reason": "Scene changes can be intentional edits and do not fail a video by themselves.",
        }

    if issue_type == "flicker":
        return {
            "status": "review",
            "confirmed": False,
            "reason": "Brightness variation is retained for review but is not severe enough to fail the video without corroboration.",
        }

    if issue_type == "freeze":
        if start < 1.0:
            return {
                "status": "review",
                "confirmed": False,
                "reason": "A static opening can be an intentional title/hold, so the first second is not auto-failed.",
            }
        if near_scene_change:
            return {
                "status": "review",
                "confirmed": False,
                "reason": "The static interval occurs beside a scene transition, so it is retained for review instead of automatic failure.",
            }
        if 0.6 <= duration <= 3.0 and confidence >= 0.995:
            return {
                "status": "confirmed",
                "confirmed": True,
                "reason": "A sustained high-confidence freeze away from an edit boundary survived second-pass filtering.",
            }
        if duration > 3.0:
            return {
                "status": "review",
                "confirmed": False,
                "reason": "A long static segment may be intentional content, so it requires human review instead of automatic failure.",
            }
        return {
            "status": "review",
            "confirmed": False,
            "reason": "The static interval is too short or insufficiently confident for automatic failure.",
        }

    if issue_type == "motion_discontinuity":
        if near_scene_change:
            return {
                "status": "review",
                "confirmed": False,
                "reason": "The motion spike overlaps an edit boundary and is therefore treated as likely transition motion.",
                "metrics": targeted_metrics or {},
            }

        metrics = targeted_metrics or {}
        contrast_ratio = float(metrics.get("contrast_ratio", 0.0))
        inside_samples = int(metrics.get("inside_samples", 0))
        if (
            raw_event_count >= 6
            and duration >= 0.25
            and confidence >= 0.97
            and inside_samples >= 3
            and contrast_ratio >= 3.0
        ):
            return {
                "status": "confirmed",
                "confirmed": True,
                "reason": "Targeted optical-flow re-analysis found motion inside the suspect window at least 3× stronger than nearby motion.",
                "metrics": metrics,
            }
        return {
            "status": "review",
            "confirmed": False,
            "reason": "The candidate did not remain sufficiently distinct from nearby motion during targeted OpenCV re-analysis.",
            "metrics": metrics,
        }

    return {
        "status": "review",
        "confirmed": False,
        "reason": "Unknown issue type retained for human review.",
    }


def apply_second_pass(
    issues: list[dict],
    *,
    video_label: str,
    video_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    reviewed: list[dict] = []
    trace: list[dict] = []
    scene_times = [float(item["start_time"]) for item in issues if item["type"] == "scene_change"]

    for issue in issues:
        start = float(issue.get("start_time", 0.0))
        end = float(issue.get("end_time", start))
        near_scene_change = any((start - 0.35) <= scene_time <= (end + 0.35) for scene_time in scene_times)

        metrics = None
        if issue["type"] == "motion_discontinuity" and video_path is not None:
            metrics = _targeted_motion_metrics(video_path, start_time=start, end_time=end)

        decision = confirm_issue(
            issue,
            targeted_metrics=metrics,
            near_scene_change=near_scene_change,
        )
        enriched = dict(issue)
        enriched["confirmation"] = decision
        reviewed.append(enriched)
        trace.append({
            "video": video_label,
            "type": issue["type"],
            "start_time": issue["start_time"],
            "end_time": issue["end_time"],
            "action": "targeted_opencv_confirmation" if issue["type"] == "motion_discontinuity" else "policy_confirmation",
            "result": decision["status"],
            "reason": decision["reason"],
            "metrics": decision.get("metrics", {}),
        })

    return reviewed, trace
