from __future__ import annotations

from pathlib import Path

from app.evidence.builder import build_evidence_cards, build_timeline
from app.vision.flicker_detection import detect_flicker_windows
from app.vision.freeze_detection import detect_freeze_windows
from app.vision.motion_analysis import analyze_motion
from app.vision.scene_change import detect_scene_changes


def analyze_video_first_pass(video_path: Path) -> dict:
    freeze_windows = detect_freeze_windows(video_path)
    _, motion_anomalies = analyze_motion(video_path)
    flicker_windows = detect_flicker_windows(video_path)
    scene_changes = detect_scene_changes(video_path)

    issues: list[dict] = []

    for item in freeze_windows:
        issues.append({
            "type": "freeze",
            "start_time": item.start_time,
            "end_time": item.end_time,
            "confidence": item.confidence,
            "details": {
                "start_frame": item.start_frame,
                "end_frame": item.end_frame,
            },
        })

    for item in motion_anomalies:
        issues.append({
            "type": "motion_discontinuity",
            "start_time": item.time_seconds,
            "end_time": item.time_seconds,
            "confidence": min(1.0, item.spike_ratio / 10.0),
            "details": {
                "frame": item.frame_index,
                "magnitude": item.magnitude,
                "baseline": item.baseline,
                "spike_ratio": item.spike_ratio,
            },
        })

    for item in flicker_windows:
        issues.append({
            "type": "flicker",
            "start_time": item.start_time,
            "end_time": item.end_time,
            "confidence": min(1.0, item.score / 10.0),
            "details": {
                "start_frame": item.start_frame,
                "end_frame": item.end_frame,
                "score": item.score,
            },
        })

    for item in scene_changes:
        issues.append({
            "type": "scene_change",
            "start_time": item.time,
            "end_time": item.time,
            "confidence": min(1.0, item.score),
            "details": {
                "frame": item.frame,
                "score": item.score,
            },
        })

    issues.sort(key=lambda issue: (issue["start_time"], issue["type"]))

    return {
        "video": video_path.name,
        "issue_count": len(issues),
        "issues": issues,
    }


def compare_first_pass(
    video_a: Path,
    video_b: Path,
    *,
    evidence_root: Path | None = None,
) -> dict:
    result_a = analyze_video_first_pass(video_a)
    result_b = analyze_video_first_pass(video_b)

    total_issues = result_a["issue_count"] + result_b["issue_count"]
    disposition = "PASS" if total_issues == 0 else "RECHECK"

    response = {
        "disposition": disposition,
        "video_a": result_a,
        "video_b": result_b,
        "total_issues": total_issues,
    }

    if evidence_root is not None:
        cards_a = build_evidence_cards(
            video_a,
            result_a["issues"],
            evidence_root,
            video_label="A",
        )
        cards_b = build_evidence_cards(
            video_b,
            result_b["issues"],
            evidence_root,
            video_label="B",
        )
        cards = cards_a + cards_b
        response["evidence_cards"] = cards
        response["timeline"] = build_timeline(cards)

    return response
