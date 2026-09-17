from __future__ import annotations

from pathlib import Path

from app.agent.confirmation import apply_second_pass
from app.evidence.builder import build_evidence_cards, build_timeline
from app.vision.flicker_detection import detect_flicker_windows
from app.vision.freeze_detection import detect_freeze_windows
from app.vision.motion_analysis import analyze_motion
from app.vision.scene_change import detect_scene_changes


def _cluster_issues(issues: list[dict], *, merge_gap_seconds: float = 0.45) -> list[dict]:
    """Merge nearby findings of the same type into one human-reviewable event."""
    if not issues:
        return []

    grouped: dict[str, list[dict]] = {}
    for issue in issues:
        grouped.setdefault(issue["type"], []).append(issue)

    clustered: list[dict] = []
    for issue_type, same_type in grouped.items():
        same_type.sort(key=lambda item: item["start_time"])
        current = dict(same_type[0])
        current["details"] = dict(current.get("details", {}))
        current["details"]["raw_event_count"] = 1

        for item in same_type[1:]:
            if float(item["start_time"]) <= float(current["end_time"]) + merge_gap_seconds:
                current["end_time"] = max(float(current["end_time"]), float(item["end_time"]))
                current["confidence"] = max(float(current.get("confidence", 0.0)), float(item.get("confidence", 0.0)))
                current["details"]["raw_event_count"] += 1
                if "spike_ratio" in item.get("details", {}):
                    current["details"]["max_spike_ratio"] = max(
                        float(current["details"].get("max_spike_ratio", current["details"].get("spike_ratio", 0.0))),
                        float(item["details"]["spike_ratio"]),
                    )
            else:
                clustered.append(current)
                current = dict(item)
                current["details"] = dict(current.get("details", {}))
                current["details"]["raw_event_count"] = 1

        clustered.append(current)

    clustered.sort(key=lambda issue: (issue["start_time"], issue["type"]))
    return clustered


def _final_verdict(result_a: dict, result_b: dict) -> dict:
    all_issues = [("A", issue) for issue in result_a["issues"]] + [("B", issue) for issue in result_b["issues"]]

    if not all_issues:
        return {
            "status": "PASS",
            "action": "PASS",
            "summary": "No significant visual anomalies were detected in either video.",
        }

    confirmed = []
    review_only = []
    for video, issue in all_issues:
        state = issue.get("confirmation", {})
        if state.get("confirmed"):
            confirmed.append((video, issue))
        elif state.get("status") == "review":
            review_only.append((video, issue))

    if confirmed:
        affected_videos = sorted({video for video, _ in confirmed})
        affected = f"Video {affected_videos[0]}" if len(affected_videos) == 1 else "Both videos"
        return {
            "status": "FAIL",
            "action": "HUMAN_REVIEW",
            "summary": f"{affected} contains one or more anomalies that survived targeted second-pass confirmation.",
        }

    if review_only:
        return {
            "status": "REVIEW",
            "action": "HUMAN_REVIEW",
            "summary": "Potential anomalies were detected, but none met the stricter second-pass threshold for automatic failure.",
        }

    return {
        "status": "PASS",
        "action": "PASS",
        "summary": "Only contextual scene changes were detected; no actionable visual defect was confirmed.",
    }


def analyze_video_first_pass(video_path: Path, *, video_label: str = "?") -> dict:
    freeze_windows = detect_freeze_windows(video_path)
    _, motion_anomalies = analyze_motion(video_path)
    flicker_windows = detect_flicker_windows(video_path)
    scene_changes = detect_scene_changes(video_path)

    raw_issues: list[dict] = []

    for item in freeze_windows:
        raw_issues.append({
            "type": "freeze",
            "start_time": item.start_time,
            "end_time": item.end_time,
            "confidence": item.confidence,
            "details": {"start_frame": item.start_frame, "end_frame": item.end_frame},
        })

    for item in motion_anomalies:
        raw_issues.append({
            "type": "motion_discontinuity",
            "start_time": item.time_seconds,
            "end_time": item.time_seconds,
            "confidence": min(1.0, item.spike_ratio / 10.0),
            "details": {
                "frame": item.frame_index,
                "magnitude": item.magnitude,
                "baseline": item.baseline,
                "spike_ratio": item.spike_ratio,
                "max_spike_ratio": item.spike_ratio,
            },
        })

    for item in flicker_windows:
        raw_issues.append({
            "type": "flicker",
            "start_time": item.start_time,
            "end_time": item.end_time,
            "confidence": min(1.0, item.score / 10.0),
            "details": {"start_frame": item.start_frame, "end_frame": item.end_frame, "score": item.score},
        })

    for item in scene_changes:
        raw_issues.append({
            "type": "scene_change",
            "start_time": item.time,
            "end_time": item.time,
            "confidence": min(1.0, item.score),
            "details": {"frame": item.frame, "score": item.score},
        })

    raw_issues.sort(key=lambda issue: (issue["start_time"], issue["type"]))
    clustered = _cluster_issues(raw_issues)
    issues, trace = apply_second_pass(clustered, video_label=video_label, video_path=video_path)

    return {
        "video": video_path.name,
        "raw_event_count": len(raw_issues),
        "issue_count": len(issues),
        "confirmed_issue_count": sum(1 for item in issues if item.get("confirmation", {}).get("confirmed")),
        "review_issue_count": sum(1 for item in issues if item.get("confirmation", {}).get("status") == "review"),
        "issues": issues,
        "agent_trace": trace,
    }


def compare_first_pass(video_a: Path, video_b: Path, *, evidence_root: Path | None = None) -> dict:
    result_a = analyze_video_first_pass(video_a, video_label="A")
    result_b = analyze_video_first_pass(video_b, video_label="B")

    total_issues = result_a["issue_count"] + result_b["issue_count"]
    total_raw_events = result_a["raw_event_count"] + result_b["raw_event_count"]
    total_confirmed = result_a["confirmed_issue_count"] + result_b["confirmed_issue_count"]
    total_review = result_a["review_issue_count"] + result_b["review_issue_count"]
    verdict = _final_verdict(result_a, result_b)

    response = {
        "disposition": verdict["action"],
        "final_verdict": verdict,
        "video_a": result_a,
        "video_b": result_b,
        "total_issues": total_issues,
        "total_raw_events": total_raw_events,
        "total_confirmed_issues": total_confirmed,
        "total_review_issues": total_review,
        "agent_trace": result_a["agent_trace"] + result_b["agent_trace"],
    }

    if evidence_root is not None:
        confirmed_a = [i for i in result_a["issues"] if i.get("confirmation", {}).get("confirmed")]
        confirmed_b = [i for i in result_b["issues"] if i.get("confirmation", {}).get("confirmed")]

        # If nothing is confirmed, show only a small review sample rather than dozens of cards.
        if confirmed_a or confirmed_b:
            display_a, display_b = confirmed_a, confirmed_b
        else:
            review_a = [i for i in result_a["issues"] if i.get("confirmation", {}).get("status") == "review"][:3]
            review_b = [i for i in result_b["issues"] if i.get("confirmation", {}).get("status") == "review"][:3]
            display_a, display_b = review_a, review_b

        cards_a = build_evidence_cards(video_a, display_a, evidence_root, video_label="A")
        cards_b = build_evidence_cards(video_b, display_b, evidence_root, video_label="B")
        cards = cards_a + cards_b
        response["evidence_cards"] = cards
        response["timeline"] = build_timeline(cards)

    return response
