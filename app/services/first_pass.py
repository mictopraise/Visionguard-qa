from __future__ import annotations

from pathlib import Path

from app.agent.confirmation import apply_second_pass
from app.evidence.builder import build_evidence_cards, build_timeline
from app.vision.flicker_detection import detect_flicker_windows
from app.vision.freeze_detection import detect_freeze_windows
from app.vision.motion_analysis import analyze_motion
from app.vision.pairwise_comparison import compare_aligned_videos
from app.vision.scene_change import detect_scene_changes
from app.vision.calibration import ARTIFACT_CALIBRATION, promotable_artifacts
from app.vision.spatial_quality import analyze_spatial_quality


def _cluster_issues(issues: list[dict], *, merge_gap_seconds: float = 0.45) -> list[dict]:
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


def _final_verdict(result_a: dict, result_b: dict, comparative: dict | None = None) -> dict:
    comparative_issues = list((comparative or {}).get("issues", []))
    if comparative_issues:
        affected = sorted({item.get("affected_video", "B") for item in comparative_issues})
        affected_text = f"Video {affected[0]}" if len(affected) == 1 else "The video pair"
        strong = [item for item in comparative_issues if float(item.get("confidence", 0.0)) >= 0.75]
        if strong:
            return {
                "status": "FAIL",
                "action": "REGENERATE" if affected == ["B"] else "HUMAN_REVIEW",
                "summary": f"{affected_text} shows sustained degradation relative to its aligned counterpart. Comparative evidence survived pairwise QA.",
            }
        return {
            "status": "RECHECK",
            "action": "HUMAN_REVIEW",
            "summary": "Aligned A/B comparison found sustained differences, but confidence is not high enough for automatic regeneration.",
        }

    all_issues = [("A", issue) for issue in result_a["issues"]] + [("B", issue) for issue in result_b["issues"]]
    if not all_issues:
        return {"status": "PASS", "action": "PASS", "summary": "No significant visual anomalies were detected in either video."}

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
        return {"status": "RECHECK", "action": "HUMAN_REVIEW", "summary": f"{affected} contains independently confirmed anomalies, but pairwise comparison did not establish comparative degradation."}

    if review_only:
        return {"status": "REVIEW", "action": "HUMAN_REVIEW", "summary": "Potential anomalies were detected, but none were confirmed as comparative degradation after alignment."}

    return {"status": "PASS", "action": "PASS", "summary": "Only contextual scene changes were detected; no actionable visual defect was confirmed."}


def _playback_quality_from_issues(issues: list[dict]) -> dict:
    """Map confirmed/reviewed temporal evidence to the human-facing playback result."""
    actionable_types = {"freeze", "motion_discontinuity", "flicker"}
    confirmed = [
        issue for issue in issues
        if issue.get("type") in actionable_types
        and issue.get("confirmation", {}).get("confirmed")
    ]
    review = [
        issue for issue in issues
        if issue.get("type") in actionable_types
        and issue.get("confirmation", {}).get("status") == "review"
    ]

    if confirmed:
        confidence = max(float(item.get("confidence", 0.0)) for item in confirmed)
        return {
            "label": "temporal_issues",
            "confidence": confidence,
            "reason": f"{len(confirmed)} temporal issue(s) survived confirmation.",
            "supporting_types": sorted({item["type"] for item in confirmed}),
        }

    if review:
        confidence = max(float(item.get("confidence", 0.0)) for item in review)
        return {
            "label": "uncertain",
            "confidence": confidence,
            "reason": f"{len(review)} temporal candidate(s) remain unresolved after confirmation.",
            "supporting_types": sorted({item["type"] for item in review}),
        }

    return {
        "label": "smooth",
        "confidence": 0.85,
        "reason": "No confirmed or unresolved playback irregularity remained after temporal QA.",
        "supporting_types": [],
    }


def _quality_record(video_label: str, spatial: dict, issues: list[dict]) -> dict:
    playback = _playback_quality_from_issues(issues)
    candidates = list(spatial.get("candidate_artifacts", []))
    allowed = promotable_artifacts()
    final_artifacts = [
        item["artifact"] for item in candidates
        if item.get("artifact") in allowed
    ]
    return {
        "video_label": video_label,
        "mos_score": None,
        "final_artifacts": final_artifacts,
        "spatial_candidates": candidates,
        "playback_quality": playback,
        "calibration_status": {
            name: calibration.status
            for name, calibration in ARTIFACT_CALIBRATION.items()
        },
        "notes": [
            "MOS is withheld until benchmark calibration is complete.",
            "Only artifacts with calibrated registry status may appear as final labels.",
        ],
    }


def analyze_video_first_pass(video_path: Path, *, video_label: str = "?") -> dict:
    freeze_windows = detect_freeze_windows(video_path)
    _, motion_anomalies = analyze_motion(video_path)
    flicker_windows = detect_flicker_windows(video_path)
    scene_changes = detect_scene_changes(video_path)
    spatial = analyze_spatial_quality(video_path)

    raw_issues: list[dict] = []
    for item in freeze_windows:
        raw_issues.append({"type": "freeze", "start_time": item.start_time, "end_time": item.end_time, "confidence": item.confidence, "details": {"start_frame": item.start_frame, "end_frame": item.end_frame}})
    for item in motion_anomalies:
        raw_issues.append({"type": "motion_discontinuity", "start_time": item.time_seconds, "end_time": item.time_seconds, "confidence": min(1.0, item.spike_ratio / 10.0), "details": {"frame": item.frame_index, "magnitude": item.magnitude, "baseline": item.baseline, "spike_ratio": item.spike_ratio, "max_spike_ratio": item.spike_ratio}})
    for item in flicker_windows:
        raw_issues.append({"type": "flicker", "start_time": item.start_time, "end_time": item.end_time, "confidence": min(1.0, item.score / 10.0), "details": {"start_frame": item.start_frame, "end_frame": item.end_frame, "score": item.score}})
    for item in scene_changes:
        raw_issues.append({"type": "scene_change", "start_time": item.time, "end_time": item.time, "confidence": min(1.0, item.score), "details": {"frame": item.frame, "score": item.score}})

    raw_issues.sort(key=lambda issue: (issue["start_time"], issue["type"]))
    clustered = _cluster_issues(raw_issues)
    issues, trace = apply_second_pass(clustered, video_label=video_label, video_path=video_path)
    quality_record = _quality_record(video_label, spatial, issues)
    return {
        "video": video_path.name,
        "raw_event_count": len(raw_issues),
        "issue_count": len(issues),
        "confirmed_issue_count": sum(1 for item in issues if item.get("confirmation", {}).get("confirmed")),
        "review_issue_count": sum(1 for item in issues if item.get("confirmation", {}).get("status") == "review"),
        "issues": issues,
        "spatial_quality": spatial,
        "quality_record": quality_record,
        "agent_trace": trace,
    }


def compare_first_pass(video_a: Path, video_b: Path, *, evidence_root: Path | None = None) -> dict:
    result_a = analyze_video_first_pass(video_a, video_label="A")
    result_b = analyze_video_first_pass(video_b, video_label="B")
    comparative = compare_aligned_videos(video_a, video_b)

    total_issues = result_a["issue_count"] + result_b["issue_count"]
    total_raw_events = result_a["raw_event_count"] + result_b["raw_event_count"]
    total_confirmed = result_a["confirmed_issue_count"] + result_b["confirmed_issue_count"]
    total_review = result_a["review_issue_count"] + result_b["review_issue_count"]
    total_spatial_candidates = len(result_a["spatial_quality"].get("candidate_artifacts", [])) + len(result_b["spatial_quality"].get("candidate_artifacts", []))
    verdict = _final_verdict(result_a, result_b, comparative)

    comparative_trace = [{
        "video": item.get("affected_video", "B"),
        "type": item["type"],
        "start_time": item["start_time"],
        "end_time": item["end_time"],
        "action": "pairwise_comparison",
        "result": "confirmed" if float(item.get("confidence", 0.0)) >= 0.75 else "review",
        "reason": "Sustained difference detected after temporal alignment and matched-frame comparison.",
    } for item in comparative.get("issues", [])]

    response = {
        "disposition": verdict["action"],
        "final_verdict": verdict,
        "video_a": result_a,
        "video_b": result_b,
        "comparative_analysis": comparative,
        "comparative_issue_count": len(comparative.get("issues", [])),
        "spatial_candidate_count": total_spatial_candidates,
        "total_issues": total_issues,
        "total_raw_events": total_raw_events,
        "total_confirmed_issues": total_confirmed,
        "total_review_issues": total_review,
        "agent_trace": result_a["agent_trace"] + result_b["agent_trace"] + comparative_trace,
    }

    if evidence_root is not None:
        confirmed_a = [i for i in result_a["issues"] if i.get("confirmation", {}).get("confirmed")]
        confirmed_b = [i for i in result_b["issues"] if i.get("confirmation", {}).get("confirmed")]
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
