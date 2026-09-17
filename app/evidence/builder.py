from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.evidence.frame_export import export_evidence_frame


def severity_for_issue(issue_type: str, confidence: float, duration: float) -> str:
    if issue_type in {"freeze", "motion_discontinuity"}:
        if confidence >= 0.9 or duration >= 1.0:
            return "major"
        return "moderate"
    if issue_type in {"flicker", "scene_change"}:
        return "moderate" if confidence >= 0.75 else "minor"
    return "minor"


def build_evidence_cards(
    video_path: Path,
    issues: list[dict],
    evidence_root: Path,
    *,
    video_label: str,
) -> list[dict]:
    cards: list[dict] = []

    for issue in issues:
        start_time = float(issue["start_time"])
        end_time = float(issue["end_time"])
        midpoint = start_time if end_time <= start_time else (start_time + end_time) / 2.0
        confidence = float(issue.get("confidence", 0.0))
        duration = max(0.0, end_time - start_time)

        issue_id = f"VG-{uuid4().hex[:8].upper()}"
        frame_path = export_evidence_frame(
            video_path,
            evidence_root / video_label,
            timestamp=midpoint,
            label=issue_id,
        )

        cards.append(
            {
                "issue_id": issue_id,
                "video": video_label,
                "type": issue["type"],
                "severity": severity_for_issue(issue["type"], confidence, duration),
                "confidence": round(confidence, 4),
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration_seconds": round(duration, 3),
                "evidence_frames": [frame_path],
                "details": issue.get("details", {}),
            }
        )

    return cards


def build_timeline(cards: list[dict]) -> list[dict]:
    return sorted(
        [
            {
                "issue_id": card["issue_id"],
                "video": card["video"],
                "type": card["type"],
                "severity": card["severity"],
                "start_time": card["start_time"],
                "end_time": card["end_time"],
            }
            for card in cards
        ],
        key=lambda item: (item["start_time"], item["video"], item["type"]),
    )
