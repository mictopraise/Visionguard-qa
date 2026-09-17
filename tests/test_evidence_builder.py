from pathlib import Path

from app.evidence.builder import build_evidence_cards, build_timeline, severity_for_issue
from benchmark.create_synthetic import create_moving_square_video


def test_evidence_cards_export_frames_and_build_timeline(tmp_path: Path) -> None:
    video = tmp_path / "sample.avi"
    create_moving_square_video(video, fps=24.0, seconds=2.0)

    issues = [
        {
            "type": "freeze",
            "start_time": 0.5,
            "end_time": 1.0,
            "confidence": 0.95,
            "details": {"start_frame": 12, "end_frame": 24},
        }
    ]

    cards = build_evidence_cards(
        video,
        issues,
        tmp_path / "evidence",
        video_label="A",
    )

    assert len(cards) == 1
    card = cards[0]
    assert card["type"] == "freeze"
    assert card["severity"] == "major"
    assert card["video"] == "A"
    assert len(card["evidence_frames"]) == 1
    assert Path(card["evidence_frames"][0]).exists()

    timeline = build_timeline(cards)
    assert timeline[0]["issue_id"] == card["issue_id"]


def test_severity_rules() -> None:
    assert severity_for_issue("freeze", 0.95, 0.2) == "major"
    assert severity_for_issue("motion_discontinuity", 0.5, 0.2) == "moderate"
    assert severity_for_issue("flicker", 0.8, 0.1) == "moderate"
    assert severity_for_issue("scene_change", 0.5, 0.0) == "minor"
