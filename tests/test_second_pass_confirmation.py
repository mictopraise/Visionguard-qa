from app.agent.confirmation import confirm_issue


def _issue(issue_type: str, start: float, end: float, confidence: float = 1.0, raw_events: int = 1) -> dict:
    return {
        "type": issue_type,
        "start_time": start,
        "end_time": end,
        "confidence": confidence,
        "details": {"raw_event_count": raw_events},
    }


def test_scene_change_is_context_only() -> None:
    result = confirm_issue(_issue("scene_change", 2.0, 2.0, 0.95))
    assert result["status"] == "context"
    assert result["confirmed"] is False


def test_long_static_segment_does_not_auto_fail() -> None:
    result = confirm_issue(_issue("freeze", 10.0, 21.0, 1.0))
    assert result["status"] == "review"
    assert result["confirmed"] is False


def test_opening_freeze_is_review_only() -> None:
    result = confirm_issue(_issue("freeze", 0.0, 0.8, 0.999))
    assert result["status"] == "review"
    assert result["confirmed"] is False


def test_short_high_confidence_freeze_is_confirmed() -> None:
    result = confirm_issue(_issue("freeze", 3.0, 4.0, 0.999))
    assert result["status"] == "confirmed"
    assert result["confirmed"] is True


def test_freeze_near_scene_change_is_review_only() -> None:
    result = confirm_issue(_issue("freeze", 3.0, 4.0, 0.999), near_scene_change=True)
    assert result["status"] == "review"
    assert result["confirmed"] is False


def test_isolated_motion_spike_is_review_only() -> None:
    result = confirm_issue(
        _issue("motion_discontinuity", 5.0, 5.0, 1.0, raw_events=1),
        targeted_metrics={"contrast_ratio": 10.0, "inside_samples": 5},
    )
    assert result["status"] == "review"
    assert result["confirmed"] is False


def test_motion_cluster_requires_targeted_contrast() -> None:
    issue = _issue("motion_discontinuity", 5.0, 5.5, 1.0, raw_events=6)
    result = confirm_issue(
        issue,
        targeted_metrics={"contrast_ratio": 1.4, "inside_samples": 8},
    )
    assert result["status"] == "review"
    assert result["confirmed"] is False


def test_persistent_motion_cluster_is_confirmed_with_targeted_evidence() -> None:
    issue = _issue("motion_discontinuity", 5.0, 5.5, 1.0, raw_events=6)
    result = confirm_issue(
        issue,
        targeted_metrics={"contrast_ratio": 3.8, "inside_samples": 8},
    )
    assert result["status"] == "confirmed"
    assert result["confirmed"] is True


def test_motion_near_scene_change_is_review_only() -> None:
    issue = _issue("motion_discontinuity", 5.0, 5.5, 1.0, raw_events=8)
    result = confirm_issue(
        issue,
        targeted_metrics={"contrast_ratio": 5.0, "inside_samples": 8},
        near_scene_change=True,
    )
    assert result["status"] == "review"
    assert result["confirmed"] is False
