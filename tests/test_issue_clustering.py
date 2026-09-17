from app.services.first_pass import _cluster_issues, _final_verdict


def test_nearby_motion_events_are_grouped() -> None:
    issues = [
        {"type": "motion_discontinuity", "start_time": 4.00, "end_time": 4.00, "confidence": 0.90, "details": {}},
        {"type": "motion_discontinuity", "start_time": 4.03, "end_time": 4.03, "confidence": 0.95, "details": {}},
        {"type": "motion_discontinuity", "start_time": 4.10, "end_time": 4.10, "confidence": 0.92, "details": {}},
        {"type": "motion_discontinuity", "start_time": 6.00, "end_time": 6.00, "confidence": 0.91, "details": {}},
    ]

    grouped = _cluster_issues(issues)

    assert len(grouped) == 2
    assert grouped[0]["start_time"] == 4.00
    assert grouped[0]["end_time"] == 4.10
    assert grouped[0]["confidence"] == 0.95
    assert grouped[0]["details"]["raw_event_count"] == 3


def test_clean_pair_passes() -> None:
    result = {"issues": []}
    verdict = _final_verdict(result, result)
    assert verdict["status"] == "PASS"
    assert verdict["action"] == "PASS"


def test_major_issue_produces_fail_and_human_review() -> None:
    result_a = {
        "issues": [
            {"type": "freeze", "start_time": 1.0, "end_time": 2.2, "confidence": 0.99, "details": {}}
        ]
    }
    result_b = {"issues": []}

    verdict = _final_verdict(result_a, result_b)
    assert verdict["status"] == "FAIL"
    assert verdict["action"] == "HUMAN_REVIEW"
    assert "Video A" in verdict["summary"]
