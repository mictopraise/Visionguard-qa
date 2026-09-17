from __future__ import annotations


def confirm_issue(issue: dict) -> dict:
    """Second-pass policy that decides whether a grouped first-pass issue is actionable.

    This deliberately favors precision over recall for the competition MVP. Scene cuts are
    context only, flicker remains review-only, and freeze/motion findings must satisfy
    stronger duration/persistence rules before they can drive a FAIL verdict.
    """
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
        if 0.6 <= duration <= 3.0 and confidence >= 0.995:
            return {
                "status": "confirmed",
                "confirmed": True,
                "reason": "Sustained near-identical frames fall inside the high-confidence freeze window used for the MVP.",
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
        if raw_event_count >= 4 and duration >= 0.15 and confidence >= 0.95:
            return {
                "status": "confirmed",
                "confirmed": True,
                "reason": "Multiple adjacent high-confidence motion spikes persisted long enough to survive second-pass filtering.",
            }
        return {
            "status": "review",
            "confirmed": False,
            "reason": "Isolated or very short motion spikes are treated as possible camera/edit motion rather than automatic failure.",
        }

    return {
        "status": "review",
        "confirmed": False,
        "reason": "Unknown issue type retained for human review.",
    }


def apply_second_pass(issues: list[dict], *, video_label: str) -> tuple[list[dict], list[dict]]:
    reviewed: list[dict] = []
    trace: list[dict] = []

    for issue in issues:
        decision = confirm_issue(issue)
        enriched = dict(issue)
        enriched["confirmation"] = decision
        reviewed.append(enriched)
        trace.append({
            "video": video_label,
            "type": issue["type"],
            "start_time": issue["start_time"],
            "end_time": issue["end_time"],
            "action": "targeted_confirmation",
            "result": decision["status"],
            "reason": decision["reason"],
        })

    return reviewed, trace
