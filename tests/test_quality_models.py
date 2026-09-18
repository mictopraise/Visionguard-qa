from app.core.models import (
    ArtifactLabel,
    MOSScore,
    PairPreference,
    PairwiseQualityDecision,
    PlaybackQuality,
    VideoQualityAssessment,
)


def test_video_quality_assessment_accepts_human_aligned_outputs() -> None:
    assessment = VideoQualityAssessment(
        video_label="A",
        mos_score=MOSScore.FAIR,
        artifacts=[ArtifactLabel.BLURRING, ArtifactLabel.BLOCKY],
        playback_quality=PlaybackQuality.SMOOTH,
        confidence=0.82,
    )

    assert assessment.mos_score == 3
    assert assessment.artifacts == [ArtifactLabel.BLURRING, ArtifactLabel.BLOCKY]
    assert assessment.playback_quality == PlaybackQuality.SMOOTH


def test_pairwise_preference_supports_same_and_inconclusive() -> None:
    same = PairwiseQualityDecision(preference=PairPreference.SAME, confidence=0.76)
    uncertain = PairwiseQualityDecision()

    assert same.preference == PairPreference.SAME
    assert uncertain.preference == PairPreference.INCONCLUSIVE


from app.services.first_pass import _playback_quality_from_issues


def test_playback_quality_reports_temporal_issues_for_confirmed_event() -> None:
    issues = [{
        "type": "motion_discontinuity",
        "confidence": 0.98,
        "confirmation": {"status": "confirmed", "confirmed": True},
    }]
    result = _playback_quality_from_issues(issues)
    assert result["label"] == "temporal_issues"
    assert "motion_discontinuity" in result["supporting_types"]


def test_playback_quality_reports_uncertain_for_review_only_event() -> None:
    issues = [{
        "type": "flicker",
        "confidence": 0.8,
        "confirmation": {"status": "review", "confirmed": False},
    }]
    result = _playback_quality_from_issues(issues)
    assert result["label"] == "uncertain"


def test_playback_quality_ignores_scene_context() -> None:
    issues = [{
        "type": "scene_change",
        "confidence": 0.9,
        "confirmation": {"status": "context", "confirmed": False},
    }]
    result = _playback_quality_from_issues(issues)
    assert result["label"] == "smooth"
