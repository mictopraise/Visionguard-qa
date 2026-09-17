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
