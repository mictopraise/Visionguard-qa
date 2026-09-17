from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Disposition(StrEnum):
    PASS = "PASS"
    RECHECK = "RECHECK"
    REGENERATE = "REGENERATE"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class MOSScore(IntEnum):
    BAD = 1
    POOR = 2
    FAIR = 3
    VISIBLE_ARTIFACT = 4
    NO_PERCEIVABLE_ISSUE = 5


class ArtifactLabel(StrEnum):
    UNNATURAL_OBJECTS = "unnatural_objects"
    OVERSHARPENING = "oversharpening"
    BLURRING = "blurring"
    OVERSMOOTH = "oversmooth"
    BLOCKY = "blocky"
    COLOR_SHIFT = "color_shift"
    OTHER = "other"
    NO_ARTIFACTS = "no_artifacts"


class PlaybackQuality(StrEnum):
    SMOOTH = "smooth"
    TEMPORAL_ISSUES = "temporal_issues"
    UNCERTAIN = "uncertain"


class PairPreference(StrEnum):
    VIDEO_A = "video_a"
    VIDEO_B = "video_b"
    SAME = "same"
    INCONCLUSIVE = "inconclusive"


class VideoMetadata(BaseModel):
    filename: str
    fps: float = Field(ge=0)
    frame_count: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    opencv_version: str


class VideoQualityAssessment(BaseModel):
    """Human-aligned quality output for one video.

    MOS, artifact labels, and playback quality are intentionally modeled separately.
    Detectors may contribute evidence without forcing a final MOS until calibration is
    benchmark-backed.
    """

    video_label: str
    mos_score: MOSScore | None = None
    artifacts: list[ArtifactLabel] = []
    playback_quality: PlaybackQuality = PlaybackQuality.UNCERTAIN
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[dict[str, Any]] = []
    notes: list[str] = []


class PairwiseQualityDecision(BaseModel):
    preference: PairPreference = PairPreference.INCONCLUSIVE
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = ""


class ComparisonJob(BaseModel):
    job_id: str
    video_a: VideoMetadata
    video_b: VideoMetadata
    status: str
    notes: list[str] = []
    evidence: list[dict[str, Any]] = []
