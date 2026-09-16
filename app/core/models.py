from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Disposition(StrEnum):
    PASS = "PASS"
    RECHECK = "RECHECK"
    REGENERATE = "REGENERATE"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class VideoMetadata(BaseModel):
    filename: str
    fps: float = Field(ge=0)
    frame_count: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    opencv_version: str


class ComparisonJob(BaseModel):
    job_id: str
    video_a: VideoMetadata
    video_b: VideoMetadata
    status: str
    notes: list[str] = []
    evidence: list[dict[str, Any]] = []
