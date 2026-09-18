from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactCalibration:
    label: str
    status: str
    source: str
    notes: str


# Milestone 6E guardrail:
# Synthetic directional calibration can establish that a signal behaves sensibly,
# but it is not sufficient to claim human-level artifact classification.
ARTIFACT_CALIBRATION: dict[str, ArtifactCalibration] = {
    name: ArtifactCalibration(
        label=name,
        status="synthetic_validated",
        source="benchmark/run_spatial_calibration.py",
        notes="Synthetic directional benchmark passed; final label promotion still requires human/example-labeled calibration.",
    )
    for name in ("blurring", "blocky", "oversharpening", "oversmooth", "color_shift")
}


def promotable_artifacts() -> set[str]:
    return {
        name
        for name, calibration in ARTIFACT_CALIBRATION.items()
        if calibration.status == "calibrated"
    }
