from __future__ import annotations

import json
from pathlib import Path

from app.vision.spatial_quality import analyze_spatial_quality
from app.vision.threshold_fitting import fit_threshold, promotion_ready

DEFAULT_MANIFEST = Path("benchmark/real_world_calibration_manifest.json")
RESULT_PATH = Path("benchmark/results/real_world_calibration.json")

SIGNALS = {
    "blurring": ("median_laplacian_variance", "lower"),
    "blocky": ("median_blockiness_ratio", "higher"),
    "oversharpening": ("median_oversharpen_edge_residual", "higher"),
    "oversmooth": ("median_fine_to_coarse_ratio", "lower"),
    "color_shift": ("median_color_bias", "higher"),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    if not manifest_path.exists():
        report = {
            "status": "awaiting_real_labeled_media",
            "manifest": str(manifest_path),
            "message": "Calibration harness is ready, but no real-world labeled manifest is present.",
            "promotion_ready": [],
        }
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return report

    manifest = _load(manifest_path)
    rows = []
    for clip in manifest.get("clips", []):
        path = Path(clip["path"])
        if not path.exists():
            continue
        spatial = analyze_spatial_quality(path)
        rows.append({
            "clip": clip,
            "metrics": spatial.get("metrics", {}),
        })

    artifact_reports = {}
    promotable = []
    for artifact, (metric_name, direction) in SIGNALS.items():
        values = []
        labels = []
        intentional_negative_count = 0
        for row in rows:
            metrics = row["metrics"]
            if metric_name not in metrics:
                continue
            clip = row["clip"]
            is_positive = artifact in set(clip.get("artifacts", []))
            values.append(float(metrics[metric_name]))
            labels.append(is_positive)
            intentional = set(clip.get("intentional_effects", []))
            if not is_positive and (
                (artifact == "blurring" and "blur" in intentional)
                or (artifact == "color_shift" and "color_grade" in intentional)
                or (artifact == "oversmooth" and "smooth_surface" in intentional)
            ):
                intentional_negative_count += 1

        positives = sum(labels)
        negatives = len(labels) - positives
        if not values or positives == 0 or negatives == 0:
            artifact_reports[artifact] = {
                "status": "insufficient_labels",
                "positive_count": positives,
                "negative_count": negatives,
                "intentional_negative_count": intentional_negative_count,
            }
            continue

        fitted = fit_threshold(values, labels, direction=direction)
        ready = promotion_ready(
            fitted,
            positive_count=positives,
            negative_count=negatives,
            intentional_negative_count=intentional_negative_count,
        )
        artifact_reports[artifact] = {
            "status": "promotion_ready" if ready else "needs_more_calibration",
            "metric": metric_name,
            "direction": direction,
            "threshold": fitted.threshold,
            "precision": fitted.precision,
            "recall": fitted.recall,
            "f1": fitted.f1,
            "false_positive_rate": fitted.false_positive_rate,
            "tp": fitted.tp,
            "fp": fitted.fp,
            "tn": fitted.tn,
            "fn": fitted.fn,
            "positive_count": positives,
            "negative_count": negatives,
            "intentional_negative_count": intentional_negative_count,
        }
        if ready:
            promotable.append(artifact)

    report = {
        "status": "evaluated",
        "clip_count": len(rows),
        "artifact_results": artifact_reports,
        "promotion_ready": promotable,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
