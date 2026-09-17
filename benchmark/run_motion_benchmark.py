from __future__ import annotations

import json
from pathlib import Path

from app.vision.motion_analysis import analyze_motion
from benchmark.create_synthetic import create_moving_square_video
from benchmark.generate_defects import inject_motion_jump


def run(output_dir: Path = Path("benchmark/results/motion_baseline")) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / "clean.avi"
    defect_path = output_dir / "motion_jump.avi"
    manifest_path = output_dir / "manifest.jsonl"

    create_moving_square_video(clean_path, fps=24.0, seconds=4.0)

    truth = inject_motion_jump(
        clean_path,
        defect_path,
        jump_frame=48,
        shift_pixels=90,
        manifest_path=manifest_path,
    )

    _, clean_anomalies = analyze_motion(clean_path)
    _, defect_anomalies = analyze_motion(defect_path)

    nearest = None
    if defect_anomalies:
        nearest = min(
            defect_anomalies,
            key=lambda item: abs(item.frame_index - truth["start_frame"]),
        )

    result = {
        "benchmark": "motion_baseline_v1",
        "ground_truth": truth,
        "clean_anomaly_count": len(clean_anomalies),
        "detected_count": len(defect_anomalies),
        "nearest_detection": None,
        "frame_error": None,
        "pass": False,
    }

    if nearest is not None:
        result["nearest_detection"] = {
            "frame_index": nearest.frame_index,
            "time_seconds": nearest.time_seconds,
            "magnitude": nearest.magnitude,
            "baseline": nearest.baseline,
            "threshold": nearest.threshold,
            "spike_ratio": nearest.spike_ratio,
        }
        result["frame_error"] = abs(nearest.frame_index - truth["start_frame"])
        result["pass"] = (
            len(clean_anomalies) <= 1
            and result["frame_error"] <= 1
        )

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
