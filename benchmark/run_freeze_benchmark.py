from __future__ import annotations

import json
from pathlib import Path

from app.vision.freeze_detection import detect_freeze_windows
from benchmark.create_synthetic import create_moving_square_video
from benchmark.generate_defects import inject_freeze


def run(output_dir: Path = Path("benchmark/results/freeze_baseline")) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / "clean.avi"
    defect_path = output_dir / "freeze_defect.mp4"
    manifest_path = output_dir / "manifest.jsonl"

    create_moving_square_video(clean_path)

    truth = inject_freeze(
        clean_path,
        defect_path,
        start_frame=60,
        freeze_frames=18,
        manifest_path=manifest_path,
    )

    clean_windows = detect_freeze_windows(clean_path)
    defect_windows = detect_freeze_windows(defect_path)

    detected = defect_windows[0] if defect_windows else None

    result = {
        "benchmark": "freeze_baseline_v1",
        "ground_truth": truth,
        "clean_false_positive_count": len(clean_windows),
        "detected_count": len(defect_windows),
        "first_detection": None,
        "start_frame_error": None,
        "end_frame_error": None,
        "pass": False,
    }

    if detected is not None:
        result["first_detection"] = {
            "start_frame": detected.start_frame,
            "end_frame": detected.end_frame,
            "start_time": detected.start_time,
            "end_time": detected.end_time,
            "confidence": detected.confidence,
        }
        result["start_frame_error"] = abs(detected.start_frame - truth["start_frame"])
        result["end_frame_error"] = abs(detected.end_frame - truth["end_frame"])
        result["pass"] = (
            len(clean_windows) == 0
            and result["start_frame_error"] <= 2
            and result["end_frame_error"] <= 2
        )

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2))
