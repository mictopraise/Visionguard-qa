from app.vision.freeze_detection import detect_freeze_windows
from benchmark.create_synthetic import create_moving_square_video
from benchmark.generate_defects import inject_freeze


def test_freeze_detector_finds_injected_window(tmp_path) -> None:
    clean = tmp_path / "clean.avi"
    defect = tmp_path / "defect.avi"

    create_moving_square_video(clean, fps=24.0, seconds=4.0)

    truth = inject_freeze(
        clean,
        defect,
        start_frame=36,
        freeze_frames=18,
    )

    clean_windows = detect_freeze_windows(clean)
    defect_windows = detect_freeze_windows(defect)

    assert clean_windows == []
    assert defect_windows

    detected = defect_windows[0]
    assert abs(detected.start_frame - truth["start_frame"]) <= 2
    assert abs(detected.end_frame - truth["end_frame"]) <= 2
    assert detected.confidence >= 0.997
