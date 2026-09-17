from app.vision.motion_analysis import analyze_motion
from benchmark.create_synthetic import create_moving_square_video
from benchmark.generate_defects import inject_motion_jump


def test_motion_detector_flags_injected_jump(tmp_path) -> None:
    clean = tmp_path / "clean.avi"
    defect = tmp_path / "motion_jump.avi"

    create_moving_square_video(clean, fps=24.0, seconds=4.0)
    truth = inject_motion_jump(
        clean,
        defect,
        jump_frame=48,
        shift_pixels=90,
    )

    _, clean_anomalies = analyze_motion(clean)
    _, defect_anomalies = analyze_motion(defect)

    assert len(clean_anomalies) <= 1
    assert defect_anomalies
    assert min(abs(item.frame_index - truth["start_frame"]) for item in defect_anomalies) <= 1
