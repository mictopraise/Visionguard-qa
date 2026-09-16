from app.video.normalize import choose_normalization_plan

def test_choose_normalization_plan_uses_lower_common_geometry_and_fps() -> None:
    plan = choose_normalization_plan(30.0, 1920, 1080, 24.0, 1280, 720)
    assert plan.target_fps == 24.0
    assert plan.target_width == 1280
    assert plan.target_height == 720
