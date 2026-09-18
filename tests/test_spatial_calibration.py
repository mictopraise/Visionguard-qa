from benchmark.run_spatial_calibration import run


def test_spatial_calibration_signals_are_directionally_separable(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    report = run()

    assert report["summary"]["artifact_count"] == 5
    assert report["summary"]["all_directionally_separated"]
    for artifact in ("blur", "blocky", "oversharpening", "oversmooth", "color_shift"):
        assert report[artifact]["separated"]
