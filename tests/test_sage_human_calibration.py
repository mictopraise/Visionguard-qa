from app.vision.human_calibration import (
    guideline_artifact_labels,
    load_sage_manifest,
    usable_preference_examples,
    validate_preference_consistency,
)
from benchmark.run_sage_human_calibration import run


def test_official_artifact_taxonomy_is_preserved() -> None:
    manifest = load_sage_manifest()
    labels = guideline_artifact_labels(manifest)
    assert labels == {
        "unnatural_objects",
        "oversharpening",
        "blurring",
        "oversmooth",
        "blocky",
        "color_shift",
        "other",
        "no_artifacts",
    }


def test_conflicting_source_example_is_not_silently_corrected() -> None:
    manifest = load_sage_manifest()
    example = next(
        item for item in manifest["textual_gold_examples"]
        if item["id"] == "sage-example-4"
    )
    assert example["left"]["mos"] == 4
    assert example["right"]["mos"] == 5
    assert example["preference"] == "left"
    assert validate_preference_consistency(example) == "conflict"


def test_only_consistent_examples_are_used_for_preference_fit() -> None:
    manifest = load_sage_manifest()
    ids = {item["id"] for item in usable_preference_examples(manifest)}
    assert ids == {"sage-example-2", "sage-example-3"}


def test_human_calibration_audit_flags_source_conflict(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # The manifest path is repo-relative; point loader-facing cwd back via copy.
    import json
    from pathlib import Path
    source = load_sage_manifest()
    target = Path("benchmark/sage_human_calibration_manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(source), encoding="utf-8")
    report = run()
    assert report["source_conflicts"] == ["sage-example-4"]
    assert report["preference_fit_example_count"] == 2
    assert report["promotion_policy"]["requires_real_labeled_media"]
