from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("benchmark/sage_human_calibration_manifest.json")


def load_sage_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_preference_consistency(example: dict[str, Any]) -> str:
    """Return consistent, ambiguous, or conflict without rewriting source labels."""
    left = example["left"]["mos"]
    right = example["right"]["mos"]
    preference = example["preference"]

    if not isinstance(left, int) or not isinstance(right, int):
        return "ambiguous"

    expected = "same"
    if left > right:
        expected = "left"
    elif right > left:
        expected = "right"

    return "consistent" if preference == expected else "conflict"


def usable_preference_examples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Exclude ambiguous and source-conflicting examples from preference fitting."""
    return [
        example
        for example in manifest.get("textual_gold_examples", [])
        if validate_preference_consistency(example) == "consistent"
        and "mos_preference_consistency" in example.get("usable_for", [])
    ]


def guideline_artifact_labels(manifest: dict[str, Any]) -> set[str]:
    return set(manifest["rules"]["artifact_labels"])
