from __future__ import annotations

import json
from pathlib import Path

from app.vision.human_calibration import (
    load_sage_manifest,
    usable_preference_examples,
    validate_preference_consistency,
)

RESULT_PATH = Path("benchmark/results/sage_human_calibration_audit.json")


def run() -> dict:
    manifest = load_sage_manifest()
    examples = manifest["textual_gold_examples"]

    statuses = {
        example["id"]: validate_preference_consistency(example)
        for example in examples
    }
    usable = usable_preference_examples(manifest)

    report = {
        "source": manifest["source"],
        "example_count": len(examples),
        "preference_fit_example_count": len(usable),
        "consistency_status": statuses,
        "source_conflicts": [
            example["id"]
            for example in examples
            if statuses[example["id"]] == "conflict"
        ],
        "ambiguous_examples": [
            example["id"]
            for example in examples
            if statuses[example["id"]] == "ambiguous"
        ],
        "promotion_policy": {
            "synthetic_validation_alone": false,
            "textual_examples_alone": false,
            "requires_real_labeled_media": true
        }
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
