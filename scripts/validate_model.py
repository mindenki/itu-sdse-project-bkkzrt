"""Validate the exported serving model against the expected predictions.

Runs the exported model on the held-out test set (via vial_classifier.modeling.predict)
and compares the result with the committed expected predictions:

- predicted classes must match exactly
- confidences must match within --tolerance

Always writes to --output-dir:
- result.json                 pass/fail and every mismatch (read by CI)
- expected_predictions.csv    this model's predictions in the format of the expected
                              file, ready to commit when the change is intended
                              (e.g. after a data refresh)

Exit code 0 when validation passes, 1 otherwise.

Usage:
    python scripts/validate_model.py --model-dir models/model \
        --images-dir data/raw/images/test \
        --expected tests/fixtures/expected_predictions.csv --output-dir validation
"""

import argparse
import csv
import json
from pathlib import Path
import sys

from vial_classifier.modeling.predict import predict_images

FIELDS = ["filename", "prediction", "confidence"]


def read_expected(path: Path) -> dict[str, dict]:
    """Read the committed expected predictions: filename -> prediction, confidence."""
    with open(path, newline="") as f:
        return {
            row["filename"]: {
                "prediction": int(row["prediction"]),
                "confidence": float(row["confidence"]),
            }
            for row in csv.DictReader(f)
        }


def write_predictions(path: Path, predictions: dict[str, dict]) -> None:
    """Write predictions in the expected-file format."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for name, pred in sorted(predictions.items()):
            writer.writerow({"filename": name, **pred})


def compare(expected: dict, actual: dict, tolerance: float) -> list[dict]:
    """Return one entry per image whose prediction or confidence does not match."""
    mismatches = []
    for name, exp in expected.items():
        act = actual.get(name)
        if act is None:
            mismatches.append({"filename": name, "problem": "no prediction for this image"})
        elif act["prediction"] != exp["prediction"]:
            mismatches.append(
                {"filename": name, "problem": "prediction", "expected": exp, "actual": act}
            )
        elif act["confidence"] is None or abs(act["confidence"] - exp["confidence"]) > tolerance:
            mismatches.append(
                {"filename": name, "problem": "confidence", "expected": exp, "actual": act}
            )
    for name in sorted(actual.keys() - expected.keys()):
        mismatches.append({"filename": name, "problem": "image missing from expected file"})
    return mismatches


def validate(
    actual: dict[str, dict], expected_path: Path, output_dir: Path, tolerance: float
) -> dict:
    """Compare predictions with the expected file and write result.json + the new CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "expected_predictions.csv", actual)

    if expected_path.exists():
        reason = None
        mismatches = compare(read_expected(expected_path), actual, tolerance)
    else:
        reason = f"expected file {expected_path} not found"
        mismatches = []

    result = {
        "passed": reason is None and not mismatches,
        "reason": reason,
        "num_images": len(actual),
        "tolerance": tolerance,
        "mismatches": mismatches,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the exported serving model against the expected predictions."
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/model"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/images/test"))
    parser.add_argument(
        "--expected", type=Path, default=Path("tests/fixtures/expected_predictions.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("validation"))
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args(argv)

    actual = predict_images(args.model_dir, args.images_dir)
    result = validate(actual, args.expected, args.output_dir, args.tolerance)

    if result["passed"]:
        print(f"Validation PASSED: {result['num_images']} images match the expected predictions.")
        return 0
    reason = result["reason"] or f"{len(result['mismatches'])} mismatch(es)"
    print(f"Validation FAILED: {reason}")
    for mismatch in result["mismatches"]:
        print(f"  {mismatch}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
