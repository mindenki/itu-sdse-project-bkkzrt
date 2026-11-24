"""Tests for the comparison logic of scripts/validate_model.py (no model, no data)."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_model.py"
spec = importlib.util.spec_from_file_location("validate_model", SCRIPT)
assert spec is not None and spec.loader is not None
validate_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_model)


def write_expected(path: Path, rows: dict[str, tuple[int, float]]) -> Path:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction", "confidence"])
        for name, (pred, conf) in rows.items():
            writer.writerow([name, pred, conf])
    return path


@pytest.fixture
def expected(tmp_path):
    return write_expected(tmp_path / "expected.csv", {"a.jpg": (1, 0.5144), "b.jpg": (0, 0.4000)})


def test_matching_predictions_pass(tmp_path, expected):
    actual = {
        "a.jpg": {"prediction": 1, "confidence": 0.5146},
        "b.jpg": {"prediction": 0, "confidence": 0.4000},
    }
    result = validate_model.validate(actual, expected, tmp_path / "out", tolerance=1e-3)
    assert result["passed"]
    assert result["mismatches"] == []


def test_wrong_class_fails(tmp_path, expected):
    actual = {
        "a.jpg": {"prediction": 0, "confidence": 0.5144},
        "b.jpg": {"prediction": 0, "confidence": 0.4000},
    }
    result = validate_model.validate(actual, expected, tmp_path / "out", tolerance=1e-3)
    assert not result["passed"]
    assert [m["problem"] for m in result["mismatches"]] == ["prediction"]


def test_confidence_outside_tolerance_fails(tmp_path, expected):
    actual = {
        "a.jpg": {"prediction": 1, "confidence": 0.5200},
        "b.jpg": {"prediction": 0, "confidence": 0.4000},
    }
    result = validate_model.validate(actual, expected, tmp_path / "out", tolerance=1e-3)
    assert [m["problem"] for m in result["mismatches"]] == ["confidence"]


def test_failed_prediction_and_unexpected_image_fail(tmp_path, expected):
    actual = {
        "a.jpg": {"prediction": None, "confidence": None},
        "c.jpg": {"prediction": 1, "confidence": 0.5},
    }
    result = validate_model.validate(actual, expected, tmp_path / "out", tolerance=1e-3)
    problems = sorted(m["problem"] for m in result["mismatches"])
    assert problems == [
        "image missing from expected file",
        "no prediction for this image",
        "prediction",
    ]


def test_missing_expected_file_fails_but_writes_new_one(tmp_path):
    actual = {"a.jpg": {"prediction": 1, "confidence": 0.5144}}
    out = tmp_path / "out"
    result = validate_model.validate(actual, tmp_path / "missing.csv", out, tolerance=1e-3)
    assert not result["passed"]
    assert "not found" in result["reason"]
    assert validate_model.read_expected(out / "expected_predictions.csv") == {
        "a.jpg": {"prediction": 1, "confidence": 0.5144}
    }
    assert json.loads((out / "result.json").read_text())["passed"] is False
