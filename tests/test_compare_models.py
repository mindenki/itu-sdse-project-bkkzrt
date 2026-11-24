"""Tests for scripts/compare_models.py (registry and MLflow are faked)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_models.py"
spec = importlib.util.spec_from_file_location("compare_models", SCRIPT)
assert spec is not None and spec.loader is not None
compare_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_models)


@pytest.mark.parametrize(
    ("before", "after", "promoted"),
    [
        (None, "1", True),  # first run: the trained model becomes Production
        ("1", "2", True),  # better model promoted
        ("1", "1", False),  # Production kept
        (None, None, False),  # nothing in Production
    ],
)
def test_decide(before, after, promoted):
    assert compare_models.decide(before, after) == {
        "promoted": promoted,
        "production_version_before": before,
        "production_version_after": after,
    }


def test_main_runs_the_registry_between_two_reads(monkeypatch, tmp_path):
    versions = iter(["3", "4"])
    calls = []
    monkeypatch.setattr(compare_models, "production_version", lambda: next(versions))
    monkeypatch.setattr(
        compare_models.registry, "main", lambda export_dir: calls.append(export_dir)
    )
    decision_path = tmp_path / "decision.json"

    assert compare_models.main(["--decision", str(decision_path), "--export-dir", "out"]) == 0

    assert calls == [Path("out")]
    assert json.loads(decision_path.read_text()) == {
        "promoted": True,
        "production_version_before": "3",
        "production_version_after": "4",
    }


def test_production_version_is_text_even_when_mlflow_returns_a_number(monkeypatch):
    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            return type("ModelVersion", (), {"version": 3})()

    monkeypatch.setattr(compare_models, "MlflowClient", FakeClient)
    monkeypatch.setattr(compare_models.mlflow, "set_tracking_uri", lambda uri: None)

    assert compare_models.production_version() == "3"


def test_production_version_is_none_without_a_production_alias(monkeypatch):
    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            raise compare_models.MlflowException("no alias")

    monkeypatch.setattr(compare_models, "MlflowClient", FakeClient)
    monkeypatch.setattr(compare_models.mlflow, "set_tracking_uri", lambda uri: None)

    assert compare_models.production_version() is None
