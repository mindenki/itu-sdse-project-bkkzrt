"""Slow integration coverage for the training CLI and its registered pyfunc model."""

import subprocess
import sys

import mlflow
import pytest

from vial_classifier import config

pytestmark = pytest.mark.slow


def test_training_cli_registers_a_loadable_pyfunc_model():
    """A small real-data run can register and serve one prediction end to end."""
    if not config.TRAIN_METADATA_PATH.exists():
        pytest.skip("Training metadata is unavailable; run `make data` first.")

    image_paths = list(config.TRAIN_DATA_DIR.glob("*.jpg"))
    if len(image_paths) < 20:
        pytest.skip("Training images are unavailable; run `make data` first.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "vial_classifier.modeling.train",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--limit",
            "20",
        ],
        check=True,
        cwd=config.PROJ_ROOT,
    )

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@best")
    result = model.predict({"instances": [str(image_paths[0])]})

    assert result["num_samples"] == 1
    assert result["predictions"][0] in (0, 1)
    assert result["confidences"][0] is not None
