"""Batch inference with the exported model.

Loads the exported MLflow pyfunc model (config.EXPORT_DIR) and predicts on a
directory of images, using the same base64 contract as POST /invocations.
Used locally and by scripts/validate_model.py (CI model validation).

CLI: `python -m vial_classifier.modeling.predict --images-dir data/raw/images/test`
"""

import base64
import csv
from pathlib import Path

from loguru import logger
import mlflow.pyfunc
import typer

from vial_classifier.config import EXPORT_DIR, TEST_DATA_DIR

app = typer.Typer()


def list_images(images_dir: Path) -> list[str]:
    """Image filenames of a data set: the ones listed in its metadata.csv, else all JPEGs."""
    metadata = Path(images_dir) / "metadata.csv"
    if metadata.exists():
        with open(metadata, newline="") as f:
            return [row["filename"] for row in csv.DictReader(f)]
    return sorted(p.name for p in Path(images_dir).glob("*.jpg"))


def predict_images(
    model_dir: Path = EXPORT_DIR, images_dir: Path = TEST_DATA_DIR
) -> dict[str, dict]:
    """Predict every image in images_dir with the exported pyfunc model.

    The images are sent base64-encoded as {"instances": [...]}, the same contract as
    POST /invocations. Returns {filename: {"prediction": int | None,
    "confidence": float | None}}; the wrapper returns None for an image it cannot process.
    """
    images_dir = Path(images_dir)
    filenames = list_images(images_dir)
    model = mlflow.pyfunc.load_model(str(model_dir))
    instances = [base64.b64encode((images_dir / name).read_bytes()).decode() for name in filenames]
    result = model.predict({"instances": instances})
    return {
        name: {"prediction": prediction, "confidence": confidence}
        for name, prediction, confidence in zip(
            filenames, result["predictions"], result["confidences"], strict=True
        )
    }


@app.command()
def main(
    model_dir: Path = EXPORT_DIR,
    images_dir: Path = TEST_DATA_DIR,
):
    """Predict clean/defective for every image in images_dir."""
    predictions = predict_images(model_dir, images_dir)
    for name, pred in predictions.items():
        logger.info(f"{name}: prediction={pred['prediction']} confidence={pred['confidence']}")


if __name__ == "__main__":
    app()
