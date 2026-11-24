"""Tests for batch inference (the model is faked, no MLflow model or data needed)."""

import base64
import csv

import pytest

from vial_classifier.modeling import predict


class FakeModel:
    """Stands in for the pyfunc model: records the request, returns fixed predictions."""

    def __init__(self, predictions, confidences):
        self.result = {"predictions": predictions, "confidences": confidences}
        self.requests = []

    def predict(self, model_input):
        self.requests.append(model_input)
        return self.result


@pytest.fixture
def images_dir(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"image-b")
    (tmp_path / "a.jpg").write_bytes(b"image-a")
    return tmp_path


def use_model(monkeypatch, model):
    loaded = []
    monkeypatch.setattr(
        predict.mlflow.pyfunc, "load_model", lambda path: loaded.append(path) or model
    )
    return loaded


def test_sends_base64_instances_and_maps_results_to_filenames(monkeypatch, images_dir):
    model = FakeModel(predictions=[1, 0], confidences=[0.51, 0.42])
    loaded = use_model(monkeypatch, model)

    result = predict.predict_images(images_dir / "model", images_dir)

    assert loaded == [str(images_dir / "model")]
    assert model.requests == [
        {
            "instances": [
                base64.b64encode(b"image-a").decode(),
                base64.b64encode(b"image-b").decode(),
            ]
        }
    ]
    assert result == {
        "a.jpg": {"prediction": 1, "confidence": 0.51},
        "b.jpg": {"prediction": 0, "confidence": 0.42},
    }


def test_uses_metadata_order_when_present(monkeypatch, images_dir):
    with open(images_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class"])
        writer.writerows([["b.jpg", 0], ["a.jpg", 1]])
    use_model(monkeypatch, FakeModel(predictions=[0, 1], confidences=[0.4, 0.6]))

    result = predict.predict_images(images_dir, images_dir)

    assert list(result) == ["b.jpg", "a.jpg"]
    assert result["a.jpg"] == {"prediction": 1, "confidence": 0.6}


def test_keeps_none_for_images_the_model_could_not_process(monkeypatch, images_dir):
    use_model(monkeypatch, FakeModel(predictions=[1, None], confidences=[0.5, None]))

    result = predict.predict_images(images_dir, images_dir)

    assert result["b.jpg"] == {"prediction": None, "confidence": None}
