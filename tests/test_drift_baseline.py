"""Tests for the drift baseline.

MLflow is replaced by a mock, so these tests never touch a real tracking store.
"""

import json
from unittest.mock import MagicMock

import pytest
import torch

from vial_classifier import config
from vial_classifier.modeling import drift_baseline
from vial_classifier.modeling.model import SimpleCNN

CPU = torch.device("cpu")


class FirstPixelModel(torch.nn.Module):
    """Uses each image's first pixel as its logit; `fc1` acts as the embedding layer."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Flatten()

    def forward(self, x):
        return self.fc1(x)[:, :1]


class FailingModel(FirstPixelModel):
    def forward(self, x):
        self.fc1(x)
        raise RuntimeError("simulated failure")


def _images_with_logits(logits):
    """Images of size 2x2 where every pixel holds the logit the model should output."""
    return torch.tensor(logits).view(-1, 1, 1, 1).expand(-1, 3, 2, 2).clone()


@pytest.fixture
def loader():
    # Probabilities are about 0.12, 0.27, 0.73 and 0.88
    images = _images_with_logits([-2.0, -1.0, 1.0, 2.0])
    return [(images[:2], torch.tensor([0, 0])), (images[2:], torch.tensor([0, 1]))]


@pytest.fixture
def stats(loader):
    return drift_baseline.compute_baseline_statistics(FirstPixelModel(), loader, device=CPU)


@pytest.fixture
def mlflow_mock(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(drift_baseline, "mlflow", mock)
    return mock


@pytest.fixture
def stats_path(tmp_path, monkeypatch):
    path = tmp_path / "artifacts" / "drift_baseline_stats.json"
    monkeypatch.setattr(config, "DRIFT_BASELINE_STATS_PATH", path)
    return path


# --- compute_baseline_statistics ---------------------------------------------


def test_baseline_has_all_the_keys_drift_detection_reads(stats):
    # The keys notebooks/02_deployment.ipynb (cell 19) reads for GET /drift
    assert {"mean", "std"} <= set(stats["confidence_statistics"])
    assert {"class_0_count", "class_1_count", "total"} <= set(stats["class_distribution"])


def test_baseline_has_the_full_notebook_structure(stats):
    assert {key: set(value) for key, value in stats.items()} == {
        "image_statistics": {"mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"},
        "class_distribution": {"class_0_count", "class_1_count", "total"},
        "prediction_distribution": {"pred_0_count", "pred_1_count", "total"},
        "confidence_statistics": {"mean", "std", "min", "max"},
        "embedding_statistics": {"mean", "std", "shape"},
    }


def test_baseline_survives_a_json_round_trip(stats):
    assert json.loads(json.dumps(stats)) == stats


def test_class_counts_come_from_the_labels(stats):
    assert stats["class_distribution"] == {"class_0_count": 3, "class_1_count": 1, "total": 4}


@pytest.mark.parametrize(
    ("threshold", "expected_pred_0", "expected_pred_1"), [(0.5, 2, 2), (0.8, 3, 1), (0.05, 0, 4)]
)
def test_prediction_counts_use_the_given_threshold(
    loader, threshold, expected_pred_0, expected_pred_1
):
    stats = drift_baseline.compute_baseline_statistics(
        FirstPixelModel(), loader, device=CPU, threshold=threshold
    )

    assert stats["prediction_distribution"] == {
        "pred_0_count": expected_pred_0,
        "pred_1_count": expected_pred_1,
        "total": 4,
    }


def test_confidence_statistics_are_computed_from_probabilities(stats):
    probabilities = torch.sigmoid(torch.tensor([-2.0, -1.0, 1.0, 2.0]))

    confidence = stats["confidence_statistics"]
    assert confidence["mean"] == pytest.approx(probabilities.mean().item())
    assert confidence["std"] == pytest.approx(probabilities.std(unbiased=False).item())
    assert confidence["min"] == pytest.approx(probabilities.min().item())
    assert confidence["max"] == pytest.approx(probabilities.max().item())


def test_image_statistics_average_the_per_batch_values(stats):
    # Batch means are -1.5 and 1.5; each batch's pixel std is that of two values 1 apart
    expected_std = torch.tensor([-2.0, -1.0]).repeat_interleave(4).std().item()

    image_statistics = stats["image_statistics"]
    for channel in "rgb":
        assert image_statistics[f"mean_{channel}"] == pytest.approx(0.0)
        assert image_statistics[f"std_{channel}"] == pytest.approx(expected_std)


def test_embedding_statistics_use_the_feature_layer():
    loader = [(torch.zeros(2, 3, 64, 64), torch.tensor([0, 1]))]

    stats = drift_baseline.compute_baseline_statistics(SimpleCNN(), loader, device=CPU)

    embedding = stats["embedding_statistics"]
    assert embedding["shape"] == 128
    assert len(embedding["mean"]) == 128
    assert len(embedding["std"]) == 128


def test_default_device_is_used_when_none_is_given(loader):
    stats = drift_baseline.compute_baseline_statistics(FirstPixelModel(), loader)

    assert stats["class_distribution"]["total"] == 4


def test_unknown_feature_layer_raises(loader):
    with pytest.raises(ValueError, match="does not have layer"):
        drift_baseline.compute_baseline_statistics(
            FirstPixelModel(), loader, device=CPU, feature_layer="fc9"
        )


def test_model_is_switched_to_eval_mode(loader):
    model = FirstPixelModel()
    model.train()

    drift_baseline.compute_baseline_statistics(model, loader, device=CPU)

    assert not model.training


def test_embedding_hook_is_removed_afterwards(loader):
    model = FirstPixelModel()

    drift_baseline.compute_baseline_statistics(model, loader, device=CPU)

    assert not model.fc1._forward_hooks


def test_embedding_hook_is_removed_after_a_failure(loader):
    model = FailingModel()

    with pytest.raises(RuntimeError, match="simulated failure"):
        drift_baseline.compute_baseline_statistics(model, loader, device=CPU)

    assert not model.fc1._forward_hooks


# --- log_baseline_metrics ----------------------------------------------------


def test_baseline_is_written_as_json(stats, mlflow_mock, stats_path):
    stats_path.parent.mkdir()

    drift_baseline.log_baseline_metrics(stats)

    assert json.loads(stats_path.read_text()) == stats


def test_missing_artifacts_folder_is_created(stats, mlflow_mock, stats_path):
    # artifacts/ isn't in git, so it doesn't exist on a fresh clone or in the container
    drift_baseline.log_baseline_metrics(stats)

    assert stats_path.exists()


def test_baseline_metrics_and_artifact_are_logged_to_mlflow(stats, mlflow_mock, stats_path):
    stats_path.parent.mkdir()

    drift_baseline.log_baseline_metrics(stats)

    logged = {c.args[0]: c.args[1] for c in mlflow_mock.log_metric.call_args_list}
    assert logged == {
        "baseline_image_mean_r": stats["image_statistics"]["mean_r"],
        "baseline_image_mean_g": stats["image_statistics"]["mean_g"],
        "baseline_image_mean_b": stats["image_statistics"]["mean_b"],
        "baseline_image_std_r": stats["image_statistics"]["std_r"],
        "baseline_image_std_g": stats["image_statistics"]["std_g"],
        "baseline_image_std_b": stats["image_statistics"]["std_b"],
        "baseline_class_0_count": 3,
        "baseline_class_1_count": 1,
        "baseline_confidence_mean": stats["confidence_statistics"]["mean"],
        "baseline_confidence_std": stats["confidence_statistics"]["std"],
    }
    mlflow_mock.log_artifact.assert_called_once_with(stats_path, "drift_detection")
