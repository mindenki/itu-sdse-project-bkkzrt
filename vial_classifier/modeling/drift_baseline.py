"""Baseline statistics that drift detection compares live requests against."""

import json

from loguru import logger
import mlflow
import numpy as np
import torch

from vial_classifier import config
from vial_classifier.modeling.model import get_device


def compute_baseline_statistics(
    model, data_loader, device=None, feature_layer="fc1", threshold=config.THRESHOLD
):
    """Compute baseline statistics for drift detection from a dataset."""
    if device is None:
        device = get_device()

    model.eval()

    image_means = []
    image_stds = []
    class_counts = {0: 0, 1: 0}
    prediction_counts = {0: 0, 1: 0}
    confidences = []
    all_embeddings = []

    # Get the feature extraction layer
    if hasattr(model, feature_layer):
        feature_extractor = getattr(model, feature_layer)
    else:
        raise ValueError(f"Model does not have layer '{feature_layer}'")

    # Hook to capture embeddings
    embeddings = []

    def embedding_hook(module, input, output):
        embeddings.append(output.detach().cpu().numpy())

    handle = feature_extractor.register_forward_hook(embedding_hook)

    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                # Get predictions (logits -> probabilities)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= threshold).astype(int)

                # Compute image statistics (per channel)
                batch_mean = images.mean(dim=(0, 2, 3)).cpu().numpy()
                batch_std = images.std(dim=(0, 2, 3)).cpu().numpy()

                image_means.append(batch_mean)
                image_stds.append(batch_std)

                # Update class counts
                for label in labels.cpu().numpy().flatten():
                    class_counts[int(label)] += 1

                # Update prediction counts
                for pred in preds:
                    prediction_counts[int(pred)] += 1

                # Collect confidences
                confidences.extend(probs)

                # Get embeddings from the hook
                if embeddings:
                    all_embeddings.extend([e for e in embeddings])
                    embeddings.clear()

        # Compute final statistics
        image_means = np.array(image_means)
        image_stds = np.array(image_stds)

        baseline_stats = {
            "image_statistics": {
                "mean_r": float(np.mean(image_means[:, 0])),
                "mean_g": float(np.mean(image_means[:, 1])),
                "mean_b": float(np.mean(image_means[:, 2])),
                "std_r": float(np.mean(image_stds[:, 0])),
                "std_g": float(np.mean(image_stds[:, 1])),
                "std_b": float(np.mean(image_stds[:, 2])),
            },
            "class_distribution": {
                "class_0_count": class_counts[0],
                "class_1_count": class_counts[1],
                "total": class_counts[0] + class_counts[1],
            },
            "prediction_distribution": {
                "pred_0_count": prediction_counts[0],
                "pred_1_count": prediction_counts[1],
                "total": prediction_counts[0] + prediction_counts[1],
            },
            "confidence_statistics": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
            },
            "embedding_statistics": {
                "mean": [],
                "std": [],
            },
        }

        # Compute embedding statistics if embeddings were captured
        if all_embeddings:
            all_embeddings_array = np.concatenate(all_embeddings, axis=0)
            embedding_mean = np.mean(all_embeddings_array, axis=0)
            embedding_std = np.std(all_embeddings_array, axis=0)
            baseline_stats["embedding_statistics"] = {
                "mean": embedding_mean.tolist(),
                "std": embedding_std.tolist(),
                "shape": int(all_embeddings_array.shape[1]),
            }

        logger.info(
            f"Baseline statistics: class 0={class_counts[0]}, class 1={class_counts[1]}, "
            f"confidence mean={baseline_stats['confidence_statistics']['mean']:.4f}"
        )

        return baseline_stats

    finally:
        handle.remove()


def log_baseline_metrics(baseline_stats):
    """Save the baseline as JSON and log it to the active MLflow run."""
    config.DRIFT_BASELINE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DRIFT_BASELINE_STATS_PATH, "w") as f:
        json.dump(baseline_stats, f, indent=2)

    # Log baseline statistics to MLflow
    mlflow.log_metric("baseline_image_mean_r", baseline_stats["image_statistics"]["mean_r"])
    mlflow.log_metric("baseline_image_mean_g", baseline_stats["image_statistics"]["mean_g"])
    mlflow.log_metric("baseline_image_mean_b", baseline_stats["image_statistics"]["mean_b"])
    mlflow.log_metric("baseline_image_std_r", baseline_stats["image_statistics"]["std_r"])
    mlflow.log_metric("baseline_image_std_g", baseline_stats["image_statistics"]["std_g"])
    mlflow.log_metric("baseline_image_std_b", baseline_stats["image_statistics"]["std_b"])
    mlflow.log_metric(
        "baseline_class_0_count", baseline_stats["class_distribution"]["class_0_count"]
    )
    mlflow.log_metric(
        "baseline_class_1_count", baseline_stats["class_distribution"]["class_1_count"]
    )
    mlflow.log_metric("baseline_confidence_mean", baseline_stats["confidence_statistics"]["mean"])
    mlflow.log_metric("baseline_confidence_std", baseline_stats["confidence_statistics"]["std"])
    mlflow.log_artifact(config.DRIFT_BASELINE_STATS_PATH, "drift_detection")
