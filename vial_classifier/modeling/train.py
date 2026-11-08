import json
import os

from loguru import logger
import mlflow
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn, optim
import typer

from vial_classifier import config
from vial_classifier.dataset import create_dataloaders, load_image_dataset, split_dataset
from vial_classifier.features import (
    calculate_image_statistics,
    define_transforms,
    save_class_names,
)
from vial_classifier.modeling.drift_baseline import (
    compute_baseline_statistics,
    log_baseline_metrics,
)
from vial_classifier.modeling.evaluate import evaluate_cnn_model
from vial_classifier.modeling.model import SimpleCNN, get_device
from vial_classifier.modeling.registry import register_model
from vial_classifier.modeling.wrapper import SimpleLoggingWrapper

app = typer.Typer()


def setup_mlflow_tracking(experiment_name=config.EXPERIMENT_NAME):
    """Point MLflow at the local state dir and create the experiment if needed."""
    os.makedirs(config.MLFLOW_STATE_DIR, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=config.MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(experiment_name)


def train_cnn_model(
    train_loader,
    val_loader,
    num_epochs=config.NUM_EPOCHS,
    learning_rate=config.LEARNING_RATE,
    device=None,
    random_seed=config.RANDOM_STATE,
):
    """Train the SimpleCNN model on glass vial images."""
    if device is None:
        device = get_device()

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = SimpleCNN().to(device)

    # Weighted BCE-with-logits loss for binary classification. The dataset is
    # imbalanced, so the positive (defective) class is down-weighted by the
    # negative/positive count ratio to rebalance the loss.
    train_labels = train_loader.dataset.labels
    num_positive = int(sum(train_labels))
    num_negative = len(train_labels) - num_positive
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Halve the learning rate when validation macro-F1 plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.SCHEDULER_PATIENCE
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": [],
        "val_f1_macro": [],
        "best_val_f1_macro": 0.0,
        "best_model_state": None,
    }
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            # Forward pass (raw logits)
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics (logits -> probabilities)
            train_running_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs >= config.THRESHOLD).float().view(-1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.view(-1)).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_accuracy = train_correct / train_total
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average="binary")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                logits = model(images)
                loss = criterion(logits, labels)

                val_running_loss += loss.item()
                probs = torch.sigmoid(logits)
                predicted = (probs >= config.THRESHOLD).float().view(-1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.view(-1)).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average="binary")
        epoch_val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_accuracy"].append(epoch_train_accuracy)
        history["val_accuracy"].append(epoch_val_accuracy)
        history["train_f1"].append(epoch_train_f1)
        history["val_f1"].append(epoch_val_f1)
        history["val_f1_macro"].append(epoch_val_f1_macro)

        # Step the LR scheduler on validation macro-F1
        scheduler.step(epoch_val_f1_macro)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} (lr={optimizer.param_groups[0]['lr']:.6f}): "
            f"train loss={epoch_train_loss:.4f}, acc={epoch_train_accuracy:.4f}, "
            f"F1={epoch_train_f1:.4f} | val loss={epoch_val_loss:.4f}, "
            f"acc={epoch_val_accuracy:.4f}, F1={epoch_val_f1:.4f}, "
            f"macro F1={epoch_val_f1_macro:.4f}"
        )

        # Save best model based on validation macro-F1
        if epoch_val_f1_macro > history["best_val_f1_macro"]:
            history["best_val_f1_macro"] = epoch_val_f1_macro
            # Deep-copy the weights so the "best" state does not keep changing
            history["best_model_state"] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                logger.info(
                    f"Early stopping: no macro F1 improvement for "
                    f"{config.EARLY_STOP_PATIENCE} consecutive epochs"
                )
                break

        # Log to MLflow (only if run is active)
        if mlflow.active_run():
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_accuracy, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)
            mlflow.log_metric("train_f1_score", epoch_train_f1, step=epoch)
            mlflow.log_metric("val_f1_score", epoch_val_f1, step=epoch)
            mlflow.log_metric("val_f1_macro", epoch_val_f1_macro, step=epoch)

    # Load best model weights
    if history["best_model_state"] is not None:
        model.load_state_dict(history["best_model_state"])
        logger.info(f"Restored best weights (val macro F1: {history['best_val_f1_macro']:.4f})")

    return model, history


def tune_threshold(model, val_loader, device=None, threshold_range=None):
    """Tune the classification threshold to maximize macro F1 on the validation set.

    Macro-F1 is used because maximizing the positive-class F1 alone rewards
    predicting "defective" for every image (defective is the majority class).
    """
    if device is None:
        device = get_device()

    if threshold_range is None:
        # Fine grid: the model's probabilities can compress into a narrow band
        threshold_range = [round(t, 3) for t in np.arange(0.025, 0.976, 0.025)]

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)

            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels)

    best_threshold = config.THRESHOLD
    best_f1 = 0.0

    for t in threshold_range:
        preds = (all_probs >= t).astype(int)
        f1_macro = f1_score(all_labels, preds, average="macro")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = t

    logger.info(f"Optimal threshold: {best_threshold} (macro F1: {best_f1:.4f})")

    # Log to MLflow
    mlflow.log_param("optimal_threshold", best_threshold)
    mlflow.log_metric("optimal_threshold_macro_f1", best_f1)

    return best_threshold


def limit_image_dataset(image_dataset, limit):
    """Keep a balanced subset for smoke-test runs.

    The metadata is ordered by class, so taking the first ``limit`` rows would
    leave a small run with only clean images. This helper is only used when
    ``--limit`` is supplied; the default notebook-parity path is unchanged.
    """
    if limit is None or limit >= len(image_dataset):
        return image_dataset

    class_0 = [item for item in image_dataset if item[1] == 0]
    class_1 = [item for item in image_dataset if item[1] == 1]
    class_0_limit = limit // 2
    class_1_limit = limit - class_0_limit

    return class_0[:class_0_limit] + class_1[:class_1_limit]


@app.command()
def main(
    epochs: int = typer.Option(
        config.NUM_EPOCHS,
        "--epochs",
        min=1,
        help="Number of training epochs.",
    ),
    batch_size: int = typer.Option(
        config.BATCH_SIZE,
        "--batch-size",
        min=1,
        help="Batch size for DataLoaders.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=14,
        help="Optional balanced image subset for smoke tests.",
    ),
):
    """Run the training job: data -> train -> tune -> evaluate -> log -> register."""
    setup_mlflow_tracking(config.EXPERIMENT_NAME)

    # Keep parity with notebook ordering: data prep before starting the MLflow run.
    image_dataset = load_image_dataset(config.TRAIN_METADATA_PATH, config.TRAIN_DATA_DIR)
    if limit is not None:
        image_dataset = limit_image_dataset(image_dataset, limit)
        logger.info(f"Using {len(image_dataset)} balanced images due to --limit={limit}")

    train_dataset, val_dataset, test_dataset = split_dataset(image_dataset)
    train_transforms, val_transforms = define_transforms()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        train_transforms,
        val_transforms,
        batch_size=batch_size,
    )

    image_stats = calculate_image_statistics(train_loader)
    config.IMAGE_STATISTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.IMAGE_STATISTICS_PATH, "w") as f:
        json.dump(image_stats, f, indent=2)
    save_class_names(config.CLASS_NAMES_PATH)

    # Notebook parity: reseed immediately before starting and running training.
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    if mlflow.active_run():
        mlflow.end_run()

    run = mlflow.start_run(run_name="training_notebook")

    try:
        mlflow.log_param("model_architecture", "SimpleCNN")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", epochs)
        mlflow.log_param("learning_rate", config.LEARNING_RATE)
        mlflow.log_param("initial_threshold", config.THRESHOLD)
        mlflow.log_param("num_classes", config.NUM_CLASSES)
        mlflow.log_param("image_size", f"{config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]}")
        mlflow.log_param("random_seed", config.RANDOM_STATE)

        model_params = sum(p.numel() for p in SimpleCNN().parameters())
        trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
        mlflow.log_param("total_parameters", model_params)
        mlflow.log_param("trainable_parameters", trainable_params)

        cnn_model, _training_history = train_cnn_model(
            train_loader,
            val_loader,
            num_epochs=epochs,
            learning_rate=config.LEARNING_RATE,
            device=get_device(),
            random_seed=config.RANDOM_STATE,
        )

        optimal_threshold = tune_threshold(cnn_model, val_loader, device=get_device())
        test_results = evaluate_cnn_model(
            cnn_model,
            test_loader,
            device=get_device(),
            threshold=optimal_threshold,
        )

        baseline_stats = compute_baseline_statistics(
            cnn_model,
            val_loader,
            device=get_device(),
            threshold=optimal_threshold,
        )
        log_baseline_metrics(baseline_stats)

        config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": cnn_model.state_dict(),
                "architecture": SimpleCNN.__name__,
                "input_shape": (3, 64, 64),
                "output_shape": (1,),
                "threshold": optimal_threshold,
                "num_classes": config.NUM_CLASSES,
            },
            config.PYTORCH_MODEL_PATH,
        )

        with open(config.MODEL_RESULTS_PATH, "w") as results_file:
            json.dump(test_results, results_file, indent=2)

        mlflow.log_param("classification_threshold", optimal_threshold)

        wrapper = SimpleLoggingWrapper(model=cnn_model)
        mlflow.pyfunc.log_model(
            config.MODEL_ARTIFACT_PATH,
            python_model=wrapper,
            code_paths=[str(config.PROJ_ROOT / "vial_classifier")],
        )

        model_version = register_model(run.info.run_id)

        logger.info(
            f"Training complete: train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}, "
            f"threshold={optimal_threshold:.4f}, "
            f"test_f1={test_results['f1_score']:.4f}, "
            f"registered={config.MODEL_NAME} v{model_version} (Staging)"
        )
    finally:
        if mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    app()
