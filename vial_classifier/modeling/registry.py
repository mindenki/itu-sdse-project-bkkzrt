"""Model registration helpers extracted from notebooks/01_training.ipynb.

This module keeps the same behavior as the training notebook:
- register `runs:/<run_id>/model_pyfunc`
- wait until the model version is READY
- transition to Staging
"""

import os
import time

import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient
import typer

from vial_classifier import config

app = typer.Typer()


def _setup_mlflow_tracking():
    os.makedirs(config.MLFLOW_STATE_DIR, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)


def wait_until_ready(model_name, model_version, max_attempts=10):
    """Wait until the model version is READY in the registry."""
    client = MlflowClient()
    for attempt in range(max_attempts):
        try:
            model_version_details = client.get_model_version(name=model_name, version=model_version)
            status = ModelVersionStatus.from_string(model_version_details.status)
            print(f"Model status: {ModelVersionStatus.to_string(status)}")
            if status == ModelVersionStatus.READY:
                print(f"Model {model_name} version {model_version} is READY")
                return True
        except Exception as e:
            print(f"Error checking model status (attempt {attempt + 1}/{max_attempts}): {e}")
        time.sleep(1)
    print(f"Model {model_name} version {model_version} did not become READY after {max_attempts} attempts")
    return False


def register_model(
    run_id,
    model_name=config.MODEL_NAME,
    artifact_path=config.MODEL_ARTIFACT_PATH,
):
    """Register one run artifact and promote it to Staging."""
    _setup_mlflow_tracking()

    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_until_ready(model_details.name, model_details.version)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(model_name, model_details.version, "deployed_to", "staging")
    client.set_model_version_tag(model_name, model_details.version, "task", "classification")
    client.set_registered_model_alias(
        name=model_name,
        alias="best",
        version=str(model_details.version),
    )

    print(f"Successfully transitioned {model_name} version {model_details.version} to Staging")
    return int(model_details.version)


@app.command()
def main(
    run_id: str = typer.Option(..., help="MLflow run id that logged model_pyfunc."),
    model_name: str = typer.Option(config.MODEL_NAME, help="Registered model name."),
):
    """Register one run's model and transition it to Staging."""
    version = register_model(run_id=run_id, model_name=model_name)
    print(f"Model registered successfully: {model_name} version {version}")


if __name__ == "__main__":
    app()
