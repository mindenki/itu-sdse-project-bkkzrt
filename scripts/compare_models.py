"""Compare the trained model with the deployed (Production) model and record the decision.

Runs the registry step of notebook 02 (`vial_classifier.modeling.registry`: select the
best run, promote it if it beats Production, export Production to models/model). The
registry does not report what it decided, so this reads the Production version before
and after it and writes:

    {"promoted": true|false, "production_version_before": ..., "production_version_after": ...}

Usage:
    python scripts/compare_models.py --decision decision.json
"""

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from vial_classifier import config
from vial_classifier.modeling import registry


def production_version() -> str | None:
    """The version that holds the Production alias (as text), or None if there is none."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        version = (
            MlflowClient().get_model_version_by_alias(config.MODEL_NAME, "Production").version
        )
    except MlflowException:
        return None
    # MLflow 3 returns the version as a number, MLflow 2 as a string
    return str(version)


def decide(before: str | None, after: str | None) -> dict:
    """The new model was promoted if the Production alias now points at another version."""
    return {
        "promoted": after is not None and after != before,
        "production_version_before": before,
        "production_version_after": after,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the trained model with Production and record the decision."
    )
    parser.add_argument("--decision", type=Path, default=Path("decision.json"))
    parser.add_argument("--export-dir", type=Path, default=config.EXPORT_DIR)
    args = parser.parse_args(argv)

    before = production_version()
    registry.main(export_dir=args.export_dir)
    decision = decide(before, production_version())

    args.decision.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
