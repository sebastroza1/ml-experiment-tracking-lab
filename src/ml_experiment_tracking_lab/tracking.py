from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .data import Splits
from .models import metrics, train_mlp


def _revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _sklearn_metrics(model, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def _tracking_uri(database: Path) -> str:
    return f"sqlite:///{database.resolve().as_posix()}"


def run_experiments(
    splits: Splits,
    config: dict[str, object],
    output_dir: str | Path,
    database: str | Path,
    data_source: str,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    database = Path(database)
    mlflow.set_tracking_uri(_tracking_uri(database))
    mlflow.set_experiment(str(config["experiment_name"]))
    common = {
        "seed": int(config["seed"]),
        "dataset": "UCI Covertype (id=31)",
        "data_source": data_source,
        "training_rows": len(splits.y_train),
        "validation_rows": len(splits.y_validation),
        "test_rows": len(splits.y_test),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "git_revision": _revision() or "uncommitted",
    }
    rows: list[dict[str, object]] = []
    models: dict[str, object] = {}
    artifacts = output / "artifacts"
    artifacts.mkdir(exist_ok=True)

    for name, model in (
        ("dummy-prior", DummyClassifier(strategy="prior")),
        ("logistic-regression", LogisticRegression(max_iter=500, solver="lbfgs")),
    ):
        started = time.perf_counter()
        model.fit(splits.x_train, splits.y_train)
        duration = time.perf_counter() - started
        validation = _sklearn_metrics(model, splits.x_validation, splits.y_validation)
        artifact = artifacts / f"{name}.joblib"
        joblib.dump(model, artifact)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({**common, "model": name})
            mlflow.log_metrics(
                {
                    "validation_accuracy": validation["accuracy"],
                    "validation_macro_f1": validation["macro_f1"],
                    "duration_seconds": duration,
                }
            )
            mlflow.log_artifact(str(artifact), artifact_path="model")
            run_id = run.info.run_id
        rows.append(
            {
                "run_id": run_id,
                "model": name,
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "duration_seconds": duration,
                "artifact": str(artifact),
            }
        )
        models[run_id] = model

    for index, search in enumerate(config["search"]):
        search = dict(search)
        name = f"mlp-{index + 1}-{search['architecture']}-{search['optimizer']}"
        artifact = artifacts / f"{name}.pt"
        model, validation, history = train_mlp(
            splits=splits,
            architecture=str(search["architecture"]),
            learning_rate=float(search["learning_rate"]),
            batch_size=int(search["batch_size"]),
            optimizer_name=str(search["optimizer"]),
            epochs=int(config["epochs"]),
            seed=int(config["seed"]),
            checkpoint=artifact,
        )
        history_path = artifacts / f"{name}-history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({**common, "model": "mlp", **search, "epochs": config["epochs"]})
            mlflow.log_metrics(
                {
                    "validation_accuracy": validation["accuracy"],
                    "validation_macro_f1": validation["macro_f1"],
                    "duration_seconds": validation["duration_seconds"],
                }
            )
            mlflow.log_artifact(str(artifact), artifact_path="model")
            mlflow.log_artifact(str(history_path), artifact_path="history")
            run_id = run.info.run_id
        rows.append(
            {
                "run_id": run_id,
                "model": name,
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "duration_seconds": validation["duration_seconds"],
                "artifact": str(artifact),
                **search,
            }
        )
        models[run_id] = model

    best = max(rows, key=lambda row: float(row["validation_macro_f1"]))
    best_model = models[str(best["run_id"])]
    test_metrics = (
        metrics(best_model, splits.x_test, splits.y_test)
        if isinstance(best_model, torch.nn.Module)
        else _sklearn_metrics(best_model, splits.x_test, splits.y_test)
    )
    best_copy = output / Path(str(best["artifact"])).name
    shutil.copy2(str(best["artifact"]), best_copy)
    client = MlflowClient(tracking_uri=_tracking_uri(database))
    downloaded = client.download_artifacts(str(best["run_id"]), "model", dst_path=str(output))
    result = {
        "status": "completed",
        "selection_metric": "validation_macro_f1",
        "best_run": best,
        "test_metrics_selected_model_only": test_metrics,
        "mlflow_database": str(database),
        "retrieved_artifact_path": str(downloaded),
        "retrieval_verified": Path(downloaded).exists(),
        "environment": common,
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    frame.to_csv(output / "runs.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(frame["model"], frame["validation_macro_f1"])
    axis.set(title="Validation macro-F1 by tracked run", ylabel="Macro-F1")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output / "validation_comparison.png", dpi=160)
    plt.close(figure)
    return result
