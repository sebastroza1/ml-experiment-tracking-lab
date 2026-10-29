from __future__ import annotations

import pandas as pd

from ml_experiment_tracking_lab.tracking import run_experiments


def test_tracking_selects_by_validation_and_recovers_artifact(tmp_path, splits) -> None:
    config = {
        "sample_size": 630,
        "epochs": 1,
        "seed": 42,
        "experiment_name": "test-search",
        "search": [
            {
                "architecture": "small",
                "learning_rate": 0.001,
                "batch_size": 64,
                "optimizer": "adamw",
            }
        ],
    }
    result = run_experiments(
        splits,
        config,
        output_dir=tmp_path / "reports",
        database=tmp_path / "mlflow.db",
        data_source="synthetic-test-fixture",
    )
    runs = pd.read_csv(tmp_path / "reports" / "runs.csv")
    assert result["status"] == "completed"
    assert result["retrieval_verified"] is True
    assert len(runs) == 3
    assert result["best_run"]["validation_macro_f1"] == runs["validation_macro_f1"].max()
