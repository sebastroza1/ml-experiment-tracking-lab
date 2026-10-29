# Reproduction Guide

1. Create a Python 3.12 virtual environment.
2. Install the CPU PyTorch wheel and `.[dev]` as shown in `README.md`.
3. Run `ruff check .` and `pytest`.
4. Run `python -m ml_experiment_tracking_lab.cli`.
5. Inspect `reports/quick-search/selection.json` and `runs.csv`.
6. Start the MLflow UI against `sqlite:///mlflow.db` to inspect all tracked runs locally.

The data cache, SQLite database, and default MLflow artifact store are intentionally ignored. `selection.json` contains the best run ID, selection metric, held-out test metrics, and a path proving the selected run's model artifact was retrieved through `MlflowClient`.

Do not compare a rerun with this result unless the dataset sample, split seed, package set, and search configuration are identical.
