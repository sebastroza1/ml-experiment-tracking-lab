# Reproducible Machine Learning Experiment Tracking

This project compares a prior baseline, logistic regression, and a controlled PyTorch MLP search on UCI Covertype. In the executed six-run quick search, logistic regression was selected with **0.5011 validation macro-F1** and achieved **0.7108 accuracy / 0.5097 macro-F1** on the untouched test split. MLflow records every run in a local SQLite backend, including parameters, validation metrics, duration, seed, environment, and model artifacts.

> Independent portfolio project by Sebastián Astroza. Not affiliated with or endorsed by CERN.

## What this project demonstrates

- Reproducible data splits and training-only preprocessing.
- Honest baseline, classical ML, and neural model comparison.
- Small resource-aware hyperparameter search across architecture, learning rate, batch size, and optimiser.
- MLflow parameters, metrics, artifacts, SQLite persistence, and artifact retrieval.
- Validation-based selection without test-set tuning.
- Offline unit/smoke tests and CPU-only CI.

## Dataset

The project uses [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype), DOI [10.24432/C50K5N](https://doi.org/10.24432/C50K5N), licensed CC BY 4.0. Quick mode uses a reproducible 12,000-row stratified sample. See [DATASET_CARD.md](DATASET_CARD.md).

## Experiment design

The tracked search contains:

- `DummyClassifier(strategy="prior")` as a non-learning baseline.
- Multinomial logistic regression as the classical model.
- Four MLP configurations covering two architectures, three learning rates, two batch sizes, and AdamW/SGD.

Each run logs the seed, dataset identity, split sizes, Python/platform/PyTorch versions, code revision when available, hyperparameters, validation accuracy, validation macro-F1, duration, and serialized model artifact.

## Run locally

```powershell
& 'C:\Users\sebas\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ml_experiment_tracking_lab.cli
```

Linux uses the same commands after `python3.12 -m venv .venv` and `source .venv/bin/activate`.

Inspect runs locally with:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

The MLflow database and default artifact store are ignored by Git. The compact comparison, selected-model record, retrieved-artifact proof, and plot under `reports/` are committed.

## Results

The best small MLP reached 0.4107 validation macro-F1, below logistic regression's 0.5011. This is a useful negative result: a short neural search did not outperform the simpler model. The selected model artifact was successfully recovered through `MlflowClient` using run ID `de75b766f74e4286ab8e44620239cb66`.

See [RESULTS.md](RESULTS.md) and `reports/quick-search/` for the full validation ranking, test result, plot, serialized candidates, and retrieval evidence.

## Limitations

- Quick mode is a resource-bounded search, not exhaustive optimisation.
- One seed is used for the initial portfolio run; repeated seeds are the next experiment.
- Random stratification does not assess geographic generalisation.
- The local host provides CPU evidence only.

## Author contribution

Sebastián Astroza defined the portfolio requirements and experiment constraints. This repository preserves the implementation, executed comparison, selection rule, artifacts, and verification evidence needed for technical review.
