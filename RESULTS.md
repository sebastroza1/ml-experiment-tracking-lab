# Experiment Results

## Run summary

- Date: 2026-10-29
- Mode: quick CPU search
- Dataset: UCI Covertype, 12,000 stratified observations
- Split: 7,200 train / 2,400 validation / 2,400 test
- Seed: 42
- Python: 3.12.14
- PyTorch: 2.9.1+cpu
- MLflow: 3.17.2 with local SQLite backend
- Selection metric: validation macro-F1

## Validation ranking

| Rank | Run | Key configuration | Accuracy | Macro-F1 | Duration |
|---:|---|---|---:|---:|---:|
| 1 | Logistic regression | `lbfgs`, max 500 iterations | 0.7158 | 0.5011 | 0.403 s |
| 2 | Small MLP | AdamW, lr 0.003, batch 256 | 0.7158 | 0.4107 | 0.653 s |
| 3 | Small MLP | AdamW, lr 0.001, batch 128 | 0.7142 | 0.3511 | 0.931 s |
| 4 | Medium MLP | AdamW, lr 0.001, batch 256 | 0.7058 | 0.3133 | 0.828 s |
| 5 | Prior baseline | Class prior only | 0.4875 | 0.0936 | 0.001 s |
| 6 | Medium MLP | SGD, lr 0.01, batch 256 | 0.4875 | 0.0936 | 0.675 s |

The MLP search was intentionally small and resource-bounded. Five epochs were insufficient for the tested networks to beat logistic regression on macro-F1. The SGD configuration effectively matched the non-learning prior baseline, which is evidence of an unsuitable configuration rather than a successful neural result.

## Selected model and test evaluation

- Selected run ID: `de75b766f74e4286ab8e44620239cb66`
- Model: logistic regression
- Selection value: validation macro-F1 0.5011
- Held-out test accuracy: 0.7108
- Held-out test macro-F1: 0.5097

Only the selected model was evaluated on the test split. The model artifact was downloaded through `MlflowClient`; `selection.json` records `retrieval_verified: true` and the retrieved path.

## What the experiment shows

1. A tracked neural model is not automatically better than a well-matched linear baseline.
2. Macro-F1 changes the ranking interpretation: logistic regression and the best MLP had identical validation accuracy, but substantially different macro-F1.
3. MLflow preserved six independently identifiable runs, configurations, metrics, durations, and artifacts.
4. The result can be recovered without selecting on test performance.

## Limitations

- One seed and a 12,000-row sample were used.
- MLPs trained for only five epochs and no scheduler or class weighting was searched.
- Duration covers fitting/training only, not dataset loading or MLflow artifact I/O.
- The experiment ran before the repository's initial commit, so MLflow records the code revision as `uncommitted`; the exact source and result are committed together.
