# Model Card

## Candidates

- Prior-probability dummy baseline.
- Multinomial logistic regression.
- Small MLP: 54-64-32-7.
- Medium MLP: 54-128-64-32-7.

## Selection protocol

All candidates use the same seeded train/validation/test split. The best run is selected using validation macro-F1. Only the selected run is evaluated on the held-out test split. MLflow stores each run's parameters, validation metrics, duration, environment, and serialized artifact.

## Intended use

This is an experiment-management demonstration, not an operational forestry model. It is intended to show reproducibility, traceability, comparison, selection, and artifact recovery.

## Limitations

The quick search uses one seed, a 12,000-row sample, and five MLP epochs. Results do not establish full-dataset convergence or geographic generalisation.

## Executed selection result

Logistic regression was selected by validation macro-F1 (0.5011). It obtained 0.7108 accuracy and 0.5097 macro-F1 on the held-out 2,400-row test split. The best MLP validation macro-F1 was 0.4107, so this experiment provides no evidence that the tested neural configurations outperform the classical model.
