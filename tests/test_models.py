from __future__ import annotations

import torch

from ml_experiment_tracking_lab.models import MLP, metrics, train_mlp


def test_mlp_shape() -> None:
    assert MLP("small")(torch.zeros(4, 54)).shape == (4, 7)


def test_train_checkpoint_and_metrics(tmp_path, splits) -> None:
    model, validation, history = train_mlp(
        splits,
        architecture="small",
        learning_rate=0.001,
        batch_size=64,
        optimizer_name="adamw",
        epochs=2,
        seed=42,
        checkpoint=tmp_path / "model.pt",
    )
    assert (tmp_path / "model.pt").exists()
    assert len(history) == 2
    assert 0 <= validation["macro_f1"] <= 1
    assert 0 <= metrics(model, splits.x_test, splits.y_test)["accuracy"] <= 1
