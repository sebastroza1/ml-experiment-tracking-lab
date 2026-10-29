from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import Splits

ARCHITECTURES = {"small": (64, 32), "medium": (128, 64, 32)}


class MLP(nn.Module):
    def __init__(self, architecture: str) -> None:
        super().__init__()
        widths = ARCHITECTURES[architecture]
        layers: list[nn.Module] = []
        inputs = 54
        for width in widths:
            layers.extend((nn.Linear(inputs, width), nn.ReLU()))
            inputs = width
        layers.append(nn.Linear(inputs, 7))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(model: nn.Module, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        predictions = model(torch.from_numpy(features)).argmax(1).numpy()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def train_mlp(
    splits: Splits,
    architecture: str,
    learning_rate: float,
    batch_size: int,
    optimizer_name: str,
    epochs: int,
    seed: int,
    checkpoint: str | Path,
) -> tuple[MLP, dict[str, float], list[dict[str, float]]]:
    seed_everything(seed)
    model = MLP(architecture)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=learning_rate)
        if optimizer_name == "adamw"
        else torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    )
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(splits.x_train), torch.from_numpy(splits.y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        validation = metrics(model, splits.x_validation, splits.y_validation)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total / len(splits.y_train),
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
            }
        )
    duration = time.perf_counter() - started
    target = Path(checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": architecture, "state_dict": model.state_dict()}, target)
    final = metrics(model, splits.x_validation, splits.y_validation)
    final["duration_seconds"] = duration
    return model, final, history
