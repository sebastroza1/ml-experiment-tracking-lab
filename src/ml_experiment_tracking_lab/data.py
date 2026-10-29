from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


@dataclass(frozen=True)
class Splits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_data(cache_dir: str | Path, sample_size: int, seed: int) -> tuple[Splits, str]:
    cache = Path(cache_dir) / "covertype.csv.gz"
    if cache.exists():
        frame = pd.read_csv(cache)
        source = "local-cache"
    else:
        dataset = fetch_ucirepo(id=31)
        features = dataset.data.features.copy()
        target = dataset.data.targets.iloc[:, 0].rename("Cover_Type")
        frame = pd.concat((features, target), axis=1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False, compression="gzip")
        source = "uci-ml-repository"

    if frame.shape[1] != 55 or frame.isna().any().any():
        raise ValueError("Covertype cache must contain 54 complete features and Cover_Type.")
    target = frame.pop("Cover_Type").astype(int)
    if set(target.unique()) != set(range(1, 8)):
        raise ValueError("Covertype target must contain classes 1 through 7.")
    features, _, target, _ = train_test_split(
        frame,
        target,
        train_size=min(sample_size, len(frame)),
        stratify=target,
        random_state=seed,
    )
    indices = np.arange(len(features))
    train_validation, test = train_test_split(
        indices, test_size=0.2, stratify=target, random_state=seed
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=target.iloc[train_validation],
        random_state=seed,
    )
    values = features.to_numpy(dtype=np.float32)
    labels = target.to_numpy(dtype=np.int64) - 1
    scaler = StandardScaler().fit(values[train, :10])
    values[train, :10] = scaler.transform(values[train, :10])
    values[validation, :10] = scaler.transform(values[validation, :10])
    values[test, :10] = scaler.transform(values[test, :10])
    return (
        Splits(
            values[train],
            labels[train],
            values[validation],
            labels[validation],
            values[test],
            labels[test],
        ),
        source,
    )
