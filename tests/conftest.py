from __future__ import annotations

import numpy as np
import pytest

from ml_experiment_tracking_lab.data import Splits


@pytest.fixture
def splits() -> Splits:
    generator = np.random.default_rng(42)

    def sample(rows: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.tile(np.arange(7), rows // 7 + 1)[:rows]
        features = generator.normal(size=(rows, 54)).astype(np.float32)
        features[:, 0] += labels * 0.2
        return features, labels.astype(np.int64)

    train_x, train_y = sample(350)
    validation_x, validation_y = sample(140)
    test_x, test_y = sample(140)
    return Splits(train_x, train_y, validation_x, validation_y, test_x, test_y)
