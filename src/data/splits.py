"""Train and validation splits.

The same page appears in dozens of photographs, every layout exists in a
Hungarian and an English version typeset identically, and each photographer's
photos share a phone and a room. `train.csv` has two grouping columns:

* `group`: no page and no photographer appears in two groups;
* `group_strict`: coarser, and in addition both language versions of a
  layout are in the same unit. The competition's test set was cut along it.

Split by `group_strict`. Measured on this data with a ResNet-18 at 768 px,
RMSE by validation scheme against the held-out test:

    random 5-fold   0.113
    by group        0.136
    by group_strict 0.143
    held-out test   0.148

The random split is a fifth too good. `group` is still 8 % too good, because
the two language versions of a layout sit in two neighbouring groups and the
network recognises the layout. `group_strict` lands on the test.

TASK 1 is `build_splits`. The test in `tests/test_splits.py` is red until it
is written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def build_splits(df: pd.DataFrame, group_column: str = "group_strict", val_fraction: float = 0.2,
                 seed: int = config.SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """TASK 1. Hold out whole units of `group_column` until at least `val_fraction` of the rows are held out."""
    groups = list(df[group_column].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    
    val_indices = []
    train_indices = []
    total_rows = len(df)
    target_val_rows = total_rows * val_fraction

    for g in groups:
        indices = df.index[df[group_column] == g].to_numpy()
        if len(val_indices) < target_val_rows:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    val_idx = np.array(val_indices)
    train_idx = np.array(train_indices)
    return train_idx, val_idx
