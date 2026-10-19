"""Grading interface: load the fitted model and predict one-day-ahead chlorophyll."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import build_single_row


def load_model(model_path: str | Path) -> Any:
    """Load and return the submitted fitted model artifact."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: Any, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> np.ndarray:
    """Return the predicted chlorophyll-a concentration as a NumPy array (1,).

    Steps (all explicit):
    1. build the lagged feature row from history + current exogenous data
    2. order features as during training
    3. clip exogenous values at the stored training percentiles
    4. standardize with the stored training means/stds
    5. predict and enforce a non-negative finite output
    """
    state = model
    row = build_single_row(history_df, day_features_df, state["lag_spec"])

    x = np.array([row.loc[row.index[-1], c] for c in state["feature_cols"]], dtype=float)

    x = np.clip(x, state["clip_lo"], state["clip_hi"])
    x = (x - state["mu"]) / state["sd"]

    y_hat = float(state["model"].predict(x.reshape(1, -1))[0])
    y_hat = max(y_hat, 0.0)   # chlorophyll concentration is non-negative
    return np.asarray([y_hat], dtype=float)