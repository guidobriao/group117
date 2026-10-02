"""Structurally valid zero-prediction example; replace it with the final model."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from utils import constant_prediction


def load_model(model_path: str | Path) -> Any:
    """Load and return the submitted fitted model artifact."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: Any, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> np.ndarray:
    """Return the dummy model's constant as a NumPy array with shape (1,).

    history_df contains all observations strictly before the prediction date,
    including past chlorophyll. day_features_df contains exactly the current
    day's exogenous variables and date, but never current-day chlorophyll.

    This example deliberately does not use either DataFrame. Students must
    replace it with their lag construction, saved input preprocessing, and
    prediction procedure. Fitted sklearn.preprocessing input objects are
    allowed; library pipelines and output-processing wrappers are not.
    """
    return np.asarray([constant_prediction(model)], dtype=float)
