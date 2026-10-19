"""Auxiliary prediction functions imported by my_model.py.

Lag construction is implemented explicitly here (no library helpers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATE = "date"
TARGET = "chlorophyll_a_mg_m3"


def build_single_row(history_df: pd.DataFrame, day_features_df: pd.DataFrame,
                     lag_spec: dict) -> pd.DataFrame:
    """Build exactly one lagged feature row for the prediction date.

    history_df: all observations strictly before the prediction date (14 cols).
    day_features_df: the current date + 12 current-day exogenous variables.
    Inputs are never modified (copies only).
    """
    today = day_features_df.copy()
    if TARGET not in today:
        today[TARGET] = np.nan
    combined = pd.concat([history_df.copy(), today], ignore_index=True, sort=False)
    combined[DATE] = pd.to_datetime(combined[DATE])
    combined = combined.sort_values(DATE).reset_index(drop=True)

    # Engineered features, identical to training (train.py / 2_model_selection.py)
    for col in ("upwelling_index", "nitrate_umol_l", "wind_speed_m_s", "sst_c"):
        combined[f"{col}_ma7"] = combined[col].rolling(7, min_periods=1).mean()
    combined["upw_x_wind"] = combined["upwelling_index"] * combined["wind_speed_m_s"]
    combined["nitrate_x_par"] = combined["nitrate_umol_l"] * combined["par_umol_m2_s"]

    columns = {}
    for col, lags in lag_spec.items():
        values = pd.to_numeric(combined[col], errors="coerce")
        for lag in lags:
            if col == TARGET and lag == 0:
                raise ValueError("target lag 0 is forbidden")
            columns[f"{col}__lag_{lag}"] = values.shift(lag)

    day = combined[DATE].dt.dayofyear.iloc[-1]
    columns["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
    columns["annual_cos"] = np.cos(2 * np.pi * day / 365.25)

    row = pd.DataFrame(columns)     # last row = prediction date
    return row.tail(1)