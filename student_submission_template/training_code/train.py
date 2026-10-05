"""Training workflow: reproduces best_model/model.pkl from the raw released data.

Recipe found by exploratory analysis (see exploratory_analysis.ipynb):
- Reduced lag spec: AR lags 1-5 for chlorophyll, lags 0-4 for all exogenous
- Exogenous features clipped at the training 1st/99th percentiles
- Standardization computed on the (clipped) training set only
- Ridge regression (alpha=0.1)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DATE = "date"
TARGET = "chlorophyll_a_mg_m3"

REDUCED_LAGS = {
    TARGET: list(range(1, 6)),
    "sst_c": list(range(0, 5)),
    "par_umol_m2_s": list(range(0, 5)),
    "nitrate_umol_l": list(range(0, 5)),
    "wind_speed_m_s": list(range(0, 5)),
    "upwelling_index": list(range(0, 5)),
    "mixed_layer_depth_m": list(range(0, 5)),
    "salinity_psu": list(range(0, 5)),
    "current_speed_m_s": list(range(0, 5)),
    "river_discharge_index": list(range(0, 5)),
    "cloud_fraction": list(range(0, 5)),
    "surface_pressure_hpa": list(range(0, 5)),
    "turbidity_ntu": list(range(0, 5)),
}
EXO_P = 0.99          # percentile for exogenous clipping
ALPHA = 0.1           # Ridge regularization
VAL_START = "2023-01-01"


def read_raw(path: Path) -> pd.DataFrame:
    df = (pd.read_excel(path, sheet_name="daily_data") if path.suffix.lower() == ".xlsx"
          else pd.read_csv(path))
    df[DATE] = pd.to_datetime(df[DATE])
    return df.sort_values(DATE).reset_index(drop=True)


def build_lagged_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Explicit lag construction (target lag 0 is never used)."""
    frame = raw.copy().sort_values(DATE).reset_index(drop=True)
    columns = {DATE: pd.to_datetime(frame[DATE])}
    for col, lags in REDUCED_LAGS.items():
        for lag in lags:
            columns[f"{col}__lag_{lag}"] = pd.to_numeric(frame[col], errors="coerce").shift(lag)
    day = frame[DATE].dt.dayofyear.to_numpy()
    columns["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
    columns["annual_cos"] = np.cos(2 * np.pi * day / 365.25)
    out = pd.DataFrame(columns)
    out[TARGET] = frame[TARGET].astype(float)
    max_lag = max(max(v) for v in REDUCED_LAGS.values())
    return out.iloc[max_lag:].reset_index(drop=True)


def metrics(y_true, y_pred):
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return r2, rmse, mae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--final", action="store_true",
                        help="after the validation report, refit on the full dataset")
    args = parser.parse_args()

    raw = read_raw(args.data)
    lagged = build_lagged_frame(raw)

    feature_cols = [c for c in lagged.columns if c not in (DATE, TARGET)]
    X = lagged[feature_cols].to_numpy(dtype=float)
    y = lagged[TARGET].to_numpy(dtype=float)

    # Chronological validation split: train <= 2022, validation 2023
    train_mask = lagged[DATE] < VAL_START
    val_mask = ~train_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # Preprocessing parameters: training split only
    lo = np.percentile(X_train, (1 - EXO_P) * 100, axis=0)
    hi = np.percentile(X_train, EXO_P * 100, axis=0)
    X_train_c = np.clip(X_train, lo, hi)
    X_val_c = np.clip(X_val, lo, hi)

    mu = X_train_c.mean(axis=0)
    sd = X_train_c.std(axis=0)
    sd[sd == 0] = 1.0
    X_train_s = (X_train_c - mu) / sd
    X_val_s = (X_val_c - mu) / sd

    # Validation model (train <= 2022)
    model = Ridge(alpha=ALPHA)
    model.fit(X_train_s, y_train)

    r2, rmse, mae = metrics(y_val, model.predict(X_val_s))
    print(f"Validation ({VAL_START} onwards): R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f}")

    if args.final:
        # Refit on the FULL released dataset; recompute preprocessing
        # parameters on the full data (they must match the fitting data)
        lo = np.percentile(X, (1 - EXO_P) * 100, axis=0)
        hi = np.percentile(X, EXO_P * 100, axis=0)
        X_c = np.clip(X, lo, hi)
        mu = X_c.mean(axis=0)
        sd = X_c.std(axis=0)
        sd[sd == 0] = 1.0

        model = Ridge(alpha=ALPHA)
        model.fit((X_c - mu) / sd, y)
        print("Final refit on the full released dataset (2015-2023)")

    # Save artifact: fitted estimator + plain numerical state only
    state = {
        "model": model,
        "feature_cols": feature_cols,
        "lag_spec": REDUCED_LAGS,
        "clip_lo": lo,
        "clip_hi": hi,
        "mu": mu,
        "sd": sd,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(state, f)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()