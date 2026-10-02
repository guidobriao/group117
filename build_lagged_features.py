"""Leakage-safe construction of one-day-ahead lagged regression rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATE = "date"
TARGET = "chlorophyll_a_mg_m3"
DEFAULT_LAGS = {
    TARGET: list(range(1, 15)),
    "sst_c": list(range(0, 15)),
    "par_umol_m2_s": list(range(0, 8)),
    "nitrate_umol_l": list(range(0, 15)),
    "wind_speed_m_s": list(range(0, 8)),
    "upwelling_index": list(range(0, 15)),
    "mixed_layer_depth_m": list(range(0, 8)),
    "salinity_psu": list(range(0, 8)),
    "current_speed_m_s": list(range(0, 8)),
    "river_discharge_index": list(range(0, 8)),
    "cloud_fraction": list(range(0, 8)),
    "surface_pressure_hpa": list(range(0, 8)),
    "turbidity_ntu": list(range(0, 8)),
}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, sheet_name="daily_data") if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    df[DATE] = pd.to_datetime(df[DATE])
    return df.sort_values(DATE).reset_index(drop=True)


def build_lagged_frame(raw: pd.DataFrame, lag_spec: dict[str, list[int]] | None = None,
                       require_target: bool = True) -> pd.DataFrame:
    """Return rows indexed by prediction date, never using target lag 0."""
    lag_spec = lag_spec or DEFAULT_LAGS
    if DATE not in raw:
        raise ValueError(f"missing required column: {DATE}")
    if require_target and TARGET not in raw:
        raise ValueError(f"missing required column: {TARGET}")
    frame = raw.copy().sort_values(DATE).reset_index(drop=True)
    columns: dict[str, pd.Series | np.ndarray] = {DATE: pd.to_datetime(frame[DATE])}
    if TARGET in frame:
        columns[TARGET] = frame[TARGET].astype(float)
    for col, lags in lag_spec.items():
        if col not in frame:
            raise ValueError(f"lag specification refers to missing column: {col}")
        for lag in lags:
            if col == TARGET and lag == 0:
                raise ValueError("target lag 0 is forbidden: it leaks the answer")
            columns[f"{col}__lag_{lag}"] = pd.to_numeric(frame[col], errors="coerce").shift(lag)
    day = pd.to_datetime(frame[DATE]).dt.dayofyear.to_numpy()
    columns["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
    columns["annual_cos"] = np.cos(2 * np.pi * day / 365.25)
    out = pd.DataFrame(columns)
    if require_target:
        out = out.dropna(subset=[TARGET])
    max_lag = max(max(v) for v in lag_spec.values())
    return out.iloc[max_lag:].reset_index(drop=True)


def one_day_row(history: pd.DataFrame, day_features: pd.DataFrame,
                lag_spec: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """Construct exactly one feature row from observed history plus today's exogenous data."""
    if len(day_features) != 1:
        raise ValueError("day_features must contain exactly one row")
    today = day_features.copy()
    if TARGET not in today:
        today[TARGET] = np.nan
    combined = pd.concat([history, today], ignore_index=True, sort=False)
    lagged = build_lagged_frame(combined, lag_spec=lag_spec, require_target=False)
    row = lagged.tail(1).drop(columns=[DATE, TARGET], errors="ignore")
    if len(row) != 1:
        raise ValueError("insufficient history to construct the requested lags")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    raw = read_table(args.input)
    lagged = build_lagged_frame(raw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(output, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
            lagged.to_excel(writer, sheet_name="lagged_data", index=False)
            ws = writer.book["lagged_data"]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            ws.column_dimensions["A"].width = 13
            for cell in ws["A"][1:]:
                cell.number_format = "yyyy-mm-dd"
            for cell in ws[1]:
                cell.font = __import__("openpyxl").styles.Font(bold=True, color="FFFFFF")
                cell.fill = __import__("openpyxl").styles.PatternFill("solid", fgColor="176B87")
                ws.column_dimensions[cell.column_letter].width = min(31, max(14, len(str(cell.value)) + 2))
    else:
        lagged.to_csv(output, index=False)
    print(f"wrote {len(lagged)} rows and {lagged.shape[1]} columns to {output}")


if __name__ == "__main__":
    main()
