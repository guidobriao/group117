# -*- coding: utf-8 -*-
"""exploratory_analysis.py

Daily chlorophyll-a forecasting with linear models (OLS / Ridge / Lasso)
on lagged features.

Protocol (assignment, Section 5): the automated evaluation of the hidden
2024 test is primarily R2, so R2 is THE selection metric; RMSE and MAE
(mg/m3) are the advised physical-unit errors and sMAPE (%) the bounded
relative error (MAPE is not used: it explodes on days where the observed
value is close to zero). The walk-forward is computed on the three folds
2021 / 2022 / 2023, but every selection decision (model family, alpha,
exo_p, target cap) uses only the folds 2021+2022 (mean per-fold R2 on the
raw truth, blind to 2023): 2023 is the one-shot FINAL validation (train
2015-2022 -> predict 2023, the official split of the project), where every
configuration is scored once, raw and capped, after the selection is
sealed. After that the winner is retrained from scratch on all of
2015-2023; the 2024 test stays hidden with the professors. Figures are
saved to ../results via save_fig().
"""

"""## Cell 0 - Setup, data loading and feature engineering"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from build_lagged_features import read_table, build_lagged_frame

# --- Figure saving setup (replaces every plt.show()) ---
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_fig(name, dpi=150):
    """Save the current figure to ../results/<name>.png, then close it."""
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")

df = read_table("../../data/chlorophyll_student_2015_2023.xlsx")

# --- Feature engineering (nothing here looks into the future) ---
# No manual seasonal encoding: the lagged-features module appends its own
# annual_sin / annual_cos pair to every frame it builds, and seasonality is
# shown in the EDA below through the monthly-mean bars.
# 7-day rolling means: ocean memory beyond single-day lags
# (min_periods=1 -> no NaNs, partial means on the first 6 days of 2015)
ROLL_COLS = ["upwelling_index", "nitrate_umol_l", "wind_speed_m_s", "sst_c"]
for col in ROLL_COLS:
    df[f"{col}_ma7"] = df[col].rolling(7, min_periods=1).mean()

# Physically motivated interactions
df["upw_x_wind"] = df["upwelling_index"] * df["wind_speed_m_s"]
df["nitrate_x_par"] = df["nitrate_umol_l"] * df["par_umol_m2_s"]

print(df.shape, "|", df["date"].min().date(), "->", df["date"].max().date())
df.head()

"""## Cell 1 - Basic exploratory plots"""

# Chlorophyll over time (target)
plt.figure(figsize=(14, 4))
plt.plot(df["date"], df["chlorophyll_a_mg_m3"], lw=0.5)
plt.title("Chlorophyll-a (2015-2023)")
plt.ylabel("mg/m3")
save_fig("cell01_chlorophyll_time_series")

# Seasonality: monthly mean
monthly = df.groupby(df["date"].dt.month)["chlorophyll_a_mg_m3"].mean()
plt.figure(figsize=(8, 3))
plt.bar(monthly.index, monthly.values)
plt.title("Mean chlorophyll per month (seasonality)")
plt.xlabel("Month")
plt.ylabel("mg/m3")
save_fig("cell01_monthly_seasonality")

# Other variables over time (auto-sized grid: 12 exogenous + 6 engineered)
n = len(df.columns[2:])
ncols = 2
nrows = (n + 1) // 2
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.2 * nrows), sharex=True)
for ax, col in zip(axes.flat, df.columns[2:]):
    ax.plot(df["date"], df[col], lw=0.4)
    ax.set_title(col, fontsize=9)
plt.tight_layout()
save_fig("cell01_exogenous_time_series")

# Correlation with the target
corr = df.drop(columns=["date"]).corr()["chlorophyll_a_mg_m3"].sort_values()
print(corr)

"""## Cell 2 - Correlation matrix (heatmap)"""

cols = df.drop(columns=["date"]).columns
corr_matrix = df[cols].corr()

plt.figure(figsize=(13, 11))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(cols)), cols, rotation=90, fontsize=8)
plt.yticks(range(len(cols)), cols, fontsize=8)
plt.title("Correlation matrix (current variables)")
save_fig("cell02_correlation_matrix")

# Most correlated pairs of exogenous variables (multicollinearity)
corr_pairs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
              .stack().sort_values(key=abs, ascending=False))
print("Top 15 correlated exogenous pairs:")
print(corr_pairs.head(15))

"""## Cell 3 - Cross-correlation target vs exogenous per lag

Each panel shows corr(y(t), x(t-j)) for j = 0..21. The lag with the peak
correlation indicates the response delay of that variable: this is the
guide for choosing the lags q_k of each exogenous variable.
"""

MAX_LAG = 21
exo_cols = [c for c in df.columns if c not in ("date", "chlorophyll_a_mg_m3")]

n = len(exo_cols)
ncols = 2
nrows = (n + 1) // 2
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.2 * nrows))
for ax, col in zip(axes.flat, exo_cols):
    ccf = [df["chlorophyll_a_mg_m3"].corr(df[col].shift(j)) for j in range(MAX_LAG + 1)]
    ax.bar(range(MAX_LAG + 1), ccf)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("lag (days)")
plt.tight_layout()
save_fig("cell03_cross_correlation_per_lag")

"""## Cell 4 - Autocorrelation (ACF) and partial autocorrelation (PACF) of the target

AR lags outside the red significance band are the ones to keep for p.
"""

y_series = df["chlorophyll_a_mg_m3"]

# Simple autocorrelation
acf = [y_series.autocorr(lag=j) for j in range(31)]
plt.figure(figsize=(8, 3))
plt.bar(range(31), acf)
plt.title("Chlorophyll ACF")
plt.xlabel("lag")
save_fig("cell04_acf")

# PACF: Yule-Walker solution with numpy (no external libraries)
from numpy.linalg import solve
r = np.array([y_series.autocorr(lag=j) for j in range(1, 31)])
pacf = []
for k in range(1, 31):
    R = np.array([[r[abs(i-j)] for j in range(k)] for i in range(k)])
    pacf.append(solve(R, r[:k])[-1])
plt.figure(figsize=(8, 3))
plt.bar(range(1, 31), pacf)
plt.axhline(1.96/np.sqrt(len(y_series)), color="red", ls="--", lw=0.7)
plt.axhline(-1.96/np.sqrt(len(y_series)), color="red", ls="--", lw=0.7)
plt.title("Chlorophyll PACF (band = significance)")
plt.xlabel("lag")
save_fig("cell04_pacf")

"""## Cell 5 - Outlier analysis (retrieval errors)

The manual warns that occasional retrieval errors can be much larger than
ordinary noise. Look for extreme spikes in the target and in the exogenous
variables. Nothing is removed or capped here: this cell only motivates the
winsorizing options that the sweep in Cell 9 will test blindly.
"""

# 1) Target distribution and extreme values
plt.figure(figsize=(10, 3))
plt.hist(df["chlorophyll_a_mg_m3"], bins=100)
plt.title("Chlorophyll distribution")
save_fig("cell05_target_distribution")

print("Target statistics:")
print(df["chlorophyll_a_mg_m3"].describe())
print("\nValues above the 99.5th percentile:",
      (df["chlorophyll_a_mg_m3"] > df["chlorophyll_a_mg_m3"].quantile(0.995)).sum())

# 2) Distribution of each exogenous variable: sudden spikes = retrieval errors
n = len(exo_cols)
ncols = 2
nrows = (n + 1) // 2
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.0 * nrows))
for ax, col in zip(axes.flat, exo_cols):
    ax.boxplot(df[col], orientation="horizontal")   # vert=False deprecated in mpl 3.11
    ax.set_title(col, fontsize=9)
plt.tight_layout()
save_fig("cell05_exogenous_boxplots")

"""## Cell 6 - Metrics (assignment, Section 5) + THE validation protocol

Metrics: the assignment (Section 5) says the automated evaluation is
PRIMARILY the coefficient of determination R2 (computed against the mean
of the evaluated days); RMSE and MAE in mg/m3 are the advised physical-unit
errors. sMAPE (%) is kept as the bounded relative error. MAPE is not used:
it explodes when the observed value is close to zero (target min ~0.28).
One metrics() and one table format for models, variants and baselines.

THE validation protocol (rolling origin, two stages, no other hold-out
before the hidden 2024 test):
    fold 2021: train on 2015-2020 -> predict 2021   } SELECTION (mean
    fold 2022: train on 2015-2021 -> predict 2022   } per-fold metrics)
    fold 2023: train on 2015-2022 -> predict 2023   -> ONE-SHOT final
                                                          validation
Every selection decision (model family, alpha, exo_p, target cap) uses
only the 2021+2022 folds, blind to 2023; 2023 scores every configuration
once, after the selection is sealed (Cell 10). The per-fold errors on
2021/2022 are deliberately not tabulated: the selection only needs their
mean, and the performance picture comes from the one-shot table on 2023.
Inside each fold the winsorizing limits and the standardization mu/sigma
are computed from that fold's TRAINING data only: no day is ever predicted
by a model that has seen it (not for the fit, not for mu/sigma, not for
the clip limits). After the one-shot check the winner is retrained from
scratch on the full 2015-2023 data (Cell 12); the hidden 2024 test stays
with the professors.
"""

FOLD_YEARS = [2021, 2022, 2023]          # folds for the tables (rolling origin)
SELECT_YEARS = [2021, 2022]              # selection looks ONLY here (blind to 2023)
FINAL_YEAR = 2023                        # one-shot final validation
SEL_LABEL = "+".join(str(y) for y in SELECT_YEARS)

def metrics(y_true, y_pred):
    """R2 (vs the mean of the evaluated days), RMSE, MAE (mg/m3), sMAPE (%)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    e = y_true - y_pred
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "R2": 1.0 - ss_res / ss_tot,
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "sMAPE": float(np.mean(2.0 * np.abs(e) / (np.abs(y_true) + np.abs(y_pred))) * 100.0),
    }

# Selection metric: R2, the metric the assignment (Section 5) grades the
# hidden 2024 test on. Ranking key: the MEAN per-fold R2 over 2021+2022,
# with ties at the printed third decimal broken by the secondary metric.
SELECT_BY = "R2"
OTHER = "sMAPE" if SELECT_BY == "R2" else "R2"
ASCENDING = {"R2": False, "RMSE": True, "MAE": True, "sMAPE": True}

n_sel_days = int(df["date"].dt.year.isin(SELECT_YEARS).sum())
n_tab_days = int(df["date"].dt.year.isin(FOLD_YEARS).sum())
print("VALIDATION PROTOCOL (single, for models, variants and baselines)")
for y in FOLD_YEARS:
    tag = ("   <- ONE-SHOT FINAL VALIDATION (never used for selection)"
           if y == FINAL_YEAR else "")
    print(f"  fold {y}: train {df['date'].min().year}-{y-1} -> predict {y}{tag}")
print(f"  tables on the {n_tab_days} days of the {len(FOLD_YEARS)} folds (raw truth)")
print(f"  SELECTION only on {SEL_LABEL} ({n_sel_days} days, mean per-fold "
      f"{SELECT_BY}, tie-break {OTHER}, blind to {FINAL_YEAR})")
print("  final step: retrain the winner from scratch on 2015-2023 (Cell 12)")
print("  never touched: the hidden 2024 test")

"""## Cell 7 - Lagged frame (engineered spec) + walk-forward engine

Lag spec: AR set [1, 7, 8] (dominant AR(1) plus weekly harmonics), the 12
original exogenous variables at lags 0-4 (their cross-correlations peak at
lags 2-3), the 7-day rolling means at lags 0-1 and the two interactions at
lags 0-2. The module appends its own annual_sin / annual_cos pair: the
sanity print below must show exactly ['annual_sin', 'annual_cos'] (79
features expected: 77 from the spec + 2 from the module).

The engine is THE SAME for every model, variant and baseline: per fold -
winsorize every feature column at the fold-train percentiles (the sin/cos
pair is bounded in [-1, 1] and stays in practice unchanged), optional
target cap in TRAIN only, standardize once with fold-train mu/sigma, fit,
predict non-negative.
"""

ENGINEERED_LAGS = {
    "chlorophyll_a_mg_m3": [1, 7, 8],   # dominant AR(1) + weekly harmonics
    # 12 original exogenous, lags 0-4 (cross-correlations peak at lags 2-3)
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
    # engineered in Cell 0 (the seasonal pair comes from the module itself)
    "upwelling_index_ma7": [0, 1],
    "nitrate_umol_l_ma7": [0, 1],
    "wind_speed_m_s_ma7": [0, 1],
    "sst_c_ma7": [0, 1],
    "upw_x_wind": [0, 1, 2],
    "nitrate_x_par": [0, 1, 2],
}

lagged = build_lagged_frame(df, lag_spec=ENGINEERED_LAGS)
COLS = [c for c in lagged.columns if c not in ("date", "chlorophyll_a_mg_m3")]
print("Frame:", lagged.shape, "| auto-added by the module:",
      [c for c in COLS if "__lag_" not in c])   # must be ['annual_sin', 'annual_cos']
print("Features:", len(COLS))                    # expected 79

def make_model(name, alpha):
    if name == "OLS":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=alpha, max_iter=50000)
    if name == "Lasso":
        return Lasso(alpha=alpha, max_iter=50000)
    raise ValueError(name)

def fit_predict(X_tr, y_tr, X_va, exo_p, tgt_clip, name, alpha):
    """Winsorize features at TRAIN percentiles -> optional target cap in
    TRAIN only -> standardize once (train mu/sigma) -> fit -> non-negative
    predictions. Everything explicit: no pipeline objects anywhere."""
    if exo_p is not None:
        lo = np.percentile(X_tr, (1.0 - exo_p) * 100.0, axis=0)
        hi = np.percentile(X_tr, exo_p * 100.0, axis=0)
        X_tr = np.clip(X_tr, lo, hi)
        X_va = np.clip(X_va, lo, hi)          # same TRAIN limits on validation
    if tgt_clip is not None:
        y_tr = np.clip(y_tr, None, tgt_clip)
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    m = make_model(name, alpha).fit((X_tr - mu) / sd, y_tr)
    yp = np.clip(m.predict((X_va - mu) / sd), 0.0, None)   # non-negative output
    return yp, m

def wf_run(exo_p=None, tgt_clip=None, name="Ridge", alpha=0.1):
    """Full walk-forward over FOLD_YEARS on the engineered frame. Returns
    (dates, y_true, y_pred, year) pooled over the folds: the model never
    sees the days it predicts. Cell 9 keeps only what it needs from these
    arrays (the selection aggregates and the one-shot fold)."""
    dates_all, yt_all, yp_all, yr_all = [], [], [], []
    for y in FOLD_YEARS:
        d = lagged["date"]
        tr = d < f"{y}-01-01"
        va = (d >= f"{y}-01-01") & (d < f"{y + 1}-01-01")
        yp, _ = fit_predict(lagged.loc[tr, COLS].to_numpy(),
                            lagged.loc[tr, "chlorophyll_a_mg_m3"].to_numpy(),
                            lagged.loc[va, COLS].to_numpy(),
                            exo_p, tgt_clip, name, alpha)
        dates_all.append(lagged.loc[va, "date"])
        yt_all.append(lagged.loc[va, "chlorophyll_a_mg_m3"].to_numpy())
        yp_all.append(yp)
        yr_all.append(np.full(int(va.sum()), y))
    return (pd.concat(dates_all).to_numpy(),
            np.concatenate(yt_all), np.concatenate(yp_all), np.concatenate(yr_all))

def score_sel(yt, yp, yr):
    """Selection aggregates: per-fold metrics on SELECT_YEARS (2021+2022),
    averaged fold by fold. These are the ONLY numbers allowed to drive a
    selection decision."""
    per = [metrics(yt[yr == y], yp[yr == y]) for y in SELECT_YEARS]
    return {f"{k}_sel": float(np.mean([m[k] for m in per]))
            for k in ("R2", "RMSE", "MAE", "sMAPE")}

"""## Cell 8 - The three baselines (same folds, same metrics, same truth)

- persistence: tomorrow = today (yesterday's OBSERVED value, exactly what
  the grading protocol feeds us at prediction time); the bar that any ARX
  model must clear;
- seasonal naive: tomorrow = same calendar day one year earlier; the
  "last year replays itself" reference;
- train mean: a constant at the mean of the fold's training years. R2 is
  built against the mean of the EVALUATED days, so a constant at the TEST
  mean would score exactly R2 = 0 by construction; the train mean is not
  that constant, so its R2 is not exactly 0 - it hovers around 0 (positive
  when train and test distributions agree, negative under drift), which
  makes it an honest floor: any model worth its features must beat it.
All three run on the same walk-forward folds used for the models, so they
land in the same final table.
"""

def wf_run_baseline(kind):
    """Walk-forward over FOLD_YEARS for a baseline; same output shape as
    wf_run. Positional day alignment (the data is contiguous daily, no
    gaps): no date arithmetic, no per-day lookups."""
    d_idx = pd.DatetimeIndex(df["date"])
    y = df["chlorophyll_a_mg_m3"].to_numpy()
    yr = d_idx.year.to_numpy()
    # value of the previous calendar day (Jan 1st 2015 has none: never validated)
    persist = np.concatenate([[np.nan], y[:-1]])
    # value of the same calendar day one year earlier (leap-safe)
    s_daily = df.set_index("date")["chlorophyll_a_mg_m3"]
    last_year = s_daily.reindex(d_idx - pd.DateOffset(years=1)).to_numpy()
    dates_all, yt_all, yp_all, yr_all = [], [], [], []
    for year in FOLD_YEARS:
        idx = np.flatnonzero(yr == year)          # positions of the fold days
        y_tr = y[yr < year]
        if kind == "persistence":
            yp = persist[idx]
        elif kind == "seasonal naive":
            yp = last_year[idx]
        elif kind == "train mean":
            yp = np.full(idx.size, y_tr.mean())
        else:
            raise ValueError(kind)
        assert not np.isnan(yp).any()   # complete daily data: this must hold
        dates_all.append(df["date"].to_numpy()[idx])
        yt_all.append(y[idx])
        yp_all.append(yp)
        yr_all.append(np.full(idx.size, year))
    return (np.concatenate(dates_all), np.concatenate(yt_all),
            np.concatenate(yp_all), np.concatenate(yr_all))

BASELINES = ["persistence", "seasonal naive", "train mean"]

BASE_RUNS = {}
baseline_sel = {}
for kind in BASELINES:
    BASE_RUNS[kind] = wf_run_baseline(kind)
    _, yt, yp, yr = BASE_RUNS[kind]
    baseline_sel[kind] = score_sel(yt, yp, yr)
print("Baselines, mean R2 on the selection folds " + SEL_LABEL + ": "
      + " | ".join(f"{k} {baseline_sel[k]['R2_sel']:+.3f}" for k in BASELINES))

"""## Cell 9 - The sweep: SELECTION on 2021+2022 only

Grid: model family x alpha x exogenous winsorizing percentile (exo_p,
applied to the exogenous FEATURES at the fold-train percentiles) x
optional target cap in TRAIN (tgt). Every configuration runs the same
walk-forward; the selection stage looks only at the mean per-fold metrics
on 2021+2022 (raw truth): the 2023 fold is reserved for the one-shot final
table of Cell 10 and never influences any decision here. The winner is
ranked by mean R2 (the grading metric of the hidden 2024 test), with ties
at the printed third decimal broken by mean sMAPE. Printed here:
- the selected configuration, plus the sMAPE-only winner as a secondary
  check (warning when the two disagree);
- the best baseline on the same folds (the bar to clear);
- the general picture: averages of the selection metrics across the grid,
  by model family/alpha, by exo_p and by exo_p x tgt - because the single
  best row is often a coin flip inside a class of equivalent models, and
  the robust conclusions are the ones that hold on average.
"""

EXO_P_GRID = [None, 0.95, 0.975, 0.99, 0.995]
TGT_GRID = [None, 5.0]
MODEL_ALPHA = [("OLS", None),
               ("Ridge", 0.03), ("Ridge", 0.1), ("Ridge", 0.3), ("Ridge", 1.0),
               ("Lasso", 0.003), ("Lasso", 0.01), ("Lasso", 0.03)]

GRID = [(exo_p, tgt, name, alpha)
        for exo_p in EXO_P_GRID
        for tgt in TGT_GRID
        for (name, alpha) in MODEL_ALPHA]

print(f"{len(GRID)} configurations x {len(FOLD_YEARS)} folds = "
      f"{len(GRID) * len(FOLD_YEARS)} fits (Lasso is the slow one: be patient)\n")

model_rows, yp23_list = [], []
dates23 = yt23 = None
for i, (exo_p, tgt, name, alpha) in enumerate(GRID, 1):
    d, yt, yp, yr = wf_run(exo_p, tgt, name, alpha)
    m23 = yr == FINAL_YEAR
    if dates23 is None:
        dates23, yt23 = d[m23], yt[m23]        # one-shot fold (same days for all)
    r = {"kind": "model", "exo_p": exo_p, "tgt": tgt,
         "model": name, "alpha": alpha}
    r.update(score_sel(yt, yp, yr))            # selection: 2021+2022 only
    r.update(metrics(yt23, yp[m23]))           # one-shot 2023, RAW truth
    model_rows.append(r)
    yp23_list.append(yp[m23])
    if i % 10 == 0:
        print(f"[sweep] {i}/{len(GRID)}")

def cfg_str(row):
    # plain None stays None inside these dicts (no DataFrame round-trip yet)
    p = "none" if row["exo_p"] is None else row["exo_p"]
    t = "none" if row["tgt"] is None else row["tgt"]
    a = "-" if row["alpha"] is None else row["alpha"]
    return f"exo_p={p}, tgt={t}, {row['model']} (alpha={a})"

def rank_key(row, primary):
    """Round the primary metric to the printed 3 decimals: configurations
    that look tied in the tables are then ordered by the other metric."""
    other = "sMAPE" if primary == "R2" else "R2"
    p, o = round(row[f"{primary}_sel"], 3), row[f"{other}_sel"]
    return (-p, o) if not ASCENDING[primary] else (p, -o)

best_pos = min(range(len(model_rows)),
               key=lambda i: rank_key(model_rows[i], SELECT_BY))
best_row = model_rows[best_pos]
other_pos = min(range(len(model_rows)),
                key=lambda i: rank_key(model_rows[i], OTHER))

print(f"\nSELECTED by mean {SELECT_BY} on {SEL_LABEL} "
      f"({SELECT_BY}_sel = {best_row[f'{SELECT_BY}_sel']:.3f}, tie-break "
      f"{OTHER}_sel = {best_row[f'{OTHER}_sel']:.2f}): {cfg_str(best_row)}"
      + ("   <- grading metric of the hidden 2024 test" if SELECT_BY == "R2" else ""))
print(f"winner by {OTHER} alone (secondary check): {cfg_str(model_rows[other_pos])}")
if best_pos != other_pos:
    print(f"WARNING: the {SELECT_BY} winner and the {OTHER} winner DISAGREE ->")
    print("check both rows in the one-shot table below before trusting the choice.")
else:
    print(f"Agreement check: the {SELECT_BY} and {OTHER} rankings pick the SAME configuration. OK.")

best_base = min(BASELINES, key=lambda k: rank_key(baseline_sel[k], SELECT_BY))
print(f"Best baseline on {SEL_LABEL} ({SELECT_BY}_sel = "
      f"{baseline_sel[best_base][f'{SELECT_BY}_sel']:.2f}): {best_base} "
      "- the model must beat this.")

def _fmt_p(p):
    return "none" if p is None else f"{p:g}"

def _fmt_a(a):
    return "-" if a is None else f"{a:g}"

def _avg(rows, key):
    return float(np.mean([r[key] for r in rows]))

print(f"\nGENERAL PICTURE on the selection folds {SEL_LABEL} "
      "(averages across the grid; R2_sel / sMAPE_sel)")
print("-- by model family and alpha --")
for name, alpha in MODEL_ALPHA:
    g = [r for r in model_rows
         if r["model"] == name and r["alpha"] == alpha]
    print(f"  {name:<6} alpha={_fmt_a(alpha):<6} n={len(g):>2}   "
          f"R2_sel={_avg(g, 'R2_sel'):+.3f}   sMAPE_sel={_avg(g, 'sMAPE_sel'):6.2f}")
print("-- by exogenous winsorizing (exo_p) --")
for p in EXO_P_GRID:
    g = [r for r in model_rows if r["exo_p"] == p]
    print(f"  exo_p={_fmt_p(p):<6} n={len(g):>2}   "
          f"R2_sel={_avg(g, 'R2_sel'):+.3f}   sMAPE_sel={_avg(g, 'sMAPE_sel'):6.2f}")
print("-- exo_p x target cap (mean R2_sel) --")
for p in EXO_P_GRID:
    cells = []
    for t in TGT_GRID:
        g = [r for r in model_rows if r["exo_p"] == p and r["tgt"] == t]
        cells.append(f"tgt={_fmt_p(t):>4}: {_avg(g, 'R2_sel'):+.3f}")
    print(f"  exo_p={_fmt_p(p):<6} " + "   ".join(cells))

"""## Cell 10 - ONE-SHOT FINAL VALIDATION on 2023: every model, raw + capped

The 2023 fold (train 2015-2022 -> predict 2023, the official split of the
project) is where all the configurations are finally scored, AFTER the
selection has been sealed on 2021+2022. One single table:
- RAW (official): R2, RMSE, MAE, sMAPE against the 2023 truth as it is,
  outliers included - the honest preview of what the hidden 2024 test will
  look like;
- CAPPED (sensitivity): the same predictions scored against the 2023 truth
  clipped at one COMMON cap for all rows (the cap selected in training, or
  the 99.5th percentile of the one-shot train when the winner uses none),
  so the reading stays comparable across rows. It shows how much of the
  error concentrates on the extreme bloom days.
Sorted by RAW R2, with ties at the printed third decimal broken by RAW
sMAPE; the CAPPED columns never drive any ordering. The predictions are
never modified here - only the yardstick changes.
"""

BEST = {"exo_p": best_row["exo_p"], "tgt": best_row["tgt"],
        "model": best_row["model"], "alpha": best_row["alpha"]}

# common sensitivity cap: the selected target cap, else the one-shot train p99.5
y_seed = lagged.loc[lagged["date"] < f"{FINAL_YEAR}-01-01",
                    "chlorophyll_a_mg_m3"].to_numpy()
CAP_DUAL = BEST["tgt"] if BEST["tgt"] is not None else float(np.percentile(y_seed, 99.5))
cap_src = ("selected tgt" if BEST["tgt"] is not None
           else f"99.5th pct of the train before {FINAL_YEAR}")

# capped one-shot metrics for every model row (common yardstick)
for r, yp23 in zip(model_rows, yp23_list):
    r.update({f"{k}_cap": v for k, v in
              metrics(np.clip(yt23, None, CAP_DUAL), yp23).items()})

# baseline rows: same metrics, same folds, same one-shot year
rows_all = list(model_rows)
for kind in BASELINES:
    _, yt_b, yp_b, yr_b = BASE_RUNS[kind]
    m23 = yr_b == FINAL_YEAR
    r = {"kind": "baseline", "exo_p": None, "tgt": None,
         "model": kind, "alpha": None}
    r.update(baseline_sel[kind])
    r.update(metrics(yt_b[m23], yp_b[m23]))
    r.update({f"{k}_cap": v for k, v in
              metrics(np.clip(yt_b[m23], None, CAP_DUAL), yp_b[m23]).items()})
    rows_all.append(r)

table_df = pd.DataFrame(rows_all)
table_df.to_csv(os.path.join(RESULTS_DIR, "sweep.csv"), index=False)

def fmt_id(r):
    p = "none" if pd.isna(r["exo_p"]) else f"{r['exo_p']:.3f}"
    t = "none" if pd.isna(r["tgt"]) else f"{r['tgt']:.0f}"
    a = "-" if pd.isna(r["alpha"]) else f"{r['alpha']:g}"
    return f"{r['kind']:<9} {p:>6} {t:>5} {r['model']:<14} {a:>6}"

def fmt_cell(col, v):
    return f"{v:>10.2f}" if col.startswith("sMAPE") else f"{v:>10.3f}"

SHOW = ["R2", "RMSE", "MAE", "sMAPE",
        "R2_cap", "RMSE_cap", "MAE_cap", "sMAPE_cap"]
tab = (table_df.assign(_r2=table_df["R2"].round(3))
       .sort_values(["_r2", "sMAPE"], ascending=[False, True]))

print(f"\nONE-SHOT FINAL VALIDATION {FINAL_YEAR}: train 2015-{FINAL_YEAR - 1} "
      f"-> predict {FINAL_YEAR} ({len(yt23)} days)")
print(f"RAW = official truth | CAPPED = truth clipped at {CAP_DUAL:.2f} mg/m3 "
      f"({cap_src}) | sorted by R2 (raw), ties at the 3rd decimal by sMAPE (raw)")
header = " " * 44 + "".join(f"{c:>10}" for c in SHOW)
print(header)
print("-" * len(header))
for pos, r in tab.iterrows():
    mark = "  <- SELECTED on 2021+2022 (blind to 2023)" if pos == best_pos else ""
    print(fmt_id(r) + "".join(fmt_cell(c, r[c]) for c in SHOW) + mark)

"""## Cell 11 - The selected configuration on the one-shot 2023 validation (plots)"""

yp_win = yp23_list[best_pos]                  # winner's 2023 predictions (from the sweep)
d_b, _, yp_b, yr_b = BASE_RUNS[best_base]     # best baseline by the selection metric
m23b = yr_b == FINAL_YEAR                     # same one-shot days for the baseline

plt.figure(figsize=(14, 4))
plt.plot(dates23, yt23, lw=0.6, color="black", label="Observed")
plt.plot(dates23, yp_win, lw=0.9, color="tab:blue",
         label=f"winner ({BEST['model']})")
plt.plot(d_b[m23b], yp_b[m23b], lw=0.7, color="tab:orange", ls="--",
         label=f"{best_base} baseline")
plt.legend()
plt.title(f"One-shot validation {FINAL_YEAR} (train 2015-{FINAL_YEAR - 1}): "
          "winner vs observed vs baseline")
plt.ylabel("mg/m3")
save_fig("cell11_final_predictions_vs_observed")

# Zoom on the last 90 days (Nov-Dec: the extreme bloom)
plt.figure(figsize=(14, 4))
plt.plot(dates23[-90:], yt23[-90:], lw=1, color="black", label="Observed")
plt.plot(dates23[-90:], yp_win[-90:], lw=1, color="tab:blue",
         label=f"winner ({BEST['model']})")
plt.plot(d_b[m23b][-90:], yp_b[m23b][-90:], lw=1, color="tab:orange", ls="--",
         label=f"{best_base} baseline")
plt.legend()
plt.title(f"Zoom: last 90 days of {FINAL_YEAR} (Nov-Dec bloom)")
plt.ylabel("mg/m3")
save_fig("cell11_final_zoom_last_90_days")

residuals = yt23 - yp_win
plt.figure(figsize=(14, 3))
plt.plot(dates23, residuals, lw=0.5)
plt.axhline(3 * residuals.std(), color="red", ls="--", lw=0.8)
plt.axhline(-3 * residuals.std(), color="red", ls="--", lw=0.8)
plt.title(f"Residuals {FINAL_YEAR}, winner ({BEST['model']}), bands = +/- 3 sigma")
plt.ylabel("mg/m3")
save_fig("cell11_final_residuals")

"""## Cell 12 - Final retrain on ALL released data (2015-2023) + coefficients

The winning recipe is retrained from scratch on everything released so far
(2015-2023, validation days included): the validation folds have done their
job (selection on 2021+2022, one-shot final check on 2023) and their data
are legitimate training material now. NO score is computed here: the only
data never used for anything is the hidden 2024 test, which stays with the
professors. This block is also the skeleton of training_code/train.py: the
winsorizing limits, mu/sigma and the target cap below are the plain numbers
that must be saved together with the fitted model into best_model/model.pkl
(assignment Section 7).
"""

X_all = lagged[COLS].to_numpy()
y_all = lagged["chlorophyll_a_mg_m3"].to_numpy()

# transforms recomputed on the FULL 2015-2023 training data, in the same
# order used at validation time: winsorize -> cap target -> standardize
Xa = X_all
if BEST["exo_p"] is not None:
    lo_f = np.percentile(Xa, (1.0 - BEST["exo_p"]) * 100.0, axis=0)
    hi_f = np.percentile(Xa, BEST["exo_p"] * 100.0, axis=0)
    Xa = np.clip(Xa, lo_f, hi_f)
ya = np.clip(y_all, None, BEST["tgt"]) if BEST["tgt"] is not None else y_all
mu_f = Xa.mean(axis=0)
sd_f = Xa.std(axis=0)
sd_f[sd_f == 0] = 1.0
model_final = make_model(BEST["model"], BEST["alpha"]).fit((Xa - mu_f) / sd_f, ya)

print(f"Final model: {BEST['model']} (alpha={BEST['alpha']}), "
      f"exo_p={BEST['exo_p']}, tgt={BEST['tgt']}")
print(f"trained on {len(ya)} days (2015-2023), {len(COLS)} features")
print(f"Intercept c = {model_final.intercept_:.4f} mg/m3")
print("No score here: the 2024 test is hidden (assignment Section 5).\n")

coefs = pd.Series(model_final.coef_, index=COLS)
print("Autoregressive coefficients (chlorophyll):")
print(coefs[coefs.index.str.startswith("chlorophyll")].to_string())
print("\nExogenous features with highest |coefficient| (top 20):")
exo = coefs.drop(index=[c for c in coefs.index if c.startswith("chlorophyll")])
for c in exo.abs().sort_values(ascending=False).head(20).index:
    print(f"{c}: {coefs[c]:+.4f}")

order = coefs.abs().sort_values(ascending=False).index
colors = ["tab:red" if c.startswith("chlorophyll") else "tab:blue" for c in order]
plt.figure(figsize=(14, 5))
plt.bar(range(len(order)), coefs[order], color=colors)
plt.xticks(range(len(order)), order, rotation=90, fontsize=6)
plt.title(f"Final model coefficients ({BEST['model']}, standardized features; red = AR lags)")
plt.ylabel("mg/m3 per z-unit")
save_fig("cell12_final_coefficients")