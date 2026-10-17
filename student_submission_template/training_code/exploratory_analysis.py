# -*- coding: utf-8 -*-
"""exploratory_analysis.py (v3)

Metriche e protocollo di valutazione allineati alla consegna (sez. 5):
R2 (metrica principale della valutazione automatica), RMSE e MAE in mg/m3,
sMAPE in %. MAPE rimossa. UNA sola validazione walk-forward (fold 2022 e
2023, pooled) su cui avviene TUTTA la selezione; il vincitore viene
riaddestrato from scratch su tutto il 2015-2023; il 2024 resta il test
nascosto dei professori. Il changelog vive nel messaggio di commit / PR /
README, non piu' qui. Figure salvate in ../results tramite save_fig().
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
# Seasonal encoding: REMOVED from df in v3. The module auto-adds its own
# annual_sin / annual_cos pair to every lagged frame, so our manual
# syear/cyear columns were duplicates (they had already left the lag spec
# in v2; now they leave df too). Seasonality in the EDA below is shown by
# the monthly-mean bars: no sin/cos columns are needed for that.
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
errors. sMAPE (%) is kept as the bounded relative error. MAPE is dropped:
it explodes when the observed value is close to zero (target min ~0.28).
One metrics() and one table format for models, variants and baselines.

THE validation protocol (the ONLY selection stage - there is no other
hold-out before the hidden 2024 test):
    fold 2022: train on 2015-2021 -> predict 2022
    fold 2023: train on 2015-2022 -> predict 2023
The 730 pooled validation days (RAW truth) are the basis for every
selection decision (model family, alpha, exo_p, target cap). Inside each
fold the winsorizing limits and the standardization mu/sigma are computed
from that fold's TRAINING data only: no day is ever predicted by a model
that has seen it (not for the fit, not for mu/sigma, not for the clip
limits). After selection the winner is retrained from scratch on the full
2015-2023 data (Cell 12); the hidden 2024 test stays with the professors.
"""

VAL_YEARS = [2022, 2023]

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

# Selection metric for the final ranking. "sMAPE" (bounded, robust) is the
# default; switch to "R2" to align the selection with the metric the
# assignment uses for the automated grading (Cell 9 prints both winners
# and warns when they differ, so the choice is informed either way).
SELECT_BY = "sMAPE"
ASCENDING = {"R2": False, "RMSE": True, "MAE": True, "sMAPE": True}

print("VALIDATION PROTOCOL (single, for models, variants and baselines)")
for y in VAL_YEARS:
    print(f"  fold {y}: train {df['date'].min().year}-{y-1} -> predict {y}")
print(f"  selection on the 730 pooled validation days, ranked by {SELECT_BY} (raw truth)")
print("  final step: retrain the winner from scratch on 2015-2023 (Cell 12)")
print("  never touched: the hidden 2024 test")

"""## Cell 7 - Lagged frame (engineered spec) + walk-forward engine

ENGINEERED_LAGS vs v2: the AR set loses lag 14 too (negligible
coefficient, like lag 2 already removed in v2) and is now [1, 7, 8]
(dominant AR(1) + weekly harmonics). The module auto-adds its seasonal
pair: the sanity print below must show exactly ['annual_sin',
'annual_cos'] (79 features expected: 77 from the spec + 2 from the module).

The engine below is THE SAME for every model, variant and baseline: per
fold - winsorize every feature column at the fold-train percentiles (the
sin/cos pair is bounded in [-1, 1] and stays in practice unchanged),
optional target cap in TRAIN only, standardize once with fold-train
mu/sigma, fit, predict non-negative.
"""

ENGINEERED_LAGS = {
    "chlorophyll_a_mg_m3": [1, 7, 8],   # dominant AR(1) + weekly harmonics (2, 12, 14 dropped)
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
    """Full walk-forward over VAL_YEARS on the engineered frame. Returns
    (dates, y_true, y_pred, year) pooled over the folds: the 730 validation
    predictions. The model never sees the days it predicts."""
    dates_all, yt_all, yp_all, yr_all = [], [], [], []
    for y in VAL_YEARS:
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

def score(yt, yp, yr):
    """Pooled metrics + per-year metrics: one dict, one table format."""
    out = metrics(yt, yp)
    for y in VAL_YEARS:
        m = yr == y
        for k, v in metrics(yt[m], yp[m]).items():
            out[f"{k}_{y}"] = v
    return out

"""## Cell 8 - The two baselines (same folds, same metrics, same truth)

Kept (the two solid, non-redundant references):
- persistence: tomorrow = today (yesterday's OBSERVED value - exactly what
  the grading protocol will feed us at prediction time); the bar that any
  ARX model must clear;
- climatology: tomorrow = mean of the same day-of-year over the fold's
  training years; the "pure seasonality" reference.
Dropped (v2 had four): train mean - it is the null model already built
into R2 (a constant predictor at the mean gives R2 = 0 by construction);
seasonal naive (same day LAST year) - a one-year, noisier version of the
climatology that replays last year's specific events.
"""

def wf_run_baseline(kind):
    """Walk-forward over VAL_YEARS for a baseline; same output shape as
    wf_run, so baselines and models land in the same table."""
    s = df.set_index("date")["chlorophyll_a_mg_m3"]
    dates_all, yt_all, yp_all, yr_all = [], [], [], []
    for y in VAL_YEARS:
        tr = df["date"] < f"{y}-01-01"
        va = (df["date"] >= f"{y}-01-01") & (df["date"] < f"{y + 1}-01-01")
        y_tr = df.loc[tr, "chlorophyll_a_mg_m3"].to_numpy()
        d_va = df.loc[va, "date"]
        if kind == "persistence":
            # tomorrow = today: yesterday's value is always available
            yp = np.array([s.get(d - pd.Timedelta(days=1), np.nan) for d in d_va])
        elif kind == "climatology":
            # tomorrow = mean of the same day-of-year over the training years
            train_df = df.loc[tr]
            clim = train_df.groupby(train_df["date"].dt.dayofyear)["chlorophyll_a_mg_m3"].mean()
            yp = np.array([clim.get(d.dayofyear, y_tr.mean()) for d in d_va])
        else:
            raise ValueError(kind)
        assert not np.isnan(yp).any()   # daily data has no gaps
        dates_all.append(d_va.to_numpy())
        yt_all.append(df.loc[va, "chlorophyll_a_mg_m3"].to_numpy())
        yp_all.append(yp)
        yr_all.append(np.full(int(va.sum()), y))
    return (np.concatenate(dates_all), np.concatenate(yt_all),
            np.concatenate(yp_all), np.concatenate(yr_all))

BASELINES = ["persistence", "climatology"]

BASE_RUNS = {}
baseline_scores = {}
for kind in BASELINES:
    BASE_RUNS[kind] = wf_run_baseline(kind)
    _, yt, yp, yr = BASE_RUNS[kind]
    baseline_scores[kind] = score(yt, yp, yr)
    pooled = {k: round(v, 3) for k, v in baseline_scores[kind].items() if "_" not in k}
    print(f"{kind:<13} pooled:", pooled)

"""## Cell 9 - The sweep: EVERY selection decision on the pooled validation

Grid: model family x alpha x exogenous winsorizing percentile x optional
target cap in TRAIN. Every configuration runs the SAME walk-forward
(2 folds, 730 pooled validation days, RAW truth) and lands in ONE league
table together with the baselines. The winner is the best MODEL row by
SELECT_BY; baselines race in the same table but are references, not
candidates. The assignment grades primarily on R2, so the script also
prints the R2-winner and warns if it differs from the SELECT_BY winner.
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

print(f"{len(GRID)} configurations x {len(VAL_YEARS)} folds = "
      f"{len(GRID) * len(VAL_YEARS)} fits (Lasso is the slow one: be patient)\n")

rows = []
for i, (exo_p, tgt, name, alpha) in enumerate(GRID, 1):
    _, yt, yp, yr = wf_run(exo_p, tgt, name, alpha)
    r = {"kind": "model", "exo_p": exo_p, "tgt": tgt,
         "model": name, "alpha": alpha}
    r.update(score(yt, yp, yr))
    rows.append(r)
    if i % 10 == 0:
        print(f"[sweep] {i}/{len(GRID)}")

for kind in BASELINES:
    r = {"kind": "baseline", "exo_p": None, "tgt": None,
         "model": kind, "alpha": None}
    r.update(baseline_scores[kind])
    rows.append(r)

sweep_df = pd.DataFrame(rows)
sweep_df.to_csv(os.path.join(RESULTS_DIR, "v3_sweep.csv"), index=False)

# --- One league table: models + baselines, same protocol, same truth ---
SHOW = ["R2", "RMSE", "MAE", "sMAPE", "R2_2022", "R2_2023", "sMAPE_2022", "sMAPE_2023"]

def fmt_id(r):
    p = "none" if pd.isna(r["exo_p"]) else f"{r['exo_p']:.3f}"
    t = "none" if pd.isna(r["tgt"]) else f"{r['tgt']:.0f}"
    a = "-" if pd.isna(r["alpha"]) else f"{r['alpha']:g}"
    return f"{r['kind']:<9} {p:>6} {t:>5} {r['model']:<12} {a:>6}"

league = sweep_df.sort_values(SELECT_BY, ascending=ASCENDING[SELECT_BY])
header = " " * 42 + "".join(f"{c:>10}" for c in SHOW)
print("\nPOOLED WALK-FORWARD VALIDATION 2022+2023 (raw truth), sorted by " + SELECT_BY)
print(header)
print("-" * len(header))
for _, r in league.iterrows():
    print(fmt_id(r) + "".join(f"{r[c]:>10.3f}" for c in SHOW))

# --- Selection + R2 agreement check ---
models_df = sweep_df[sweep_df["kind"] == "model"]
best_row = models_df.sort_values(SELECT_BY, ascending=ASCENDING[SELECT_BY]).iloc[0]
best_r2_row = models_df.sort_values("R2", ascending=False).iloc[0]

def cfg_str(row):
    # pandas turns None into NaN inside a DataFrame: normalize for printing
    p = "none" if pd.isna(row["exo_p"]) else row["exo_p"]
    t = "none" if pd.isna(row["tgt"]) else row["tgt"]
    a = "-" if pd.isna(row["alpha"]) else row["alpha"]
    return f"exo_p={p}, tgt={t}, {row['model']} (alpha={a})"

def nan_to_none(v):
    """None -> NaN inside sweep_df: turn it back into a real None so the
    `is not None` checks in Cells 10 and 12 behave correctly."""
    return None if pd.isna(v) else v

print(f"\nSELECTED by {SELECT_BY}: {cfg_str(best_row)}")
print(f"winner by R2   : {cfg_str(best_r2_row)}   "
      f"(the assignment grades primarily on R2)")
same = (best_row["exo_p"], best_row["tgt"], best_row["model"], best_row["alpha"]) == \
       (best_r2_row["exo_p"], best_r2_row["tgt"], best_r2_row["model"], best_r2_row["alpha"])
if same:
    print("Agreement check: the two rankings pick the SAME configuration. OK.")
else:
    print("WARNING: the two rankings DISAGREE -> look at both rows in the table")
    print("above before trusting the selection (or set SELECT_BY = \"R2\").")

best_base = min(BASELINES, key=lambda k: baseline_scores[k][SELECT_BY])
print(f"Best baseline: {best_base} "
      f"({SELECT_BY} = {baseline_scores[best_base][SELECT_BY]:.2f}) - "
      f"the model must beat this.")

"""## Cell 10 - Final report of the selected configuration: raw vs capped truth

Two SEPARATE readings of the SAME pooled walk-forward predictions:
- RAW truth (official): all 730 validation days as they are, outliers
  included - the honest preview of what the hidden 2024 test will look
  like;
- CAPPED truth (sensitivity): the same predictions scored against the
  validation truth winsorized at the cap used in training (or, when the
  winner uses no cap, at the 99.5th percentile of the 2015-2021 train).
  It shows how much of the error concentrates on the extreme bloom days.
The predictions are NEVER modified here - only the yardstick changes.
"""

BEST = {"exo_p": nan_to_none(best_row["exo_p"]),
        "tgt": nan_to_none(best_row["tgt"]),
        "model": best_row["model"],
        "alpha": nan_to_none(best_row["alpha"])}

d_win, yt_win, yp_win, yr_win = wf_run(BEST["exo_p"], BEST["tgt"],
                                       BEST["model"], BEST["alpha"])

y_seed = lagged.loc[lagged["date"] < "2022-01-01", "chlorophyll_a_mg_m3"].to_numpy()
CAP_DUAL = BEST["tgt"] if BEST["tgt"] is not None else float(np.percentile(y_seed, 99.5))

print(f"Selected configuration: {cfg_str(best_row)}")
print(f"Sensitivity cap on the truth: {CAP_DUAL:.2f} mg/m3 "
      f"({'selected tgt' if BEST['tgt'] is not None else '99.5th pct of the 2015-2021 train'})")

def _s(v):
    return "none" if v is None else v

win_label = (f"winner {BEST['model']} a={_s(BEST['alpha'])} "
             f"exo_p={_s(BEST['exo_p'])} tgt={_s(BEST['tgt'])}")
preds = {win_label: (yt_win, yp_win, yr_win)}
for kind in BASELINES:
    _, yt_b, yp_b, yr_b = BASE_RUNS[kind]
    preds[kind] = (yt_b, yp_b, yr_b)

MET = ["R2", "RMSE", "MAE", "sMAPE"]
hdr = f"{'variant':<44}" + "".join(f"{m:>10}" for m in MET)

print("\nPOOLED 2022+2023 vs RAW truth (OFFICIAL reading)")
print(hdr)
print("-" * len(hdr))
for label, (yt, yp, yr) in preds.items():
    m = metrics(yt, yp)
    print(f"{label:<44}" + "".join(f"{m[k]:>10.3f}" for k in MET))

print(f"\nPOOLED 2022+2023 vs CAPPED truth @ {CAP_DUAL:.2f} mg/m3 (SENSITIVITY reading)")
print(hdr)
print("-" * len(hdr))
for label, (yt, yp, yr) in preds.items():
    m = metrics(np.clip(yt, None, CAP_DUAL), yp)
    print(f"{label:<44}" + "".join(f"{m[k]:>10.3f}" for k in MET))

print("\nPer-year detail (raw truth): the normal year vs the hard year")
hdr2 = (f"{'variant':<44}{'sMAPE 22':>10}{'sMAPE 23':>10}"
        f"{'R2 22':>10}{'R2 23':>10}")
print(hdr2)
print("-" * len(hdr2))
for label, (yt, yp, yr) in preds.items():
    m22 = metrics(yt[yr == 2022], yp[yr == 2022])
    m23 = metrics(yt[yr == 2023], yp[yr == 2023])
    print(f"{label:<44}{m22['sMAPE']:>10.2f}{m23['sMAPE']:>10.2f}"
          f"{m22['R2']:>10.3f}{m23['R2']:>10.3f}")

"""## Cell 11 - The selected configuration on the pooled validation (plots)"""

best_base = min(BASELINES, key=lambda k: baseline_scores[k]["sMAPE"])
d_b, _, yp_b, _ = BASE_RUNS[best_base]

plt.figure(figsize=(14, 4))
plt.plot(d_win, yt_win, lw=0.6, color="black", label="Observed")
plt.plot(d_win, yp_win, lw=0.9, color="tab:blue",
         label=f"winner ({BEST['model']})")
plt.plot(d_b, yp_b, lw=0.7, color="tab:orange", ls="--",
         label=f"{best_base} baseline")
plt.legend()
plt.title("Pooled walk-forward validation 2022+2023: winner vs observed vs baseline")
plt.ylabel("mg/m3")
save_fig("cell11_final_predictions_vs_observed")

# Zoom on the last 90 days (Nov-Dec 2023: the extreme bloom)
plt.figure(figsize=(14, 4))
plt.plot(d_win[-90:], yt_win[-90:], lw=1, color="black", label="Observed")
plt.plot(d_win[-90:], yp_win[-90:], lw=1, color="tab:blue",
         label=f"winner ({BEST['model']})")
plt.plot(d_b[-90:], yp_b[-90:], lw=1, color="tab:orange", ls="--",
         label=f"{best_base} baseline")
plt.legend()
plt.title("Zoom: last 90 days of validation (Nov-Dec 2023 bloom)")
plt.ylabel("mg/m3")
save_fig("cell11_final_zoom_last_90_days")

residuals = yt_win - yp_win
plt.figure(figsize=(14, 3))
plt.plot(d_win, residuals, lw=0.5)
plt.axhline(3 * residuals.std(), color="red", ls="--", lw=0.8)
plt.axhline(-3 * residuals.std(), color="red", ls="--", lw=0.8)
plt.title(f"Residuals, winner ({BEST['model']}), bands = +/- 3 sigma")
plt.ylabel("mg/m3")
save_fig("cell11_final_residuals")

"""## Cell 12 - Final retrain on ALL released data (2015-2023) + coefficients

The winning recipe is retrained from scratch on everything we have
(2015-2023, validation days included): the validation has done its job
(selection) and its data are legitimate training material now. NO score
is computed here: the only data never used for anything is the hidden
2024 test, and it stays with the professors. This block is also the
skeleton of training_code/train.py: the winsorizing limits, mu/sigma and
the target cap below are the plain numbers that must be saved together
with the fitted model into best_model/model.pkl (assignment Section 7).
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