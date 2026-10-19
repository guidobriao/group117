# -*- coding: utf-8 -*-
"""1_exploratory_analysis.py

Exploratory analysis of the daily chlorophyll-a series and its exogenous
drivers (2015-2023): evolution and seasonality of the target, exogenous
time series, correlation structure (target correlations and
multicollinearity among the drivers), lagged cross-correlations (the
response delays that guide the lag specification of the models), the
autocorrelation / partial autocorrelation of the target (the AR lags), and
the outlier / retrieval-error diagnostics that motivate the target- and
feature-winsorizing options evaluated by 2_model_selection.py under its
validation protocol. Figures are saved to ../results via save_fig().
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

from build_lagged_features import read_table

# ============================================================================
# Setup: figure saving, data loading, feature engineering
# ============================================================================

# --- Figure saving setup (replaces every plt.show()) ---
RESULTS_DIR = "../../results"
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
# annual_sin / annual_cos pair to every frame it builds; the seasonal
# structure is visible in the monthly-mean bars below.
# 7-day rolling means: ocean memory beyond single-day lags
# (min_periods=1 -> no NaNs, partial means on the first 6 days of 2015)
ROLL_COLS = ["upwelling_index", "nitrate_umol_l", "wind_speed_m_s", "sst_c"]
for col in ROLL_COLS:
    df[f"{col}_ma7"] = df[col].rolling(7, min_periods=1).mean()

# Physically motivated interactions
df["upw_x_wind"] = df["upwelling_index"] * df["wind_speed_m_s"]
df["nitrate_x_par"] = df["nitrate_umol_l"] * df["par_umol_m2_s"]

exo_cols = [c for c in df.columns if c not in ("date", "chlorophyll_a_mg_m3")]
print(df.shape, "|", df["date"].min().date(), "->", df["date"].max().date())

# ============================================================================
# Target evolution and seasonality
# ============================================================================

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

# ============================================================================
# Exogenous time series
# ============================================================================

nrows = (len(exo_cols) + 1) // 2
fig, axes = plt.subplots(nrows, 2, figsize=(14, 2.2 * nrows), sharex=True)
for ax, col in zip(axes.flat, exo_cols):
    ax.plot(df["date"], df[col], lw=0.4)
    ax.set_title(col, fontsize=9)
plt.tight_layout()
save_fig("cell01_exogenous_time_series")

# ============================================================================
# Correlation structure
# ============================================================================

# Correlation of every variable with the target
corr = df.drop(columns=["date"]).corr()["chlorophyll_a_mg_m3"].sort_values()
print(corr)

# Correlation matrix (heatmap)
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

# ============================================================================
# Cross-correlation target vs exogenous per lag
# ============================================================================
# Each panel shows corr(y(t), x(t-j)) for j = 0..21. The lag with the peak
# correlation indicates the response delay of that variable: this is the
# guide for choosing the lags q_k of each exogenous variable.

MAX_LAG = 21

nrows = (len(exo_cols) + 1) // 2
fig, axes = plt.subplots(nrows, 2, figsize=(14, 2.2 * nrows))
for ax, col in zip(axes.flat, exo_cols):
    ccf = [df["chlorophyll_a_mg_m3"].corr(df[col].shift(j)) for j in range(MAX_LAG + 1)]
    ax.bar(range(MAX_LAG + 1), ccf)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("lag (days)")
plt.tight_layout()
save_fig("cell03_cross_correlation_per_lag")

# ============================================================================
# Autocorrelation (ACF) and partial autocorrelation (PACF) of the target
# ============================================================================
# AR lags outside the red significance band are the ones to keep for p.

y_series = df["chlorophyll_a_mg_m3"]

# Simple autocorrelation
acf = [y_series.autocorr(lag=j) for j in range(31)]
plt.figure(figsize=(8, 3))
plt.bar(range(31), acf)
plt.title("Chlorophyll ACF")
plt.xlabel("lag")
save_fig("cell04_acf")

# PACF: Yule-Walker solution with numpy (no external libraries)
r = np.array([y_series.autocorr(lag=j) for j in range(1, 31)])
pacf = []
for k in range(1, 31):
    R = np.array([[r[abs(i - j)] for j in range(k)] for i in range(k)])
    pacf.append(solve(R, r[:k])[-1])
plt.figure(figsize=(8, 3))
plt.bar(range(1, 31), pacf)
plt.axhline(1.96 / np.sqrt(len(y_series)), color="red", ls="--", lw=0.7)
plt.axhline(-1.96 / np.sqrt(len(y_series)), color="red", ls="--", lw=0.7)
plt.title("Chlorophyll PACF (band = significance)")
plt.xlabel("lag")
save_fig("cell04_pacf")

# ============================================================================
# Outlier analysis (retrieval errors)
# ============================================================================
# The manual warns that occasional retrieval errors can be much larger than
# ordinary noise. Look for extreme spikes in the target and in the exogenous
# variables. Nothing is removed or capped here: this section only motivates
# the target- and feature-winsorizing options that 2_model_selection.py
# evaluates under its validation protocol.

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
nrows = (len(exo_cols) + 1) // 2
fig, axes = plt.subplots(nrows, 2, figsize=(14, 2.0 * nrows))
for ax, col in zip(axes.flat, exo_cols):
    ax.boxplot(df[col], orientation="horizontal")   # vert=False deprecated in mpl 3.11
    ax.set_title(col, fontsize=9)
plt.tight_layout()
save_fig("cell05_exogenous_boxplots")
