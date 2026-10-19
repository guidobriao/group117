# Training code - One-day-ahead chlorophyll-a forecasting (Group 117)

## Overview

The model is an ARX linear regressor: chlorophyll-a is predicted one day
ahead from its own past values, from current/past exogenous variables and
from a small set of engineered features - 7-day rolling means of upwelling
index, nitrate, wind speed and sea-surface temperature, plus the two
physically motivated interactions upwelling x wind and nitrate x PAR. The
annual sin/cos pair is appended by the lagged-features module itself (no
manual seasonal encoding; the EDA reads seasonality through the monthly
means).

Recipe (lags and winsorizing options are motivated by the exploratory
analysis in `1_exploratory_analysis.py`; every choice is made by the
walk-forward validation in `2_model_selection.py`):

- **Lags**: AR lags 1, 7 and 8 for chlorophyll (dominant AR(1) plus weekly
  harmonics; the other AR lags showed negligible coefficients); lags 0-4
  for the 12 original exogenous variables (their cross-correlations with
  the target peak within that window, mostly around lags 2-3); lags 0-1
  for the rolling means and lags 0-2 for the interactions - 79 features in
  total
- **Outlier handling**: exogenous features winsorized at the training
  0.5th/99.5th percentiles (the data contains retrieval errors much larger
  than ordinary noise, mainly in the exogenous variables); training target
  capped at 5.0 mg/m3 (a thin tail of extreme bloom days would otherwise
  dominate the squared-error loss)
- **Standardization**: explicit mean/std computed on the (winsorized)
  training data only, reused unchanged at prediction time; predictions
  clipped at zero (concentrations are non-negative)
- **Estimator**: Ridge regression, alpha = 1 (OLS, Ridge and Lasso are
  compared in the sweep; shrinkage handles the strong collinearity between
  the exogenous features, with pairs up to |r| ~ 0.99)

## Validation protocol

A single walk-forward validation with annual folds (rolling origin):

- fold 2021: train 2015-2020 -> predict 2021
- fold 2022: train 2015-2021 -> predict 2022
- fold 2023: train 2015-2022 -> predict 2023

Inside every fold the winsorizing limits and the standardization mean/std
are recomputed from that fold's training data only: no day is ever
predicted by a model that has seen it - not for the fit, not for mu/sigma,
not for the clip limits. Annual folds mirror the deployment scenario (a
model frozen at the end of its training period predicting a whole year
ahead); monthly windows would re-fit mid-year and give a slightly
optimistic picture.

Every selection decision (model family, alpha, exogenous winsorizing
percentile, target cap) is taken exclusively on folds 2021+2022, ranked by
mean per-fold R2 on the raw truth, with ties at the printed third decimal
broken by sMAPE. R2 is the primary metric - the hidden 2024 test is scored
primarily on it - and the sMAPE-only winner is printed as a secondary
check, with a warning when the two rankings disagree. 2023 is deliberately
kept out of the selection: it is the anomalous year (an exceptional
November bloom; R2 there is typically ~0.57 against ~0.84 on normal
years), and tuning the recipe on it would mean optimizing for that one
specific peak.

Fold 2023 is used exactly once, as an a posteriori one-shot validation of
the whole grid, after the selection is sealed. The selected model lands
within sampling noise of the best 2023 configuration (ΔR2 = 0.004,
ΔsMAPE = 0.07 percentage points; by capped-truth R2 it ranks first), so
no change is made: the winner is retrained from scratch on the full
2015-2023 dataset, same recipe and all transformations recomputed, and no
score is computed after that. The 2024 data is never touched and stays as
the hidden test period.

## Metrics

R2 (computed against the mean of the evaluated days), RMSE and MAE in
mg/m3, sMAPE in % - one metrics implementation and one table format for
models, preprocessing variants and baselines. MAPE is not used: it
explodes when the observed value is close to zero (target minimum ~0.28
mg/m3). The one-shot table reads every configuration twice:

- **RAW** (official): metrics against the truth as it is, outliers
  included - the honest preview of the test reading
- **CAPPED** (sensitivity): the same predictions scored against one common
  clipped truth (the selected training cap, or the 99.5th percentile of
  the one-shot train when the winner uses none), showing how much of the
  error concentrates on the extreme bloom days - predictions are never
  modified, only the yardstick changes

Rows are sorted by raw R2, with ties at the third decimal broken by raw
sMAPE; the capped columns never drive any ordering.

## Baselines

Same folds, same metrics, same truth as the models:

- **Persistence** (tomorrow = today, the previous day's observed value -
  exactly what is available one day ahead): the bar that any ARX model
  must clear
- **Seasonal naive** (same calendar day one year earlier): the
  "last year replays itself" reference
- **Train mean** (constant at the mean of the fold's training years): R2
  is built against the mean of the evaluated days, so this constant
  hovers around R2 = 0 - the honest floor that any model worth its
  features must beat

## Usage

Exploratory analysis and model selection (run from this folder; figures
and sweep.csv are written to ../../results):

    python 1_exploratory_analysis.py
    python 2_model_selection.py

Validation run of the training script (train 2015-2022, one-shot
validation on 2023, saves the validated model):

    python train.py --data <path to xlsx/csv> --output <path to model.pkl>

Final run (reports the same validation metrics, then refits on the full
2015-2023 dataset and saves the final model - this is the submitted
artifact):

    python train.py --data <path to xlsx/csv> --output <path to model.pkl> --final

Example:

    python train.py \
        --data ../../data/chlorophyll_student_2015_2023.xlsx \
        --output ../best_model/model.pkl \
        --final

## Files

- `1_exploratory_analysis.py` - exploratory analysis: target evolution and
  seasonality (monthly means), exogenous time series, correlations with
  the target and multicollinearity among the drivers, lagged
  cross-correlations (the response delays that guide the lag
  specification), ACF/PACF of the target (the AR lags), outlier and
  retrieval-error diagnostics (the motivation for the winsorizing options)
- `2_model_selection.py` - model-selection workflow: walk-forward engine,
  baselines, training-target strategies (raw / clip / drop / log, with
  sigma-rule thresholds), 80-configuration sweep (model family x alpha x
  exogenous winsorizing x target cap), selection on 2021+2022 with
  general-picture averages across the grid, one-shot 2023 table (every
  configuration and baseline, raw + capped), winner diagnostics, final
  retrain on 2015-2023 with coefficient report
- `train.py` - complete training workflow (loading, sorting, lag
  construction, chronological split, preprocessing, validation report,
  final fit, saving)
- `build_lagged_features.py` - lagged-frame construction shared by all
  scripts (appends the annual sin/cos pair to every frame)

## Validation results

Selection folds 2021+2022 (mean per-fold metrics, raw truth):

- selected configuration (Ridge, alpha = 1, exogenous winsorizing
  0.5/99.5 percentiles, target cap 5.0 mg/m3): R2 = 0.846, sMAPE = 9.79%
- persistence: R2 = +0.704
- train mean: R2 = -0.003
- seasonal naive: R2 = -0.602

One-shot 2023 validation (train 2015-2022 -> predict 2023): R2 = 0.564
for the selected configuration, against R2 = 0.568 for the best 2023
configuration in the grid - within sampling noise, as detailed in the
validation protocol above. The full tables (R2, RMSE, MAE, sMAPE, raw and
capped, for every configuration and baseline) are printed by
`2_model_selection.py` and saved to `../../results/sweep.csv`.