# Training code - One-day-ahead chlorophyll-a forecasting (Group 117)

## Overview

The model is an ARX linear regressor: chlorophyll-a is predicted from its own
past values and from current/past exogenous variables, plus two periodic
seasonal features (sin/cos of the day of year).

Recipe (selected by exploratory analysis, see `exploratory_analysis.ipynb`):

- **Lags**: AR lags 1-5 for chlorophyll, lags 0-4 for every exogenous variable
  (the PACF showed only lag 1 is significant for the target; cross-correlation
  analysis showed all exogenous response peaks within lags 0-4)
- **Outlier handling**: exogenous features clipped at the training 1st/99th
  percentiles (the data contains retrieval errors much larger than ordinary
  noise, mainly in the exogenous variables)
- **Standardization**: explicit mean/std computed on the (clipped) training
  data only, reused unchanged at prediction time
- **Estimator**: Ridge regression, alpha = 0.1

## Usage

Validation run (train <= 2022, validation on 2023, saves the validated model):

    python train.py --data <path to xlsx/csv> --output <path to model.pkl>

Final run (reports the same validation metrics, then refits on the full
2015-2023 dataset and saves the final model - this is the submitted artifact):

    python train.py --data <path to xlsx/csv> --output <path to model.pkl> --final

Example:

    python train.py \
        --data ../data/chlorophyll_student_2015_2023.xlsx \
        --output ../best_model/model.pkl \
        --final

## Files

- `train.py` - complete training workflow (loading, sorting, lag construction,
  chronological split, preprocessing, validation report, final fit, saving)
- `exploratory_analysis.ipynb` - supplementary notebook documenting the EDA:
  correlation matrix, cross-correlation per lag, ACF/PACF, outlier analysis,
  model comparison (OLS/Ridge/Lasso), alpha tuning, lag and clipping sweeps

## Validation results (train <= 2022, validation 2023)

R2 = 0.5670, RMSE = 0.5109 mg/m3, MAE = 0.2077 mg/m3
