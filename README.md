# 🏠 House Price Prediction

A machine learning pipeline for the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), achieving a best score of **0.126 RMSLE** (top ~15% on the leaderboard).

## Overview

This project builds an end-to-end ML pipeline to predict residential property sale prices using the Ames Housing dataset. The workflow covers exploratory data analysis, feature engineering, preprocessing, model training, hyperparameter tuning, and submission generation.

## Project Structure
```
house-price-prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_modelling.ipynb    # Model training and evaluation
├── predictions/
│   ├── submission_rf.csv
│   ├── submission_xg_tuned.csv
│   └── submission_lgbm_tuned.csv
├── src/
│   ├── __init__.py
│   ├── features.py           # Feature engineering transformer
│   ├── model.py              # Model building and pipeline
│   └── preprocessing.py      # Numerical and categorical preprocessing
└── README.md
```

## Approach

### Exploratory Data Analysis
- Identified high-cardinality and near-constant columns to drop (e.g. `Utilities`, `BsmtHalfBath`)
- Flagged domain-meaningful missingness (e.g. no garage → `GarageYrBlt` is NaN by design, not error)
- Confirmed strong right-skew in `SalePrice`, justifying a log transform as the target

### Feature Engineering
Custom `FeatureEngineer` sklearn transformer (`src/features.py`) applied inside the pipeline:
- **Type conversions** — `MSSubClass`, `MoSold` cast to string (they are nominal, not ordinal)
- **Domain-based missing value fills** — categorical NaNs filled with `"None"` where absence is meaningful (e.g. `PoolQC`, `FireplaceQu`)
- **New features** — `house_age`, `garage_age`, `has_garage`
- **Binary flags** — `has_second_floor`, `has_pool`, `has_ScreenPorch`, etc.
- **Column drops** — low-signal and highly skewed features removed

### Preprocessing
Separate pipelines for numerical and categorical features via `ColumnTransformer`:
- **Numerical** — median imputation + standard scaling
- **Categorical** — mode imputation + one-hot encoding

### Models
Three sklearn-compatible pipelines were trained and evaluated:

| Model | CV RMSLE |
|---|---|
| Random Forest | — |
| XGBoost (tuned) | — |
| LightGBM (tuned) | **0.126** ✅ |

Hyperparameter tuning was performed using [Optuna](https://optuna.org/) with 50 trials per model, optimising 5-fold cross-validated RMSLE.

## Results

Best Kaggle submission: **0.126 RMSLE** using a tuned LightGBM pipeline — placing in approximately the top 15% of the leaderboard.

## Requirements
```
numpy
pandas
scikit-learn
xgboost
lightgbm
optuna
matplotlib
seaborn
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

## Usage

1. Download the data from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place `train.csv` and `test.csv` in the `data/` folder
2. Run `notebooks/01_eda.ipynb` to explore the data
3. Run `notebooks/02_modelling.ipynb` to train models and generate submissions
