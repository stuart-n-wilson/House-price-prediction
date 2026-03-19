# 🏠 House Price Prediction

A machine learning pipeline for the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), achieving a best score of **0.126 RMSLE** (top ~23% on the leaderboard).

## Overview

This project builds an end-to-end ML pipeline to predict property sales prices. Using the Ames Housing Dataset, provided on Kaggle, this project covers exploratory data analysis, feature engineering, preprocessing, model training and tuning, and competition submission, with AI assistance where necessary.

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
- Identified highly imbalanced or near constant columns to drop (e.g. `Utilities`)
- Identified domain specific missingness (e.g. `GarageYrBlt` is NaN by when there is no garage, by design)
- Confirmed strong right-skew in target variable `SalePrice`, justifying a log transform as the target

### Feature Engineering
FeatureEngineer sklearn transformer (`src/features.py`) applied inside the pipeline:
- **Type conversions** — `MSSubClass`, `MoSold` converted to string as they are nominal, not ordinal
- **Domain-based missing value fills** — categorical NaNs filled with `"None"` where meaningful (e.g. `PoolQC` when there is no pool)
- **New features** —  created new features `house_age`, `garage_age`, `has_garage`
- **Binary conversion** — converted heavily imbalanced features to binary e.g. `has_second_floor`, `has_pool`, `has_ScreenPorch`
- **Column drops** — low importance and highly skewed features removed

### Preprocessing
Separate pipelines for numerical and categorical features with ColumnTransformer:
- **Numerical** — median imputation + standard scaling
- **Categorical** — mode imputation + one-hot encoding

### Models
Three pipelines were trained and evaluated:

| Model | Kaggle RMSE of  |
|---|---|
| Random Forest | 0.14520 |
| XGBoost (tuned) | 0.13430 |
| LightGBM (tuned) | **0.12651** |

Hyperparameter tuning using Optuna was performed on XGBoost and LightGBM, optimising 5-fold cross-validated scores.

## Results

Best Kaggle submission: **0.12651 RMSLE** using a tuned LightGBM pipelnie — placing in approximately the top 23% of the leaderboard.

**Most importantly, gained a lot of understanding of end-to-end ML pipelines.**

## Package requirements
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

## How to run

1. Download the data from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place `train.csv` and `test.csv` in the `data/` folder
2. Download the src folder to the same location as the data folder
3. Download the notebooks into the same location
4. Run `notebooks/01_eda.ipynb` to explore the data
5. Run `notebooks/02_modelling.ipynb` to train models and generate submissions
