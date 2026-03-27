# 🏠 House Price Prediction

![Python](https://img.shields.io/badge/Python-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-green)
![XGBoost](https://img.shields.io/badge/XGBoost-GBDT-red)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-purple)

End-to-end machine learning pipeline achieving a **top 23% Kaggle ranking (RMSE: 0.1265)** using gradient boosting models.

---

## 🚀 Overview

This project builds a full ML pipeline to predict house prices using the Ames Housing dataset, covering:

* exploratory data analysis (EDA)
* feature engineering
* preprocessing pipelines
* model training and hyperparameter tuning

---

## 📊 Key Results

* 🥇 Best model: **LightGBM (tuned)**
* 📉 Score: **0.12651 RMSE (log transformed target)**
* 📊 Ranking: **Top ~23% on Kaggle leaderboard**

| Model            | RMSE       |
| ---------------- | ---------- |
| Random Forest    | 0.1452     |
| XGBoost (tuned)  | 0.1343     |
| LightGBM (tuned) | **0.1265** |

---

## 🔍 Exploratory Data Analysis

* Identified skewed target (`SalePrice`) → applied log transformation
* Detected features with structural missingness (e.g. no garage/pool)
* Removed low-information and near-constant features
* Analysed feature distributions and correlations to guide modelling

---

## ⚙️ Feature Engineering

Custom `FeatureEngineer` transformer:

* Converted nominal variables to categorical (`MSSubClass`, `MoSold`)
* Domain-aware missing value handling (e.g. `"None"` for absent features)
* Created new binary features:
  * `has_garage`
  * `has_pool`
  * `has_second_floor`
* Derived new descriptive features:
  * `house_age`
  * `garage_age`
* Removed redundant or low importance variables

---

## 🧠 Modelling & Preprocessing

* Separate pipelines using `ColumnTransformer`:

  * Numerical → median imputation + scaling
  * Categorical → mode imputation + one-hot encoding
* Models:

  * Random Forest
  * XGBoost
  * LightGBM
* Hyperparameter tuning with **Optuna (5-fold CV)**

---

## 📂 Project Structure

```
house-price-prediction/
├── data/                    # Train/test datasets
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory analysis
│   └── 02_modelling.ipynb   # Model training & evaluation
├── src/
│   ├── features.py          # Feature engineering
│   ├── preprocessing.py     # Data preprocessing
│   └── model.py             # Model pipelines
├── predictions/             # Kaggle submissions
└── README.md
```

---

## 🧠 Key Insights

* Gradient boosting models outperform tree based models
* Informed feature engineering imporved model performance
* Target transformation proved critical for handling skewed target

---

## ▶️ How to Run

1. Download the data from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place `train.csv` and `test.csv` in the `data/` folder.
2. Download the src folder to the same location as the data folder.
3. Download the notebooks into the same location.
4. Run `notebooks/01_eda.ipynb` to explore the data.
5. Run `notebooks/02_modelling.ipynb` to train models and generate submissions.
