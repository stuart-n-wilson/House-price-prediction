from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.features import FeatureEngineer
from src.preprocessing import build_preprocessor


def get_model(model_type="random_forest", random_state=42):
    """
    Returns a model based on the specified type. Default is RF.
    """

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),

        "xgboost": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        ),

        "lightgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state
        )
    }

    if model_type not in models:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return models[model_type]


def build_model(num_cols, cat_cols, model_type="random_forest", random_state=42):
    """
    Builds full pipeline: feature engineering, preprocessing, modelling
    """

    # Preprocessing
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Model
    model = get_model(model_type=model_type, random_state=random_state)

    # Full pipeline
    pipeline = Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    return pipeline