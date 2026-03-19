from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for House Prices dataset.
    
    Applies:
        - Type conversions
        - Domain-based missing values
        - Feature creation
        - Binary encoding
        - Column dropping (based on EDA)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # --- Feature type conversions ---

        X["MoSold"] = X["MoSold"].astype("str")
        X["MSSubClass"] = X["MSSubClass"].astype("str")

        # --- Domain-based missing value fill --- 

        cols_to_fill_with_NA = ["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu", "GarageType", "GarageQual",
                                "GarageCond", "BsmtExposure", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtFinType1", "GarageFinish"]
        existing_cols = [col for col in cols_to_fill_with_NA if col in X.columns]
        X[existing_cols] = X[existing_cols].fillna("None")

        # --- Feature creation ---

        X["has_garage"] = X["GarageYrBlt"].notnull().astype(int)
        X["garage_age"] = (X["YrSold"] - X["GarageYrBlt"]).clip(lower=0).fillna(0)
        
        X["house_age"] = X["YrSold"] - X["YearBuilt"]

        # --- Binary features ---

        X["has_second_floor"] = (X["2ndFlrSF"] > 0).astype(int)
        X["has_3SsnPorch"] = (X["3SsnPorch"] > 0).astype(int)
        X["has_ScreenPorch"] = (X["ScreenPorch"] > 0).astype(int)
        X["has_pool"] = (X["PoolArea"] > 0).astype(int)
        X["has_EnclosedPorch"] = (X["EnclosedPorch"] > 0).astype(int)

        # --- Column dropping ---

        cols_to_drop = ["GarageYrBlt", "YrSold", "YearBuilt", "3SsnPorch", "ScreenPorch", "PoolArea", "EnclosedPorch",
                        "BedroomAbvGr", "BsmtFinSF2", "BsmtHalfBath", "LowQualFinSF", "MiscVal", "KitchenAbvGr", "BsmtFullBath", "HalfBath", "Utilities"]
        X = X.drop(columns=cols_to_drop, errors="ignore")

        return X