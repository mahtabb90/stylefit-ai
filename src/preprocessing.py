"""Stateful preprocessing module for StyleFit AI dataset.

Provides train-fitted Scikit-Learn ColumnTransformer builders supporting:
  - Explicit allowlist feature selection (select_general_fit_features)
  - Leakage column and metadata exclusion
  - Collision-safe missing token ('__missing__')
  - Train-fitted infrequent/rare category handling ('handle_unknown="infrequent_if_exist"')
  - Configurable size strategies ('size_strategy="numeric"' vs 'size_strategy="categorical"')
  - Type-safe categorical size string casting
  - Configurable numeric feature scaling ('scale_numeric=True' vs 'scale_numeric=False')
"""

from typing import Any, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import (
    GENERAL_FIT_FEATURES,
    LEAKAGE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)


def select_general_fit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the approved pre-purchase features using an explicit allowlist.

    Positive selection based on GENERAL_FIT_FEATURES. Does NOT rely on negative
    column dropping. Guarantees leakage columns and metadata identifiers are excluded.
    """
    missing_cols = [col for col in GENERAL_FIT_FEATURES if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required general fit feature columns: {missing_cols}. "
            "Ensure clean_dataset(df) has been run prior to feature selection."
        )

    return df[GENERAL_FIT_FEATURES].copy()


class CategoricalSizeStringCaster(BaseEstimator, TransformerMixin):
    """Type-safe transformer that converts non-missing size values to strings.

    Preserves NaNs so SimpleImputer can handle them cleanly with fill_value='__missing__'.
    Ensures OneHotEncoder never receives mixed int/float and str types.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            # Convert non-null to str, preserving nulls
            s = X_df[col]
            X_df[col] = s.apply(
                lambda v: str(int(v)) if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) and float(v).is_integer()
                else (str(v) if not pd.isna(v) else np.nan)
            )
        return X_df.values


def build_preprocessor(
    scale_numeric: bool = True,
    size_strategy: Literal["numeric", "categorical"] = "numeric",
    min_frequency: Union[int, float] = 100,
    sparse_output: bool = False,
) -> ColumnTransformer:
    """Build stateful ColumnTransformer for General Fit Model features.

    Args:
        scale_numeric (bool): If True, applies StandardScaler to numeric features (for LR, KNN, MLP).
                              If False, leaves numeric features unscaled (for tree models RF, XGBoost).
        size_strategy (str): 'numeric' treats size as ordinal/numerical.
                             'categorical' treats size as categorical string feature.
        min_frequency (int | float): Frequency threshold for rare category grouping in OneHotEncoder,
                                     learned strictly from X_train.
        sparse_output (bool): Return sparse one-hot features when possible. The
                              default remains dense for backward compatibility.

    Returns:
        ColumnTransformer: Configured Scikit-Learn transformer ready to be fit on X_train.
    """
    if size_strategy not in ("numeric", "categorical"):
        raise ValueError(
            f"Invalid size_strategy '{size_strategy}'. Must be 'numeric' or 'categorical'."
        )

    # 1. Define Feature Lists based on size_strategy
    if size_strategy == "numeric":
        numeric_features = [
            "parsed_height_inches",
            "parsed_weight_lbs",
            "bust_band_size",
            "age",
            "size",
        ]
        categorical_features = [
            "bust_cup_size",
            "body_type",
            "category",
            "rented_for",
        ]
    else:  # categorical
        numeric_features = [
            "parsed_height_inches",
            "parsed_weight_lbs",
            "bust_band_size",
            "age",
        ]
        categorical_features = [
            "bust_cup_size",
            "body_type",
            "category",
            "rented_for",
            "size",
        ]

    # 2. Build Numeric Pipeline
    num_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=num_steps)

    # 3. Build Categorical Pipeline
    cat_steps: List[Tuple[str, Any]] = []
    if size_strategy == "categorical":
        cat_steps.append(("string_caster", CategoricalSizeStringCaster()))

    cat_steps.extend(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__missing__"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    min_frequency=min_frequency,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=sparse_output,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(steps=cat_steps)

    # 4. Assemble ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",  # Drops any extra columns not in approved list
    )

    return preprocessor
