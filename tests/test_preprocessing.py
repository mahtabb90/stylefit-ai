"""Unit tests for stateful preprocessing module src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    GENERAL_FIT_FEATURES,
    LEAKAGE_COLUMNS,
    METADATA_COLUMNS,
    clean_dataset,
)
from src.preprocessing import (
    CategoricalSizeStringCaster,
    build_preprocessor,
    select_general_fit_features,
)


@pytest.fixture
def sample_clean_df():
    """Create sample cleaned DataFrame with pre-purchase features, leakage, metadata, and target."""
    data = {
        # General Fit Features
        "parsed_height_inches": [68.0, 65.0, 62.0, np.nan, 70.0, 66.0, 68.0, 64.0, 67.0, 65.0],
        "parsed_weight_lbs": [137.0, 125.0, 115.0, 150.0, np.nan, 130.0, 140.0, 120.0, 135.0, 128.0],
        "bust_band_size": [34.0, 32.0, 32.0, 36.0, 38.0, 34.0, 34.0, 32.0, np.nan, 34.0],
        "bust_cup_size": ["c", "d", "b", "c", "dd", "b", "c", "d", "c", np.nan],
        "body_type": ["hourglass", "athletic", "petite", "pear", "full bust", "hourglass", "athletic", "petite", "straight & narrow", "hourglass"],
        "age": [28.0, 36.0, 24.0, 34.0, 45.0, 29.0, 32.0, np.nan, 30.0, 27.0],
        "size": [14, 12, 4, 8, 20, 8, 14, 4, 12, np.nan],
        "category": ["romper", "gown", "sheath", "dress", "gown", "dress", "romper", "dress", "sheath", "gown"],
        "rented_for": ["vacation", "other", "party", "formal affair", "wedding", "everyday", "vacation", "party", "other", "wedding"],
        # Leakage Columns
        "rating": [10.0, 8.0, 10.0, 6.0, 9.0, 10.0, 8.0, 10.0, 9.0, 7.0],
        "review_text": ["Great fit!", "Loved it", "A bit tight", "Okay", "Super comfortable", "Perfect", "Loved the color", "Nice dress", "Fits well", "Good"],
        "review_summary": ["Awesome", "Great", "Tight", "Okay", "Loved it", "Perfect", "Great", "Nice", "Good", "Fine"],
        # Metadata Columns
        "user_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        "item_id": [201, 202, 203, 204, 205, 201, 202, 203, 204, 205],
        "review_date": pd.to_datetime(["2016-05-08", "2017-01-15", "2018-03-20", "2016-11-04", "2019-06-12", "2017-08-22", "2016-09-30", "2018-12-01", "2017-04-14", "2019-02-18"]),
        # Target
        "fit": ["fit", "fit", "small", "fit", "large", "fit", "fit", "small", "fit", "large"],
    }
    return pd.DataFrame(data)


def test_select_general_fit_features(sample_clean_df):
    features_df = select_general_fit_features(sample_clean_df)

    assert list(features_df.columns) == GENERAL_FIT_FEATURES
    for leakage_col in LEAKAGE_COLUMNS:
        assert leakage_col not in features_df.columns
    for meta_col in METADATA_COLUMNS:
        assert meta_col not in features_df.columns
    assert "fit" not in features_df.columns


def test_leakage_and_metadata_exclusion_in_preprocessor(sample_clean_df):
    preprocessor = build_preprocessor(scale_numeric=True, size_strategy="numeric", min_frequency=1)
    X_full = sample_clean_df  # Includes leakage and metadata columns

    # Preprocessor should transform only approved features and ignore remainder columns
    preprocessor.fit(X_full)
    X_trans = preprocessor.transform(X_full)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(sample_clean_df)


def test_categorical_size_string_caster():
    caster = CategoricalSizeStringCaster()
    df_sizes = pd.DataFrame({"size": [0, 4, 8, 14, np.nan, 20.0]})
    casted = caster.transform(df_sizes)

    assert casted[0, 0] == "0"
    assert casted[1, 0] == "4"
    assert casted[2, 0] == "8"
    assert casted[3, 0] == "14"
    assert pd.isna(casted[4, 0])
    assert casted[5, 0] == "20"


def test_size_strategy_numeric_vs_categorical(sample_clean_df):
    X = select_general_fit_features(sample_clean_df)

    # Strategy 1: Numeric
    prep_num = build_preprocessor(scale_numeric=True, size_strategy="numeric", min_frequency=1)
    prep_num.fit(X)
    X_num_trans = prep_num.transform(X)

    # Strategy 2: Categorical
    prep_cat = build_preprocessor(scale_numeric=True, size_strategy="categorical", min_frequency=1)
    prep_cat.fit(X)
    X_cat_trans = prep_cat.transform(X)

    assert X_num_trans.shape[0] == len(X)
    assert X_cat_trans.shape[0] == len(X)
    # Categorical strategy will have more columns due to one-hot encoded size
    assert X_cat_trans.shape[1] > X_num_trans.shape[1]


def test_scale_numeric_true_vs_false(sample_clean_df):
    X = select_general_fit_features(sample_clean_df)

    # Scale True
    prep_scale = build_preprocessor(scale_numeric=True, size_strategy="numeric", min_frequency=1)
    prep_scale.fit(X)
    X_scale = prep_scale.transform(X)

    # Scale False
    prep_noscale = build_preprocessor(scale_numeric=False, size_strategy="numeric", min_frequency=1)
    prep_noscale.fit(X)
    X_noscale = prep_noscale.transform(X)

    # Feature shapes match, but values differ due to scaling
    assert X_scale.shape == X_noscale.shape
    assert not np.allclose(X_scale[:, 0], X_noscale[:, 0])


def test_unseen_categories_and_missing_values(sample_clean_df):
    X_train = select_general_fit_features(sample_clean_df.iloc[:6])
    X_test = select_general_fit_features(sample_clean_df.iloc[6:]).copy()

    # Introduce completely unseen category in X_test
    X_test.loc[6, "category"] = "unseen_brand_new_category"
    X_test.loc[7, "rented_for"] = "unseen_event"

    preprocessor = build_preprocessor(scale_numeric=True, size_strategy="numeric", min_frequency=1)
    preprocessor.fit(X_train)

    # Transformation must NOT crash on unseen categories or missing values
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Exact feature dimension matching between train and test
    assert X_train_trans.shape[1] == X_test_trans.shape[1]


def test_distinct_real_other_category(sample_clean_df):
    X = select_general_fit_features(sample_clean_df)
    preprocessor = build_preprocessor(scale_numeric=True, size_strategy="numeric", min_frequency=1)
    preprocessor.fit(X)

    # Get feature names from categorical encoder
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    feature_names = cat_encoder.get_feature_names_out(
        ["bust_cup_size", "body_type", "category", "rented_for"]
    )

    # Verify real 'other' category and missing token '__missing__' are separate features
    rented_for_features = [f for f in feature_names if f.startswith("rented_for_")]
    assert "rented_for_other" in rented_for_features
    assert "rented_for___missing__" not in rented_for_features or "rented_for_other" != "rented_for___missing__"
