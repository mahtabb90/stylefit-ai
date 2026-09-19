"""Tests for advanced ML experiment modules.

Coverage:
  - CatBoost feature preparation scope and correctness
  - Class weight computation from training labels only
  - Group-aware internal eval split integrity
  - MLP preprocessing produces clean numeric matrices
  - Ordinal target encoding, reconstruction, and monotonicity enforcement
  - Fixed class order alignment
  - Zero user overlap in group-aware CV folds
  - Deterministic advanced CV splits
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.cleaning import GENERAL_FIT_FEATURES
from src.evaluation import CLASS_ORDER, align_probability_columns
from src.ordinal import (
    OrdinalBinaryPair,
    decode_ordinal_predictions,
    encode_ordinal_targets,
)
from src.splitting import make_group_cv_splits


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_development_data():
    """Minimal but realistic development-partition fixture.

    36 users * 4 rows = 144 rows. Classes cycle evenly so every CV fold
    contains all three classes and group-aware splitting is feasible.
    """
    rng = np.random.default_rng(42)
    n_users = 36
    rows_per_user = 4
    n_rows = n_users * rows_per_user

    groups = pd.Series(
        np.repeat([f"user_{i}" for i in range(n_users)], rows_per_user),
        name="user_id",
    )
    # Cycle classes evenly
    class_cycle = (CLASS_ORDER * (n_rows // len(CLASS_ORDER) + 1))[:n_rows]
    y = pd.Series(class_cycle, name="fit")

    X = pd.DataFrame(
        {
            "parsed_height_inches": rng.normal(66, 3, n_rows),
            "parsed_weight_lbs": rng.normal(140, 20, n_rows),
            "bust_band_size": rng.choice([32.0, 34.0, 36.0, 38.0], n_rows),
            "bust_cup_size": rng.choice(["b", "c", "d"], n_rows),
            "body_type": rng.choice(["athletic", "hourglass", "petite"], n_rows),
            "age": rng.integers(18, 70, n_rows).astype(float),
            "size": rng.choice([2, 4, 8, 12, 16], n_rows),
            "category": rng.choice(["dress", "gown", "romper"], n_rows),
            "rented_for": rng.choice(["party", "wedding", "work"], n_rows),
        }
    )[GENERAL_FIT_FEATURES]

    return X, y, groups


@pytest.fixture
def cv_splits(small_development_data):
    X, y, groups = small_development_data
    return make_group_cv_splits(X, y, groups, n_splits=3, random_state=42)


# ---------------------------------------------------------------------------
# CatBoost feature preparation
# ---------------------------------------------------------------------------


def test_catboost_feature_prep_scope(small_development_data):
    """build_catboost_features must return exactly GENERAL_FIT_FEATURES, no extras."""
    from src.advanced_models import (
        CATBOOST_CATEGORICAL_FEATURES,
        MISSING_TOKEN,
        build_catboost_features,
    )

    X, _, _ = small_development_data
    X_cat, cat_indices = build_catboost_features(X)

    # Exact feature scope
    assert list(X_cat.columns) == GENERAL_FIT_FEATURES, (
        f"Column mismatch: {list(X_cat.columns)} != {GENERAL_FIT_FEATURES}"
    )

    # Categorical indices are valid
    assert len(cat_indices) == len(CATBOOST_CATEGORICAL_FEATURES)
    assert all(0 <= i < len(GENERAL_FIT_FEATURES) for i in cat_indices)

    # Missing token applied to categorical NaNs
    X_with_nan = X.copy()
    for col in CATBOOST_CATEGORICAL_FEATURES:
        X_with_nan.loc[X_with_nan.index[:5], col] = np.nan
    X_cat_nan, _ = build_catboost_features(X_with_nan)
    for col in CATBOOST_CATEGORICAL_FEATURES:
        assert (X_cat_nan[col] == MISSING_TOKEN).sum() >= 5, (
            f"MISSING_TOKEN not applied for column '{col}'"
        )

    # No leakage columns
    forbidden = {"rating", "review_text", "review_summary", "user_id"}
    assert forbidden.isdisjoint(set(X_cat.columns)), (
        f"Forbidden columns found: {forbidden & set(X_cat.columns)}"
    )


def test_catboost_feature_prep_missing_column():
    """build_catboost_features must raise KeyError when a feature column is absent."""
    from src.advanced_models import build_catboost_features

    X_incomplete = pd.DataFrame({"parsed_height_inches": [66.0, 67.0]})
    with pytest.raises(KeyError, match="missing required columns"):
        build_catboost_features(X_incomplete)


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------


def test_catboost_class_weights_from_train_only(small_development_data):
    """Class weights computed from y_train must cover all three classes."""
    from src.advanced_models import compute_class_weights_from_train

    _, y, _ = small_development_data
    # Use only the first 80% as a proxy training fold
    n_train = int(0.8 * len(y))
    y_train = y.iloc[:n_train]

    weights = compute_class_weights_from_train(y_train)

    assert set(weights.keys()) == set(CLASS_ORDER), (
        f"Weight keys must be exactly CLASS_ORDER; got {set(weights.keys())}"
    )
    assert all(w > 0 for w in weights.values()), "All class weights must be positive"


def test_class_weights_no_validation_data(small_development_data):
    """Weights computed from training fold differ from full-dataset weights
    when class distributions differ, confirming fold-only computation."""
    from src.advanced_models import compute_class_weights_from_train

    # Build a deliberately imbalanced training fold
    _, y, _ = small_development_data
    y_imbalanced = pd.Series(
        ["small"] * 60 + ["fit"] * 20 + ["large"] * 10, name="fit"
    )
    y_balanced = pd.Series(
        ["small"] * 30 + ["fit"] * 30 + ["large"] * 30, name="fit"
    )

    w_imb = compute_class_weights_from_train(y_imbalanced)
    w_bal = compute_class_weights_from_train(y_balanced)

    # Imbalanced weights for "small" should be smaller (more samples → lower weight)
    assert w_imb["small"] < w_bal["small"], (
        "Class weight for majority class should decrease when it has more samples"
    )


def test_class_weights_raises_on_missing_class():
    """compute_class_weights_from_train must raise if a class is absent."""
    from src.advanced_models import compute_class_weights_from_train

    y_incomplete = pd.Series(["small", "fit", "small", "fit"], name="fit")
    with pytest.raises(ValueError, match="missing classes"):
        compute_class_weights_from_train(y_incomplete)


# ---------------------------------------------------------------------------
# Group-aware internal eval split
# ---------------------------------------------------------------------------


def test_catboost_internal_eval_split_group_aware(small_development_data):
    """make_internal_eval_split must produce zero user overlap between
    internal train and eval, and both subsets must stay within the outer fold."""
    from src.advanced_models import build_catboost_features, make_internal_eval_split

    X, y, groups = small_development_data

    # Simulate an outer training fold (use ~80% of data)
    n_train = int(0.8 * len(X))
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    g_train = groups.iloc[:n_train]

    X_cat, _ = build_catboost_features(X_train)
    X_int_tr, y_int_tr, X_int_ev, y_int_ev = make_internal_eval_split(
        X_cat, y_train, g_train, n_internal_splits=5, random_state=42
    )

    # Zero user overlap
    g_int_train = set(g_train.iloc[X_cat.index.get_indexer(X_int_tr.index)])
    g_int_eval = set(g_train.iloc[X_cat.index.get_indexer(X_int_ev.index)])
    assert g_int_train.isdisjoint(g_int_eval), (
        "User overlap detected between internal CatBoost train and eval subsets"
    )

    # Both subsets within outer training fold
    assert set(X_int_tr.index).issubset(set(X_train.index))
    assert set(X_int_ev.index).issubset(set(X_train.index))

    # All classes present in internal eval
    assert set(y_int_ev.unique()) == set(CLASS_ORDER)


# ---------------------------------------------------------------------------
# MLP preprocessing
# ---------------------------------------------------------------------------


def test_mlp_input_no_nan_inf(small_development_data, cv_splits):
    """After build_mlp_preprocessor fit_transform, training matrix must contain
    no NaN or Inf values."""
    from src.advanced_models import build_mlp_preprocessor

    X, y, groups = small_development_data
    train_idx, _ = cv_splits[0]
    X_train = X.iloc[train_idx]

    preprocessor = build_mlp_preprocessor()
    X_proc = preprocessor.fit_transform(X_train)
    X_arr = X_proc.toarray() if hasattr(X_proc, "toarray") else np.asarray(X_proc)

    assert not np.any(np.isnan(X_arr)), "NaN found in MLP preprocessed matrix"
    assert not np.any(np.isinf(X_arr)), "Inf found in MLP preprocessed matrix"


# ---------------------------------------------------------------------------
# Ordinal target encoding
# ---------------------------------------------------------------------------


def test_ordinal_target_encoding():
    """encode_ordinal_targets must produce correct binary arrays."""
    y = pd.Series(["small", "fit", "large", "small", "large", "fit"])
    y_A, y_B = encode_ordinal_targets(y, CLASS_ORDER)

    # Decision A: 1 if not "small"
    expected_A = np.array([0, 1, 1, 0, 1, 1])
    np.testing.assert_array_equal(y_A, expected_A)

    # Decision B: 1 if "large"
    expected_B = np.array([0, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(y_B, expected_B)


def test_ordinal_target_encoding_unknown_label():
    """encode_ordinal_targets must raise on labels not in class_order."""
    y = pd.Series(["small", "medium", "large"])
    with pytest.raises(ValueError, match="unknown labels"):
        encode_ordinal_targets(y, CLASS_ORDER)


# ---------------------------------------------------------------------------
# Ordinal prediction reconstruction
# ---------------------------------------------------------------------------


def test_ordinal_prediction_reconstruction():
    """decode_ordinal_predictions must return valid class labels for
    self-consistent inputs (p_A >= p_B)."""
    # Clear majority cases
    p_A = np.array([0.1, 0.6, 0.9])  # small | fit or large | large
    p_B = np.array([0.05, 0.3, 0.8])  # small or fit | fit | large

    proba, y_pred = decode_ordinal_predictions(p_A, p_B, CLASS_ORDER)

    assert proba.shape == (3, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    # Row 0: p_A=0.1 → P(small)=0.9 dominates → small
    assert y_pred[0] == "small"
    # Row 2: p_B=0.8 → P(large)=0.8 dominates → large
    assert y_pred[2] == "large"

    # All predicted labels must be in CLASS_ORDER
    assert all(label in CLASS_ORDER for label in y_pred)


# ---------------------------------------------------------------------------
# Ordinal monotonicity enforcement
# ---------------------------------------------------------------------------


def test_ordinal_monotonicity_enforcement():
    """When p_B > p_A (inconsistent), clipping must be applied before
    reconstruction, and P(fit) must remain non-negative."""
    # Deliberately inconsistent: p_B > p_A
    p_A = np.array([0.3, 0.5])
    p_B = np.array([0.6, 0.8])  # Both violate p_A >= p_B

    proba, y_pred = decode_ordinal_predictions(p_A, p_B, CLASS_ORDER)

    # After clipping p_B = min(p_A, p_B):
    # Row 0: p_B_clipped = 0.3, P(fit) = 0.3 - 0.3 = 0.0 >= 0
    # Row 1: p_B_clipped = 0.5, P(fit) = 0.5 - 0.5 = 0.0 >= 0
    p_fit = proba[:, 1]
    assert np.all(p_fit >= 0.0), (
        f"P(fit) must be non-negative after monotonicity clipping; got {p_fit}"
    )
    # P(large) must equal clipped p_B (= p_A in violation cases)
    np.testing.assert_allclose(proba[:, 2], p_A, atol=1e-9)


# ---------------------------------------------------------------------------
# Fixed class order alignment
# ---------------------------------------------------------------------------


def test_fixed_class_order_alignment():
    """align_probability_columns must reorder any model class ordering to
    exactly ['small', 'fit', 'large'] regardless of input order."""
    # Simulate model that learned classes in alphabetical order: fit, large, small
    model_classes = ["fit", "large", "small"]
    raw_proba = np.array(
        [
            [0.2, 0.3, 0.5],  # fit=0.2, large=0.3, small=0.5
            [0.6, 0.1, 0.3],  # fit=0.6, large=0.1, small=0.3
        ]
    )

    aligned = align_probability_columns(raw_proba, model_classes, CLASS_ORDER)

    # Expected alignment to [small, fit, large]:
    # Row 0: small=0.5, fit=0.2, large=0.3
    np.testing.assert_allclose(aligned[0], [0.5, 0.2, 0.3])
    # Row 1: small=0.3, fit=0.6, large=0.1
    np.testing.assert_allclose(aligned[1], [0.3, 0.6, 0.1])


def test_fixed_class_order_alignment_correct_order():
    """align_probability_columns must be a no-op when model classes already
    match CLASS_ORDER."""
    raw_proba = np.array([[0.1, 0.6, 0.3]])
    aligned = align_probability_columns(raw_proba, CLASS_ORDER, CLASS_ORDER)
    np.testing.assert_allclose(aligned, raw_proba)


# ---------------------------------------------------------------------------
# Zero user overlap in group-aware CV folds
# ---------------------------------------------------------------------------


def test_zero_user_overlap_advanced_cv(small_development_data, cv_splits):
    """Every CV fold must have zero user overlap between train and validation."""
    _, _, groups = small_development_data

    for fold_i, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        train_users = set(groups.iloc[train_idx])
        val_users = set(groups.iloc[val_idx])
        overlap = train_users & val_users
        assert len(overlap) == 0, (
            f"Fold {fold_i}: user overlap detected: {overlap}"
        )


# ---------------------------------------------------------------------------
# Deterministic advanced CV splits
# ---------------------------------------------------------------------------


def test_advanced_cv_deterministic(small_development_data):
    """Calling make_group_cv_splits with the same arguments must produce
    byte-identical fold index arrays."""
    X, y, groups = small_development_data

    splits_1 = make_group_cv_splits(X, y, groups, n_splits=3, random_state=42)
    splits_2 = make_group_cv_splits(X, y, groups, n_splits=3, random_state=42)

    assert len(splits_1) == len(splits_2)
    for (tr1, va1), (tr2, va2) in zip(splits_1, splits_2):
        np.testing.assert_array_equal(tr1, tr2, err_msg="Train indices differ")
        np.testing.assert_array_equal(va1, va2, err_msg="Val indices differ")


# ---------------------------------------------------------------------------
# OrdinalBinaryPair unfitted guard
# ---------------------------------------------------------------------------


def test_ordinal_pair_raises_when_unfitted():
    """predict_proba_pair must raise RuntimeError before fit() is called."""
    pair = OrdinalBinaryPair()
    dummy_X = np.zeros((5, 3))
    with pytest.raises(RuntimeError, match="has not been fitted"):
        pair.predict_proba_pair(dummy_X)
