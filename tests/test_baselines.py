"""Unit tests for StyleFit AI baseline models module (splitting, evaluation, leakage boundary)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.cleaning import GENERAL_FIT_FEATURES, LEAKAGE_COLUMNS, METADATA_COLUMNS
from src.evaluation import (
    CLASS_ORDER,
    align_probability_columns,
    compute_confusion_matrices,
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_probability_metrics,
)
from src.preprocessing import select_general_fit_features
from src.splitting import (
    stratified_random_split,
    unseen_user_split,
    verify_split_integrity,
)


@pytest.fixture
def dummy_dataset() -> pd.DataFrame:
    """Construct a mock DataFrame with users, target labels, features, and leakage columns."""
    np.random.seed(42)
    n_samples = 200
    n_users = 40

    user_ids = [f"user_{i % n_users}" for i in range(n_samples)]
    targets = np.random.choice(["fit", "small", "large"], size=n_samples, p=[0.7, 0.15, 0.15])

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": [f"item_{i}" for i in range(n_samples)],
            "review_date": pd.date_range("2023-01-01", periods=n_samples, freq="D"),
            "rating": np.random.randint(1, 11, size=n_samples),
            "review_text": [f"Review text {i}" for i in range(n_samples)],
            "review_summary": [f"Summary {i}" for i in range(n_samples)],
            "fit": targets,
            "parsed_height_inches": np.random.uniform(60, 72, size=n_samples),
            "parsed_weight_lbs": np.random.uniform(100, 200, size=n_samples),
            "bust_band_size": np.random.choice([32.0, 34.0, 36.0], size=n_samples),
            "bust_cup_size": np.random.choice(["b", "c", "d"], size=n_samples),
            "body_type": np.random.choice(["hourglass", "athletic"], size=n_samples),
            "age": np.random.uniform(20, 50, size=n_samples),
            "size": np.random.choice([4, 8, 12], size=n_samples),
            "category": np.random.choice(["dress", "gown"], size=n_samples),
            "rented_for": np.random.choice(["wedding", "party"], size=n_samples),
        }
    )
    return df


def test_select_general_fit_features_leakage_exclusion(dummy_dataset):
    """Test that select_general_fit_features returns ONLY allowed features and excludes leakage/metadata."""
    features = select_general_fit_features(dummy_dataset)

    # Check exact feature set
    assert list(features.columns) == GENERAL_FIT_FEATURES

    # Assert no leakage column present
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in features.columns

    # Assert no metadata column present
    for meta_col in METADATA_COLUMNS:
        assert meta_col not in features.columns


def test_stratified_random_split_integrity(dummy_dataset):
    """Test stratified random split row ratios, index disjointness, and target presence."""
    X_tr, X_te, y_tr, y_te = stratified_random_split(
        dummy_dataset, target_col="fit", test_size=0.20, random_state=42
    )

    assert len(X_tr) + len(X_te) == len(dummy_dataset)
    assert abs(len(X_te) / len(dummy_dataset) - 0.20) < 0.05
    assert len(set(X_tr.index).intersection(set(X_te.index))) == 0

    integrity = verify_split_integrity(X_tr, X_te, y_tr, y_te)
    assert integrity["index_disjoint"] is True
    assert integrity["row_counts_match"] is True


def test_unseen_user_split_zero_overlap(dummy_dataset):
    """Test unseen-user split guarantees zero user_id overlap and evaluates candidate folds."""
    X_tr, X_te, y_tr, y_te, users_tr, users_te = unseen_user_split(
        dummy_dataset, user_col="user_id", target_col="fit", n_splits=5, random_state=42
    )

    # Assert zero user overlap
    user_overlap = set(users_tr).intersection(set(users_te))
    assert len(user_overlap) == 0

    # Verify split integrity
    integrity = verify_split_integrity(X_tr, X_te, y_tr, y_te, users_tr, users_te)
    assert integrity["index_disjoint"] is True
    assert integrity["row_counts_match"] is True
    assert integrity["zero_user_overlap"] is True


def test_align_probability_columns():
    """Test probability column reordering when model.classes_ is scrambled."""
    # Raw probability matrix corresponding to model classes ['large', 'small', 'fit']
    raw_proba = np.array([
        [0.1, 0.7, 0.2],  # sample 0: large=0.1, small=0.7, fit=0.2
        [0.6, 0.1, 0.3],  # sample 1: large=0.6, small=0.1, fit=0.3
    ])
    model_classes = np.array(["large", "small", "fit"])

    aligned = align_probability_columns(raw_proba, model_classes, target_order=CLASS_ORDER)

    # CLASS_ORDER = ["small", "fit", "large"]
    # Sample 0 aligned probabilities: small=0.7, fit=0.2, large=0.1
    np.testing.assert_allclose(aligned[0], [0.7, 0.2, 0.1])
    # Sample 1 aligned probabilities: small=0.1, fit=0.3, large=0.6
    np.testing.assert_allclose(aligned[1], [0.1, 0.3, 0.6])


def test_evaluation_fixed_class_ordering():
    """Test overall and per-class metrics structure with fixed CLASS_ORDER."""
    y_true = np.array(["fit", "small", "large", "fit", "small"])
    y_pred = np.array(["fit", "fit", "large", "fit", "small"])

    overall = compute_overall_metrics(y_true, y_pred)
    assert "accuracy" in overall
    assert "balanced_accuracy" in overall
    assert "macro_f1" in overall
    assert "weighted_f1" in overall

    per_class = compute_per_class_metrics(y_true, y_pred)
    assert list(per_class.keys()) == ["small", "fit", "large"]
    for cls_name in ["small", "fit", "large"]:
        assert "precision" in per_class[cls_name]
        assert "recall" in per_class[cls_name]
        assert "f1" in per_class[cls_name]
        assert "support" in per_class[cls_name]

    raw_cm, norm_cm = compute_confusion_matrices(y_true, y_pred)
    assert raw_cm.shape == (3, 3)
    assert norm_cm.shape == (3, 3)
    np.testing.assert_allclose(norm_cm.sum(axis=1), [1.0, 1.0, 1.0])


def test_split_reproducibility(dummy_dataset):
    """Test that random_state produces identical splits across runs."""
    X_tr1, X_te1, y_tr1, y_te1 = stratified_random_split(
        dummy_dataset, target_col="fit", test_size=0.20, random_state=42
    )
    X_tr2, X_te2, y_tr2, y_te2 = stratified_random_split(
        dummy_dataset, target_col="fit", test_size=0.20, random_state=42
    )

    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_frame_equal(X_te1, X_te2)
    pd.testing.assert_series_equal(y_tr1, y_tr2)
    pd.testing.assert_series_equal(y_te1, y_te2)
