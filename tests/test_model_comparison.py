"""Tests for leakage-safe, reproducible model comparison."""

import json

import numpy as np
import pandas as pd
import pytest

from src.cleaning import GENERAL_FIT_FEATURES
from src.evaluation import CLASS_ORDER
from src.model_comparison import (
    METRIC_COLUMNS,
    ComparisonConfig,
    build_model_specs,
    run_cross_validation,
    save_results,
    select_candidate,
)
from src.splitting import make_group_cv_splits


@pytest.fixture
def comparison_data():
    rng = np.random.default_rng(42)
    n_users = 36
    rows_per_user = 4
    n_rows = n_users * rows_per_user
    groups = pd.Series(
        np.repeat([f"user_{index}" for index in range(n_users)], rows_per_user),
        name="user_id",
    )
    class_cycle = np.asarray(CLASS_ORDER * (n_rows // len(CLASS_ORDER)))
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
    )
    return X[GENERAL_FIT_FEATURES], y, groups


def test_group_cv_has_no_user_overlap_and_all_classes(comparison_data):
    X, y, groups = comparison_data
    for train_idx, validation_idx in make_group_cv_splits(
        X, y, groups, n_splits=3, random_state=42
    ):
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[validation_idx]))
        assert set(y.iloc[validation_idx]) == set(CLASS_ORDER)


def test_group_cv_is_deterministic(comparison_data):
    X, y, groups = comparison_data
    first = make_group_cv_splits(X, y, groups, n_splits=3, random_state=42)
    second = make_group_cv_splits(X, y, groups, n_splits=3, random_state=42)
    for (first_train, first_validation), (second_train, second_validation) in zip(
        first, second
    ):
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_validation, second_validation)


def test_model_registry_preserves_references_and_explicit_balancing():
    specs = build_model_specs(random_state=42, include_xgboost=False)
    by_name = {spec.name: spec for spec in specs}
    assert by_name["Dummy (Most Frequent)"].role == "reference"
    assert by_name["Logistic Regression (Unweighted, Numeric Size)"].role == "reference"
    assert by_name["Logistic Regression (Unweighted, Categorical Size)"].role == "reference"
    assert by_name["Logistic Regression (Balanced)"].parameters["class_weight"] == "balanced"
    assert by_name["Random Forest (Balanced)"].parameters["class_weight"] == "balanced_subsample"


def test_cross_validation_result_schema_and_class_order(comparison_data):
    X, y, groups = comparison_data
    all_specs = build_model_specs(random_state=42, include_xgboost=False)
    specs = [
        spec
        for spec in all_specs
        if spec.name in {"Dummy (Most Frequent)", "Logistic Regression (Balanced)"}
    ]
    config = ComparisonConfig(cv_n_splits=3, min_category_frequency=1)
    fold_results, summary, confusion = run_cross_validation(X, y, groups, specs, config)

    assert len(fold_results) == len(specs) * config.cv_n_splits
    assert set(METRIC_COLUMNS).issubset(fold_results.columns)
    assert {f"{metric}_mean" for metric in METRIC_COLUMNS}.issubset(summary.columns)
    assert all(
        np.asarray(matrix).shape == (len(CLASS_ORDER), len(CLASS_ORDER))
        for matrix in confusion.values()
    )
    assert select_candidate(summary) == "Logistic Regression (Balanced)"


def test_results_artifacts_are_machine_readable(comparison_data, tmp_path):
    X, y, groups = comparison_data
    specs = [
        spec
        for spec in build_model_specs(random_state=42, include_xgboost=False)
        if spec.name in {"Dummy (Most Frequent)", "Logistic Regression (Balanced)"}
    ]
    config = ComparisonConfig(
        cv_n_splits=3,
        min_category_frequency=1,
        output_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
    )
    fold_results, summary, confusion = run_cross_validation(X, y, groups, specs, config)
    selected = select_candidate(summary)
    paths = save_results(
        fold_results,
        summary,
        confusion,
        specs,
        {"status": "sealed_not_evaluated"},
        selected,
        config,
    )

    assert all(path.exists() for path in paths.values())
    payload = json.loads(paths["results"].read_text(encoding="utf-8"))
    assert payload["class_order"] == CLASS_ORDER
    assert payload["selection_metric"] == "macro_f1_mean"
    assert payload["selected_candidate"] == selected
    assert payload["final_holdout_evaluated"] is False
