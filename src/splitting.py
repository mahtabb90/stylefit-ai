"""Dataset splitting module for StyleFit AI baseline evaluations.

Provides functions for:
  - Stratified random train/test splitting.
  - New-user generalization train/test splitting using candidate fold selection
    from StratifiedGroupKFold to minimize test size and target distribution drift.
  - Verification of split integrity (zero user overlap, row totals, target distributions).
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def stratified_random_split(
    df: pd.DataFrame,
    target_col: str = "fit",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified random train/test split.

    Args:
        df: Input DataFrame containing features and target column.
        target_col: Target column name.
        test_size: Fraction of samples to include in test set (default 0.20).
        random_state: Seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def unseen_user_split(
    df: pd.DataFrame,
    user_col: str = "user_id",
    target_col: str = "fit",
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Perform a group-aware train/test split ensuring zero user_id overlap.

    Evaluates all `n_splits` candidate folds from StratifiedGroupKFold based on:
      1. Size deviation from desired ~20% test ratio.
      2. Deviation from overall target-class proportions.

    Selects the optimal fold strictly using data distribution criteria (NO model predictions).

    Args:
        df: Input DataFrame containing user_col, target_col, and feature columns.
        user_col: Column name identifying users.
        target_col: Column name identifying target label.
        n_splits: Number of folds for StratifiedGroupKFold (n_splits=5 gives ~20% test size).
        random_state: Seed for StratifiedGroupKFold.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, users_train, users_test)
    """
    if user_col not in df.columns:
        raise KeyError(f"User column '{user_col}' not found in DataFrame.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    total_rows = len(df)
    target_classes = ["small", "fit", "large"]
    overall_dist = df[target_col].value_counts(normalize=True)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    candidate_evaluations: List[Dict[str, float]] = []

    for fold_idx, (train_indices, test_indices) in enumerate(
        sgkf.split(df, df[target_col], groups=df[user_col])
    ):
        test_subset = df.iloc[test_indices]
        test_dist = test_subset[target_col].value_counts(normalize=True)

        size_ratio = len(test_subset) / total_rows
        size_deviation = abs(size_ratio - (1.0 / n_splits))

        dist_deviation = sum(
            (float(test_dist.get(c, 0.0)) - float(overall_dist.get(c, 0.0))) ** 2
            for c in target_classes
        )

        combined_score = size_deviation + dist_deviation

        candidate_evaluations.append(
            {
                "fold_idx": fold_idx,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "size_ratio": size_ratio,
                "size_deviation": size_deviation,
                "dist_deviation": dist_deviation,
                "score": combined_score,
            }
        )

    # Select candidate fold minimizing combined deviation score
    best_candidate = min(candidate_evaluations, key=lambda c: c["score"])
    train_idx = best_candidate["train_indices"]
    test_idx = best_candidate["test_indices"]

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    users_train = train_df[user_col].copy()
    users_test = test_df[user_col].copy()

    y_train = train_df[target_col].copy()
    y_test = test_df[target_col].copy()

    X_train = train_df.drop(columns=[target_col]).copy()
    X_test = test_df.drop(columns=[target_col]).copy()

    return X_train, X_test, y_train, y_test, users_train, users_test


def verify_split_integrity(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    users_train: Optional[pd.Series] = None,
    users_test: Optional[pd.Series] = None,
) -> Dict[str, bool]:
    """Verify split integrity and return status dictionary.

    Checks:
      1. Train and test indices are completely disjoint.
      2. Row count sum matches.
      3. Zero user overlap (if user series provided).

    Returns:
        Dict[str, bool]: Verification results for each check.
    """
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    index_disjoint = len(train_indices.intersection(test_indices)) == 0
    row_counts_match = (len(y_train) == len(X_train)) and (len(y_test) == len(X_test))

    results = {
        "index_disjoint": index_disjoint,
        "row_counts_match": row_counts_match,
    }

    if users_train is not None and users_test is not None:
        user_set_train = set(users_train)
        user_set_test = set(users_test)
        user_overlap_count = len(user_set_train.intersection(user_set_test))
        results["zero_user_overlap"] = (user_overlap_count == 0)

    return results
