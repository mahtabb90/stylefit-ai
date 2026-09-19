"""Advanced tabular ML experiments for StyleFit AI.

Provides:
  - CatBoost multiclass experiment with native categorical handling, class
    weights derived from training fold only, and group-aware internal
    early-stopping split.
  - MLP/ANN experiment reusing the existing leakage-safe preprocessing
    pipeline with training-fold balanced sample weights and transparent
    convergence reporting.

Design invariants
-----------------
* No leakage fields (rating, review_text, review_summary) ever enter any
  feature matrix.
* user_id is used exclusively for group-aware splitting — never as a feature.
* Class weights / sample weights are computed from the training fold labels
  only, never from validation or full-dataset labels.
* Early stopping for CatBoost uses a group-aware internal sub-split of the
  outer training fold — the outer validation fold is never seen during fit.
* MLP built-in early_stopping is disabled (not group-aware); convergence
  warnings are captured and surfaced transparently.
* Probability columns are always explicitly aligned to CLASS_ORDER
  ["small", "fit", "large"] via align_probability_columns().
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from src.cleaning import GENERAL_FIT_FEATURES
from src.evaluation import CLASS_ORDER, align_probability_columns, evaluate_predictions
from src.preprocessing import build_preprocessor
from src.splitting import make_group_cv_splits  # noqa: F401 – re-exported for convenience

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Categorical features to be passed natively to CatBoost (no one-hot encoding)
CATBOOST_CATEGORICAL_FEATURES: List[str] = [
    "bust_cup_size",
    "body_type",
    "category",
    "rented_for",
    "size",
]

# Numeric features passed as-is; CatBoost handles NaN natively
CATBOOST_NUMERIC_FEATURES: List[str] = [
    "parsed_height_inches",
    "parsed_weight_lbs",
    "bust_band_size",
    "age",
]

# Sentinel token for missing categoricals — matches project convention
MISSING_TOKEN: str = "__missing__"


# ---------------------------------------------------------------------------
# CatBoost feature preparation
# ---------------------------------------------------------------------------


def build_catboost_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """Prepare feature matrix for CatBoost native categorical handling.

    Keeps exactly the approved GENERAL_FIT_FEATURES scope:
      - Categorical columns: missing values filled with MISSING_TOKEN (__missing__).
        Values cast to str so CatBoost receives a uniform dtype.
      - Numeric columns: left as-is; CatBoost handles NaN internally.

    No one-hot encoding is applied. No leakage columns may enter X.

    Args:
        X: DataFrame with GENERAL_FIT_FEATURES columns (or a subset thereof).

    Returns:
        Tuple:
          - DataFrame with categorical NaNs filled by MISSING_TOKEN.
          - List of integer column indices for categorical features (for CatBoost
            ``cat_features`` argument).

    Raises:
        KeyError: If any expected feature column is missing from X.
    """
    missing_cols = [c for c in GENERAL_FIT_FEATURES if c not in X.columns]
    if missing_cols:
        raise KeyError(
            f"build_catboost_features: missing required columns: {missing_cols}"
        )

    # Work on exact approved feature order
    X_cat = X[GENERAL_FIT_FEATURES].copy()

    for col in CATBOOST_CATEGORICAL_FEATURES:
        X_cat[col] = X_cat[col].fillna(MISSING_TOKEN).astype(str)

    cat_feature_indices = [
        list(X_cat.columns).index(col) for col in CATBOOST_CATEGORICAL_FEATURES
    ]

    return X_cat, cat_feature_indices


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights_from_train(y_train: pd.Series) -> Dict[str, float]:
    """Compute balanced class weights from training-fold labels only.

    Uses sklearn's ``compute_class_weight("balanced", ...)`` so that each
    class weight = total_samples / (n_classes * class_count).

    Args:
        y_train: Series of target labels from the training fold.

    Returns:
        Dict mapping each class label (str) to its weight (float).

    Raises:
        ValueError: If y_train does not contain every class in CLASS_ORDER.
    """
    present = set(y_train.unique())
    missing = set(CLASS_ORDER) - present
    if missing:
        raise ValueError(
            f"compute_class_weights_from_train: training fold is missing classes "
            f"{sorted(missing)}. Cannot compute balanced weights."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(CLASS_ORDER),
        y=y_train.to_numpy(),
    )
    return {cls: float(w) for cls, w in zip(CLASS_ORDER, weights)}


# ---------------------------------------------------------------------------
# Group-aware internal early-stopping split
# ---------------------------------------------------------------------------


def make_internal_eval_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_internal_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create a group-aware internal train/eval split from the outer training fold.

    Uses StratifiedGroupKFold to produce an internal eval subset with zero user
    overlap against the internal training subset. Both subsets remain entirely
    within the outer training fold — the outer validation fold is never involved.

    The first fold of the internal splitter is used as the internal eval set,
    giving approximately 1/n_internal_splits of the outer training data.

    Args:
        X_train: Feature DataFrame for the outer training fold.
        y_train: Target labels for the outer training fold.
        groups_train: user_id Series for the outer training fold.
        n_internal_splits: Number of folds for the internal StratifiedGroupKFold.
        random_state: Seed for reproducibility.

    Returns:
        Tuple: (X_int_train, y_int_train, X_int_eval, y_int_eval)

    Raises:
        RuntimeError: If user overlap exists between internal train and eval.
        ValueError: If internal eval fold is missing any target class.
    """
    splitter = StratifiedGroupKFold(
        n_splits=n_internal_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Use the first available fold as the internal eval split
    int_train_idx, int_eval_idx = next(
        splitter.split(X_train, y_train, groups=groups_train)
    )

    # Verify zero user overlap
    int_train_users = set(groups_train.iloc[int_train_idx])
    int_eval_users = set(groups_train.iloc[int_eval_idx])
    overlap = int_train_users & int_eval_users
    if overlap:
        raise RuntimeError(
            f"make_internal_eval_split: user overlap detected in internal split: "
            f"{len(overlap)} user(s) appear in both internal train and eval."
        )

    # Verify all classes present in internal eval
    eval_classes = set(y_train.iloc[int_eval_idx].unique())
    if eval_classes != set(CLASS_ORDER):
        raise ValueError(
            f"make_internal_eval_split: internal eval fold missing classes "
            f"{sorted(set(CLASS_ORDER) - eval_classes)}."
        )

    X_int_train = X_train.iloc[int_train_idx]
    y_int_train = y_train.iloc[int_train_idx]
    X_int_eval = X_train.iloc[int_eval_idx]
    y_int_eval = y_train.iloc[int_eval_idx]

    return X_int_train, y_int_train, X_int_eval, y_int_eval


# ---------------------------------------------------------------------------
# CatBoost CV spec and runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatBoostSpec:
    """Documented, inspectable CatBoost configuration.

    Parameters are conservative and suitable for a portfolio experiment.
    No hyperparameter search is performed.
    """

    iterations: int = 500
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    loss_function: str = "MultiClass"
    # eval_metric monitors unweighted Macro F1 -- same definition used for
    # cross-model comparison -- despite using class_weights for training.
    eval_metric: str = "TotalF1:average=Macro;use_weights=false"
    early_stopping_rounds: int = 50
    bootstrap_type: str = "Bayesian"
    random_seed: int = 42
    thread_count: int = -1
    verbose: int = 100
    # Disable all file writes so CatBoost never creates catboost_info/ artifacts
    allow_writing_files: bool = False
    # Internal group-aware split configuration
    internal_eval_n_splits: int = 5
    internal_eval_random_state: int = 42


def run_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    spec: Optional[CatBoostSpec] = None,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    """Run CatBoost multiclass experiment on pre-computed user-disjoint CV folds.

    For each outer fold:
      1. Prepare CatBoost feature matrix (native categoricals, __missing__ fill).
      2. Compute class weights from outer training labels only.
      3. Build a group-aware internal eval split from the outer training fold
         (zero user overlap verified; outer validation fold never seen).
      4. Fit CatBoost with early_stopping_rounds on the internal eval set.
      5. Evaluate on the outer validation fold with metrics aligned to CLASS_ORDER.

    Args:
        X: Development feature DataFrame (GENERAL_FIT_FEATURES columns).
        y: Target Series (values in CLASS_ORDER).
        groups: user_id Series (same length as X/y).
        splits: Pre-computed fold index pairs from make_group_cv_splits().
        spec: CatBoostSpec configuration (uses defaults if None).

    Returns:
        Tuple:
          - fold_results: DataFrame with one row per fold.
          - oof_y_true: List of y_true arrays per fold.
          - oof_y_pred: List of y_pred arrays per fold.
    """
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is required for this experiment. "
            "Install with: python3 -m pip install catboost"
        ) from exc

    if spec is None:
        spec = CatBoostSpec()

    fold_rows: List[Dict[str, Any]] = []
    oof_y_true: List[np.ndarray] = []
    oof_y_pred: List[np.ndarray] = []

    for fold_number, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        g_train = groups.iloc[train_idx]

        # 1. Prepare feature matrices
        X_train_cat, cat_feature_indices = build_catboost_features(X_train)
        X_val_cat, _ = build_catboost_features(X_val)

        # 2. Class weights from training fold only
        class_weights = compute_class_weights_from_train(y_train)

        # 3. Group-aware internal eval split (outer val fold never touched)
        X_int_tr, y_int_tr, X_int_ev, y_int_ev = make_internal_eval_split(
            X_train_cat,
            y_train,
            g_train,
            n_internal_splits=spec.internal_eval_n_splits,
            random_state=spec.internal_eval_random_state,
        )
        print(
            f"  [Fold {fold_number}] Internal eval split: "
            f"{len(X_int_tr)} train rows / {len(X_int_ev)} eval rows | "
            f"zero user overlap verified"
        )

        train_pool = Pool(
            X_int_tr,
            label=y_int_tr,
            cat_features=cat_feature_indices,
        )
        eval_pool = Pool(
            X_int_ev,
            label=y_int_ev,
            cat_features=cat_feature_indices,
        )
        outer_val_pool = Pool(
            X_val_cat,
            label=y_val,
            cat_features=cat_feature_indices,
        )

        # 4. Fit CatBoost
        # Note: classes_count must NOT be set when training with string class labels;
        # CatBoost infers the class count from the Pool labels automatically.
        # allow_writing_files=False prevents creation of catboost_info/ artifacts.
        model = CatBoostClassifier(
            iterations=spec.iterations,
            learning_rate=spec.learning_rate,
            depth=spec.depth,
            l2_leaf_reg=spec.l2_leaf_reg,
            loss_function=spec.loss_function,
            eval_metric=spec.eval_metric,
            early_stopping_rounds=spec.early_stopping_rounds,
            bootstrap_type=spec.bootstrap_type,
            random_seed=spec.random_seed,
            thread_count=spec.thread_count,
            verbose=spec.verbose,
            class_weights=class_weights,
            allow_writing_files=spec.allow_writing_files,
        )

        started = perf_counter()
        model.fit(train_pool, eval_set=eval_pool)
        fit_seconds = perf_counter() - started

        best_iter = model.get_best_iteration()
        print(
            f"  [Fold {fold_number}] CatBoost best_iteration={best_iter} "
            f"(of {spec.iterations}) | fit_seconds={fit_seconds:.1f}"
        )

        # 5. Evaluate on outer validation fold
        raw_proba = model.predict_proba(outer_val_pool)
        # CatBoost returns class labels (strings) from model.classes_ in sorted
        # alphabetical order — not necessarily CLASS_ORDER. align_probability_columns
        # reorders the probability columns to the fixed [small, fit, large] order.
        model_classes = [str(c) for c in model.classes_]
        aligned_proba = align_probability_columns(raw_proba, model_classes, CLASS_ORDER)
        # CatBoost predict() returns the class label strings directly when trained
        # with string labels — do NOT treat as integer indices.
        y_pred = np.asarray([str(p) for p in model.predict(outer_val_pool).flatten()])

        evaluation = evaluate_predictions(
            y_true=y_val,
            y_pred=y_pred,
            aligned_proba=aligned_proba,
            model_name="CatBoost",
            split_type=f"group_cv_fold_{fold_number}",
        )

        oof_y_true.append(np.asarray(y_val))
        oof_y_pred.append(y_pred)

        fold_rows.append(
            _flatten_advanced_fold(
                evaluation, fold_number, fit_seconds, best_iteration=best_iter
            )
        )

    fold_results = pd.DataFrame(fold_rows)
    return fold_results, oof_y_true, oof_y_pred


# ---------------------------------------------------------------------------
# MLP CV spec and runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLPSpec:
    """Documented MLP configuration.

    early_stopping is disabled because sklearn's built-in implementation
    uses a row-level internal split that is not group-aware. max_iter=300
    provides a fixed training budget. Convergence warnings are captured and
    surfaced transparently per fold.
    """

    hidden_layer_sizes: Tuple[int, ...] = (128, 64)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.001
    learning_rate_init: float = 0.001
    max_iter: int = 300
    early_stopping: bool = False  # Not group-aware; disabled for this phase
    random_state: int = 42


def build_mlp_preprocessor():
    """Return a fresh leakage-safe ColumnTransformer for MLP input.

    Reuses the project's existing build_preprocessor with:
      - scale_numeric=True  (StandardScaler on numeric features)
      - size_strategy="categorical"  (size treated as OHE categorical)

    Must be fit only on training data per fold.
    """
    return build_preprocessor(scale_numeric=True, size_strategy="categorical")


def run_mlp_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    spec: Optional[MLPSpec] = None,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    """Run MLP multiclass experiment on pre-computed user-disjoint CV folds.

    For each outer fold:
      1. Fit the leakage-safe preprocessor on outer training data only.
      2. Transform outer training and validation data.
      3. Compute balanced sample_weight from outer training labels only.
      4. Fit MLPClassifier with early_stopping=False and max_iter=300.
         Convergence warnings are captured and reported per fold.
      5. Evaluate on the outer validation fold with metrics aligned to CLASS_ORDER.

    Args:
        X: Development feature DataFrame (GENERAL_FIT_FEATURES columns).
        y: Target Series (values in CLASS_ORDER).
        groups: user_id Series (same length as X/y).
        splits: Pre-computed fold index pairs from make_group_cv_splits().
        spec: MLPSpec configuration (uses defaults if None).

    Returns:
        Tuple:
          - fold_results: DataFrame with one row per fold.
          - oof_y_true: List of y_true arrays per fold.
          - oof_y_pred: List of y_pred arrays per fold.
    """
    if spec is None:
        spec = MLPSpec()

    fold_rows: List[Dict[str, Any]] = []
    oof_y_true: List[np.ndarray] = []
    oof_y_pred: List[np.ndarray] = []

    for fold_number, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 1. Fit preprocessor on training fold only
        preprocessor = build_mlp_preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        # 2. Balanced sample weights from training labels only
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        # 3. Fit MLP — capture convergence warnings transparently
        mlp = MLPClassifier(
            hidden_layer_sizes=spec.hidden_layer_sizes,
            activation=spec.activation,
            solver=spec.solver,
            alpha=spec.alpha,
            learning_rate_init=spec.learning_rate_init,
            max_iter=spec.max_iter,
            early_stopping=spec.early_stopping,
            random_state=spec.random_state,
        )

        converged = True
        convergence_msg = ""
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            mlp.fit(X_train_proc, y_train, sample_weight=sample_weight)
            fit_seconds = perf_counter() - started
            for w in caught_warnings:
                if issubclass(w.category, Warning) and "Stochastic Optimizer" in str(
                    w.message
                ):
                    converged = False
                    convergence_msg = str(w.message)

        n_iter_actual = mlp.n_iter_
        print(
            f"  [Fold {fold_number}] MLP n_iter={n_iter_actual} | "
            f"converged={converged} | fit_seconds={fit_seconds:.1f}"
        )
        if not converged:
            print(f"  [Fold {fold_number}] ConvergenceWarning: {convergence_msg}")

        # 4. Predict and align probabilities
        raw_proba = mlp.predict_proba(X_val_proc)
        aligned_proba = align_probability_columns(
            raw_proba, mlp.classes_, CLASS_ORDER
        )
        y_pred = mlp.predict(X_val_proc)

        evaluation = evaluate_predictions(
            y_true=y_val,
            y_pred=y_pred,
            aligned_proba=aligned_proba,
            model_name="MLP (128-64, relu, adam)",
            split_type=f"group_cv_fold_{fold_number}",
        )

        oof_y_true.append(np.asarray(y_val))
        oof_y_pred.append(y_pred)

        row = _flatten_advanced_fold(evaluation, fold_number, fit_seconds)
        row["mlp_n_iter"] = n_iter_actual
        row["mlp_converged"] = converged
        fold_rows.append(row)

    fold_results = pd.DataFrame(fold_rows)
    return fold_results, oof_y_true, oof_y_pred


# ---------------------------------------------------------------------------
# Shared fold-result flattener
# ---------------------------------------------------------------------------


def _flatten_advanced_fold(
    evaluation: Dict[str, Any],
    fold: int,
    fit_seconds: float,
    best_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    """Flatten a single fold's evaluation dict into a flat metrics row."""
    overall = evaluation["overall"]
    per_class = evaluation["per_class"]
    prob = evaluation.get("probability_metrics") or {}

    row: Dict[str, Any] = {
        "fold": fold,
        "fit_seconds": fit_seconds,
        "accuracy": overall["accuracy"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "macro_f1": overall["macro_f1"],
        "weighted_f1": overall["weighted_f1"],
        "macro_pr_auc": prob.get("macro_pr_auc", np.nan),
        "macro_roc_auc": prob.get("macro_roc_auc", np.nan),
    }
    for cls in CLASS_ORDER:
        row[f"precision_{cls}"] = per_class[cls]["precision"]
        row[f"recall_{cls}"] = per_class[cls]["recall"]
        row[f"f1_{cls}"] = per_class[cls]["f1"]
        row[f"pr_auc_{cls}"] = prob.get("per_class_pr_auc", {}).get(cls, np.nan)

    if best_iteration is not None:
        row["best_iteration"] = best_iteration

    return row


# ---------------------------------------------------------------------------
# Aggregate fold results
# ---------------------------------------------------------------------------


def aggregate_advanced_fold_results(fold_results: pd.DataFrame) -> pd.Series:
    """Compute mean and std across folds for all numeric metric columns.

    Returns a Series with keys like ``macro_f1_mean``, ``macro_f1_std``, etc.
    """
    numeric_cols = fold_results.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude fold index and timing from mean/std
    exclude = {"fold"}
    metric_cols = [c for c in numeric_cols if c not in exclude]

    stats: Dict[str, float] = {}
    for col in metric_cols:
        stats[f"{col}_mean"] = float(fold_results[col].mean())
        stats[f"{col}_std"] = float(fold_results[col].std(ddof=0))

    return pd.Series(stats)


# ---------------------------------------------------------------------------
# Guard: prevent accidental import of forbidden column names
# ---------------------------------------------------------------------------

_FORBIDDEN_FEATURES = frozenset(
    ["rating", "review_text", "review_summary", "user_id", "item_id", "review_date"]
)


def assert_no_forbidden_features(X: pd.DataFrame, context: str = "") -> None:
    """Raise ValueError if any forbidden feature column appears in X."""
    found = _FORBIDDEN_FEATURES & set(X.columns)
    if found:
        raise ValueError(
            f"Forbidden feature(s) detected{' in ' + context if context else ''}: "
            f"{sorted(found)}. These columns must not enter any model feature matrix."
        )
