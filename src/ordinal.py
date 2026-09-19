"""Ordinal classification experiment for StyleFit AI.

Implements a cumulative binary decomposition approach that exploits the
natural ordering of the target:

    small < fit < large

Two binary logistic classifiers are trained:

    Decision A:  P(outcome > small)  =  P(fit or large)
    Decision B:  P(outcome > fit)    =  P(large)

The three-class probabilities are then reconstructed as:

    P(small)  = 1 - p_A
    P(fit)    = p_A - p_B
    P(large)  = p_B

subject to a monotonicity constraint:  p_A >= p_B.
When this is violated (i.e. p_B > p_A), p_B is clipped to p_A before
reconstruction, preventing logically inconsistent probability combinations.

The final class prediction is argmax over [P(small), P(fit), P(large)].

This experiment is explicitly modular and labeled as experimental.
It does not replace the original multiclass target in stored or raw data.

Preprocessing
-------------
The existing leakage-safe ColumnTransformer:

    build_preprocessor(scale_numeric=True, size_strategy="categorical")

is fitted on the outer training fold only per CV iteration, then used to
transform both the training and validation folds. No leakage fields or
user identifiers may enter the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation import CLASS_ORDER, align_probability_columns, evaluate_predictions
from src.preprocessing import build_preprocessor


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

# Fixed class order for ordinal decisions
_ORDINAL_CLASS_ORDER = CLASS_ORDER  # ["small", "fit", "large"]


def encode_ordinal_targets(
    y: pd.Series,
    class_order: List[str] = _ORDINAL_CLASS_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 3-class labels into two cumulative binary targets.

    Decision A: Is the outcome greater than the first class (small)?
        y_A = 1 if y in {fit, large}, else 0

    Decision B: Is the outcome greater than the second class (fit)?
        y_B = 1 if y in {large},      else 0

    Args:
        y: Series of target labels. Values must be in class_order.
        class_order: Ordered list of class labels [small, fit, large].

    Returns:
        Tuple[np.ndarray, np.ndarray]: (y_A, y_B) as integer arrays.

    Raises:
        ValueError: If y contains labels not in class_order.
    """
    if len(class_order) != 3:
        raise ValueError(
            f"encode_ordinal_targets requires exactly 3 classes; got {class_order}."
        )
    unknown = set(y.unique()) - set(class_order)
    if unknown:
        raise ValueError(
            f"encode_ordinal_targets: unknown labels in y: {sorted(unknown)}. "
            f"Expected values from {class_order}."
        )

    y_arr = np.asarray(y)
    # A: P(outcome > class_order[0])  —  "greater than small"
    y_A = (y_arr != class_order[0]).astype(int)
    # B: P(outcome > class_order[1])  —  "greater than fit"
    y_B = (y_arr == class_order[2]).astype(int)

    return y_A, y_B


# ---------------------------------------------------------------------------
# Probability reconstruction
# ---------------------------------------------------------------------------


def decode_ordinal_predictions(
    p_A: np.ndarray,
    p_B: np.ndarray,
    class_order: List[str] = _ORDINAL_CLASS_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct 3-class probabilities and labels from cumulative binary estimates.

    These probability estimates come from the two binary logistic models.

    Monotonicity enforcement:
        The ordinal constraint requires p_A >= p_B (if the outcome is
        "greater than fit", it must also be "greater than small").
        When p_B > p_A (inconsistent), p_B is clipped to p_A.

    Reconstruction:
        P(small) = 1 - p_A
        P(fit)   = p_A - p_B          (guaranteed >= 0 after clipping)
        P(large) = p_B

    Args:
        p_A: Probability estimates from Decision A binary model, shape (n,).
        p_B: Probability estimates from Decision B binary model, shape (n,).
        class_order: Ordered class labels [small, fit, large].

    Returns:
        Tuple[np.ndarray, np.ndarray]:
          - proba_matrix: shape (n, 3) aligned to class_order.
          - y_pred: string label array, argmax over proba_matrix.
    """
    p_A = np.asarray(p_A, dtype=float)
    p_B = np.asarray(p_B, dtype=float)

    # Monotonicity enforcement
    p_B_clipped = np.minimum(p_B, p_A)

    # Reconstruct class probabilities
    p_small = 1.0 - p_A
    p_fit = p_A - p_B_clipped
    p_large = p_B_clipped

    proba_matrix = np.column_stack([p_small, p_fit, p_large])
    class_array = np.asarray(class_order)
    y_pred = class_array[np.argmax(proba_matrix, axis=1)]

    return proba_matrix, y_pred


# ---------------------------------------------------------------------------
# OrdinalBinaryPair
# ---------------------------------------------------------------------------


@dataclass
class OrdinalBinaryPair:
    """Pair of binary logistic classifiers implementing cumulative ordinal decoding.

    Attributes:
        estimator_A: Fitted LogisticRegression for Decision A (P(outcome > small)).
        estimator_B: Fitted LogisticRegression for Decision B (P(outcome > fit)).
        class_order: Ordered class labels used during fit.
    """

    class_order: List[str] = None  # type: ignore[assignment]
    _estimator_A: Optional[LogisticRegression] = None
    _estimator_B: Optional[LogisticRegression] = None

    def __post_init__(self):
        if self.class_order is None:
            self.class_order = list(_ORDINAL_CLASS_ORDER)

    @staticmethod
    def _make_lr(random_state: int = 42) -> LogisticRegression:
        return LogisticRegression(
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=random_state,
        )

    def fit(
        self,
        X: np.ndarray,
        y_A: np.ndarray,
        y_B: np.ndarray,
        random_state: int = 42,
    ) -> "OrdinalBinaryPair":
        """Fit both binary classifiers independently.

        Args:
            X: Preprocessed feature matrix (training fold).
            y_A: Binary labels for Decision A.
            y_B: Binary labels for Decision B.
            random_state: Seed for both estimators.

        Returns:
            self
        """
        self._estimator_A = self._make_lr(random_state)
        self._estimator_B = self._make_lr(random_state)
        self._estimator_A.fit(X, y_A)
        self._estimator_B.fit(X, y_B)
        return self

    def predict_proba_pair(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return probability estimates from each binary logistic model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (p_A, p_B) — P(class=1) per estimator.

        Raises:
            RuntimeError: If the pair has not been fitted.
        """
        if self._estimator_A is None or self._estimator_B is None:
            raise RuntimeError(
                "OrdinalBinaryPair has not been fitted. Call fit() first."
            )
        p_A = self._estimator_A.predict_proba(X)[:, 1]
        p_B = self._estimator_B.predict_proba(X)[:, 1]
        return p_A, p_B


# ---------------------------------------------------------------------------
# Ordinal CV runner
# ---------------------------------------------------------------------------


def run_ordinal_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    """Run ordinal classification experiment on pre-computed user-disjoint CV folds.

    For each outer fold:
      1. Fit the leakage-safe preprocessor on outer training data only.
      2. Transform outer training and validation data.
      3. Encode ordinal binary targets from outer training labels.
      4. Fit OrdinalBinaryPair on the transformed training matrix.
      5. Predict probability estimates from the two binary logistic models on
         outer validation data.
      6. Reconstruct 3-class probabilities and predictions using
         decode_ordinal_predictions() with monotonicity enforcement.
      7. Evaluate reconstructed predictions with the shared evaluation contract.

    No leakage fields or identifiers enter the feature matrix.
    The original multiclass target is not modified in stored or raw data.

    Args:
        X: Development feature DataFrame (GENERAL_FIT_FEATURES columns).
        y: Target Series (values in CLASS_ORDER).
        groups: user_id Series (same length as X/y).
        splits: Pre-computed fold index pairs from make_group_cv_splits().
        random_state: Seed for OrdinalBinaryPair estimators.

    Returns:
        Tuple:
          - fold_results: DataFrame with one row per fold.
          - oof_y_true: List of y_true arrays per fold.
          - oof_y_pred: List of reconstructed y_pred arrays per fold.
    """
    fold_rows: List[Dict[str, Any]] = []
    oof_y_true: List[np.ndarray] = []
    oof_y_pred: List[np.ndarray] = []

    for fold_number, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 1. Fit preprocessor on outer training fold only
        preprocessor = build_preprocessor(
            scale_numeric=True, size_strategy="categorical"
        )
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        # 2. Encode ordinal binary targets from training labels
        y_A_train, y_B_train = encode_ordinal_targets(y_train, CLASS_ORDER)

        # 3. Fit ordinal binary pair
        started = perf_counter()
        pair = OrdinalBinaryPair(class_order=list(CLASS_ORDER))
        pair.fit(X_train_proc, y_A_train, y_B_train, random_state=random_state)
        fit_seconds = perf_counter() - started

        # 4. Reconstruct predictions on outer validation fold
        p_A, p_B = pair.predict_proba_pair(X_val_proc)
        proba_matrix, y_pred = decode_ordinal_predictions(p_A, p_B, CLASS_ORDER)

        print(
            f"  [Fold {fold_number}] Ordinal | fit_seconds={fit_seconds:.1f} | "
            f"monotonicity violations clipped: "
            f"{int((p_B > p_A).sum())} / {len(p_A)} samples"
        )

        evaluation = evaluate_predictions(
            y_true=y_val,
            y_pred=y_pred,
            aligned_proba=proba_matrix,
            model_name="Ordinal LR (Cumulative Binary)",
            split_type=f"group_cv_fold_{fold_number}",
        )

        oof_y_true.append(np.asarray(y_val))
        oof_y_pred.append(y_pred)

        from src.advanced_models import _flatten_advanced_fold

        fold_rows.append(
            _flatten_advanced_fold(evaluation, fold_number, fit_seconds)
        )

    fold_results = pd.DataFrame(fold_rows)
    return fold_results, oof_y_true, oof_y_pred
