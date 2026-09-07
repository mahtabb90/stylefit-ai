"""Evaluation and metrics module for StyleFit AI baseline models.

Enforces a fixed class ordering across all calculations:
    CLASS_ORDER = ["small", "fit", "large"]

Provides functions for:
  - Probability column alignment based on model.classes_.
  - Overall classification metrics (Accuracy, Balanced Accuracy, Macro F1, Weighted F1).
  - Per-class classification metrics (Precision, Recall, F1, Support).
  - Raw and row-normalized confusion matrices.
  - One-vs-Rest ROC-AUC and PR-AUC / Average Precision (per-class and macro).
  - Comprehensive model evaluation pipeline and summary table formatting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

CLASS_ORDER = ["small", "fit", "large"]


def align_probability_columns(
    raw_proba: np.ndarray,
    model_classes: Union[np.ndarray, List[Any]],
    target_order: List[str] = CLASS_ORDER,
) -> np.ndarray:
    """Align model.predict_proba() columns to target_order ("small", "fit", "large").

    Ensures probability matrices match exact fixed class order regardless of
    how model.classes_ is sorted or if any class is missing.

    Args:
        raw_proba: Predicted probability matrix of shape (n_samples, n_model_classes).
        model_classes: List or array of class labels corresponding to raw_proba columns.
        target_order: Desired class column order (default ["small", "fit", "large"]).

    Returns:
        np.ndarray: Reordered probability matrix of shape (n_samples, len(target_order)).
    """
    n_samples = raw_proba.shape[0]
    aligned = np.zeros((n_samples, len(target_order)), dtype=float)
    model_classes_list = [str(c) for c in model_classes]

    for target_idx, target_class in enumerate(target_order):
        if target_class in model_classes_list:
            col_idx = model_classes_list.index(target_class)
            aligned[:, target_idx] = raw_proba[:, col_idx]
        else:
            aligned[:, target_idx] = 0.0

    return aligned


def compute_overall_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    target_order: List[str] = CLASS_ORDER,
) -> Dict[str, float]:
    """Compute overall classification performance metrics.

    Metrics:
      - Accuracy
      - Balanced Accuracy
      - Macro F1
      - Weighted F1

    Returns:
        Dict[str, float]: Dictionary of metric names and floating point values.
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=target_order))
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", labels=target_order)
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def compute_per_class_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    target_order: List[str] = CLASS_ORDER,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, F1, and support using target_order.

    Returns:
        Dict[str, Dict[str, float]]: Nested dict mapping class label to its metrics.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=target_order, zero_division=0
    )

    per_class_res: Dict[str, Dict[str, float]] = {}
    for idx, cls_name in enumerate(target_order):
        per_class_res[cls_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return per_class_res


def compute_confusion_matrices(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    target_order: List[str] = CLASS_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw count and row-normalized confusion matrices using target_order.

    Row-normalization normalizes over true class labels (each row sums to 1.0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (raw_cm, normalized_cm)
    """
    raw_cm = confusion_matrix(y_true, y_pred, labels=target_order)
    norm_cm = confusion_matrix(y_true, y_pred, labels=target_order, normalize="true")
    return raw_cm, norm_cm


def compute_probability_metrics(
    y_true: Union[pd.Series, np.ndarray],
    aligned_proba: np.ndarray,
    target_order: List[str] = CLASS_ORDER,
) -> Dict[str, Any]:
    """Compute One-vs-Rest ROC-AUC and Average Precision (PR-AUC) metrics.

    Expects `aligned_proba` columns to already match `target_order`.

    Returns:
        Dict[str, Any]: Dictionary containing macro and per-class ROC-AUC and PR-AUC scores.
    """
    y_bin = label_binarize(y_true, classes=target_order)

    # If y_bin has only 1 column (e.g. binary), reshape for safety
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    per_class_roc_auc: Dict[str, float] = {}
    per_class_pr_auc: Dict[str, float] = {}

    for idx, cls_name in enumerate(target_order):
        # Calculate per-class ROC-AUC if class present in y_true
        if len(np.unique(y_bin[:, idx])) > 1:
            roc_val = float(roc_auc_score(y_bin[:, idx], aligned_proba[:, idx]))
            pr_val = float(
                average_precision_score(y_bin[:, idx], aligned_proba[:, idx])
            )
        else:
            roc_val = np.nan
            pr_val = np.nan

        per_class_roc_auc[cls_name] = roc_val
        per_class_pr_auc[cls_name] = pr_val

    try:
        macro_roc_auc = float(
            roc_auc_score(y_bin, aligned_proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        macro_roc_auc = np.nan

    try:
        macro_pr_auc = float(
            average_precision_score(y_bin, aligned_proba, average="macro")
        )
    except ValueError:
        macro_pr_auc = np.nan

    return {
        "macro_roc_auc": macro_roc_auc,
        "macro_pr_auc": macro_pr_auc,
        "per_class_roc_auc": per_class_roc_auc,
        "per_class_pr_auc": per_class_pr_auc,
    }


def evaluate_baseline_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    split_type: str,
    preprocessor: Optional[Any] = None,
    target_order: List[str] = CLASS_ORDER,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Execute complete evaluation workflow for a baseline model.

    Workflow:
      1. Fit preprocessor ONLY on X_train (if preprocessor provided).
      2. Transform X_train and X_test.
      3. Fit model on transformed X_train and y_train, optionally with sample weights.
      4. Generate predictions and probability outputs on transformed X_test.
      5. Align probability columns using model.classes_.
      6. Compute overall metrics, per-class metrics, confusion matrices, and probability metrics.

    Returns:
        Dict[str, Any]: Complete results payload for reporting and visualization.
    """
    if preprocessor is not None:
        X_tr_proc = preprocessor.fit_transform(X_train)
        X_te_proc = preprocessor.transform(X_test)
    else:
        X_tr_proc = X_train
        X_te_proc = X_test

    # Fit model on training data
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(X_tr_proc, y_train, **fit_kwargs)

    # Predict on test data
    y_pred = model.predict(X_te_proc)

    # Extract & align probabilities if supported
    has_proba = hasattr(model, "predict_proba")
    aligned_proba = None
    if has_proba:
        raw_proba = model.predict_proba(X_te_proc)
        model_classes = getattr(model, "classes_", target_order)
        aligned_proba = align_probability_columns(
            raw_proba, model_classes, target_order
        )

    evaluated = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        aligned_proba=aligned_proba,
        model_name=model_name,
        split_type=split_type,
        target_order=target_order,
    )
    return {
        **evaluated,
        "model": model,
        "preprocessor": preprocessor,
    }


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    aligned_proba: Optional[np.ndarray],
    model_name: str,
    split_type: str,
    target_order: List[str] = CLASS_ORDER,
) -> Dict[str, Any]:
    """Evaluate predictions through the project's shared metric contract.

    Keeping this logic independent of fitting lets cross-validation reuse a
    fold-fitted preprocessor across models with the same feature strategy.
    """
    overall = compute_overall_metrics(y_true, y_pred, target_order)
    per_class = compute_per_class_metrics(y_true, y_pred, target_order)
    raw_cm, norm_cm = compute_confusion_matrices(y_true, y_pred, target_order)
    probability_metrics = None
    if aligned_proba is not None:
        probability_metrics = compute_probability_metrics(
            y_true, aligned_proba, target_order
        )

    return {
        "model_name": model_name,
        "split_type": split_type,
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": raw_cm,
        "normalized_confusion_matrix": norm_cm,
        "probability_metrics": probability_metrics,
        "aligned_proba": aligned_proba,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def build_summary_dataframe(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format evaluation results into a clean pandas DataFrame for comparative reporting.

    Columns included:
      - Model
      - Benchmark Split
      - Accuracy
      - Balanced Accuracy
      - Macro F1
      - Weighted F1
      - Macro ROC-AUC
      - Macro PR-AUC
      - F1 (small)
      - F1 (fit)
      - F1 (large)
    """
    rows = []
    for res in results_list:
        model_name = res["model_name"]
        split_type = res["split_type"]
        ov = res["overall"]
        pc = res["per_class"]
        pm = res.get("probability_metrics") or {}

        rows.append(
            {
                "Model": model_name,
                "Benchmark Split": split_type,
                "Accuracy": ov["accuracy"],
                "Balanced Accuracy": ov["balanced_accuracy"],
                "Macro F1": ov["macro_f1"],
                "Weighted F1": ov["weighted_f1"],
                "Macro ROC-AUC": pm.get("macro_roc_auc", np.nan),
                "Macro PR-AUC": pm.get("macro_pr_auc", np.nan),
                "F1 (small)": pc["small"]["f1"],
                "F1 (fit)": pc["fit"]["f1"],
                "F1 (large)": pc["large"]["f1"],
                "Recall (small)": pc["small"]["recall"],
                "Recall (fit)": pc["fit"]["recall"],
                "Recall (large)": pc["large"]["recall"],
            }
        )

    df_summary = pd.DataFrame(rows)
    return df_summary
