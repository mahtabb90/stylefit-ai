"""Reproducible, leakage-safe model comparison for StyleFit AI.

This module deliberately stops before final-holdout evaluation, threshold tuning,
or model serialization. Candidate selection is based only on user-disjoint
cross-validation within the development partition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from src.cleaning import (
    GENERAL_FIT_FEATURES,
    TARGET_COLUMN,
    clean_dataset,
    remove_exact_duplicates,
)
from src.dataset_audit import DATA_PATH, load_dataset
from src.evaluation import (
    CLASS_ORDER,
    align_probability_columns,
    evaluate_predictions,
)
from src.model_comparison_visualization import save_comparison_figures
from src.preprocessing import build_preprocessor, select_general_fit_features
from src.splitting import make_group_cv_splits, unseen_user_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_comparison"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "model_comparison"
RANDOM_STATE = 42

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_pr_auc",
    "macro_roc_auc",
    *[
        f"{metric}_{class_name}"
        for class_name in CLASS_ORDER
        for metric in ("precision", "recall", "f1", "pr_auc", "roc_auc")
    ],
]


@dataclass(frozen=True)
class ComparisonConfig:
    """All choices that affect data partitioning or model comparison."""

    random_state: int = RANDOM_STATE
    holdout_n_splits: int = 5
    cv_n_splits: int = 3
    min_category_frequency: int = 100
    output_dir: Path = DEFAULT_OUTPUT_DIR
    figures_dir: Path = DEFAULT_FIGURES_DIR


@dataclass(frozen=True)
class ModelSpec:
    """Inspectable model and preprocessing configuration."""

    name: str
    role: str
    factory: Callable[[], Any]
    parameters: Dict[str, Any]
    size_strategy: Optional[str] = None
    scale_numeric: bool = False
    balanced_sample_weight: bool = False

    @property
    def preprocessing_key(self) -> Optional[Tuple[str, bool]]:
        if self.size_strategy is None:
            return None
        return self.size_strategy, self.scale_numeric


class FixedClassXGBClassifier:
    """Adapt XGBoost's numeric targets to StyleFit's fixed string class order."""

    def __init__(self, model: Any):
        self.model = model
        self.classes_ = np.asarray(CLASS_ORDER)

    def fit(self, X: Any, y: Sequence[str], sample_weight: Optional[np.ndarray] = None):
        class_to_index = {class_name: index for index, class_name in enumerate(CLASS_ORDER)}
        encoded_y = np.asarray([class_to_index[str(value)] for value in y])
        self.model.fit(X, encoded_y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded = self.model.predict(X).astype(int)
        return self.classes_[encoded]

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)


def build_model_specs(
    random_state: int = RANDOM_STATE,
    include_xgboost: bool = True,
) -> List[ModelSpec]:
    """Return the small, pre-declared model set used for comparison."""
    logistic_common = {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 500,
        "random_state": random_state,
    }
    specs = [
        ModelSpec(
            name="Dummy (Most Frequent)",
            role="reference",
            factory=lambda: DummyClassifier(strategy="most_frequent"),
            parameters={"strategy": "most_frequent"},
        ),
        ModelSpec(
            name="Logistic Regression (Unweighted, Numeric Size)",
            role="reference",
            factory=lambda: LogisticRegression(class_weight=None, **logistic_common),
            parameters={**logistic_common, "class_weight": None},
            size_strategy="numeric",
            scale_numeric=True,
        ),
        ModelSpec(
            name="Logistic Regression (Unweighted, Categorical Size)",
            role="reference",
            factory=lambda: LogisticRegression(class_weight=None, **logistic_common),
            parameters={**logistic_common, "class_weight": None},
            size_strategy="categorical",
            scale_numeric=True,
        ),
        ModelSpec(
            name="Logistic Regression (Balanced)",
            role="candidate",
            factory=lambda: LogisticRegression(class_weight="balanced", **logistic_common),
            parameters={**logistic_common, "class_weight": "balanced"},
            size_strategy="categorical",
            scale_numeric=True,
        ),
        ModelSpec(
            name="Random Forest (Balanced)",
            role="candidate",
            factory=lambda: RandomForestClassifier(
                n_estimators=180,
                max_depth=18,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            parameters={
                "n_estimators": 180,
                "max_depth": 18,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
                "random_state": random_state,
            },
            size_strategy="categorical",
        ),
    ]

    if include_xgboost:
        xgb_parameters = {
            "objective": "multi:softprob",
            "num_class": len(CLASS_ORDER),
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.08,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "n_jobs": -1,
            "random_state": random_state,
        }

        def build_xgboost() -> FixedClassXGBClassifier:
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise RuntimeError(
                    "XGBoost is required for the default comparison. Install the "
                    "ML dependencies or pass --skip-xgboost."
                ) from exc
            return FixedClassXGBClassifier(XGBClassifier(**xgb_parameters))

        specs.append(
            ModelSpec(
                name="XGBoost (Balanced Sample Weights)",
                role="candidate",
                factory=build_xgboost,
                parameters=xgb_parameters,
                size_strategy="categorical",
                balanced_sample_weight=True,
            )
        )

    return specs


def prepare_development_data(
    raw_df: pd.DataFrame,
    config: ComparisonConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Clean data and isolate an untouched, user-disjoint final holdout.

    Only development features, labels, and groups are returned. Holdout labels and
    features remain local to this function and are never exposed to model selection.
    """
    cleaned = remove_exact_duplicates(clean_dataset(raw_df))
    approved_features = select_general_fit_features(cleaned)
    split_frame = approved_features.copy()
    split_frame[TARGET_COLUMN] = cleaned[TARGET_COLUMN]
    split_frame["user_id"] = cleaned["user_id"]

    (
        development_frame,
        holdout_frame,
        development_y,
        holdout_y,
        development_users,
        holdout_users,
    ) = unseen_user_split(
        split_frame,
        user_col="user_id",
        target_col=TARGET_COLUMN,
        n_splits=config.holdout_n_splits,
        random_state=config.random_state,
    )

    if set(development_users).intersection(set(holdout_users)):
        raise RuntimeError("Final holdout contains users present in development data.")

    development_X = development_frame[GENERAL_FIT_FEATURES].copy()
    holdout_manifest = {
        "status": "sealed_not_evaluated",
        "selection_role": "none",
        "split_method": "StratifiedGroupKFold candidate fold",
        "rows": int(len(holdout_frame)),
        "unique_users": int(pd.Series(holdout_users).nunique()),
        "development_rows": int(len(development_X)),
        "development_unique_users": int(pd.Series(development_users).nunique()),
        "all_classes_present": set(holdout_y.unique()) == set(CLASS_ORDER),
    }
    return development_X, development_y, development_users, holdout_manifest


def _flatten_fold_result(
    evaluation: Dict[str, Any],
    spec: ModelSpec,
    fold: int,
    train_rows: int,
    validation_rows: int,
    train_groups: int,
    validation_groups: int,
    fit_seconds: float,
) -> Dict[str, Any]:
    overall = evaluation["overall"]
    per_class = evaluation["per_class"]
    probability = evaluation["probability_metrics"] or {}
    row: Dict[str, Any] = {
        "model": spec.name,
        "role": spec.role,
        "fold": fold,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "fit_seconds": fit_seconds,
        **overall,
        "macro_pr_auc": probability.get("macro_pr_auc", np.nan),
        "macro_roc_auc": probability.get("macro_roc_auc", np.nan),
    }
    for class_name in CLASS_ORDER:
        row[f"precision_{class_name}"] = per_class[class_name]["precision"]
        row[f"recall_{class_name}"] = per_class[class_name]["recall"]
        row[f"f1_{class_name}"] = per_class[class_name]["f1"]
        row[f"pr_auc_{class_name}"] = probability.get("per_class_pr_auc", {}).get(
            class_name, np.nan
        )
        row[f"roc_auc_{class_name}"] = probability.get("per_class_roc_auc", {}).get(
            class_name, np.nan
        )
    row["confusion_matrix"] = evaluation["confusion_matrix"].tolist()
    return row


def aggregate_fold_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics into a stable, selection-ready result schema."""
    missing = set(["model", "role", *METRIC_COLUMNS]) - set(fold_results.columns)
    if missing:
        raise ValueError(f"Fold results are missing required columns: {sorted(missing)}")

    rows = []
    for (model_name, role), model_rows in fold_results.groupby(
        ["model", "role"], sort=False
    ):
        row: Dict[str, Any] = {
            "model": model_name,
            "role": role,
            "n_folds": int(len(model_rows)),
            "fit_seconds_total": float(model_rows["fit_seconds"].sum()),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(model_rows[metric].mean())
            row[f"{metric}_std"] = float(model_rows[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean"], ascending=False
    ).reset_index(drop=True)


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_specs: Sequence[ModelSpec],
    config: ComparisonConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[List[int]]]]:
    """Evaluate every model on identical user-disjoint validation folds."""
    if list(X.columns) != GENERAL_FIT_FEATURES:
        raise ValueError("Cross-validation received features outside the approved allowlist.")
    if set(pd.Series(y).unique()) != set(CLASS_ORDER):
        raise ValueError(f"Target classes must be exactly {CLASS_ORDER}.")

    splits = make_group_cv_splits(
        X,
        y,
        groups,
        n_splits=config.cv_n_splits,
        random_state=config.random_state,
    )
    fold_rows: List[Dict[str, Any]] = []
    confusion_totals = {
        spec.name: np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)
        for spec in model_specs
    }

    for fold_number, (train_idx, validation_idx) in enumerate(splits, start=1):
        X_train, X_validation = X.iloc[train_idx], X.iloc[validation_idx]
        y_train, y_validation = y.iloc[train_idx], y.iloc[validation_idx]
        train_groups = groups.iloc[train_idx]
        validation_groups = groups.iloc[validation_idx]
        transformed: Dict[Optional[Tuple[str, bool]], Tuple[Any, Any]] = {None: (X_train, X_validation)}

        preprocessing_keys = sorted(
            {spec.preprocessing_key for spec in model_specs if spec.preprocessing_key}
        )
        for key in preprocessing_keys:
            size_strategy, scale_numeric = key
            preprocessor = build_preprocessor(
                scale_numeric=scale_numeric,
                size_strategy=size_strategy,
                min_frequency=config.min_category_frequency,
                sparse_output=True,
            )
            transformed[key] = (
                preprocessor.fit_transform(X_train),
                preprocessor.transform(X_validation),
            )

        for spec in model_specs:
            model = spec.factory()
            X_train_model, X_validation_model = transformed[spec.preprocessing_key]
            sample_weight = (
                compute_sample_weight(class_weight="balanced", y=y_train)
                if spec.balanced_sample_weight
                else None
            )
            started = perf_counter()
            fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
            model.fit(X_train_model, y_train, **fit_kwargs)
            predictions = model.predict(X_validation_model)
            probabilities = align_probability_columns(
                model.predict_proba(X_validation_model),
                model.classes_,
                CLASS_ORDER,
            )
            fit_seconds = perf_counter() - started
            evaluation = evaluate_predictions(
                y_true=y_validation,
                y_pred=predictions,
                aligned_proba=probabilities,
                model_name=spec.name,
                split_type=f"group_cv_fold_{fold_number}",
            )
            confusion_totals[spec.name] += evaluation["confusion_matrix"]
            fold_rows.append(
                _flatten_fold_result(
                    evaluation,
                    spec,
                    fold_number,
                    len(train_idx),
                    len(validation_idx),
                    train_groups.nunique(),
                    validation_groups.nunique(),
                    fit_seconds,
                )
            )
            print(
                f"[{fold_number}/{config.cv_n_splits}] {spec.name}: "
                f"macro_f1={evaluation['overall']['macro_f1']:.4f}, "
                f"balanced_accuracy={evaluation['overall']['balanced_accuracy']:.4f}"
            )

    fold_results = pd.DataFrame(fold_rows)
    summary = aggregate_fold_results(fold_results)
    confusion_json = {name: matrix.tolist() for name, matrix in confusion_totals.items()}
    return fold_results, summary, confusion_json


def select_candidate(summary: pd.DataFrame) -> str:
    """Select the strongest non-reference model by Macro F1, then balanced accuracy."""
    candidates = summary.loc[summary["role"] == "candidate"]
    if candidates.empty:
        raise ValueError("At least one candidate model is required for selection.")
    ordered = candidates.sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean"], ascending=False
    )
    return str(ordered.iloc[0]["model"])


def save_results(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    confusion_matrices: Dict[str, List[List[int]]],
    model_specs: Sequence[ModelSpec],
    holdout_manifest: Dict[str, Any],
    selected_model: str,
    config: ComparisonConfig,
) -> Dict[str, Path]:
    """Persist machine-readable results and professional comparison figures."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    fold_csv = config.output_dir / "model_comparison_fold_metrics.csv"
    summary_csv = config.output_dir / "model_comparison_summary.csv"
    results_json = config.output_dir / "model_comparison_results.json"
    fold_results.drop(columns=["confusion_matrix"]).to_csv(fold_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    def portable_path(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "macro_f1_mean",
        "secondary_selection_metric": "balanced_accuracy_mean",
        "class_order": CLASS_ORDER,
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "config": {
            **asdict(config),
            "output_dir": portable_path(config.output_dir),
            "figures_dir": portable_path(config.figures_dir),
        },
        "holdout": holdout_manifest,
        "models": [
            {
                "name": spec.name,
                "role": spec.role,
                "parameters": spec.parameters,
                "size_strategy": spec.size_strategy,
                "scale_numeric": spec.scale_numeric,
                "balanced_sample_weight": spec.balanced_sample_weight,
            }
            for spec in model_specs
        ],
        "fold_results": fold_results.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "aggregate_confusion_matrices": confusion_matrices,
        "selected_candidate": selected_model,
        "final_holdout_evaluated": False,
    }
    results_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    figure_paths = save_comparison_figures(
        summary,
        confusion_matrices[selected_model],
        selected_model,
        config.figures_dir,
    )
    return {
        "fold_metrics": fold_csv,
        "summary": summary_csv,
        "results": results_json,
        **figure_paths,
    }


def run_model_comparison(
    config: ComparisonConfig,
    include_xgboost: bool = True,
) -> Tuple[pd.DataFrame, str, Dict[str, Path]]:
    """Run the complete comparison while leaving the final holdout untouched."""
    print(f"Loading Rent the Runway data from {DATA_PATH}")
    raw_df = load_dataset()
    X, y, groups, holdout_manifest = prepare_development_data(raw_df, config)
    print(
        f"Development partition: {len(X):,} rows, {groups.nunique():,} users; "
        f"sealed holdout: {holdout_manifest['rows']:,} rows"
    )
    specs = build_model_specs(config.random_state, include_xgboost=include_xgboost)
    fold_results, summary, confusion_matrices = run_cross_validation(
        X, y, groups, specs, config
    )
    selected_model = select_candidate(summary)
    paths = save_results(
        fold_results,
        summary,
        confusion_matrices,
        specs,
        holdout_manifest,
        selected_model,
        config,
    )
    print("\nCross-validation summary (sorted by Macro F1):")
    print(
        summary[
            [
                "model",
                "macro_f1_mean",
                "balanced_accuracy_mean",
                "macro_pr_auc_mean",
                "recall_small_mean",
                "recall_large_mean",
            ]
        ].to_string(index=False)
    )
    print(f"\nSelected candidate: {selected_model}")
    print("Final holdout status: sealed and not evaluated")
    return summary, selected_model, paths


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--holdout-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--min-category-frequency", type=int, default=100)
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Run a reduced comparison when XGBoost is unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    config = ComparisonConfig(
        random_state=args.random_state,
        holdout_n_splits=args.holdout_folds,
        cv_n_splits=args.cv_folds,
        min_category_frequency=args.min_category_frequency,
    )
    run_model_comparison(config, include_xgboost=not args.skip_xgboost)


if __name__ == "__main__":
    main()
