"""Reporting figures for the StyleFit AI model-comparison phase."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import CLASS_ORDER


PRIMARY = "#6C4AB6"
SECONDARY = "#2A9D8F"
REFERENCE = "#8D99AE"
SMALL = "#E76F51"
LARGE = "#457B9D"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFC",
            "axes.edgecolor": "#D9D9E3",
            "axes.titleweight": "bold",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
        }
    )


def _model_colors(summary: pd.DataFrame) -> List[str]:
    best_candidate = summary.loc[summary["role"] == "candidate", "macro_f1_mean"].idxmax()
    return [
        PRIMARY if index == best_candidate else (REFERENCE if role == "reference" else SECONDARY)
        for index, role in zip(summary.index, summary["role"])
    ]


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_comparison_table(summary: pd.DataFrame, output_path: Path) -> Path:
    columns = [
        "model",
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "macro_pr_auc_mean",
        "recall_small_mean",
        "recall_large_mean",
    ]
    display = summary[columns].copy()
    display.columns = [
        "Model",
        "Macro F1",
        "Balanced Acc.",
        "Macro PR-AUC",
        "Recall: small",
        "Recall: large",
    ]
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")

    fig_height = max(3.0, 1.1 + 0.48 * len(display))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title("StyleFit AI — Group-Aware Cross-Validation Model Comparison", pad=18)
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.38, 0.12, 0.14, 0.14, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#E3E3EA")
        if row == 0:
            cell.set_facecolor(PRIMARY)
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F0F8")
        if column == 0 and row > 0:
            cell.set_text_props(ha="left")
    return _save(fig, output_path)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    x_label: str,
    output_path: Path,
) -> Path:
    ordered = summary.sort_values(metric, ascending=True)
    errors = ordered[metric.replace("_mean", "_std")]
    colors = _model_colors(ordered)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.barh(ordered["model"], ordered[metric], xerr=errors, color=colors, capsize=3)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_xlim(left=0)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.text(
        0,
        -0.17,
        "Bars show mean across user-disjoint folds; error bars show fold standard deviation.",
        transform=ax.transAxes,
        color="#5C5C66",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.34, bottom=0.2)
    return _save(fig, output_path)


def plot_minority_recall(summary: pd.DataFrame, output_path: Path) -> Path:
    ordered = summary.sort_values("macro_f1_mean", ascending=False)
    positions = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    small_bars = ax.bar(
        positions - width / 2,
        ordered["recall_small_mean"],
        width,
        yerr=ordered["recall_small_std"],
        label="small",
        color=SMALL,
        capsize=3,
    )
    large_bars = ax.bar(
        positions + width / 2,
        ordered["recall_large_mean"],
        width,
        yerr=ordered["recall_large_std"],
        label="large",
        color=LARGE,
        capsize=3,
    )
    ax.set_title("Minority-Class Recall by Model")
    ax.set_ylabel("Recall")
    ax.set_xticks(positions, ordered["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.bar_label(small_bars, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(large_bars, fmt="%.2f", padding=3, fontsize=8)
    fig.subplots_adjust(bottom=0.28)
    return _save(fig, output_path)


def plot_confusion_matrix(
    matrix: List[List[int]],
    model_name: str,
    output_path: Path,
) -> Path:
    raw = np.asarray(matrix, dtype=float)
    normalized = np.divide(
        raw,
        raw.sum(axis=1, keepdims=True),
        out=np.zeros_like(raw),
        where=raw.sum(axis=1, keepdims=True) != 0,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(normalized, cmap="Purples", vmin=0, vmax=1)
    for row in range(len(CLASS_ORDER)):
        for column in range(len(CLASS_ORDER)):
            value = normalized[row, column]
            ax.text(
                column,
                row,
                f"{value:.1%}\n(n={int(raw[row, column]):,})",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#252532",
                fontsize=10,
            )
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Aggregate CV Confusion Matrix\n{model_name}")
    fig.colorbar(image, ax=ax, label="Row-normalized share")
    return _save(fig, output_path)


def save_comparison_figures(
    summary: pd.DataFrame,
    selected_confusion_matrix: List[List[int]],
    selected_model: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate the standard report bundle for a comparison run."""
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "comparison_table_figure": plot_comparison_table(
            summary, output_dir / "01_model_comparison_table.png"
        ),
        "macro_f1_figure": plot_metric(
            summary,
            "macro_f1_mean",
            "Macro F1 by Model (Primary Selection Metric)",
            "Macro F1",
            output_dir / "02_macro_f1_comparison.png",
        ),
        "balanced_accuracy_figure": plot_metric(
            summary,
            "balanced_accuracy_mean",
            "Balanced Accuracy by Model",
            "Balanced accuracy",
            output_dir / "03_balanced_accuracy_comparison.png",
        ),
        "minority_recall_figure": plot_minority_recall(
            summary, output_dir / "04_minority_class_recall.png"
        ),
        "selected_confusion_matrix_figure": plot_confusion_matrix(
            selected_confusion_matrix,
            selected_model,
            output_dir / "05_selected_model_confusion_matrix.png",
        ),
    }
