"""Visualization module for StyleFit AI Exploratory Data Analysis.

Generates publication-quality, read-only Matplotlib charts and saves them to reports/figures/.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Ensure output directory exists
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Custom color palette for StyleFit AI
COLORS = {
    "fit": "#2ca02c",  # Green
    "small": "#ff7f0e",  # Orange
    "large": "#1f77b4",  # Blue
    "primary": "#333333",
    "accent": "#d62728",
    "gray": "#7f7f7f",
    "light_gray": "#e0e0e0",
}


def setup_matplotlib_style() -> None:
    """Set global matplotlib style parameters for clean, publication-ready figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
        }
    )


def plot_target_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plot fit target distribution (small, fit, large)."""
    if output_path is None:
        output_path = FIGURES_DIR / "01_target_distribution.png"

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df["fit"].value_counts()
    percentages = df["fit"].value_counts(normalize=True) * 100

    labels = counts.index.tolist()
    color_list = [COLORS.get(lbl, COLORS["primary"]) for lbl in labels]

    bars = ax.bar(labels, counts.values, color=color_list, width=0.5, edgecolor="black", alpha=0.85)

    for bar, pct in zip(bars, percentages.values):
        height = bar.get_height()
        ax.annotate(
            f"{height:,}\n({pct:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Target Distribution ('fit') - Pre-Purchase Classification Goal")
    ax.set_xlabel("Fit Outcome")
    ax.set_ylabel("Transaction Count")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_missing_values(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plot NaN percentages per column, highlighting post-purchase leakage fields."""
    if output_path is None:
        output_path = FIGURES_DIR / "02_missing_values.png"

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    missing_pct = (df.isna().mean() * 100).sort_values(ascending=True)

    # Differentiate leakage features
    leakage_cols = {"rating", "review_text", "review_summary"}
    colors = [
        COLORS["accent"] if col in leakage_cols else (COLORS["primary"] if pct > 0 else COLORS["gray"])
        for col, pct in zip(missing_pct.index, missing_pct.values)
    ]

    bars = ax.barh(missing_pct.index, missing_pct.values, color=colors, edgecolor="black", alpha=0.85)

    for bar, val, col in zip(bars, missing_pct.values, missing_pct.index):
        width = bar.get_width()
        label = f"{val:.2f}%"
        if col in leakage_cols:
            label += " [Leakage Field - Post-Purchase]"
        ax.annotate(
            label,
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=COLORS["accent"] if col in leakage_cols else "black",
        )

    # Note on whitespace-only strings
    note_text = (
        "Missingness Definition Note:\n"
        "Bar chart displays NaN percentages.\n"
        "Whitespace-only strings evaluated separately:\n"
        "  - review_text: 2 rows\n"
        "  - review_summary: 16 rows"
    )
    ax.text(
        0.97,
        0.25,
        note_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_title("NaN Percentage per Column")
    ax.set_xlabel("Missing NaN Percentage (%)")
    ax.set_ylabel("Column Name")
    ax.set_xlim(0, max(missing_pct.values) * 1.45 if max(missing_pct.values) > 0 else 10)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_age_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plot customer age histogram (focusing on ages 16-90 for plot legibility).

    Note: Annotates missing ages, ages < 16, and ages > 90. Clearly states that
    filtering is visualization-only and no dataset rows are removed.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "03_age_distribution.png"

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    missing_count = df["age"].isna().sum()
    below_16_count = (df["age"] < 16).sum()
    above_90_count = (df["age"] > 90).sum()

    valid_ages = df["age"].dropna()
    visible_ages = valid_ages[(valid_ages >= 16) & (valid_ages <= 90)]

    ax.hist(
        visible_ages,
        bins=35,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.75,
    )

    stats_text = (
        "Visualization Range Note:\n"
        "Histogram displays ages 16-90 for visual clarity.\n"
        "NO dataset rows have been removed.\n\n"
        f"Missing ages: {missing_count:,}\n"
        f"Ages < 16: {below_16_count:,}\n"
        f"Ages > 90: {above_90_count:,}\n"
        f"Displayed (16-90): {len(visible_ages):,}"
    )

    ax.text(
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_title("Customer Age Distribution (Histogram)")
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Transaction Frequency")
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_size_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plot size frequency distribution using logarithmic y-axis for high legibility."""
    if output_path is None:
        output_path = FIGURES_DIR / "04_size_distribution.png"

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    size_counts = df["size"].value_counts().sort_index()
    x_positions = np.arange(len(size_counts))

    bars = ax.bar(
        x_positions,
        size_counts.values,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.75,
        log=True,
    )

    # Neutral annotation for size 0
    if 0 in size_counts.index:
        idx_0 = list(size_counts.index).index(0)
        bars[idx_0].set_color("#ff7f0e")
        bars[idx_0].set_edgecolor("black")
        ax.annotate(
            f"Size value 0 ({size_counts[0]:,} rows)\nPresent & not automatically treated as invalid",
            xy=(idx_0, size_counts[0]),
            xytext=(15, 30),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.5),
        )

    ax.set_title("Selected Size Value Distribution (Logarithmic Scale)")
    ax.set_xlabel("Selected Size Value")
    ax.set_ylabel("Transaction Count (Log Scale)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(s) for s in size_counts.index], rotation=90, fontsize=8)
    ax.set_ylim(0.8, max(size_counts.values) * 3)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_physical_distributions(
    df: pd.DataFrame,
    h_parsed: pd.Series,
    w_parsed: pd.Series,
    output_path: Optional[Path] = None,
) -> Path:
    """Plot in-memory parsed height (inches) and weight (lbs) distributions."""
    if output_path is None:
        output_path = FIGURES_DIR / "05_height_weight_distributions.png"

    setup_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Height Subplot
    h_valid = h_parsed.dropna()
    h_missing = df["height"].isna().sum()
    h_failed = df["height"].notna().sum() - len(h_valid)

    ax1.hist(h_valid, bins=25, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax1.set_title("Height Distribution (In-Memory Parsed)")
    ax1.set_xlabel("Height (Inches)")
    ax1.set_ylabel("Count")

    h_info = (
        f"Total Non-Null: {df['height'].notna().sum():,}\n"
        f"Parsed Valid: {len(h_valid):,}\n"
        f"Failed Parse: {h_failed}\n"
        f"Missing (NaN): {h_missing:,}\n"
        f"Min: {int(h_valid.min())} in ({int(h_valid.min()//12)}'{int(h_valid.min()%12)}\")\n"
        f"Max: {int(h_valid.max())} in ({int(h_valid.max()//12)}'{int(h_valid.max()%12)}\")"
    )
    ax1.text(
        0.03,
        0.95,
        h_info,
        transform=ax1.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # Weight Subplot
    w_valid = w_parsed.dropna()
    w_missing = df["weight"].isna().sum()
    w_failed = df["weight"].notna().sum() - len(w_valid)

    ax2.hist(w_valid, bins=30, color="#2ca02c", edgecolor="black", alpha=0.75)
    ax2.set_title("Weight Distribution (In-Memory Parsed)")
    ax2.set_xlabel("Weight (lbs)")
    ax2.set_ylabel("Count")

    w_info = (
        f"Total Non-Null: {df['weight'].notna().sum():,}\n"
        f"Parsed Valid: {len(w_valid):,}\n"
        f"Failed Parse: {w_failed}\n"
        f"Missing (NaN): {w_missing:,}\n"
        f"Min: {int(w_valid.min())} lbs\n"
        f"Max: {int(w_valid.max())} lbs"
    )
    ax2.text(
        0.97,
        0.95,
        w_info,
        transform=ax2.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_category_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plot top clothing categories and long-tail category counts."""
    if output_path is None:
        output_path = FIGURES_DIR / "06_category_distribution.png"

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    cat_counts = df["category"].value_counts()
    top_15 = cat_counts.head(15)

    bars = ax.bar(top_15.index, top_15.values, color="#1f77b4", edgecolor="black", alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fewer_10 = (cat_counts < 10).sum()
    fewer_100 = (cat_counts < 100).sum()
    total_cats = len(cat_counts)

    info_text = (
        f"Category Summary:\n"
        f"Total Unique Categories: {total_cats}\n"
        f"Categories < 100 rows: {fewer_100}\n"
        f"Categories < 10 rows: {fewer_10}"
    )
    ax.text(
        0.97,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_title("Top 15 Clothing Categories & Long-Tail Overview")
    ax.set_xlabel("Category")
    ax.set_ylabel("Transaction Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rented_for_body_type(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plot rental occasion (rented for) and body type distributions."""
    if output_path is None:
        output_path = FIGURES_DIR / "07_rented_for_body_type.png"

    setup_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Rented for
    rf_counts = df["rented for"].value_counts(dropna=False)
    rf_labels = [str(k) if pd.notna(k) else "Missing (NaN)" for k in rf_counts.index]

    bars1 = ax1.barh(rf_labels, rf_counts.values, color="#9467bd", edgecolor="black", alpha=0.85)

    for bar, label, val in zip(bars1, rf_labels, rf_counts.values):
        ax1.annotate(
            f"{val:,}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
        )
        if label == "party: cocktail":
            ax1.annotate(
                "Anomaly (count=1)",
                xy=(bar.get_width() + 1500, bar.get_y() + bar.get_height() / 2),
                fontsize=8.5,
                color="red",
                weight="bold",
                va="center",
            )

    ax1.set_title("Rental Occasion Distribution ('rented for')")
    ax1.set_xlabel("Transaction Count")
    ax1.set_xlim(0, max(rf_counts.values) * 1.18)

    # Body type
    bt_counts = df["body type"].value_counts(dropna=False)
    bt_labels = [str(k) if pd.notna(k) else "Missing (NaN)" for k in bt_counts.index]

    bars2 = ax2.barh(bt_labels, bt_counts.values, color="#8c564b", edgecolor="black", alpha=0.85)

    for bar, val in zip(bars2, bt_counts.values):
        ax2.annotate(
            f"{val:,}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
        )

    ax2.set_title("Body Type Distribution")
    ax2.set_xlabel("Transaction Count")
    ax2.set_xlim(0, max(bt_counts.values) * 1.18)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fit_by_categorical_factors(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plot multi-panel figure of within-group fit proportions.

    - Horizontal 100% stacked bars for Category (N >= 1,000), Rented-For (N >= 100), Body Type (N >= 100).
    - Unsmoothed line plot with markers at actual observed size values for Selected Size (N >= 500).
    """
    if output_path is None:
        output_path = FIGURES_DIR / "08_fit_by_categorical_factors.png"

    setup_matplotlib_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Category (Horizontal 100% Stacked Bar)
    ax_cat = axes[0, 0]
    cat_counts = df["category"].value_counts()
    valid_cats = cat_counts[cat_counts >= 1000].index.tolist()
    sub_cat = df[df["category"].isin(valid_cats)]
    ct_cat = pd.crosstab(sub_cat["category"], sub_cat["fit"], normalize="index") * 100
    for c in ["small", "fit", "large"]:
        if c not in ct_cat.columns:
            ct_cat[c] = 0.0
    ct_cat = ct_cat[["small", "fit", "large"]].loc[valid_cats]  # sort by frequency

    y_labels_cat = [f"{cat} (N={cat_counts[cat]:,})" for cat in ct_cat.index]
    left = np.zeros(len(ct_cat))
    for c in ["small", "fit", "large"]:
        vals = ct_cat[c].values
        ax_cat.barh(y_labels_cat, vals, left=left, label=c, color=COLORS[c], edgecolor="black", alpha=0.85)
        left += vals

    ax_cat.set_title("Fit Outcome Proportions by Category (N >= 1,000)", pad=25)
    ax_cat.set_xlabel("Within-Group Proportion (%)")
    ax_cat.set_xlim(0, 100)
    ax_cat.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, title="Fit Outcome", frameon=True)

    # 2. Selected Size (Line Plot with Markers at Actual Observed Sizes, N >= 500)
    ax_size = axes[0, 1]
    size_counts = df["size"].value_counts()
    valid_sizes = sorted([int(s) for s in size_counts[size_counts >= 500].index])
    sub_size = df[df["size"].isin(valid_sizes)]
    ct_size = pd.crosstab(sub_size["size"], sub_size["fit"], normalize="index") * 100
    for c in ["small", "fit", "large"]:
        if c not in ct_size.columns:
            ct_size[c] = 0.0
    ct_size = ct_size.loc[valid_sizes]

    # Plot lines with markers connecting actual observed sizes without smoothing
    x_positions = np.arange(len(valid_sizes))
    ax_size.plot(x_positions, ct_size["fit"].values, marker="o", linewidth=2, color=COLORS["fit"], label="fit")
    ax_size.plot(x_positions, ct_size["small"].values, marker="s", linewidth=2, color=COLORS["small"], label="small")
    ax_size.plot(x_positions, ct_size["large"].values, marker="^", linewidth=2, color=COLORS["large"], label="large")

    ax_size.set_title("Fit Outcome Trends by Selected Size (N >= 500)")
    ax_size.set_xlabel("Selected Size Value")
    ax_size.set_ylabel("Within-Group Proportion (%)")
    ax_size.set_xticks(x_positions)
    ax_size.set_xticklabels([str(s) for s in valid_sizes], rotation=90, fontsize=8)
    ax_size.set_ylim(0, 100)
    ax_size.legend(loc="upper right", title="Fit Outcome")
    ax_size.text(
        0.03,
        0.05,
        "Note: Threshold N >= 500 applied.\nMarkers reflect actual observed size values.\nNo line smoothing or interpolation applied.",
        transform=ax_size.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # 3. Rented For (Horizontal 100% Stacked Bar)
    ax_rf = axes[1, 0]
    rf_counts = df["rented for"].value_counts().dropna()
    valid_rf = rf_counts[rf_counts >= 100].index.tolist()
    sub_rf = df[df["rented for"].isin(valid_rf)]
    ct_rf = pd.crosstab(sub_rf["rented for"], sub_rf["fit"], normalize="index") * 100
    for c in ["small", "fit", "large"]:
        if c not in ct_rf.columns:
            ct_rf[c] = 0.0
    ct_rf = ct_rf[["small", "fit", "large"]].loc[valid_rf]

    y_labels_rf = [f"{rf} (N={rf_counts[rf]:,})" for rf in ct_rf.index]
    left = np.zeros(len(ct_rf))
    for c in ["small", "fit", "large"]:
        vals = ct_rf[c].values
        ax_rf.barh(y_labels_rf, vals, left=left, label=c, color=COLORS[c], edgecolor="black", alpha=0.85)
        left += vals

    ax_rf.set_title("Fit Outcome Proportions by Rental Occasion (N >= 100)", pad=25)
    ax_rf.set_xlabel("Within-Group Proportion (%)")
    ax_rf.set_xlim(0, 100)
    ax_rf.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, title="Fit Outcome", frameon=True)

    # 4. Body Type (Horizontal 100% Stacked Bar)
    ax_bt = axes[1, 1]
    bt_counts = df["body type"].value_counts().dropna()
    valid_bt = bt_counts[bt_counts >= 100].index.tolist()
    sub_bt = df[df["body type"].isin(valid_bt)]
    ct_bt = pd.crosstab(sub_bt["body type"], sub_bt["fit"], normalize="index") * 100
    for c in ["small", "fit", "large"]:
        if c not in ct_bt.columns:
            ct_bt[c] = 0.0
    ct_bt = ct_bt[["small", "fit", "large"]].loc[valid_bt]

    y_labels_bt = [f"{bt} (N={bt_counts[bt]:,})" for bt in ct_bt.index]
    left = np.zeros(len(ct_bt))
    for c in ["small", "fit", "large"]:
        vals = ct_bt[c].values
        ax_bt.barh(y_labels_bt, vals, left=left, label=c, color=COLORS[c], edgecolor="black", alpha=0.85)
        left += vals

    ax_bt.set_title("Fit Outcome Proportions by Body Type (N >= 100)", pad=25)
    ax_bt.set_xlabel("Within-Group Proportion (%)")
    ax_bt.set_xlim(0, 100)
    ax_bt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, title="Fit Outcome", frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_user_item_sparsity(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plot binned user and item transaction frequency distributions."""
    if output_path is None:
        output_path = FIGURES_DIR / "09_user_item_sparsity.png"

    setup_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bins = [0, 1, 2, 5, 10, np.inf]
    bin_labels = ["1", "2", "3-5", "6-10", "11+"]

    # User Sparsity
    user_counts = df["user_id"].value_counts()
    user_binned = pd.cut(user_counts, bins=bins, labels=bin_labels).value_counts().loc[bin_labels]

    bars1 = ax1.bar(bin_labels, user_binned.values, color="#1f77b4", edgecolor="black", alpha=0.85)
    for bar in bars1:
        h = bar.get_height()
        pct = (h / len(user_counts)) * 100
        ax1.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax1.set_title("User Transaction Sparsity (Binned Counts)")
    ax1.set_xlabel("Transactions per User Bin")
    ax1.set_ylabel("User Count")
    ax1.set_ylim(0, max(user_binned.values) * 1.18)

    single_user_pct = (user_counts == 1).mean() * 100
    u_info = (
        f"Unique Users: {len(user_counts):,}\n"
        f"Single-Tx Users: {(user_counts == 1).sum():,} ({single_user_pct:.1f}%)\n"
        f"Mean Tx/User: {user_counts.mean():.2f}\n"
        f"Max Tx/User: {user_counts.max():,}"
    )
    ax1.text(
        0.97,
        0.95,
        u_info,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # Item Sparsity
    item_counts = df["item_id"].value_counts()
    item_binned = pd.cut(item_counts, bins=bins, labels=bin_labels).value_counts().loc[bin_labels]

    bars2 = ax2.bar(bin_labels, item_binned.values, color="#2ca02c", edgecolor="black", alpha=0.85)
    for bar in bars2:
        h = bar.get_height()
        pct = (h / len(item_counts)) * 100
        ax2.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax2.set_title("Item Transaction Sparsity (Binned Counts)")
    ax2.set_xlabel("Transactions per Item Bin")
    ax2.set_ylabel("Item Count")
    ax2.set_ylim(0, max(item_binned.values) * 1.18)

    single_item_pct = (item_counts == 1).mean() * 100
    i_info = (
        f"Unique Items: {len(item_counts):,}\n"
        f"Single-Tx Items: {(item_counts == 1).sum():,} ({single_item_pct:.1f}%)\n"
        f"Mean Tx/Item: {item_counts.mean():.2f}\n"
        f"Max Tx/Item: {item_counts.max():,}"
    )
    ax2.text(
        0.97,
        0.95,
        i_info,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
