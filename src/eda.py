"""Exploratory Data Analysis (EDA) module for StyleFit AI.

Provides reusable data analysis functions, in-memory height/weight parsing,
EDA summary calculations, and command-line execution (`python3 -m src.eda`).
"""

from pathlib import Path
import re
from typing import Dict, Tuple

import pandas as pd

from src.visualization import (
    plot_age_distribution,
    plot_category_distribution,
    plot_fit_by_categorical_factors,
    plot_missing_values,
    plot_physical_distributions,
    plot_rented_for_body_type,
    plot_size_distribution,
    plot_target_distribution,
    plot_user_item_sparsity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "renttherunway_final_data.json.gz"


def load_dataset() -> pd.DataFrame:
    """Load raw Rent the Runway dataset in read-only mode."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_json(
        DATA_PATH,
        lines=True,
        compression="gzip",
    )


def parse_height_inches(series: pd.Series) -> pd.Series:
    """Parse height strings (e.g. "5' 8\\"") in-memory to integer total inches.

    Note: Invalid or unexpected formats safely return NaN. Raw dataset is NOT
    modified.
    """

    def _parse(val):
        if pd.isna(val):
            return None
        m = re.match(r"^\s*(\d+)'\s*(\d+)\"\s*$", str(val))
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        return None

    return series.apply(_parse)


def parse_weight_lbs(series: pd.Series) -> pd.Series:
    """Parse weight strings (e.g. "137lbs") in-memory to integer lbs.

    Note: Invalid or unexpected formats safely return NaN. Raw dataset is NOT
    modified.
    """

    def _parse(val):
        if pd.isna(val):
            return None
        m = re.match(r"^\s*(\d+)\s*lbs\s*$", str(val), re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    return series.apply(_parse)


def get_parsing_report(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute in-memory parsing report for height and weight columns."""
    h_parsed = parse_height_inches(df["height"])
    w_parsed = parse_weight_lbs(df["weight"])

    h_total = df["height"].notna().sum()
    h_valid = h_parsed.notna().sum()
    h_failed = h_total - h_valid

    w_total = df["weight"].notna().sum()
    w_valid = w_parsed.notna().sum()
    w_failed = w_total - w_valid

    report = {
        "height": {
            "total_non_null": int(h_total),
            "parsed_successful": int(h_valid),
            "parsed_failed": int(h_failed),
            "min_inches": float(h_parsed.min()) if h_valid > 0 else 0.0,
            "max_inches": float(h_parsed.max()) if h_valid > 0 else 0.0,
        },
        "weight": {
            "total_non_null": int(w_total),
            "parsed_successful": int(w_valid),
            "parsed_failed": int(w_failed),
            "min_lbs": float(w_parsed.min()) if w_valid > 0 else 0.0,
            "max_lbs": float(w_parsed.max()) if w_valid > 0 else 0.0,
        },
    }
    return report


def run_eda_pipeline() -> Tuple[pd.DataFrame, list[Path]]:
    """Run full read-only EDA analysis and generate all 9 visual figures."""
    print("=" * 60)
    print("STYLEFIT AI - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    df = load_dataset()
    print(f"Dataset Loaded Successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # In-memory parsing
    h_parsed = parse_height_inches(df["height"])
    w_parsed = parse_weight_lbs(df["weight"])

    parse_report = get_parsing_report(df)
    print("\n--- In-Memory Height/Weight Parsing Report ---")
    print(
        f"Height - Total: {parse_report['height']['total_non_null']:,}, "
        f"Parsed: {parse_report['height']['parsed_successful']:,}, "
        f"Failed: {parse_report['height']['parsed_failed']}, "
        f"Min: {int(parse_report['height']['min_inches'])}in, "
        f"Max: {int(parse_report['height']['max_inches'])}in"
    )
    print(
        f"Weight - Total: {parse_report['weight']['total_non_null']:,}, "
        f"Parsed: {parse_report['weight']['parsed_successful']:,}, "
        f"Failed: {parse_report['weight']['parsed_failed']}, "
        f"Min: {int(parse_report['weight']['min_lbs'])}lbs, "
        f"Max: {int(parse_report['weight']['max_lbs'])}lbs"
    )

    print("\n--- Generating Visualizations ---")
    saved_paths = []

    p1 = plot_target_distribution(df)
    saved_paths.append(p1)

    p2 = plot_missing_values(df)
    saved_paths.append(p2)

    p3 = plot_age_distribution(df)
    saved_paths.append(p3)

    p4 = plot_size_distribution(df)
    saved_paths.append(p4)

    p5 = plot_physical_distributions(df, h_parsed, w_parsed)
    saved_paths.append(p5)

    p6 = plot_category_distribution(df)
    saved_paths.append(p6)

    p7 = plot_rented_for_body_type(df)
    saved_paths.append(p7)

    p8 = plot_fit_by_categorical_factors(df)
    saved_paths.append(p8)

    p9 = plot_user_item_sparsity(df)
    saved_paths.append(p9)

    print("\nSaved Figures List:")
    for i, path in enumerate(saved_paths, start=1):
        print(f"[{i}/9] Saved figure: {path}")

    print("\n" + "=" * 60)
    print("EDA Pipeline Completed Successfully")
    print("=" * 60)

    return df, saved_paths


def main() -> None:
    run_eda_pipeline()


if __name__ == "__main__":
    main()
