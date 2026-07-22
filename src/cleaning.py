"""Deterministic cleaning and parsing module for StyleFit AI dataset.

Provides stateless, row-level functions for parsing attributes, normalizing column names,
sanitizing age outliers, parsing dates, reporting/removing exact duplicates, and cleaning
the raw dataset read-only.
"""

from pathlib import Path
import re
from typing import Any, Dict, Tuple

import pandas as pd


# Column and Feature Constants
RAW_TO_SNAKE_COLUMNS = {
    "bust size": "bust_size",
    "rented for": "rented_for",
    "body type": "body_type",
}

LEAKAGE_COLUMNS = ["rating", "review_text", "review_summary"]
METADATA_COLUMNS = ["user_id", "item_id", "review_date"]
TARGET_COLUMN = "fit"

GENERAL_FIT_FEATURES = [
    "parsed_height_inches",
    "parsed_weight_lbs",
    "bust_band_size",
    "bust_cup_size",
    "body_type",
    "age",
    "size",
    "category",
    "rented_for",
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to snake_case.

    Renames:
      - 'bust size' -> 'bust_size'
      - 'rented for' -> 'rented_for'
      - 'body type' -> 'body_type'
    """
    rename_dict = {
        col: RAW_TO_SNAKE_COLUMNS.get(col, col.strip().lower().replace(" ", "_"))
        for col in df.columns
    }
    return df.rename(columns=rename_dict)


def parse_height_inches(series: pd.Series) -> pd.Series:
    """Parse height strings (e.g. "5' 8\\"") in-memory into total numeric inches.

    Returns float Series. Invalid or missing values safely return NaN.
    """

    def _parse(val: Any) -> Any:
        if pd.isna(val):
            return None
        m = re.match(r"^\s*(\d+)'\s*(\d+)\"\s*$", str(val))
        if m:
            return float(int(m.group(1)) * 12 + int(m.group(2)))
        return None

    return series.apply(_parse).astype(float)


def parse_weight_lbs(series: pd.Series) -> pd.Series:
    """Parse weight strings (e.g. "137lbs") in-memory into numeric pounds.

    Returns float Series. Invalid or missing values safely return NaN.
    """

    def _parse(val: Any) -> Any:
        if pd.isna(val):
            return None
        m = re.match(r"^\s*(\d+)\s*lbs\s*$", str(val), re.IGNORECASE)
        if m:
            return float(m.group(1))
        return None

    return series.apply(_parse).astype(float)


def parse_bust_size(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse bra bust size strings (e.g. "34c", "34ddd/e") into band size and cup size.

    Returns:
        Tuple[pd.Series, pd.Series]: (bust_band_size float Series, bust_cup_size string Series)
        Invalid or missing values return (NaN, NaN).
    """

    def _parse(val: Any) -> Tuple[Any, Any]:
        if pd.isna(val):
            return (None, None)
        m = re.match(r"^\s*(\d+)\s*([a-zA-Z\+\/]+)\s*$", str(val))
        if m:
            band = float(m.group(1))
            cup = m.group(2).lower().strip()
            return (band, cup)
        return (None, None)

    parsed = series.apply(_parse)
    bands = pd.Series([p[0] for p in parsed], index=series.index, dtype=float)
    cups = pd.Series([p[1] for p in parsed], index=series.index, dtype=object)
    return bands, cups


def sanitize_age(series: pd.Series) -> pd.Series:
    """Sanitize customer ages by replacing implausible values (< 16 or > 90) with NaN.

    Does NOT remove any rows.
    """

    def _sanitize(val: Any) -> Any:
        if pd.isna(val):
            return None
        try:
            v = float(val)
            if v < 16.0 or v > 90.0:
                return None
            return v
        except (ValueError, TypeError):
            return None

    return series.apply(_sanitize).astype(float)


def normalize_rented_for(series: pd.Series) -> pd.Series:
    """Normalize known inconsistency in 'rented_for' column.

    Maps 'party: cocktail' -> 'party'.
    """

    def _norm(val: Any) -> Any:
        if pd.isna(val):
            return val
        s = str(val).strip()
        if s == "party: cocktail":
            return "party"
        return s

    return series.apply(_norm)


def parse_review_date(series: pd.Series) -> pd.Series:
    """Parse review_date string column into datetime64[ns] objects with errors='coerce'."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


def identify_exact_duplicates(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for exact redundant full-row duplicates using keep='first'."""
    return df.duplicated(keep="first")


def report_exact_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Report exact duplicate row metrics unambiguously.

    Returns:
        Dict with keys:
          - redundant_duplicate_rows (int)
          - rows_involved_in_duplicate_groups (int)
          - duplicate_percentage (float)
          - unique_duplicate_patterns (int)
    """
    redundant_count = int(df.duplicated(keep="first").sum())
    total_group_rows = int(df.duplicated(keep=False).sum())
    pct = float((redundant_count / len(df)) * 100) if len(df) > 0 else 0.0

    dup_group_df = df[df.duplicated(keep=False)]
    unique_patterns = int(len(dup_group_df.drop_duplicates())) if len(dup_group_df) > 0 else 0

    return {
        "redundant_duplicate_rows": redundant_count,
        "rows_involved_in_duplicate_groups": total_group_rows,
        "duplicate_percentage": round(pct, 4),
        "unique_duplicate_patterns": unique_patterns,
    }


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact full-row redundant duplicates using keep='first'.

    Preserves repeated users, repeated items, and distinct legitimate transactions.
    """
    return df.drop_duplicates(keep="first").copy()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, stateless data cleaning to input DataFrame.

    MUST operate strictly on a copy (immutable to input df).

    Steps:
      1. Copy input DataFrame.
      2. Normalize column names to snake_case.
      3. Parse height string column to 'parsed_height_inches'.
      4. Parse weight string column to 'parsed_weight_lbs'.
      5. Parse bust size string column to 'bust_band_size' and 'bust_cup_size'.
      6. Sanitize age column (replace < 16 or > 90 with NaN).
      7. Normalize 'rented_for' ('party: cocktail' -> 'party').
      8. Parse 'review_date' to datetime.
    """
    cleaned = df.copy()
    cleaned = normalize_column_names(cleaned)

    if "height" in cleaned.columns:
        cleaned["parsed_height_inches"] = parse_height_inches(cleaned["height"])

    if "weight" in cleaned.columns:
        cleaned["parsed_weight_lbs"] = parse_weight_lbs(cleaned["weight"])

    if "bust_size" in cleaned.columns:
        bands, cups = parse_bust_size(cleaned["bust_size"])
        cleaned["bust_band_size"] = bands
        cleaned["bust_cup_size"] = cups

    if "age" in cleaned.columns:
        cleaned["age"] = sanitize_age(cleaned["age"])

    if "rented_for" in cleaned.columns:
        cleaned["rented_for"] = normalize_rented_for(cleaned["rented_for"])

    if "review_date" in cleaned.columns:
        cleaned["review_date"] = parse_review_date(cleaned["review_date"])

    return cleaned
