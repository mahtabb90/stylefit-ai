"""Unit tests for deterministic cleaning module src/cleaning.py."""

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    clean_dataset,
    identify_exact_duplicates,
    normalize_column_names,
    normalize_rented_for,
    parse_bust_size,
    parse_height_inches,
    parse_review_date,
    parse_weight_lbs,
    remove_exact_duplicates,
    report_exact_duplicates,
    sanitize_age,
)


def test_normalize_column_names():
    raw_df = pd.DataFrame(columns=["bust size", "rented for", "body type", "fit", "user_id"])
    norm_df = normalize_column_names(raw_df)
    assert list(norm_df.columns) == ["bust_size", "rented_for", "body_type", "fit", "user_id"]


def test_parse_height_inches():
    heights = pd.Series(["5' 8\"", "4' 11\"", "6' 2\"", None, "invalid", "6'0\""])
    parsed = parse_height_inches(heights)
    expected = [68.0, 59.0, 74.0, np.nan, np.nan, 72.0]

    for p, e in zip(parsed, expected):
        if np.isnan(e):
            assert np.isnan(p)
        else:
            assert p == e


def test_parse_weight_lbs():
    weights = pd.Series(["137lbs", "110 LBS", "200lbs", None, "invalid", "150"])
    parsed = parse_weight_lbs(weights)
    expected = [137.0, 110.0, 200.0, np.nan, np.nan, np.nan]

    for p, e in zip(parsed, expected):
        if np.isnan(e):
            assert np.isnan(p)
        else:
            assert p == e


def test_parse_bust_size():
    busts = pd.Series(["34c", "32d", "34ddd/e", "36d+", "28aa", None, "invalid"])
    bands, cups = parse_bust_size(busts)

    expected_bands = [34.0, 32.0, 34.0, 36.0, 28.0, np.nan, np.nan]
    expected_cups = ["c", "d", "ddd/e", "d+", "aa", None, None]

    for b, e_b in zip(bands, expected_bands):
        if np.isnan(e_b):
            assert np.isnan(b)
        else:
            assert b == e_b

    for c, e_c in zip(cups, expected_cups):
        if e_c is None:
            assert pd.isna(c) or c is None
        else:
            assert c == e_c


def test_sanitize_age():
    ages = pd.Series([28.0, 0.0, 14.0, 15.0, 16.0, 90.0, 91.0, 117.0, None])
    sanitized = sanitize_age(ages)

    expected = [28.0, np.nan, np.nan, np.nan, 16.0, 90.0, np.nan, np.nan, np.nan]
    for s, e in zip(sanitized, expected):
        if np.isnan(e):
            assert np.isnan(s)
        else:
            assert s == e


def test_normalize_rented_for():
    rf = pd.Series(["wedding", "party: cocktail", "party", "everyday", None])
    norm = normalize_rented_for(rf)
    expected = ["wedding", "party", "party", "everyday", None]

    for n, e in zip(norm, expected):
        if e is None:
            assert pd.isna(n)
        else:
            assert n == e


def test_parse_review_date():
    dates = pd.Series(["May 8, 2016", "2018-01-15", "invalid_date", None])
    parsed = parse_review_date(dates)

    assert pd.notna(parsed[0])
    assert parsed[0].year == 2016
    assert pd.notna(parsed[1])
    assert parsed[1].year == 2018
    assert pd.isna(parsed[2])
    assert pd.isna(parsed[3])


def test_exact_duplicate_functions():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "item_id": [10, 10, 20, 20, 20],
            "fit": ["fit", "fit", "small", "small", "large"],
        }
    )

    mask = identify_exact_duplicates(df)
    assert mask.tolist() == [False, True, False, True, False]

    report = report_exact_duplicates(df)
    assert report["redundant_duplicate_rows"] == 2
    assert report["rows_involved_in_duplicate_groups"] == 4
    assert report["duplicate_percentage"] == 40.0
    assert report["unique_duplicate_patterns"] == 2

    deduped = remove_exact_duplicates(df)
    assert len(deduped) == 3
    assert deduped.index.tolist() == [0, 2, 4]


def test_clean_dataset_immutability():
    original_df = pd.DataFrame(
        {
            "bust size": ["34c", "32d"],
            "rented for": ["party: cocktail", "wedding"],
            "body type": ["hourglass", "athletic"],
            "height": ["5' 8\"", "5' 5\""],
            "weight": ["137lbs", "125lbs"],
            "age": [28.0, 117.0],
            "review_date": ["May 8, 2016", "Jan 1, 2017"],
        }
    )

    original_copy = original_df.copy()
    cleaned = clean_dataset(original_df)

    # Verify original df was not mutated
    pd.testing.assert_frame_equal(original_df, original_copy)
    assert "bust size" in original_df.columns

    # Verify cleaned df has snake_case and parsed columns
    assert "bust_size" in cleaned.columns
    assert "parsed_height_inches" in cleaned.columns
    assert "parsed_weight_lbs" in cleaned.columns
    assert "bust_band_size" in cleaned.columns
    assert "bust_cup_size" in cleaned.columns
    assert np.isnan(cleaned["age"].iloc[1])  # 117.0 replaced with NaN
    assert cleaned["rented_for"].iloc[0] == "party"
