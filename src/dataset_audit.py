from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "renttherunway_final_data.json.gz"


def load_dataset() -> pd.DataFrame:
    """Load the raw Rent the Runway dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_json(
        DATA_PATH,
        lines=True,
        compression="gzip",
    )


def audit_basic_info(df: pd.DataFrame) -> None:
    """Print basic dataset information: shape, columns, dtypes, missing values."""
    print("=" * 60)
    print("STYLEFIT AI - INITIAL DATASET AUDIT")
    print("=" * 60)

    print("\n1. Dataset shape")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. Column names")
    for column in df.columns:
        print(f"- {column}")

    print("\n3. Data types")
    print(df.dtypes)

    print("\n4. Target distribution")
    target_counts = df["fit"].value_counts(dropna=False)
    target_percentages = (
        df["fit"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    target_summary = pd.DataFrame(
        {
            "count": target_counts,
            "percentage": target_percentages,
        }
    )
    print(target_summary)

    print("\n5. Missing values")
    missing_summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percentage": (
                df.isna().mean().mul(100).round(2)
            ),
        }
    ).sort_values("missing_percentage", ascending=False)
    print(missing_summary)

    print("\n6. Numeric column summary")
    numeric_columns = ["age", "size", "rating"]
    existing_numeric_columns = [
        col for col in numeric_columns if col in df.columns
    ]
    print(df[existing_numeric_columns].describe().round(2))


def audit_suspicious_ages(df: pd.DataFrame) -> None:
    """Audit suspicious age values without modifying or removing rows."""
    print("\n7. Suspicious age values")
    if "age" not in df.columns:
        print("Column 'age' not found in dataset.")
        return

    missing_age_count = df["age"].isna().sum()
    below_16_count = (df["age"] < 16).sum()
    above_90_count = (df["age"] > 90).sum()

    print(f"Missing age count: {missing_age_count:,}")
    print(f"Ages below 16: {below_16_count:,}")
    print(f"Ages above 90: {above_90_count:,}")

    valid_ages = df["age"].dropna()
    age_counts = valid_ages.value_counts().sort_index()

    print("\n10 smallest non-null ages with counts:")
    smallest_10 = age_counts.head(10)
    for age, count in smallest_10.items():
        print(f"  Age {age:g}: {count:,}")

    print("\n10 largest non-null ages with counts:")
    largest_10 = age_counts.sort_index(ascending=False).head(10)
    for age, count in largest_10.items():
        print(f"  Age {age:g}: {count:,}")


def audit_size(df: pd.DataFrame) -> None:
    """Audit size values and frequency distribution.

    Note: US clothing size 0 is a valid apparel size and must not automatically
    be treated as an error or missing value.
    """
    print("\n8. Size inspection")
    if "size" not in df.columns:
        print("Column 'size' not found in dataset.")
        return

    # US clothing size 0 is standard in women's apparel sizing and must NOT
    # automatically be flagged as an error, zero-value placeholder, or missing data.
    min_size = df["size"].min()
    max_size = df["size"].max()
    unique_sizes = df["size"].nunique()
    sorted_sizes = [int(s) for s in sorted(df["size"].dropna().unique())]

    print(f"Minimum size: {min_size}")
    print(f"Maximum size: {max_size}")
    print(f"Unique size count: {unique_sizes}")
    print(f"Sorted list of size values:\n{sorted_sizes}")

    print("\nFrequency of each size:")
    size_freq = df["size"].value_counts().sort_index()
    print(size_freq.to_string())


def audit_user_item_sparsity(df: pd.DataFrame) -> None:
    """Audit user and item transaction sparsity."""
    print("\n9. User and item sparsity")

    if "user_id" in df.columns:
        num_users = df["user_id"].nunique()
        user_tx = df["user_id"].value_counts()
        single_tx_users = (user_tx == 1).sum()

        print(f"Number of unique user_id values: {num_users:,}")
        print(f"Users with exactly one transaction: {single_tx_users:,}")
        print("\nDescriptive statistics for transactions per user:")
        print(user_tx.describe().round(2))

    if "item_id" in df.columns:
        num_items = df["item_id"].nunique()
        item_tx = df["item_id"].value_counts()
        single_tx_items = (item_tx == 1).sum()

        print(f"\nNumber of unique item_id values: {num_items:,}")
        print(f"Items with exactly one transaction: {single_tx_items:,}")
        print("\nDescriptive statistics for transactions per item:")
        print(item_tx.describe().round(2))


def audit_duplicates(df: pd.DataFrame) -> None:
    """Audit exact duplicate rows without removing them."""
    print("\n10. Duplicate inspection")
    redundant_dup_count = df.duplicated(keep="first").sum()
    redundant_dup_pct = (redundant_dup_count / len(df)) * 100
    group_dup_count = df.duplicated(keep=False).sum()

    print(f"Redundant duplicate rows: {redundant_dup_count:,}")
    print(f"Redundant duplicate percentage: {redundant_dup_pct:.2f}%")
    print(f"Rows involved in duplicate groups: {group_dup_count:,}")

    if "fit" in df.columns:
        dup_rows = df[df.duplicated(keep=False)]
        print(
            "\nTarget ('fit') distribution among all rows involved in duplicate groups:"
        )
        if len(dup_rows) > 0:
            print(dup_rows["fit"].value_counts(dropna=False))
        else:
            print("No duplicate rows found.")


def audit_empty_strings(df: pd.DataFrame) -> None:
    """Inspect object/string columns for empty or whitespace-only values."""
    print("\n11. Empty string inspection")

    obj_columns = df.select_dtypes(include=["object", "string"]).columns
    empty_counts = {}

    for col in obj_columns:
        mask = df[col].notna() & df[col].astype(str).str.strip().eq("")
        count = mask.sum()
        empty_counts[col] = count

    empty_df = pd.DataFrame(
        {
            "empty_or_whitespace_count": pd.Series(empty_counts),
            "nan_count": df[obj_columns].isna().sum(),
        }
    )
    print(empty_df)


def audit_category_sparsity(df: pd.DataFrame) -> None:
    """Audit category frequency and sparsity."""
    print("\n12. Category sparsity")
    if "category" not in df.columns:
        print("Column 'category' not found in dataset.")
        return

    cat_counts = df["category"].value_counts(dropna=False)
    fewer_than_10 = (cat_counts < 10).sum()
    fewer_than_100 = (cat_counts < 100).sum()

    print(f"Number of categories with fewer than 10 rows: {fewer_than_10}")
    print(f"Number of categories with fewer than 100 rows: {fewer_than_100}")

    print("\n15 least frequent categories:")
    least_frequent = cat_counts.tail(15)
    print(least_frequent)


def audit_rented_for_quality(df: pd.DataFrame) -> None:
    """Audit rented-for column data quality without normalizing values."""
    print("\n13. Rented-for data quality")
    if "rented for" not in df.columns:
        print("Column 'rented for' not found in dataset.")
        return

    rented_counts = df["rented for"].value_counts(dropna=False)
    print("Unique values and counts:")
    print(rented_counts)

    if "party: cocktail" in rented_counts.index:
        count = rented_counts["party: cocktail"]
        print(
            f"\nInconsistency note: Value 'party: cocktail' found with count={count}. "
            "This is a potential data inconsistency compared to 'party'."
        )


def print_dataset_audit(df: pd.DataFrame) -> None:
    """Print a comprehensive audit of the raw dataset."""
    audit_basic_info(df)
    audit_suspicious_ages(df)
    audit_size(df)
    audit_user_item_sparsity(df)
    audit_duplicates(df)
    audit_empty_strings(df)
    audit_category_sparsity(df)
    audit_rented_for_quality(df)
    print("\n" + "=" * 60)
    print("Dataset audit completed")
    print("=" * 60)


def main() -> None:
    dataset = load_dataset()
    print_dataset_audit(dataset)


if __name__ == "__main__":
    main()