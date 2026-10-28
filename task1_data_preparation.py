"""Task 1: reproducible split and initial data preparation for 1BP450.

This script deliberately does not calculate bins, WoE, or select predictors.
Those data-driven decisions belong to later tasks and must be learned on train.

Example (PowerShell or Anaconda Prompt):
  python task1_data_preparation.py --input "C:\\path\\to\\AppData.xlsx"

Outputs are written to task1_output by default:
  train.csv, validation.csv, split_summary.csv,
  training_data_quality.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
TRAIN_SHARE = 0.70
EMPLOYMENT_SENTINEL = 365243

BUILDING_COLUMNS = [
    "APARTMENTS_AVG",
    "COMMONAREA_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAREA_AVG",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]


def stratified_split(
    data: pd.DataFrame, target_col: str = "TARGET"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic 70/30 split, preserving each target share."""
    if target_col not in data.columns:
        raise ValueError(f"Required target column {target_col!r} is missing.")
    if data[target_col].isna().any():
        raise ValueError("TARGET contains missing values; resolve before splitting.")

    labels = set(data[target_col].unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected TARGET values 0 and 1; found {labels}.")

    rng = np.random.default_rng(SEED)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    # Shuffle each outcome group separately so both sets preserve the bad rate.
    for label in sorted(labels):
        group_indices = data.index[data[target_col] == label].to_numpy().copy()
        rng.shuffle(group_indices)
        n_train = int(TRAIN_SHARE * len(group_indices))
        train_indices.extend(group_indices[:n_train].tolist())
        validation_indices.extend(group_indices[n_train:].tolist())

    train_idx = np.asarray(train_indices, dtype=int)
    validation_idx = np.asarray(validation_indices, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(validation_idx)

    train = data.loc[train_idx].copy()
    validation = data.loc[validation_idx].copy()
    return train, validation


def add_initial_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, interpretable variables without fitting on data."""
    data = data.copy()

    # These fields count days before application, hence the sign change.
    data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25

    employed_days = data["DAYS_EMPLOYED"]
    is_employment_sentinel = employed_days.eq(EMPLOYMENT_SENTINEL)
    data["EMPLOYMENT_SPECIAL_CODE"] = np.where(
        employed_days.isna(), np.nan, is_employment_sentinel.astype(float)
    )
    data["EMPLOYMENT_YEARS"] = (-employed_days / 365.25).where(
        ~is_employment_sentinel
    )

    # Keep the original day fields. These year-scaled versions aid interpretation.
    for source, destination in [
        ("DAYS_REGISTRATION", "YEARS_SINCE_REGISTRATION"),
        ("DAYS_ID_PUBLISH", "YEARS_SINCE_ID_PUBLISH"),
        ("DAYS_LAST_PHONE_CHANGE", "YEARS_SINCE_PHONE_CHANGE"),
    ]:
        data[destination] = -data[source] / 365.25

    # The source data has positive denominators; replace zero defensively.
    income = data["AMT_INCOME_TOTAL"].replace(0, np.nan)
    goods_price = data["AMT_GOODS_PRICE"].replace(0, np.nan)
    data["CREDIT_TO_INCOME"] = data["AMT_CREDIT"] / income
    data["ANNUITY_TO_INCOME"] = data["AMT_ANNUITY"] / income
    data["CREDIT_TO_GOODS_PRICE"] = data["AMT_CREDIT"] / goods_price

    available_building_columns = [
        column for column in BUILDING_COLUMNS if column in data.columns
    ]
    if len(available_building_columns) != len(BUILDING_COLUMNS):
        missing = sorted(set(BUILDING_COLUMNS) - set(available_building_columns))
        raise ValueError(f"Building columns required for missingness count: {missing}")
    data["BUILDING_MISSING_COUNT"] = data[available_building_columns].isna().sum(axis=1)

    return data


def make_training_quality_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Summarize training-data missingness and numeric ranges for Task 1."""
    rows = []
    n = len(train)

    for column in train.columns:
        series = train[column]
        nonmissing = series.dropna()
        record = {
            "variable": column,
            "dtype": str(series.dtype),
            "n_rows": n,
            "n_missing": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean()),
            "n_unique_nonmissing": int(nonmissing.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series) and not nonmissing.empty:
            quantiles = nonmissing.quantile([0.01, 0.50, 0.99])
            record.update(
                {
                    "minimum": float(nonmissing.min()),
                    "p01": float(quantiles.loc[0.01]),
                    "median": float(quantiles.loc[0.50]),
                    "p99": float(quantiles.loc[0.99]),
                    "maximum": float(nonmissing.max()),
                }
            )
        else:
            record.update(
                {"minimum": np.nan, "p01": np.nan, "median": np.nan,
                 "p99": np.nan, "maximum": np.nan}
            )
        rows.append(record)

    return pd.DataFrame(rows)


def make_split_summary(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, subset in [("train", train), ("validation", validation)]:
        rows.append(
            {
                "sample": name,
                "n_total": len(subset),
                "n_good_TARGET_0": int((subset["TARGET"] == 0).sum()),
                "n_bad_TARGET_1": int((subset["TARGET"] == 1).sum()),
                "bad_rate_in_sample": float(subset["TARGET"].mean()),
                "random_seed": SEED,
                "train_share": TRAIN_SHARE,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("AppData.xlsx"),
        help="Path to AppData.xlsx (default: ./AppData.xlsx)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("task1_output"),
        help="Folder for the generated CSV files (default: ./task1_output)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Cannot find {args.input}. Pass its location with --input."
        )

    raw = pd.read_excel(args.input)
    required = {"SK_ID_CURR", "TARGET", "DAYS_BIRTH", "DAYS_EMPLOYED",
                "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
                "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")
    if raw["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR contains duplicate IDs.")

    # Split first. All subsequent data-driven choices must use train only.
    train, validation = stratified_split(raw)
    train = add_initial_features(train)
    validation = add_initial_features(validation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train.csv", index=False)
    validation.to_csv(args.output_dir / "validation.csv", index=False)
    make_split_summary(train, validation).to_csv(
        args.output_dir / "split_summary.csv", index=False
    )
    make_training_quality_summary(train).to_csv(
        args.output_dir / "training_data_quality.csv", index=False
    )

    summary = make_split_summary(train, validation)
    print(f"Seed: {SEED}; stratified split: {TRAIN_SHARE:.0%}/{1-TRAIN_SHARE:.0%}")
    print(summary.to_string(index=False))
    print(f"\nOutputs saved in: {args.output_dir.resolve()}")
    print("No imputation, binning, WoE calculation, or predictor selection was done.")


if __name__ == "__main__":
    main()
