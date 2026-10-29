"""Task 2: training-only fine/coarse classing, WoE, IV, Gini, and CS.

The script re-creates the Task 1 stratified 70/30 split (seed 42) and performs
Task 2 screening on the training sample only. It does not use validation labels.

Example (Anaconda Prompt or VS Code terminal):
  python task2_univariate_analysis.py --input "C:\\path\\to\\AppData.xlsx"

Dependencies: pandas, numpy, scipy (included in a standard Anaconda install).

Outputs in task2_output:
  variable_summary.csv  Fine/coarse IV, Gini, CS, missingness and bin counts
  fine_bins.csv         Initial fine-classing bin counts, WoE and IV contribution
  coarse_bins.csv       Proposed coarse bins, WoE, IV contribution and class counts
  merge_log.csv         Adjacent/closest-risk CS merges used for coarse bins

All values, thresholds, merges, and WoE estimates are development-sample
estimates. Treat the automatic coarse bins as proposals to review, especially
where bins are non-monotonic, special, missing, or small.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata


SEED = 42
TRAIN_SHARE = 0.70
FINE_PREBINS = 20
MIN_BIN_SHARE = 0.05
MIN_CLASS_COUNT = 20
MAX_REGULAR_COARSE_BINS = 8
WOE_ALPHA = 0.5
EMPLOYMENT_SENTINEL = 365243
POPULATION_BAD_RATE = 0.08
MISSING_TOKEN = "<MISSING>"
SPECIAL_TOKEN = "<SPECIAL_CODE>"

BUILDING_COLUMNS = [
    "APARTMENTS_AVG", "COMMONAREA_AVG", "LIVINGAREA_AVG",
    "NONLIVINGAREA_AVG", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]
DAY_COLUMNS = {
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_REGISTRATION": "YEARS_SINCE_REGISTRATION",
    "DAYS_ID_PUBLISH": "YEARS_SINCE_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE": "YEARS_SINCE_PHONE_CHANGE",
}
RAW_DAY_COLUMNS_REPLACED = set(DAY_COLUMNS) | {"DAYS_EMPLOYED"}
EXPECTED_MONOTONICITY = {
    # Higher external scores and longer ordinary histories are expected to
    # indicate lower risk; higher repayment burden ratios are expected to mean
    # higher risk. Special/missing groups are kept outside the numeric trend.
    "EXT_SOURCE_1": "decreasing",
    "EXT_SOURCE_2": "decreasing",
    "AGE_YEARS": "decreasing",
    "EMPLOYMENT_YEARS": "decreasing",
    "CREDIT_TO_GOODS_PRICE": "increasing",
    "CREDIT_TO_INCOME": "increasing",
    "ANNUITY_TO_INCOME": "increasing",
}


def split_like_task1(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match the deterministic seed-42 split made by the Task 1 script."""
    if data["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR contains duplicate IDs.")
    if data["TARGET"].isna().any() or set(data["TARGET"].unique()) != {0, 1}:
        raise ValueError("TARGET must contain only nonmissing values 0 and 1.")

    rng = np.random.default_rng(SEED)
    train_indices, validation_indices = [], []
    for label in sorted(data["TARGET"].unique()):
        indices = data.index[data["TARGET"] == label].to_numpy().copy()
        rng.shuffle(indices)
        n_train = int(TRAIN_SHARE * len(indices))
        train_indices.extend(indices[:n_train].tolist())
        validation_indices.extend(indices[n_train:].tolist())
    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = np.asarray(validation_indices, dtype=int)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return data.loc[train_indices].copy(), data.loc[validation_indices].copy()


def add_task1_features(data: pd.DataFrame) -> pd.DataFrame:
    """Recreate the deterministic feature definitions from Task 1."""
    data = data.copy()
    for source, destination in DAY_COLUMNS.items():
        data[destination] = -data[source] / 365.25

    employed_days = data["DAYS_EMPLOYED"]
    special = employed_days.eq(EMPLOYMENT_SENTINEL)
    data["EMPLOYMENT_SPECIAL_CODE"] = np.where(
        employed_days.isna(), np.nan, special.astype(float)
    )
    data["EMPLOYMENT_YEARS"] = (-employed_days / 365.25).where(~special)

    income = data["AMT_INCOME_TOTAL"].replace(0, np.nan)
    goods_price = data["AMT_GOODS_PRICE"].replace(0, np.nan)
    data["CREDIT_TO_INCOME"] = data["AMT_CREDIT"] / income
    data["ANNUITY_TO_INCOME"] = data["AMT_ANNUITY"] / income
    data["CREDIT_TO_GOODS_PRICE"] = data["AMT_CREDIT"] / goods_price
    data["BUILDING_MISSING_COUNT"] = data[BUILDING_COLUMNS].isna().sum(axis=1)
    return data


def candidate_variables(data: pd.DataFrame) -> list[str]:
    excluded = {"SK_ID_CURR", "TARGET"} | RAW_DAY_COLUMNS_REPLACED
    # EMPLOYMENT_YEARS carries the special-code group as its own bin below;
    # including its indicator as a second candidate would duplicate that signal.
    excluded.add("EMPLOYMENT_SPECIAL_CODE")
    return [column for column in data.columns if column not in excluded]


def feature_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 20:
        return "continuous"
    if pd.api.types.is_numeric_dtype(series):
        return "ordered_discrete"
    return "categorical"


def text_token(value) -> str:
    if pd.isna(value):
        return MISSING_TOKEN
    return str(value)


def make_fine_bins(
    train: pd.DataFrame, variable: str, kind: str
) -> tuple[pd.Series, list[dict]]:
    """Create 20-quantile fine bins for continuous inputs; natural levels otherwise."""
    values = train[variable]
    assignments = pd.Series(index=train.index, dtype="object")
    metadata: list[dict] = []
    next_id = 0

    if kind == "continuous":
        special_mask = pd.Series(False, index=train.index)
        if variable == "EMPLOYMENT_YEARS":
            special_mask = train["EMPLOYMENT_SPECIAL_CODE"].eq(1).fillna(False)
        missing_mask = values.isna() & ~special_mask
        valid_mask = values.notna() & ~special_mask
        valid_values = values.loc[valid_mask]

        if not valid_values.empty:
            quantiles, edges = pd.qcut(
                valid_values, q=FINE_PREBINS, labels=False,
                retbins=True, duplicates="drop"
            )
            quantiles = pd.Series(quantiles, index=valid_values.index).astype(int)
            n_quantile_bins = len(edges) - 1
            for q in range(n_quantile_bins):
                bin_id = f"B{next_id:03d}"
                member_idx = quantiles.index[quantiles == q]
                assignments.loc[member_idx] = bin_id
                metadata.append(
                    {
                        "bin_id": bin_id,
                        "fine_order": next_id,
                        "bin_label": f"[{edges[q]:.7g}, {edges[q+1]:.7g}]" if q == n_quantile_bins-1
                        else f"[{edges[q]:.7g}, {edges[q+1]:.7g})",
                        "numeric_low": float(edges[q]),
                        "numeric_high": float(edges[q + 1]),
                        "category_values": "",
                        "is_missing": False,
                        "is_special": False,
                        "is_protected": False,
                    }
                )
                next_id += 1

        if special_mask.any():
            bin_id = f"B{next_id:03d}"
            assignments.loc[special_mask] = bin_id
            metadata.append({
                "bin_id": bin_id, "fine_order": next_id,
                "bin_label": SPECIAL_TOKEN, "numeric_low": np.nan,
                "numeric_high": np.nan, "category_values": SPECIAL_TOKEN,
                "is_missing": False, "is_special": True, "is_protected": True,
            })
            next_id += 1

        if missing_mask.any():
            bin_id = f"B{next_id:03d}"
            assignments.loc[missing_mask] = bin_id
            metadata.append({
                "bin_id": bin_id, "fine_order": next_id,
                "bin_label": MISSING_TOKEN, "numeric_low": np.nan,
                "numeric_high": np.nan, "category_values": MISSING_TOKEN,
                "is_missing": True, "is_special": False, "is_protected": False,
            })
        return assignments, metadata

    if kind == "ordered_discrete":
        nonmissing = values.dropna()
        for value in sorted(nonmissing.unique().tolist()):
            bin_id = f"B{next_id:03d}"
            mask = values.eq(value).fillna(False)
            assignments.loc[mask] = bin_id
            metadata.append({
                "bin_id": bin_id, "fine_order": next_id,
                "bin_label": f"value={value}", "numeric_low": float(value),
                "numeric_high": float(value), "category_values": str(value),
                "is_missing": False, "is_special": False, "is_protected": False,
            })
            next_id += 1
        missing_mask = values.isna()
        if missing_mask.any():
            bin_id = f"B{next_id:03d}"
            assignments.loc[missing_mask] = bin_id
            metadata.append({
                "bin_id": bin_id, "fine_order": next_id,
                "bin_label": MISSING_TOKEN, "numeric_low": np.nan,
                "numeric_high": np.nan, "category_values": MISSING_TOKEN,
                "is_missing": True, "is_special": False, "is_protected": False,
            })
        return assignments, metadata

    tokens = values.map(text_token)
    levels = tokens.unique().tolist()
    levels = sorted(levels, key=lambda x: (x == MISSING_TOKEN, x.casefold()))
    for level in levels:
        bin_id = f"B{next_id:03d}"
        assignments.loc[tokens == level] = bin_id
        is_missing = level == MISSING_TOKEN
        is_special = level.strip().casefold() in {"xna", "xap", "unknown"}
        metadata.append({
            "bin_id": bin_id, "fine_order": next_id,
            "bin_label": level, "numeric_low": np.nan,
            "numeric_high": np.nan, "category_values": level,
            "is_missing": is_missing, "is_special": is_special,
            "is_protected": is_special,
        })
        next_id += 1
    return assignments, metadata


def count_bins(assignments: pd.Series, target: pd.Series, metadata: list[dict]) -> list[dict]:
    counts = pd.DataFrame({"bin_id": assignments, "target": target})
    grouped = counts.groupby(["bin_id", "target"], observed=True).size().unstack(fill_value=0)
    records = []
    for meta in metadata:
        row = dict(meta)
        row["good"] = int(grouped.loc[row["bin_id"], 0]) if row["bin_id"] in grouped.index else 0
        row["bad"] = int(grouped.loc[row["bin_id"], 1]) if row["bin_id"] in grouped.index else 0
        row["n"] = row["good"] + row["bad"]
        records.append(row)
    return records


def woe_stats(records: list[dict]) -> list[dict]:
    nonempty = [row for row in records if row["n"] > 0]
    total_good = sum(row["good"] for row in nonempty)
    total_bad = sum(row["bad"] for row in nonempty)
    has_zero_class = any(row["good"] == 0 or row["bad"] == 0 for row in nonempty)
    alpha = WOE_ALPHA if has_zero_class else 0.0
    k = len(nonempty)
    sample_bad_rate = total_bad / (total_good + total_bad)
    population_odds_multiplier = (
        (POPULATION_BAD_RATE / (1 - POPULATION_BAD_RATE))
        / (sample_bad_rate / (1 - sample_bad_rate))
    )
    result = []
    for row in records:
        out = dict(row)
        if row["n"] == 0 or total_good == 0 or total_bad == 0:
            out.update({"sample_bad_rate": np.nan, "good_share": np.nan,
                        "bad_share": np.nan, "woe_good_over_bad": np.nan,
                        "iv_contribution": np.nan, "woe_smoothing_alpha": alpha})
        else:
            good_share = (row["good"] + alpha) / (total_good + alpha * k)
            bad_share = (row["bad"] + alpha) / (total_bad + alpha * k)
            woe = math.log(good_share / bad_share)
            sample_rate_for_calibration = (row["bad"] + alpha) / (row["n"] + 2 * alpha)
            sample_odds = sample_rate_for_calibration / (1 - sample_rate_for_calibration)
            population_odds = sample_odds * population_odds_multiplier
            population_rate = population_odds / (1 + population_odds)
            out.update({
                "sample_bad_rate": row["bad"] / row["n"],
                "population_adjusted_bad_rate_8pct": population_rate,
                "good_share": good_share,
                "bad_share": bad_share,
                "woe_good_over_bad": woe,
                "iv_contribution": (good_share - bad_share) * woe,
                "woe_smoothing_alpha": alpha,
            })
        result.append(out)
    return result


def global_cs(records: list[dict]) -> tuple[float, int, float]:
    bins = [r for r in records if r["n"] > 0]
    total_n = sum(r["n"] for r in bins)
    total_bad = sum(r["bad"] for r in bins)
    if len(bins) <= 1 or total_n == total_bad or total_bad == 0:
        return 0.0, max(0, len(bins) - 1), 1.0
    pd_hat = total_bad / total_n
    statistic = 0.0
    for row in bins:
        expected_bad = row["n"] * pd_hat
        statistic += (expected_bad - row["bad"]) ** 2 / (
            row["n"] * pd_hat * (1 - pd_hat)
        )
    df = len(bins) - 1
    return statistic, df, float(chi2.sf(statistic, df))


def gini_from_woe(assignments: pd.Series, target: pd.Series, records: list[dict]) -> float:
    map_woe = {row["bin_id"]: row["woe_good_over_bad"] for row in records}
    woe = assignments.map(map_woe).astype(float)
    eligible = woe.notna() & target.notna()
    y = target.loc[eligible].to_numpy(dtype=int)
    risk = -woe.loc[eligible].to_numpy(dtype=float)
    n_bad = int(y.sum())
    n_good = len(y) - n_bad
    if n_bad == 0 or n_good == 0:
        return 0.0
    ranks = rankdata(risk, method="average")
    auc = (ranks[y == 1].sum() - n_bad * (n_bad + 1) / 2) / (n_bad * n_good)
    return float(2 * auc - 1)


def pair_cs(first: dict, second: dict) -> tuple[float, float]:
    n1, b1, g1 = first["n"], first["bad"], first["good"]
    n2, b2, g2 = second["n"], second["bad"], second["good"]
    n, bad = n1 + n2, b1 + b2
    if n == 0 or bad == 0 or bad == n:
        return 0.0, 1.0
    pooled_pd = bad / n
    statistic = 0.0
    for ni, bi in [(n1, b1), (n2, b2)]:
        statistic += (ni * pooled_pd - bi) ** 2 / (ni * pooled_pd * (1 - pooled_pd))
    return float(statistic), float(chi2.sf(statistic, 1))


def current_label(group: dict) -> str:
    return " + ".join(group["labels"])


def merge_bins(
    fine_records: list[dict], kind: str, min_n: int, max_regular: int,
    variable: str,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Propose coarser groups by minimum-support checks and adjacent CS merging."""
    groups = []
    for record in fine_records:
        groups.append({
            "fine_ids": [record["bin_id"]],
            "labels": [record["bin_label"]],
            "n": record["n"], "good": record["good"], "bad": record["bad"],
            "numeric_low": record["numeric_low"],
            "numeric_high": record["numeric_high"],
            "is_missing": record["is_missing"],
            "is_special": record["is_special"],
            "is_protected": record["is_protected"],
            "fine_order": record["fine_order"],
        })

    # Protect meaningful special-code and XNA/XAP/Unknown bins. A large genuine
    # missing group is also retained. Small missing groups remain merge candidates.
    for group in groups:
        well_supported = (
            group["n"] >= min_n and group["good"] >= MIN_CLASS_COUNT
            and group["bad"] >= MIN_CLASS_COUNT
        )
        if group["is_special"] or (group["is_missing"] and well_supported):
            group["is_protected"] = True

    merge_log = []
    iteration = 0

    def risk(group):
        return (group["bad"] + 0.5) / (group["n"] + 1.0)

    while True:
        mergeable = [g for g in groups if not g["is_protected"]]
        if kind == "categorical":
            mergeable = sorted(mergeable, key=lambda g: risk(g))
        else:
            mergeable = sorted(
                mergeable,
                key=lambda g: (
                    g["numeric_low"] if pd.notna(g["numeric_low"]) else float("inf"),
                    g["fine_order"],
                ),
            )

        undersized = [
            g for g in mergeable
            if g["n"] < min_n or g["good"] < MIN_CLASS_COUNT or g["bad"] < MIN_CLASS_COUNT
        ]
        need_reduce = len(mergeable) > max_regular
        if not undersized and not need_reduce:
            break
        if len(mergeable) < 2:
            break

        candidate_pairs = []
        if undersized:
            under_ids = {id(g) for g in undersized}
            for i, group in enumerate(mergeable):
                if id(group) not in under_ids:
                    continue
                # A numeric missing bin has no value adjacency, so compare it to
                # each eligible numeric group; normal numeric bins stay adjacent.
                if group["is_missing"] or group["is_special"]:
                    neighbors = [j for j, other in enumerate(mergeable) if j != i]
                else:
                    neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(mergeable)]
                for j in neighbors:
                    left, right = sorted((i, j))
                    if (left, right) not in candidate_pairs:
                        candidate_pairs.append((left, right))
        else:
            candidate_pairs = [(i, i + 1) for i in range(len(mergeable) - 1)]

        if not candidate_pairs:
            break

        pair_results = []
        for i, j in candidate_pairs:
            stat, pvalue = pair_cs(mergeable[i], mergeable[j])
            pair_results.append((stat, pvalue, i, j))
        stat, pvalue, i, j = min(pair_results, key=lambda item: item[0])
        left, right = mergeable[i], mergeable[j]
        merged = {
            "fine_ids": left["fine_ids"] + right["fine_ids"],
            "labels": left["labels"] + right["labels"],
            "n": left["n"] + right["n"],
            "good": left["good"] + right["good"],
            "bad": left["bad"] + right["bad"],
            "numeric_low": min(
                [x for x in [left["numeric_low"], right["numeric_low"]] if pd.notna(x)],
                default=np.nan,
            ),
            "numeric_high": max(
                [x for x in [left["numeric_high"], right["numeric_high"]] if pd.notna(x)],
                default=np.nan,
            ),
            "is_missing": left["is_missing"] or right["is_missing"],
            "is_special": left["is_special"] or right["is_special"],
            "is_protected": False,
            "fine_order": min(left["fine_order"], right["fine_order"]),
        }
        iteration += 1
        merge_log.append({
            "merge_step": iteration,
            "left_group": current_label(left),
            "right_group": current_label(right),
            "cs_statistic": stat,
            "chi_square_p_value_df1": pvalue,
            "merged_n": merged["n"],
            "merged_goods": merged["good"],
            "merged_bads": merged["bad"],
            "reason": "minimum support" if undersized else "reduce to at most 8 regular bins",
        })
        groups.remove(left)
        groups.remove(right)
        groups.append(merged)

    # For variables with an expected ordered risk relationship, pool adjacent
    # numeric groups that violate that direction. This is a transparent PAVA-like
    # step; it is not imposed on every predictor. Missing/special bins are excluded.
    expected_direction = EXPECTED_MONOTONICITY.get(variable)
    if kind == "continuous" and expected_direction:
        while True:
            regular = sorted(
                [g for g in groups if not g["is_missing"] and not g["is_special"]],
                key=lambda g: g["numeric_low"] if pd.notna(g["numeric_low"]) else float("inf"),
            )
            violations = []
            for i in range(len(regular) - 1):
                left, right = regular[i], regular[i + 1]
                left_risk, right_risk = risk(left), risk(right)
                violates = (
                    left_risk < right_risk - 1e-12
                    if expected_direction == "decreasing"
                    else left_risk > right_risk + 1e-12
                )
                if violates:
                    stat, pvalue = pair_cs(left, right)
                    violations.append((stat, pvalue, left, right))
            if not violations:
                break
            stat, pvalue, left, right = min(violations, key=lambda item: item[0])
            merged = {
                "fine_ids": left["fine_ids"] + right["fine_ids"],
                "labels": left["labels"] + right["labels"],
                "n": left["n"] + right["n"],
                "good": left["good"] + right["good"],
                "bad": left["bad"] + right["bad"],
                "numeric_low": min(left["numeric_low"], right["numeric_low"]),
                "numeric_high": max(left["numeric_high"], right["numeric_high"]),
                "is_missing": False,
                "is_special": False,
                "is_protected": False,
                "fine_order": min(left["fine_order"], right["fine_order"]),
            }
            iteration += 1
            merge_log.append({
                "merge_step": iteration,
                "left_group": current_label(left),
                "right_group": current_label(right),
                "cs_statistic": stat,
                "chi_square_p_value_df1": pvalue,
                "merged_n": merged["n"],
                "merged_goods": merged["good"],
                "merged_bads": merged["bad"],
                "reason": f"enforce expected {expected_direction} risk trend",
            })
            groups.remove(left)
            groups.remove(right)
            groups.append(merged)

    # Final order follows predictor values for numeric/ordered variables and risk
    # for nominal categories. Protected special groups are appended explicitly.
    if kind == "categorical":
        groups.sort(key=lambda g: (g["is_protected"], risk(g)))
    else:
        groups.sort(key=lambda g: (
            g["is_protected"],
            g["numeric_low"] if pd.notna(g["numeric_low"]) else float("inf"),
            g["fine_order"],
        ))

    coarse_metadata = []
    fine_to_coarse = {}
    for i, group in enumerate(groups):
        bin_id = f"C{i+1:02d}"
        label = current_label(group)
        coarse_metadata.append({
            "bin_id": bin_id,
            "fine_order": i,
            "bin_label": label,
            "numeric_low": group["numeric_low"],
            "numeric_high": group["numeric_high"],
            "category_values": label,
            "is_missing": group["is_missing"],
            "is_special": group["is_special"],
            "is_protected": group["is_protected"],
        })
        for fine_id in group["fine_ids"]:
            fine_to_coarse[fine_id] = bin_id

    return coarse_metadata, merge_log, fine_to_coarse


def monotonicity(records: list[dict], kind: str) -> str:
    if kind == "categorical":
        return "Nominal categories"
    regular = [r for r in records if not r["is_missing"] and not r["is_special"] and r["n"] > 0]
    if len(regular) <= 2:
        return "Not assessed (fewer than 3 regular bins)"
    rates = [r["sample_bad_rate"] for r in regular]
    diffs = np.diff(rates)
    if np.all(diffs >= -1e-12):
        return "Non-decreasing"
    if np.all(diffs <= 1e-12):
        return "Non-increasing"
    return "Non-monotonic; review"


def metrics_and_tables(
    train: pd.DataFrame, variables: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fine_tables, coarse_tables, summaries, logs = [], [], [], []
    target = train["TARGET"].astype(int)
    min_bin_n = int(math.ceil(MIN_BIN_SHARE * len(train)))

    for variable in variables:
        series = train[variable]
        kind = feature_kind(series)
        fine_assignments, fine_meta = make_fine_bins(train, variable, kind)
        fine_raw = count_bins(fine_assignments, target, fine_meta)
        fine = woe_stats(fine_raw)
        fine_cs, fine_df, fine_p = global_cs(fine)
        fine_iv = float(sum(r["iv_contribution"] for r in fine if pd.notna(r["iv_contribution"])))
        fine_gini = gini_from_woe(fine_assignments, target, fine)

        coarse_meta, merge_log, mapping = merge_bins(
            fine, kind, min_bin_n, MAX_REGULAR_COARSE_BINS, variable
        )
        if merge_log:
            for row in merge_log:
                row["variable"] = variable
                row["feature_type"] = kind
            logs.extend(merge_log)
        coarse_assignments = fine_assignments.map(mapping)
        coarse_raw = count_bins(coarse_assignments, target, coarse_meta)
        coarse = woe_stats(coarse_raw)
        coarse_cs, coarse_df, coarse_p = global_cs(coarse)
        coarse_iv = float(sum(r["iv_contribution"] for r in coarse if pd.notna(r["iv_contribution"])))
        coarse_gini = gini_from_woe(coarse_assignments, target, coarse)

        missing_n = int(series.isna().sum())
        special_n = 0
        if variable == "EMPLOYMENT_YEARS":
            special_n = int(train["EMPLOYMENT_SPECIAL_CODE"].eq(1).sum())
            # The derived tenure is NaN for the sentinel, so report the sentinel
            # separately from genuine missing source values.
            missing_n = max(0, missing_n - special_n)
        elif not pd.api.types.is_numeric_dtype(series):
            special_n = int(series.astype("string").str.strip().str.casefold().isin(
                {"xna", "xap", "unknown"}
            ).sum())
        all_coarse = len([r for r in coarse if r["n"] > 0])
        min_good = min((r["good"] for r in coarse if r["n"] > 0), default=0)
        min_bad = min((r["bad"] for r in coarse if r["n"] > 0), default=0)
        low_bins = sum(
            r["n"] < min_bin_n or r["good"] < MIN_CLASS_COUNT or r["bad"] < MIN_CLASS_COUNT
            for r in coarse if r["n"] > 0
        )
        is_constant = series.nunique(dropna=True) <= 1
        summary = {
            "variable": variable,
            "feature_type": kind,
            "n_missing": missing_n,
            "missing_pct": missing_n / len(train),
            "n_special_code": special_n,
            "special_code_pct": special_n / len(train),
            "n_unique_nonmissing": int(series.nunique(dropna=True)),
            "fine_bin_count": len(fine),
            "coarse_bin_count": all_coarse,
            "fine_IV": fine_iv,
            "coarse_IV": coarse_iv,
            "fine_Gini": fine_gini,
            "coarse_Gini": coarse_gini,
            "fine_CS": fine_cs,
            "fine_CS_df": fine_df,
            "fine_CS_p": fine_p,
            "coarse_CS": coarse_cs,
            "coarse_CS_df": coarse_df,
            "coarse_CS_p": coarse_p,
            "min_coarse_bin_n": min((r["n"] for r in coarse if r["n"] > 0), default=0),
            "min_coarse_good": min_good,
            "min_coarse_bad": min_bad,
            "coarse_bins_below_support_rule": low_bins,
            "coarse_monotonicity": monotonicity(coarse, kind),
            "expected_monotonicity": EXPECTED_MONOTONICITY.get(variable, "not imposed"),
            "constant_or_near_constant": is_constant,
            "woe_smoothing_used": any(r["woe_smoothing_alpha"] > 0 for r in coarse),
        }
        summaries.append(summary)

        for label, records, bins_df in [
            ("fine", fine, fine_tables),
            ("coarse", coarse, coarse_tables),
        ]:
            for row in records:
                out = dict(row)
                out["variable"] = variable
                out["feature_type"] = kind
                out["classing"] = label
                out["cs_statistic_for_full_partition"] = fine_cs if label == "fine" else coarse_cs
                out["cs_df"] = fine_df if label == "fine" else coarse_df
                out["cs_p_value"] = fine_p if label == "fine" else coarse_p
                bins_df.append(out)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["coarse_IV", "coarse_Gini"], ascending=[False, False]
    ).reset_index(drop=True)
    summary_df.insert(0, "rank_by_coarse_IV", np.arange(1, len(summary_df) + 1))
    return (
        summary_df,
        pd.DataFrame(fine_tables),
        pd.DataFrame(coarse_tables),
        pd.DataFrame(logs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("AppData.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("task2_output"))
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Cannot find {args.input}; pass its path with --input.")

    raw = pd.read_excel(args.input)
    train_raw, _validation = split_like_task1(raw)
    train = add_task1_features(train_raw)
    variables = candidate_variables(train)

    summary, fine_bins, coarse_bins, merge_log = metrics_and_tables(train, variables)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "variable_summary.csv", index=False)
    fine_bins.to_csv(args.output_dir / "fine_bins.csv", index=False)
    coarse_bins.to_csv(args.output_dir / "coarse_bins.csv", index=False)
    merge_log.to_csv(args.output_dir / "merge_log.csv", index=False)

    print(f"Training sample: {len(train):,}; seed={SEED}; bad rate={train['TARGET'].mean():.3%}")
    print(f"Candidate predictors screened: {len(variables)}")
    print(f"Initial fine pre-bins: up to {FINE_PREBINS}; coarse regular bins: at most {MAX_REGULAR_COARSE_BINS}")
    print(f"Minimum-bin review threshold: {MIN_BIN_SHARE:.0%} of training sample ({int(math.ceil(MIN_BIN_SHARE*len(train))):,})")
    print("\nTop 15 by coarse IV:")
    print(summary[["rank_by_coarse_IV", "variable", "feature_type", "coarse_IV", "coarse_Gini",
                   "coarse_bin_count", "missing_pct", "coarse_monotonicity"]].head(15).to_string(index=False))
    print(f"\nOutputs saved in: {args.output_dir.resolve()}")
    print("Validation data were not used for classing, merging, WoE, IV, or Gini.")


if __name__ == "__main__":
    main()
    
    
