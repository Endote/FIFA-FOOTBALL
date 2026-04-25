from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/baseline_modeling")
PASS_FILE = DATA_DIR / "player_appearance_pass.csv"

TRAIN_SHARE_TARGET = 0.60
VAL_SHARE_TARGET = 0.20
TEST_SHARE_TARGET = 0.20

DROP_FROM_MODEL = [
    "player_appearance_id",
    "player_id",
    "fixture_id",
    "date",
    "jersey_number",
    "checkpoint_period",
    "checkpoint_min",
    "checkpoint",
    "fixture_order",
    "minute_in",
    "minute_out",
]

TARGET_COL = "scored_after"

ROW_FILTERS = [
    ("last15_distance", "<", 1000),
    ("cumul_distance", "<", 1000),
    ("last15_mean_max_speed", "<", 10.3),
    ("cumul_mean_max_speed", "<", 10.3),
]

PERIOD_PREFIX = {
    "half_1": "H1",
    "half_2": "H2",
    "extra_time_1": "ET1",
    "extra_time_2": "ET2",
}

PERIOD_ORDER = {
    "half_1": 1,
    "half_2": 2,
    "extra_time_1": 3,
    "extra_time_2": 4,
}

ABS_MINUTE_TO_CHECKPOINT = {
    15: "H1_15",
    30: "H1_30",
    45: "H1_45",
    60: "H2_15",
    75: "H2_30",
    90: "H2_45",
    105: "ET1_15",
    120: "ET2_15",
}

CHECKPOINT_TO_ABS_MINUTE = {v: k for k, v in ABS_MINUTE_TO_CHECKPOINT.items()}


def build_fixture_split(base: pd.DataFrame) -> pd.DataFrame:
    fixtures = (
        base.groupby(["date", "fixture_id"])
        .agg(rows=("fixture_id", "size"), positives=(TARGET_COL, "sum"))
        .reset_index()
        .sort_values(["date", "fixture_id"])
        .reset_index(drop=True)
    )

    total_rows = int(fixtures["rows"].sum())
    train_target_rows = total_rows * TRAIN_SHARE_TARGET
    val_target_rows = total_rows * VAL_SHARE_TARGET

    train_end_idx = None
    best_train_diff = None
    for idx in range(len(fixtures)):
        train_rows = int(fixtures.loc[:idx, "rows"].sum())
        diff = abs(train_rows - train_target_rows)
        if best_train_diff is None or diff < best_train_diff:
            best_train_diff = diff
            train_end_idx = idx

    val_end_idx = None
    best_val_diff = None
    for idx in range(train_end_idx + 1, len(fixtures) - 1):
        val_rows = int(fixtures.loc[train_end_idx + 1 : idx, "rows"].sum())
        diff = abs(val_rows - val_target_rows)
        if best_val_diff is None or diff < best_val_diff:
            best_val_diff = diff
            val_end_idx = idx

    fixtures["split"] = "test"
    fixtures.loc[:train_end_idx, "split"] = "train"
    fixtures.loc[train_end_idx + 1 : val_end_idx, "split"] = "val"
    fixtures["fixture_order"] = range(1, len(fixtures) + 1)
    fixtures["cum_rows"] = fixtures["rows"].cumsum()
    fixtures["cum_share"] = fixtures["cum_rows"] / total_rows
    return fixtures


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("split")
        .agg(
            rows=("fixture_id", "size"),
            fixtures=("fixture_id", "nunique"),
            player_appearances=("player_appearance_id", "nunique"),
            players=("player_id", "nunique"),
            positives=(TARGET_COL, "sum"),
        )
        .reset_index()
    )
    summary["positive_rate"] = summary["positives"] / summary["rows"]
    summary["row_share"] = summary["rows"] / summary["rows"].sum()
    return summary


def apply_row_filters(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = pd.Series(True, index=base.index)
    for col, op, threshold in ROW_FILTERS:
        if op != "<":
            raise ValueError(f"Unsupported operator: {op}")
        mask &= base[col] < threshold

    removed = base.loc[~mask].copy()
    kept = base.loc[mask].copy()
    return kept, removed


def minute_to_bucket(minute: float) -> int | None:
    if pd.isna(minute):
        return None
    minute = float(minute)
    if minute <= 0:
        return None
    # Event data can include stoppage-time values (>45) within the same period.
    # We cap them at the 45-minute checkpoint for that period.
    if minute <= 15:
        return 15
    if minute <= 30:
        return 30
    return 45


def period_minute_to_checkpoint(period: str, minute: float) -> str | None:
    period_key = str(period).strip().lower()
    prefix = PERIOD_PREFIX.get(period_key)
    if prefix is None:
        return None
    bucket = minute_to_bucket(minute)
    if bucket is None:
        return None
    return f"{prefix}_{bucket}"


def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype("string").str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    )


def build_received_pass_features() -> pd.DataFrame:
    passes = pd.read_csv(PASS_FILE)
    passes = passes[passes["stage"].isin(["top", "middle"])].copy()
    passes = passes[passes["addressee_player_appearance_id"].notna()].copy()
    passes["addressee_player_appearance_id"] = passes["addressee_player_appearance_id"].astype(int)
    passes["checkpoint"] = [
        period_minute_to_checkpoint(period, minute)
        for period, minute in zip(passes["period"], passes["minute"])
    ]
    passes["checkpoint_period_order"] = (
        passes["period"].astype("string").str.lower().map(PERIOD_ORDER)
    )
    passes["checkpoint_bucket"] = passes["minute"].apply(minute_to_bucket)
    passes = passes[passes["checkpoint"].notna()].copy()
    passes["received_succ"] = parse_bool(passes["accurate"]).astype(int)
    passes["received_unsucc"] = 1 - passes["received_succ"]

    last15 = (
        passes.groupby(
            [
                "addressee_player_appearance_id",
                "checkpoint",
                "checkpoint_period_order",
                "checkpoint_bucket",
            ],
            as_index=False,
        )
        .agg(
            last_15_received_succ=("received_succ", "sum"),
            last_15_received_unsucc=("received_unsucc", "sum"),
        )
        .rename(columns={"addressee_player_appearance_id": "player_appearance_id"})
    )

    last15 = last15.sort_values(
        ["player_appearance_id", "checkpoint_period_order", "checkpoint_bucket"]
    )
    last15["cumul_received_succ"] = (
        last15.groupby("player_appearance_id")["last_15_received_succ"].cumsum()
    )
    last15["cumul_received_unsucc"] = (
        last15.groupby("player_appearance_id")["last_15_received_unsucc"].cumsum()
    )
    return last15.drop(columns=["checkpoint_period_order", "checkpoint_bucket"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(DATA_DIR / "players_quarters_final.csv", parse_dates=["date"])
    base = base.sort_values(["date", "fixture_id", "player_appearance_id", "checkpoint"]).reset_index(drop=True)
    base["cumul_in_game_time"] = (
        base["checkpoint"].map(CHECKPOINT_TO_ABS_MINUTE) - base["minute_in"]
    ).clip(lower=0)
    received_pass_features = build_received_pass_features()
    base = base.merge(received_pass_features, on=["player_appearance_id", "checkpoint"], how="left")
    base[
        [
            "last_15_received_succ",
            "last_15_received_unsucc",
            "cumul_received_succ",
            "cumul_received_unsucc",
        ]
    ] = base[
        [
            "last_15_received_succ",
            "last_15_received_unsucc",
            "cumul_received_succ",
            "cumul_received_unsucc",
        ]
    ].fillna(0).astype(int)
    filtered_base, removed_rows = apply_row_filters(base)

    fixture_split = build_fixture_split(filtered_base)
    dataset = filtered_base.merge(
        fixture_split[["date", "fixture_id", "split", "fixture_order"]],
        on=["date", "fixture_id"],
        how="left",
    )

    model_columns = [col for col in dataset.columns if col not in DROP_FROM_MODEL]
    model_dataset = dataset[model_columns].copy()
    model_dataset_export = model_dataset.drop(columns=["split"]).copy()

    split_summary = summarize_split(dataset)

    model_dataset_export.to_csv(OUTPUT_DIR / "baseline_all_model_ready.csv", index=False)
    fixture_split.to_csv(OUTPUT_DIR / "baseline_fixture_split.csv", index=False)
    split_summary.to_csv(OUTPUT_DIR / "baseline_split_summary.csv", index=False)
    removed_rows.to_csv(OUTPUT_DIR / "baseline_removed_rows_quality_filters.csv", index=False)

    for split_name in ["train", "val", "test"]:
        split_model = model_dataset[model_dataset["split"] == split_name].drop(columns=["split"]).copy()
        split_model.to_csv(OUTPUT_DIR / f"baseline_{split_name}_model_ready.csv", index=False)

    feature_manifest = pd.DataFrame(
        {
            "column": dataset.columns,
            "role": [
                "target"
                if col == TARGET_COL
                else "dropped_from_model"
                if col in DROP_FROM_MODEL
                else "split_metadata"
                if col == "split"
                else "predictor"
                for col in dataset.columns
            ],
        }
    )
    feature_manifest.to_csv(OUTPUT_DIR / "baseline_feature_manifest.csv", index=False)

    summary_md = f"""# Baseline Modeling Dataset

## Source

- Source table: [players_quarters_final.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/players_quarters_final.csv)
- Extension source: [player_appearance_pass.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/player_appearance_pass.csv)
- Output directory: [baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/data/baseline_modeling)

## Requested 60 / 20 / 20 split

- Exact 60 / 20 / 20 is not always achievable with whole-fixture chronological splits.
- The closest deterministic split on this dataset is:

{split_summary.to_string(index=False)}

## Added baseline extension features

- `last_15_received_succ`
- `last_15_received_unsucc`
- `cumul_received_succ`
- `cumul_received_unsucc`
- `cumul_in_game_time`
"""

    (OUTPUT_DIR / "README.md").write_text(summary_md)
    print("## Source")
    print("- players_quarters_final.csv")
    print("- player_appearance_pass.csv")
    print(f"- output: {OUTPUT_DIR}")
    print()
    print("## Requested 60 / 20 / 20 split")
    print(split_summary.to_string(index=False))
    print()
    print("## Added baseline extension features")
    print("- last_15_received_succ")
    print("- last_15_received_unsucc")
    print("- cumul_received_succ")
    print("- cumul_received_unsucc")
    print("- cumul_in_game_time")


if __name__ == "__main__":
    main()
