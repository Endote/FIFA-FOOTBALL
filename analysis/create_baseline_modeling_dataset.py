from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/baseline_modeling")
PASS_FILE = DATA_DIR / "player_appearance_pass.csv"
PRESSURE_FILE = DATA_DIR / "player_appearance_behaviour_under_pressure.csv"

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
    # "checkpoint",
    "fixture_order",
    "minute_in",
    "minute_out",
]

TARGET_COL = "scored_after"

ROW_FILTERS = [
    ("last15_distance", "<", 1000),
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


def build_feature_manifest(dataset: pd.DataFrame) -> pd.DataFrame:
    roles: list[dict[str, str]] = []
    for col in dataset.columns:
        if col == TARGET_COL:
            role = "target"
        elif col == "split":
            role = "split_metadata"
        elif col in DROP_FROM_MODEL:
            role = "dropped_from_model"
        else:
            role = "predictor"
        roles.append({"column": col, "role": role})
    return pd.DataFrame(roles)


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
    return build_received_event_features(PASS_FILE, pressure_suffix=False)


def build_received_pressure_features() -> pd.DataFrame:
    return build_received_event_features(PRESSURE_FILE, pressure_suffix=True)


def build_received_event_features(source_file: Path, pressure_suffix: bool) -> pd.DataFrame:
    checkpoint_grid = pd.DataFrame(
        {
            "checkpoint": ["H1_15", "H1_30", "H1_45", "H2_15", "H2_30", "H2_45", "ET1_15"],
            "checkpoint_period_order": [1, 1, 1, 2, 2, 2, 3],
            "checkpoint_bucket": [15, 30, 45, 15, 30, 45, 15],
        }
    )

    events = pd.read_csv(source_file)
    events = events[events["stage"].isin(["top", "middle"])].copy()
    events = events[events["addressee_player_appearance_id"].notna()].copy()
    events["addressee_player_appearance_id"] = events["addressee_player_appearance_id"].astype(int)
    events["checkpoint"] = [
        period_minute_to_checkpoint(period, minute)
        for period, minute in zip(events["period"], events["minute"])
    ]
    events["checkpoint_period_order"] = (
        events["period"].astype("string").str.lower().map(PERIOD_ORDER)
    )
    events["checkpoint_bucket"] = events["minute"].apply(minute_to_bucket)
    events = events[events["checkpoint"].notna()].copy()
    events["received_succ"] = parse_bool(events["accurate"]).astype(int)
    events["received_unsucc"] = 1 - events["received_succ"]

    suffix = "_pressure" if pressure_suffix else ""
    last15_succ_col = f"last_15_received_succ{suffix}"
    last15_unsucc_col = f"last_15_received_unsucc{suffix}"
    cumul_succ_col = f"cumul_received_succ{suffix}"
    cumul_unsucc_col = f"cumul_received_unsucc{suffix}"

    last15 = (
        events.groupby(
            [
                "addressee_player_appearance_id",
                "checkpoint",
                "checkpoint_period_order",
                "checkpoint_bucket",
            ],
            as_index=False,
        )
        .agg(
            **{
                last15_succ_col: ("received_succ", "sum"),
                last15_unsucc_col: ("received_unsucc", "sum"),
            }
        )
        .rename(columns={"addressee_player_appearance_id": "player_appearance_id"})
    )

    receivers = pd.DataFrame({"player_appearance_id": last15["player_appearance_id"].unique()})
    full_grid = receivers.merge(checkpoint_grid, how="cross")
    full_grid = full_grid.merge(
        last15,
        on=["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
        how="left",
    )
    full_grid[[last15_succ_col, last15_unsucc_col]] = full_grid[
        [last15_succ_col, last15_unsucc_col]
    ].fillna(0).astype(int)

    full_grid = full_grid.sort_values(
        ["player_appearance_id", "checkpoint_period_order", "checkpoint_bucket"]
    )
    full_grid[cumul_succ_col] = (
        full_grid.groupby("player_appearance_id")[last15_succ_col].cumsum()
    )
    full_grid[cumul_unsucc_col] = (
        full_grid.groupby("player_appearance_id")[last15_unsucc_col].cumsum()
    )
    return full_grid.drop(columns=["checkpoint_period_order", "checkpoint_bucket"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(DATA_DIR / "players_quarters_final.csv", parse_dates=["date"])
    base = base.sort_values(["date", "fixture_id", "player_appearance_id", "checkpoint"]).reset_index(drop=True)
    base["cumul_in_game_time"] = (
        base["checkpoint"].map(CHECKPOINT_TO_ABS_MINUTE) - base["minute_in"]
    ).clip(lower=0)
    received_pass_features = build_received_pass_features()
    received_pressure_features = build_received_pressure_features()
    base = base.merge(received_pass_features, on=["player_appearance_id", "checkpoint"], how="left")
    base = base.merge(received_pressure_features, on=["player_appearance_id", "checkpoint"], how="left")
    base[
        [
            "last_15_received_succ",
            "last_15_received_unsucc",
            "cumul_received_succ",
            "cumul_received_unsucc",
            "last_15_received_succ_pressure",
            "last_15_received_unsucc_pressure",
            "cumul_received_succ_pressure",
            "cumul_received_unsucc_pressure",
        ]
    ] = base[
        [
            "last_15_received_succ",
            "last_15_received_unsucc",
            "cumul_received_succ",
            "cumul_received_unsucc",
            "last_15_received_succ_pressure",
            "last_15_received_unsucc_pressure",
            "cumul_received_succ_pressure",
            "cumul_received_unsucc_pressure",
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
    feature_manifest = build_feature_manifest(dataset)

    model_dataset_export.to_csv(OUTPUT_DIR / "baseline_all_model_ready.csv", index=False)
    fixture_split.to_csv(OUTPUT_DIR / "baseline_fixture_split.csv", index=False)
    feature_manifest.to_csv(OUTPUT_DIR / "baseline_feature_manifest.csv", index=False)

    for split_name in ["train", "val", "test"]:
        split_model = model_dataset[model_dataset["split"] == split_name].drop(columns=["split"]).copy()
        split_model.to_csv(OUTPUT_DIR / f"baseline_{split_name}_model_ready.csv", index=False)

    summary_md = f"""# Baseline Modeling Dataset

## Source

- Source table: [players_quarters_final.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/players_quarters_final.csv)
- Extension source: [player_appearance_pass.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/player_appearance_pass.csv)
- Extension source: [player_appearance_behaviour_under_pressure.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/player_appearance_behaviour_under_pressure.csv)
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
- `last_15_received_succ_pressure`
- `last_15_received_unsucc_pressure`
- `cumul_received_succ_pressure`
- `cumul_received_unsucc_pressure`
- `cumul_in_game_time`
"""

    (OUTPUT_DIR / "README.md").write_text(summary_md)
    print("## Source")
    print("- players_quarters_final.csv")
    print("- player_appearance_pass.csv")
    print("- player_appearance_behaviour_under_pressure.csv")
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
    print("- last_15_received_succ_pressure")
    print("- last_15_received_unsucc_pressure")
    print("- cumul_received_succ_pressure")
    print("- cumul_received_unsucc_pressure")
    print("- cumul_in_game_time")
    print()
    print("## Removed from model-ready files")
    for col in DROP_FROM_MODEL:
        print(f"- {col}")


if __name__ == "__main__":
    main()
