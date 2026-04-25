from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/baseline_modeling")
PASS_FILE = DATA_DIR / "player_appearance_pass.csv"
PRESSURE_FILE = DATA_DIR / "player_appearance_behaviour_under_pressure.csv"
SHOT_FILE_CANDIDATES = [
    DATA_DIR / "player_appearance_shot_limited.csv",
    DATA_DIR / "player_appearance_shots_limited.csv",
]

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
    "fixture_order",
    "minute_in",
    "minute_out",
    "subbed",
    #"formation"
    "cumul_distance",
    "cumul_mean_max_speed",
    "last15_distance",
    "last15_mean_max_speed",
    "last15_peak_speed",
    "cumul_peak_speed",
    "last15_hsr",
    "cumul_hsr",
    #"subbed",
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

PASS_FEATURE_COLS = [
    "last_15_received_succ",
    "last_15_received_unsucc",
    "cumul_received_succ",
    "cumul_received_unsucc",
]

PRESSURE_COUNT_COLS = [
    "last_15_times_pressured",
    "last_15_pressured_succ",
    "last_15_pressured_unsucc",
    "cumul_times_pressured",
    "cumul_pressured_succ",
    "cumul_pressured_unsucc",
    "last_15_pressures_applied",
    "last_15_pressures_won",
    "last_15_pressures_lost",
    "cumul_pressures_applied",
    "cumul_pressures_won",
    "cumul_pressures_lost",
]

PRESSURE_RATE_COLS = [
    "last_15_pressured_success_rate",
    "cumul_pressured_success_rate",
    "last_15_pressure_success_rate",
    "cumul_pressure_success_rate",
]

SHOT_COUNT_COLS = [
    "last_15_shots_total",
    "cumul_shots_total",
    "last_15_shots_special",
    "cumul_shots_special",
    "last_15_shots_set_play",
    "cumul_shots_set_play",
    "last_15_shots_blocked",
    "cumul_shots_blocked",
    "last_15_shots_under_pressure",
    "cumul_shots_under_pressure",
]

SHOT_RATE_COLS = [
    "last_15_shots_under_pressure_rate",
    "cumul_shots_under_pressure_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create baseline modeling dataset with optional event feature merges."
    )
    parser.add_argument(
        "--merge-sources",
        type=str,
        default="passes",
        help="Comma-separated sources to merge: passes,pressure,none (e.g. passes,pressure).",
    )
    parser.add_argument(
        "--feature-window",
        type=str,
        default="all",
        choices=["all", "cumul", "last15"],
        help=(
            "Which time-windowed features to keep in model outputs: "
            "'all' (default), 'cumul' only, or 'last15' only."
        ),
    )
    return parser.parse_args()


def parse_merge_sources(value: str) -> set[str]:
    tokens = {token.strip().lower() for token in str(value).split(",") if token.strip()}
    valid = {"passes", "pressure", "shots", "none"}
    invalid = tokens - valid
    if invalid:
        raise ValueError(f"Unsupported merge source(s): {sorted(invalid)}. Valid: {sorted(valid)}")
    if "none" in tokens:
        return set()
    return tokens


def is_last15_feature(col: str) -> bool:
    return col.startswith("last15_") or col.startswith("last_15_")


def is_cumul_feature(col: str) -> bool:
    return col.startswith("cumul_")


def filter_model_columns_by_window(
    columns: list[str],
    target_col: str,
    window_mode: str,
) -> list[str]:
    if window_mode == "all":
        return columns

    keep: list[str] = []
    for col in columns:
        if col == target_col:
            keep.append(col)
            continue
        if window_mode == "cumul":
            if is_last15_feature(col):
                continue
            keep.append(col)
            continue
        if window_mode == "last15":
            if is_cumul_feature(col):
                continue
            keep.append(col)
            continue
        keep.append(col)
    return keep


def split_formation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "formation" not in out.columns:
        return out

    parts = out["formation"].astype("string").str.split("-")

    def to_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    defenders: list[int] = []
    midfielders: list[int] = []
    attackers: list[int] = []
    strikers: list[int] = []

    for formation_parts in parts:
        if formation_parts is None or len(formation_parts) == 0:
            defenders.append(0)
            midfielders.append(0)
            attackers.append(0)
            strikers.append(0)
            continue

        tokens = [to_int(token) for token in formation_parts]

        # 3-part formation: D-M-A, no dedicated striker
        # 4-part formation: D-M-A-S
        if len(tokens) == 3:
            d, m, a = tokens
            s = 0
        elif len(tokens) >= 4:
            d = tokens[0]
            m = tokens[1]
            a = tokens[2]
            s = tokens[3]
        else:
            d = tokens[0]
            m = tokens[1] if len(tokens) > 1 else 0
            a = 0
            s = 0

        defenders.append(d)
        midfielders.append(m)
        attackers.append(a)
        strikers.append(s)

    out["formation_defenders"] = defenders
    out["formation_midfielders"] = midfielders
    out["formation_attackers"] = attackers
    out["formation_striker"] = strikers
    return out


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
        if col not in base.columns:
            continue
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


def add_checkpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["checkpoint"] = [
        period_minute_to_checkpoint(period, minute)
        for period, minute in zip(out["period"], out["minute"])
    ]
    out["checkpoint_period_order"] = (
        out["period"].astype("string").str.lower().map(PERIOD_ORDER)
    )
    out["checkpoint_bucket"] = out["minute"].apply(minute_to_bucket)
    out = out[out["checkpoint"].notna()].copy()
    return out


def resolve_shot_file() -> Path:
    for path in SHOT_FILE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find shots file. Expected one of: "
        + ", ".join(str(p) for p in SHOT_FILE_CANDIDATES)
    )


def build_received_pass_features() -> pd.DataFrame:
    passes = pd.read_csv(PASS_FILE)
    passes = passes[passes["stage"].isin(["top", "middle"])].copy()
    passes = passes[passes["addressee_player_appearance_id"].notna()].copy()
    passes["addressee_player_appearance_id"] = passes["addressee_player_appearance_id"].astype(int)
    passes = add_checkpoint_columns(passes)
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


def build_pressure_features() -> pd.DataFrame:
    pressure = pd.read_csv(PRESSURE_FILE)
    pressure = add_checkpoint_columns(pressure)
    pressure["accurate_bool"] = parse_bool(pressure["accurate"])

    pressured = (
        pressure.groupby(
            ["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
            as_index=False,
        )
        .agg(
            last_15_times_pressured=("accurate_bool", "size"),
            last_15_pressured_succ=("accurate_bool", "sum"),
        )
    )
    pressured["last_15_pressured_unsucc"] = (
        pressured["last_15_times_pressured"] - pressured["last_15_pressured_succ"]
    )
    pressured = pressured.sort_values(
        ["player_appearance_id", "checkpoint_period_order", "checkpoint_bucket"]
    )
    pressured["cumul_times_pressured"] = (
        pressured.groupby("player_appearance_id")["last_15_times_pressured"].cumsum()
    )
    pressured["cumul_pressured_succ"] = (
        pressured.groupby("player_appearance_id")["last_15_pressured_succ"].cumsum()
    )
    pressured["cumul_pressured_unsucc"] = (
        pressured.groupby("player_appearance_id")["last_15_pressured_unsucc"].cumsum()
    )
    pressured["last_15_pressured_success_rate"] = (
        pressured["last_15_pressured_succ"] / pressured["last_15_times_pressured"]
    ).fillna(0.0)
    pressured["cumul_pressured_success_rate"] = (
        pressured["cumul_pressured_succ"] / pressured["cumul_times_pressured"]
    ).fillna(0.0)
    pressured = pressured.drop(columns=["checkpoint_period_order", "checkpoint_bucket"])

    pressure_applied = pressure[pressure["pressing_player_appearance_id"].notna()].copy()
    pressure_applied["pressing_player_appearance_id"] = (
        pressure_applied["pressing_player_appearance_id"].astype(int)
    )
    pressure_applied["pressure_won"] = (~pressure_applied["accurate_bool"]).astype(int)

    applied = (
        pressure_applied.groupby(
            [
                "pressing_player_appearance_id",
                "checkpoint",
                "checkpoint_period_order",
                "checkpoint_bucket",
            ],
            as_index=False,
        )
        .agg(
            last_15_pressures_applied=("pressure_won", "size"),
            last_15_pressures_won=("pressure_won", "sum"),
        )
        .rename(columns={"pressing_player_appearance_id": "player_appearance_id"})
    )
    applied["last_15_pressures_lost"] = (
        applied["last_15_pressures_applied"] - applied["last_15_pressures_won"]
    )
    applied = applied.sort_values(
        ["player_appearance_id", "checkpoint_period_order", "checkpoint_bucket"]
    )
    applied["cumul_pressures_applied"] = (
        applied.groupby("player_appearance_id")["last_15_pressures_applied"].cumsum()
    )
    applied["cumul_pressures_won"] = (
        applied.groupby("player_appearance_id")["last_15_pressures_won"].cumsum()
    )
    applied["cumul_pressures_lost"] = (
        applied.groupby("player_appearance_id")["last_15_pressures_lost"].cumsum()
    )
    applied["last_15_pressure_success_rate"] = (
        applied["last_15_pressures_won"] / applied["last_15_pressures_applied"]
    ).fillna(0.0)
    applied["cumul_pressure_success_rate"] = (
        applied["cumul_pressures_won"] / applied["cumul_pressures_applied"]
    ).fillna(0.0)
    applied = applied.drop(columns=["checkpoint_period_order", "checkpoint_bucket"])

    return pressured.merge(applied, on=["player_appearance_id", "checkpoint"], how="outer")


def build_shot_features() -> pd.DataFrame:
    shots_path = resolve_shot_file()
    shots = pd.read_csv(shots_path)

    # Keep only relevant spatial stages.
    shots = shots[shots["stage"].isin(["top", "middle"])].copy()
    # Remove own-goal tagged rows.
    shots = shots[shots["own_goal_player_appearance_id"].isna()].copy()

    shots = add_checkpoint_columns(shots)
    shots["under_pressure_bool"] = parse_bool(shots["under_pressure"]).astype(int)
    shots["is_special_shot"] = (
        shots["technique"].astype("string").str.strip().str.lower().isin(
            {"volley", "lob", "overhead_kick", "other"}
        )
    ).astype(int)
    shots["is_set_play"] = (
        shots["play_pattern"].astype("string").str.strip().str.lower().isin(
            {"corner_kick", "direct_free_kick", "indirect_free_kick", "penalty", "throw_in"}
        )
    ).astype(int)
    shots["is_blocked"] = shots["block_player_appearance_id"].notna().astype(int)

    grouped = (
        shots.groupby(
            ["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
            as_index=False,
        )
        .agg(
            last_15_shots_total=("id", "size"),
            last_15_shots_special=("is_special_shot", "sum"),
            last_15_shots_set_play=("is_set_play", "sum"),
            last_15_shots_blocked=("is_blocked", "sum"),
            last_15_shots_under_pressure=("under_pressure_bool", "sum"),
        )
    )

    grouped = grouped.sort_values(
        ["player_appearance_id", "checkpoint_period_order", "checkpoint_bucket"]
    )
    grouped["cumul_shots_total"] = grouped.groupby("player_appearance_id")["last_15_shots_total"].cumsum()
    grouped["cumul_shots_special"] = grouped.groupby("player_appearance_id")["last_15_shots_special"].cumsum()
    grouped["cumul_shots_set_play"] = grouped.groupby("player_appearance_id")["last_15_shots_set_play"].cumsum()
    grouped["cumul_shots_blocked"] = grouped.groupby("player_appearance_id")["last_15_shots_blocked"].cumsum()
    grouped["cumul_shots_under_pressure"] = grouped.groupby("player_appearance_id")[
        "last_15_shots_under_pressure"
    ].cumsum()

    grouped["last_15_shots_under_pressure_rate"] = (
        grouped["last_15_shots_under_pressure"] / grouped["last_15_shots_total"]
    ).fillna(0.0)
    grouped["cumul_shots_under_pressure_rate"] = (
        grouped["cumul_shots_under_pressure"] / grouped["cumul_shots_total"]
    ).fillna(0.0)

    return grouped.drop(columns=["checkpoint_period_order", "checkpoint_bucket"])


def build_feature_spine_from_selected_sources(selected_sources: set[str]) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []
    if "passes" in selected_sources:
        feature_frames.append(build_received_pass_features())
    if "pressure" in selected_sources:
        feature_frames.append(build_pressure_features())
    if "shots" in selected_sources:
        feature_frames.append(build_shot_features())

    if not feature_frames:
        raise ValueError(
            "No feature datasets selected. Use --merge-sources with one or more of: "
            "passes, pressure, shots."
        )

    spine = feature_frames[0].copy()
    for frame in feature_frames[1:]:
        spine = spine.merge(frame, on=["player_appearance_id", "checkpoint"], how="outer")
    return spine


def build_label_context_table() -> pd.DataFrame:
    base = pd.read_csv(DATA_DIR / "players_quarters_final.csv", parse_dates=["date"])
    label_cols = ["player_appearance_id", "checkpoint", "date", "fixture_id", "player_id", TARGET_COL]
    labels = base[label_cols].copy()
    labels = labels.drop_duplicates(subset=["player_appearance_id", "checkpoint"])
    labels[TARGET_COL] = pd.to_numeric(labels[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return labels


def main() -> None:
    args = parse_args()
    selected_sources = parse_merge_sources(args.merge_sources)
    window_mode = args.feature_window
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use players_quarters_final as the base dataset and merge selected aggregates into it.
    base = pd.read_csv(DATA_DIR / "players_quarters_final.csv", parse_dates=["date"])
    base = base.sort_values(
        ["date", "fixture_id", "player_appearance_id", "checkpoint"]
    ).reset_index(drop=True)
    base = split_formation_columns(base)
    base["cumul_in_game_time"] = (
        base["checkpoint"].map(CHECKPOINT_TO_ABS_MINUTE) - base["minute_in"]
    ).clip(lower=0)

    if "passes" in selected_sources:
        received_pass_features = build_received_pass_features()
        base = base.merge(received_pass_features, on=["player_appearance_id", "checkpoint"], how="left")
        for col in PASS_FEATURE_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0).astype(int)

    if "pressure" in selected_sources:
        pressure_features = build_pressure_features()
        base = base.merge(pressure_features, on=["player_appearance_id", "checkpoint"], how="left")
        for col in PRESSURE_COUNT_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0).astype(int)
        for col in PRESSURE_RATE_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0.0)

    if "shots" in selected_sources:
        shot_features = build_shot_features()
        base = base.merge(shot_features, on=["player_appearance_id", "checkpoint"], how="left")
        for col in SHOT_COUNT_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0).astype(int)
        for col in SHOT_RATE_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0.0)

    filtered_base, removed_rows = apply_row_filters(base)

    fixture_split = build_fixture_split(filtered_base)
    dataset = filtered_base.merge(
        fixture_split[["date", "fixture_id", "split", "fixture_order"]],
        on=["date", "fixture_id"],
        how="left",
    )

    model_columns = [col for col in dataset.columns if col not in DROP_FROM_MODEL]
    model_columns = filter_model_columns_by_window(
        columns=model_columns,
        target_col=TARGET_COL,
        window_mode=window_mode,
    )
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
- Extension source(s): `{",".join(sorted(selected_sources)) if selected_sources else "none"}`
- Feature-window mode: `{window_mode}`
- Output directory: [baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/data/baseline_modeling)

## Requested 60 / 20 / 20 split

- Exact 60 / 20 / 20 is not always achievable with whole-fixture chronological splits.
- The closest deterministic split on this dataset is:

{split_summary.to_string(index=False)}

## Added baseline extension features

{chr(10).join([f"- `{c}`" for c in (PASS_FEATURE_COLS if "passes" in selected_sources else []) + (PRESSURE_COUNT_COLS + PRESSURE_RATE_COLS if "pressure" in selected_sources else []) + (SHOT_COUNT_COLS + SHOT_RATE_COLS if "shots" in selected_sources else [])]) if selected_sources else "- none"}
"""

    (OUTPUT_DIR / "README.md").write_text(summary_md)
    print("## Source")
    print("- players_quarters_final.csv")
    if "passes" in selected_sources:
        print("- player_appearance_pass.csv")
    if "pressure" in selected_sources:
        print("- player_appearance_behaviour_under_pressure.csv")
    if "shots" in selected_sources:
        print(f"- {resolve_shot_file().name}")
    print(f"- feature-window: {window_mode}")
    print(f"- output: {OUTPUT_DIR}")
    print()
    print("## Requested 60 / 20 / 20 split")
    print(split_summary.to_string(index=False))
    print()
    print("## Added baseline extension features")
    if "passes" in selected_sources:
        print("- last_15_received_succ")
        print("- last_15_received_unsucc")
        print("- cumul_received_succ")
        print("- cumul_received_unsucc")
    if "pressure" in selected_sources:
        for col in PRESSURE_COUNT_COLS + PRESSURE_RATE_COLS:
            print(f"- {col}")
    if "shots" in selected_sources:
        for col in SHOT_COUNT_COLS + SHOT_RATE_COLS:
            print(f"- {col}")


if __name__ == "__main__":
    main()
