from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/baseline_modeling")
PASS_FILE = DATA_DIR / "player_appearance_pass.csv"
PRESSURE_FILE = DATA_DIR / "player_appearance_behaviour_under_pressure.csv"
RUN_FILE = DATA_DIR / "player_appearance_run.csv"
SHOT_FILE_CANDIDATES = [
    DATA_DIR / "player_appearance_shot_limited.csv",
    DATA_DIR / "player_appearance_shots_limited.csv",
]

TRAIN_SHARE_TARGET = 0.60
VAL_SHARE_TARGET = 0.20
TEST_SHARE_TARGET = 0.20

KEEP_BASE_MODEL = [
    "position",
    "checkpoint",
    "formation",

    "cumul_received_unsucc",

    "cumul_in_game_time",

    "top_distance_share",
    "avg_top_sprint_distance",

    "possessions_with_2plus_runs",
    
    "cumul_shots_on_target",
    "cumul_shots_total",

    "cumul_shots_blocked",
    "cumul_shots_under_pressure",
    # "cumul_pressure_events",
    # "last15_pressure_events",
    # "cumul_pressure_turnover_rate",
    # "last15_pressure_turnover_rate",
    # "cumul_pressure_forward_rate",
    # "last15_pressure_forward_rate",
    # "pressure_forward_minus_backward",
    # "pressure_escape_score",
    # "mean_abs_pass_angle_under_pressure",
    # "top_third_pressure_count",
    # "top_third_pressure_turnover_rate",

    "player_share_team_cumul_shots",
    "player_share_team_shots_on_target",
    "player_share_team_top_distance",
    "player_rank_team_cumul_shots",
    "player_rank_team_top_distance_share",
    "player_z_team_top_distance_share",
    "player_z_team_shots_total",
    # "team_total_cumul_shots",
    # "team_total_top_runs",
    # "opponent_total_cumul_shots",
    # "team_minus_opponent_shot_load",

]



STRUCTURAL_DROP_ALWAYS = [
    "fixture_id",
    "date",
    "player_id",
    "jersey_number",
    "player_appearance_id",
    "id",
    "checkpoint_period",
    "checkpoint_min",
    "is_home",
    "minute_in",
    "minute_out",
    "cumul_mean_max_speed",
    "cumul_peak_speed",

    "last15_top_sprint_count",
    "last15_middle_sprint_count",
    "last15_middle_hsr_count",
    "last15_bottom_sprint_count",
    "last15_bottom_hsr_count",
    "cumul_bottom_sprint_count",
    "cumul_bottom_hsr_count",
    "cumul_top_sprint_count",
    "cumul_top_hsr_count",
    "cumul_middle_sprint_count",
    "cumul_middle_hsr_count",

    "cumul_pressures_applied",
    "cumul_pressures_won",
    "cumul_pressures_lost",

    "last_15_shots_special",
    "last_15_shots_set_play",
    "cumul_shots_special",
    "cumul_shots_set_play",

    # "cumul_top_run_share",
    "cumul_middle_run_share",
    "cumul_bottom_run_share",
    "cumul_top_sprint_share",
    # "cumul_top_hsr_share",

    "top_run_repeat_possession_rate",
    "cumul_unique_run_possessions",
    "last15_unique_run_possessions",
    "runs_per_possession",
    "sprints_per_possession",

    "cumul_pressured_success_rate",

    # "last15_top_run_share",
    "last15_top_sprint_share",
    # "share_of_possessions_with_top_run",
    "share_of_possessions_with_sprint",
    "top_runs_per_possession",
    "possessions_with_2plus_top_runs",
    "top_sprint_distance",
    "top_hsr_distance",
    "distance_per_run",
    "distance_per_possession",
    "sprint_distance_share",
    "last15_top_hsr_count",

    "possessions_with_sprint_and_hsr",
    "cumul_pressure_success_rate",
    "last_15_pressured_success_rate",
    "last_15_pressure_success_rate",

    "cumul_top_hsr_share",
    "cumul_top_run_share",
    "last15_top_run_share",

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
}

CHECKPOINT_TO_ABS_MINUTE = {v: k for k, v in ABS_MINUTE_TO_CHECKPOINT.items()}

PASS_FEATURE_COLS = [
    "last_15_received_succ",
    "last_15_received_unsucc",
    "cumul_received_succ",
    "cumul_received_unsucc",
    "cumul_pass_top_count",
    "cumul_pass_middle_count",
    "cumul_pass_top_accuracy_rate",
    "cumul_pass_middle_accuracy_rate",
]
PASS_COUNT_COLS = [
    "last_15_received_succ",
    "last_15_received_unsucc",
    "cumul_received_succ",
    "cumul_received_unsucc",
    "cumul_pass_top_count",
    "cumul_pass_middle_count",
]
PASS_RATE_COLS = [
    "cumul_pass_top_accuracy_rate",
    "cumul_pass_middle_accuracy_rate",
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
    "cumul_pressure_turnover_rate",
    "last15_pressure_turnover_rate",
    "cumul_pressure_forward_rate",
    "last15_pressure_forward_rate",
    "top_third_pressure_turnover_rate",
]
PRESSURE_CONTEXT_COUNT_COLS = [
    "cumul_pressure_events",
    "last15_pressure_events",
    "pressure_forward_minus_backward",
    "pressure_escape_score",
    "top_third_pressure_count",
]
PRESSURE_CONTEXT_VALUE_COLS = [
    "mean_abs_pass_angle_under_pressure",
]
PRESSURE_CARRY_FORWARD_COLS = [
    "cumul_pressure_events",
    "cumul_pressure_turnover_rate",
    "cumul_pressure_forward_rate",
    "pressure_forward_minus_backward",
    "pressure_escape_score",
    "mean_abs_pass_angle_under_pressure",
    "top_third_pressure_count",
    "top_third_pressure_turnover_rate",
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
TEAM_BASE_CONTEXT_FEATURE_COLS = [
    "player_share_team_shots_on_target",
]
TEAM_SHOT_CONTEXT_FEATURE_COLS = [
    "player_share_team_cumul_shots",
    "player_rank_team_cumul_shots",
    "player_z_team_shots_total",
    "team_total_cumul_shots",
    "opponent_total_cumul_shots",
    "team_minus_opponent_shot_load",
]
SHOT_CUMUL_COUNT_COLS = [
    "cumul_shots_total",
    "cumul_shots_special",
    "cumul_shots_set_play",
    "cumul_shots_blocked",
    "cumul_shots_under_pressure",
]
SHOT_CUMUL_RATE_COLS = [
    "cumul_shots_under_pressure_rate",
]
RUN_COUNT_COLS = [
    "last15_top_sprint_count",
    "last15_top_hsr_count",
    "last15_middle_sprint_count",
    "last15_middle_hsr_count",
    "last15_bottom_sprint_count",
    "last15_bottom_hsr_count",
    "cumul_top_sprint_count",
    "cumul_top_hsr_count",
    "cumul_middle_sprint_count",
    "cumul_middle_hsr_count",
    "cumul_bottom_sprint_count",
    "cumul_bottom_hsr_count",
]
RUN_SHARE_COLS = [
    "cumul_top_run_share",
    "cumul_middle_run_share",
    "cumul_bottom_run_share",
    "cumul_top_sprint_share",
    "cumul_top_hsr_share",
    "cumul_bottom_sprint_share",
    "last15_top_run_share",
    "last15_top_sprint_share",
]
RUN_DISTANCE_COLS = [
    "top_sprint_distance",
    "top_hsr_distance",
    "distance_per_run",
    "distance_per_possession",
    "top_distance_share",
    "middle_distance_share",
    "bottom_distance_share",
    "sprint_distance_share",
    "avg_top_sprint_distance",
]
RUN_POSSESSION_COLS = [
    "cumul_unique_run_possessions",
    "last15_unique_run_possessions",
    "runs_per_possession",
    "sprints_per_possession",
    "top_runs_per_possession",
    "share_of_possessions_with_top_run",
    "share_of_possessions_with_sprint",
    "possessions_with_2plus_runs",
    "possessions_with_2plus_top_runs",
    "possessions_with_sprint_and_hsr",
    "top_run_repeat_possession_rate",
]
TEAM_RUN_CONTEXT_FEATURE_COLS = [
    "player_share_team_top_distance",
    "player_rank_team_top_distance_share",
    "player_z_team_top_distance_share",
    "team_total_top_runs",
]
RUN_CUMUL_COLS = [
    "cumul_top_sprint_count",
    "cumul_top_hsr_count",
    "cumul_middle_sprint_count",
    "cumul_middle_hsr_count",
    "cumul_bottom_sprint_count",
    "cumul_bottom_hsr_count",
    "cumul_unique_run_possessions",
]
CHECKPOINT_ORDER = {checkpoint: idx for idx, checkpoint in enumerate(ABS_MINUTE_TO_CHECKPOINT.values(), start=1)}
VALID_CHECKPOINT_MINUTES = sorted(ABS_MINUTE_TO_CHECKPOINT.keys())

CSV_NULL_TOKENS = ["NULL", "null", ""]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create baseline modeling dataset with optional event feature merges."
    )
    parser.add_argument(
        "--merge-sources",
        type=str,
        default="passes",
        help="Comma-separated sources to merge: passes,pressure,runs,shots,none (e.g. passes,pressure,runs).",
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
    valid = {"passes", "pressure", "shots", "runs", "none"}
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


def get_source_feature_columns(selected_sources: set[str]) -> list[str]:
    feature_cols: list[str] = TEAM_BASE_CONTEXT_FEATURE_COLS.copy()
    if "passes" in selected_sources:
        feature_cols.extend(PASS_FEATURE_COLS)
    if "pressure" in selected_sources:
        feature_cols.extend(PRESSURE_COUNT_COLS + PRESSURE_RATE_COLS)
    if "runs" in selected_sources:
        feature_cols.extend(
            RUN_COUNT_COLS + RUN_SHARE_COLS + RUN_DISTANCE_COLS + RUN_POSSESSION_COLS + TEAM_RUN_CONTEXT_FEATURE_COLS
        )
    if "shots" in selected_sources:
        feature_cols.extend(SHOT_COUNT_COLS + SHOT_RATE_COLS + TEAM_SHOT_CONTEXT_FEATURE_COLS)
    return feature_cols


def get_added_extension_features(selected_sources: set[str], window_mode: str) -> list[str]:
    return filter_model_columns_by_window(
        columns=get_source_feature_columns(selected_sources),
        target_col=TARGET_COL,
        window_mode=window_mode,
    )


def build_model_keep_columns(
    dataset_columns: list[str],
    selected_sources: set[str],
    window_mode: str,
) -> list[str]:
    structural_drop = set(STRUCTURAL_DROP_ALWAYS + ["split", "fixture_order"])

    if not KEEP_BASE_MODEL:
        return filter_model_columns_by_window(
            columns=[
                col
                for col in dataset_columns
                if col not in structural_drop
            ],
            target_col=TARGET_COL,
            window_mode=window_mode,
        )

    desired_columns = filter_model_columns_by_window(
        columns=KEEP_BASE_MODEL + [TARGET_COL],
        target_col=TARGET_COL,
        window_mode=window_mode,
    )

    available = set(dataset_columns)
    keep: list[str] = []
    seen: set[str] = set()
    for col in desired_columns:
        if col in available and col not in structural_drop and col not in seen:
            keep.append(col)
            seen.add(col)
    return keep


def get_included_extension_features(
    model_columns: list[str],
    selected_sources: set[str],
) -> list[str]:
    source_feature_set = set(get_source_feature_columns(selected_sources))
    return [col for col in model_columns if col in source_feature_set]


def carry_forward_cumulative_features(df: pd.DataFrame, cumulative_cols: list[str]) -> pd.DataFrame:
    valid_cols = list(dict.fromkeys(col for col in cumulative_cols if col in df.columns))
    if not valid_cols:
        return df

    out = df.copy()
    out["_checkpoint_order"] = out["checkpoint"].map(CHECKPOINT_ORDER)
    out = out.sort_values(["player_appearance_id", "_checkpoint_order"]).copy()
    out[valid_cols] = out.groupby("player_appearance_id")[valid_cols].ffill()
    out = out.sort_index().drop(columns=["_checkpoint_order"])
    return out


def safe_group_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, pd.NA)).fillna(0.0)


def add_team_level_context_features(base: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["fixture_id", "checkpoint", "is_home"]
    required_group_cols = set(group_keys)
    if not required_group_cols.issubset(base.columns):
        return base

    out = base.copy()

    shot_source_cols = ["cumul_shots_total", "cumul_shots_on_target"]
    run_source_cols = ["top_distance_share", "cumul_top_sprint_count", "cumul_top_hsr_count"]

    for col in shot_source_cols + run_source_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if "cumul_shots_total" in out.columns:
        out["team_total_cumul_shots"] = out.groupby(group_keys)["cumul_shots_total"].transform("sum")
        out["player_share_team_cumul_shots"] = safe_group_share(
            out["cumul_shots_total"],
            out["team_total_cumul_shots"],
        )
        out["player_rank_team_cumul_shots"] = (
            out.groupby(group_keys)["cumul_shots_total"]
            .rank(method="dense", ascending=False)
            .fillna(0)
            .astype(int)
        )
        shot_means = out.groupby(group_keys)["cumul_shots_total"].transform("mean")
        shot_stds = out.groupby(group_keys)["cumul_shots_total"].transform(lambda s: s.std(ddof=0))
        shot_stds = pd.to_numeric(shot_stds, errors="coerce").mask(lambda s: s == 0)
        out["player_z_team_shots_total"] = (
            (out["cumul_shots_total"] - shot_means).div(shot_stds).fillna(0.0)
        )

        opponent_shots = (
            out[group_keys + ["team_total_cumul_shots"]]
            .drop_duplicates()
            .rename(
                columns={
                    "is_home": "opponent_is_home",
                    "team_total_cumul_shots": "opponent_total_cumul_shots",
                }
            )
        )
        opponent_shots["is_home"] = ~opponent_shots["opponent_is_home"].astype(bool)
        opponent_shots = opponent_shots.drop(columns=["opponent_is_home"])
        out = out.merge(opponent_shots, on=group_keys, how="left")
        out["opponent_total_cumul_shots"] = out["opponent_total_cumul_shots"].fillna(0.0)
        out["team_minus_opponent_shot_load"] = (
            out["team_total_cumul_shots"] - out["opponent_total_cumul_shots"]
        )

    if "cumul_shots_on_target" in out.columns:
        team_total_cumul_shots_on_target = out.groupby(group_keys)["cumul_shots_on_target"].transform("sum")
        out["player_share_team_shots_on_target"] = safe_group_share(
            out["cumul_shots_on_target"],
            team_total_cumul_shots_on_target,
        )

    if {"cumul_top_sprint_count", "cumul_top_hsr_count"}.issubset(out.columns):
        out["_player_cumul_top_runs"] = out["cumul_top_sprint_count"] + out["cumul_top_hsr_count"]
        out["team_total_top_runs"] = out.groupby(group_keys)["_player_cumul_top_runs"].transform("sum")
        out = out.drop(columns=["_player_cumul_top_runs"])

    if "top_distance_share" in out.columns:
        team_total_top_distance_share = out.groupby(group_keys)["top_distance_share"].transform("sum")
        out["player_share_team_top_distance"] = safe_group_share(
            out["top_distance_share"],
            team_total_top_distance_share,
        )
        out["player_rank_team_top_distance_share"] = (
            out.groupby(group_keys)["top_distance_share"]
            .rank(method="dense", ascending=False)
            .fillna(0)
            .astype(int)
        )
        top_distance_means = out.groupby(group_keys)["top_distance_share"].transform("mean")
        top_distance_stds = (
            out.groupby(group_keys)["top_distance_share"]
            .transform(lambda s: s.std(ddof=0))
        )
        top_distance_stds = pd.to_numeric(top_distance_stds, errors="coerce").mask(lambda s: s == 0)
        out["player_z_team_top_distance_share"] = (
            (out["top_distance_share"] - top_distance_means).div(top_distance_stds).fillna(0.0)
        )

    return out


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
    if minute <= 15:
        return 15
    if minute <= 30:
        return 30
    if minute <= 45:
        return 45
    return None


def period_minute_to_abs_minute(period: str, minute: float) -> float | None:
    if pd.isna(minute):
        return None

    minute = float(minute)
    if minute <= 0:
        return None

    period_key = str(period).strip().lower()
    offsets = {
        "half_1": 0,
        "half_2": 45,
        "extra_time_1": 90,
        "extra_time_2": 105,
    }
    offset = offsets.get(period_key)
    if offset is None:
        return None
    return offset + minute


def abs_minute_to_checkpoint(abs_minute: float) -> str | None:
    if abs_minute is None or pd.isna(abs_minute):
        return None
    abs_minute = float(abs_minute)
    if abs_minute <= 0:
        return None

    for checkpoint_minute in VALID_CHECKPOINT_MINUTES:
        if abs_minute <= checkpoint_minute:
            return ABS_MINUTE_TO_CHECKPOINT[checkpoint_minute]
    return None


def period_minute_to_checkpoint(period: str, minute: float) -> str | None:
    abs_minute = period_minute_to_abs_minute(period, minute)
    return abs_minute_to_checkpoint(abs_minute)


def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype("string").str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    )


def add_checkpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_abs_minute"] = [
        period_minute_to_abs_minute(period, minute)
        for period, minute in zip(out["period"], out["minute"])
    ]
    out["checkpoint"] = [
        abs_minute_to_checkpoint(abs_minute)
        for abs_minute in out["event_abs_minute"]
    ]
    out["checkpoint_period_order"] = out["checkpoint"].map(CHECKPOINT_ORDER)
    out["checkpoint_bucket"] = out["checkpoint"].map(CHECKPOINT_TO_ABS_MINUTE)
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


def read_csv_with_nulls(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=CSV_NULL_TOKENS)


def build_received_pass_features() -> pd.DataFrame:
    passes = read_csv_with_nulls(PASS_FILE)
    passes = passes[passes["stage"].isin(["top", "middle"])].copy()
    sender_passes = add_checkpoint_columns(passes)
    sender_passes["accurate_bool"] = parse_bool(sender_passes["accurate"]).astype(int)

    sender_grouped = (
        sender_passes.groupby(
            ["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
            as_index=False,
        )
        .agg(
            last_15_pass_top_count=(
                "id",
                lambda s: int((sender_passes.loc[s.index, "stage"] == "top").sum()),
            ),
            last_15_pass_top_succ=(
                "accurate_bool",
                lambda s: int(
                    sender_passes.loc[s.index, "accurate_bool"][sender_passes.loc[s.index, "stage"] == "top"].sum()
                ),
            ),
            last_15_pass_middle_count=(
                "id",
                lambda s: int((sender_passes.loc[s.index, "stage"] == "middle").sum()),
            ),
            last_15_pass_middle_succ=(
                "accurate_bool",
                lambda s: int(
                    sender_passes.loc[s.index, "accurate_bool"][
                        sender_passes.loc[s.index, "stage"] == "middle"
                    ].sum()
                ),
            ),
        )
    )
    sender_grouped = sender_grouped.sort_values(
        ["player_appearance_id", "checkpoint_period_order", "checkpoint_bucket"]
    )
    sender_grouped["cumul_pass_top_count"] = (
        sender_grouped.groupby("player_appearance_id")["last_15_pass_top_count"].cumsum()
    )
    sender_grouped["cumul_pass_top_succ"] = (
        sender_grouped.groupby("player_appearance_id")["last_15_pass_top_succ"].cumsum()
    )
    sender_grouped["cumul_pass_middle_count"] = (
        sender_grouped.groupby("player_appearance_id")["last_15_pass_middle_count"].cumsum()
    )
    sender_grouped["cumul_pass_middle_succ"] = (
        sender_grouped.groupby("player_appearance_id")["last_15_pass_middle_succ"].cumsum()
    )
    sender_grouped["cumul_pass_top_accuracy_rate"] = (
        sender_grouped["cumul_pass_top_succ"] / sender_grouped["cumul_pass_top_count"].replace(0, pd.NA)
    ).fillna(0.0)
    sender_grouped["cumul_pass_middle_accuracy_rate"] = (
        sender_grouped["cumul_pass_middle_succ"] / sender_grouped["cumul_pass_middle_count"].replace(0, pd.NA)
    ).fillna(0.0)
    sender_grouped = sender_grouped.drop(
        columns=[
            "checkpoint_period_order",
            "checkpoint_bucket",
            "last_15_pass_top_count",
            "last_15_pass_top_succ",
            "last_15_pass_middle_count",
            "last_15_pass_middle_succ",
            "cumul_pass_top_succ",
            "cumul_pass_middle_succ",
        ]
    )

    receiver_passes = passes[passes["addressee_player_appearance_id"].notna()].copy()
    receiver_passes["addressee_player_appearance_id"] = receiver_passes["addressee_player_appearance_id"].astype(int)
    receiver_passes = add_checkpoint_columns(receiver_passes)
    receiver_passes["received_succ"] = parse_bool(receiver_passes["accurate"]).astype(int)
    receiver_passes["received_unsucc"] = 1 - receiver_passes["received_succ"]

    receiver_grouped = (
        receiver_passes.groupby(
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

    receiver_grouped = receiver_grouped.sort_values(
        ["player_appearance_id", "checkpoint_period_order", "checkpoint_bucket"]
    )
    receiver_grouped["cumul_received_succ"] = (
        receiver_grouped.groupby("player_appearance_id")["last_15_received_succ"].cumsum()
    )
    receiver_grouped["cumul_received_unsucc"] = (
        receiver_grouped.groupby("player_appearance_id")["last_15_received_unsucc"].cumsum()
    )
    receiver_grouped = receiver_grouped.drop(columns=["checkpoint_period_order", "checkpoint_bucket"])

    return receiver_grouped.merge(sender_grouped, on=["player_appearance_id", "checkpoint"], how="outer")


def build_pressure_features() -> pd.DataFrame:
    pressure = read_csv_with_nulls(PRESSURE_FILE)
    pressure = add_checkpoint_columns(pressure)
    pressure["accurate_bool"] = parse_bool(pressure["accurate"])
    pressure["press_induced_outcome"] = (
        pressure["press_induced_outcome"].astype("string").str.strip().str.lower()
    )
    pressure["stage"] = pressure["stage"].astype("string").str.strip().str.lower()
    pressure["pass_angle"] = pd.to_numeric(pressure["pass_angle"], errors="coerce")
    pressure["abs_pass_angle"] = pressure["pass_angle"].abs()
    pressure["pressure_turnover"] = pressure["press_induced_outcome"].eq("turnover").fillna(False).astype(int)
    pressure["pressure_forward"] = pressure["press_induced_outcome"].eq("forward_pass").fillna(False).astype(int)
    pressure["pressure_backward"] = pressure["press_induced_outcome"].eq("backward_pass").fillna(False).astype(int)
    pressure["pressure_ball_carry"] = pressure["press_induced_outcome"].eq("ball_carry").fillna(False).astype(int)
    pressure["top_third_pressure"] = pressure["stage"].eq("top").fillna(False).astype(int)
    pressure["top_third_pressure_turnover"] = (
        pressure["top_third_pressure"] * pressure["pressure_turnover"]
    )

    pressured = (
        pressure.groupby(
            ["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
            as_index=False,
        )
        .agg(
            last_15_times_pressured=("accurate_bool", "size"),
            last_15_pressured_succ=("accurate_bool", "sum"),
            last15_pressure_events=("id", "size"),
            last15_pressure_turnovers=("pressure_turnover", "sum"),
            last15_pressure_forward=("pressure_forward", "sum"),
            last15_pressure_backward=("pressure_backward", "sum"),
            last15_pressure_ball_carry=("pressure_ball_carry", "sum"),
            last15_abs_pass_angle_sum=("abs_pass_angle", "sum"),
            last15_abs_pass_angle_count=("abs_pass_angle", "count"),
            last15_top_third_pressure_turnovers=("top_third_pressure_turnover", "sum"),
            top_third_pressure_count=("top_third_pressure", "sum"),
        )
    )
    pressured["last_15_pressured_unsucc"] = (
        pressured["last_15_times_pressured"] - pressured["last_15_pressured_succ"]
    )
    pressured["last15_pressure_turnover_rate"] = (
        pressured["last15_pressure_turnovers"] / pressured["last15_pressure_events"].replace(0, pd.NA)
    ).fillna(0.0)
    pressured["last15_pressure_forward_rate"] = (
        pressured["last15_pressure_forward"] / pressured["last15_pressure_events"].replace(0, pd.NA)
    ).fillna(0.0)
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
    pressured["cumul_pressure_events"] = (
        pressured.groupby("player_appearance_id")["last15_pressure_events"].cumsum()
    )
    pressured["cumul_pressure_turnovers"] = (
        pressured.groupby("player_appearance_id")["last15_pressure_turnovers"].cumsum()
    )
    pressured["cumul_pressure_forward"] = (
        pressured.groupby("player_appearance_id")["last15_pressure_forward"].cumsum()
    )
    pressured["cumul_pressure_backward"] = (
        pressured.groupby("player_appearance_id")["last15_pressure_backward"].cumsum()
    )
    pressured["cumul_pressure_ball_carry"] = (
        pressured.groupby("player_appearance_id")["last15_pressure_ball_carry"].cumsum()
    )
    pressured["cumul_abs_pass_angle_sum"] = (
        pressured.groupby("player_appearance_id")["last15_abs_pass_angle_sum"].cumsum()
    )
    pressured["cumul_abs_pass_angle_count"] = (
        pressured.groupby("player_appearance_id")["last15_abs_pass_angle_count"].cumsum()
    )
    pressured["cumul_top_third_pressure_turnovers"] = (
        pressured.groupby("player_appearance_id")["last15_top_third_pressure_turnovers"].cumsum()
    )
    pressured["top_third_pressure_count"] = (
        pressured.groupby("player_appearance_id")["top_third_pressure_count"].cumsum()
    )
    pressured["last_15_pressured_success_rate"] = (
        pressured["last_15_pressured_succ"] / pressured["last_15_times_pressured"]
    ).fillna(0.0)
    pressured["cumul_pressured_success_rate"] = (
        pressured["cumul_pressured_succ"] / pressured["cumul_times_pressured"]
    ).fillna(0.0)
    pressured["cumul_pressure_turnover_rate"] = (
        pressured["cumul_pressure_turnovers"] / pressured["cumul_pressure_events"].replace(0, pd.NA)
    ).fillna(0.0)
    pressured["cumul_pressure_forward_rate"] = (
        pressured["cumul_pressure_forward"] / pressured["cumul_pressure_events"].replace(0, pd.NA)
    ).fillna(0.0)
    pressured["pressure_forward_minus_backward"] = (
        pressured["cumul_pressure_forward"] - pressured["cumul_pressure_backward"]
    )
    pressured["pressure_escape_score"] = (
        pressured["cumul_pressure_forward"]
        + pressured["cumul_pressure_ball_carry"]
        - pressured["cumul_pressure_turnovers"]
        - pressured["cumul_pressure_backward"]
    )
    pressured["mean_abs_pass_angle_under_pressure"] = (
        pressured["cumul_abs_pass_angle_sum"] / pressured["cumul_abs_pass_angle_count"].replace(0, pd.NA)
    ).fillna(0.0)
    pressured["top_third_pressure_turnover_rate"] = (
        pressured["cumul_top_third_pressure_turnovers"] / pressured["top_third_pressure_count"].replace(0, pd.NA)
    ).fillna(0.0)
    pressured = pressured.drop(
        columns=[
            "checkpoint_period_order",
            "checkpoint_bucket",
            "last15_pressure_turnovers",
            "last15_pressure_forward",
            "last15_pressure_backward",
            "last15_pressure_ball_carry",
            "last15_abs_pass_angle_sum",
            "last15_abs_pass_angle_count",
            "last15_top_third_pressure_turnovers",
            "cumul_pressure_turnovers",
            "cumul_pressure_forward",
            "cumul_pressure_backward",
            "cumul_pressure_ball_carry",
            "cumul_abs_pass_angle_sum",
            "cumul_abs_pass_angle_count",
            "cumul_top_third_pressure_turnovers",
        ]
    )

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
    shots = read_csv_with_nulls(shots_path)

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


def build_run_features() -> pd.DataFrame:
    runs = read_csv_with_nulls(RUN_FILE)
    runs = runs[runs["stage"].isin(["top", "middle", "bottom"])].copy()
    runs = runs[runs["run_type"].isin(["sprint", "hsr"])].copy()
    runs = add_checkpoint_columns(runs)
    runs["possession"] = pd.to_numeric(runs["possession"], errors="coerce")
    runs["distance"] = pd.to_numeric(runs["distance"], errors="coerce")
    runs["max_speed"] = pd.to_numeric(runs["max_speed"], errors="coerce")
    safe_distance_mask = runs["distance"].notna() & runs["max_speed"].notna() & (runs["distance"] < 1000) & (runs["max_speed"] <= 10.3)
    safe_runs = runs[safe_distance_mask].copy()

    grouped = (
        runs.groupby(
            ["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
            as_index=False,
        )
        .agg(
            last15_top_sprint_count=(
                "id",
                lambda s: int(
                    ((runs.loc[s.index, "stage"] == "top") & (runs.loc[s.index, "run_type"] == "sprint")).sum()
                ),
            ),
            last15_top_hsr_count=(
                "id",
                lambda s: int(
                    ((runs.loc[s.index, "stage"] == "top") & (runs.loc[s.index, "run_type"] == "hsr")).sum()
                ),
            ),
            last15_middle_sprint_count=(
                "id",
                lambda s: int(
                    ((runs.loc[s.index, "stage"] == "middle") & (runs.loc[s.index, "run_type"] == "sprint")).sum()
                ),
            ),
            last15_middle_hsr_count=(
                "id",
                lambda s: int(
                    ((runs.loc[s.index, "stage"] == "middle") & (runs.loc[s.index, "run_type"] == "hsr")).sum()
                ),
            ),
            last15_bottom_sprint_count=(
                "id",
                lambda s: int(
                    ((runs.loc[s.index, "stage"] == "bottom") & (runs.loc[s.index, "run_type"] == "sprint")).sum()
                ),
            ),
            last15_bottom_hsr_count=(
                "id",
                lambda s: int(
                    ((runs.loc[s.index, "stage"] == "bottom") & (runs.loc[s.index, "run_type"] == "hsr")).sum()
                ),
            ),
        )
    )
    safe_grouped = (
        safe_runs.groupby(
            ["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
            as_index=False,
        )
        .agg(
            top_sprint_distance=(
                "distance",
                lambda s: float(
                    safe_runs.loc[s.index, "distance"][
                        (safe_runs.loc[s.index, "stage"] == "top")
                        & (safe_runs.loc[s.index, "run_type"] == "sprint")
                    ].sum()
                ),
            ),
            top_hsr_distance=(
                "distance",
                lambda s: float(
                    safe_runs.loc[s.index, "distance"][
                        (safe_runs.loc[s.index, "stage"] == "top")
                        & (safe_runs.loc[s.index, "run_type"] == "hsr")
                    ].sum()
                ),
            ),
            safe_total_distance=("distance", "sum"),
            safe_run_count=("id", "size"),
            safe_total_sprint_distance=(
                "distance",
                lambda s: float(
                    safe_runs.loc[s.index, "distance"][
                        safe_runs.loc[s.index, "run_type"] == "sprint"
                    ].sum()
                ),
            ),
            safe_top_total_distance=(
                "distance",
                lambda s: float(
                    safe_runs.loc[s.index, "distance"][
                        safe_runs.loc[s.index, "stage"] == "top"
                    ].sum()
                ),
            ),
            safe_middle_total_distance=(
                "distance",
                lambda s: float(
                    safe_runs.loc[s.index, "distance"][
                        safe_runs.loc[s.index, "stage"] == "middle"
                    ].sum()
                ),
            ),
            safe_bottom_total_distance=(
                "distance",
                lambda s: float(
                    safe_runs.loc[s.index, "distance"][
                        safe_runs.loc[s.index, "stage"] == "bottom"
                    ].sum()
                ),
            ),
            safe_top_sprint_count=(
                "id",
                lambda s: int(
                    ((safe_runs.loc[s.index, "stage"] == "top") & (safe_runs.loc[s.index, "run_type"] == "sprint")).sum()
                ),
            ),
            safe_unique_possessions=("possession", "nunique"),
        )
    )
    possession_runs = runs[runs["possession"].notna()].copy()
    possession_level = (
        possession_runs.groupby(
            [
                "player_appearance_id",
                "checkpoint",
                "checkpoint_period_order",
                "checkpoint_bucket",
                "possession",
            ],
            as_index=False,
        )
        .agg(
            possession_run_count=("id", "size"),
            possession_sprint_count=(
                "id",
                lambda s: int((possession_runs.loc[s.index, "run_type"] == "sprint").sum()),
            ),
            possession_hsr_count=(
                "id",
                lambda s: int((possession_runs.loc[s.index, "run_type"] == "hsr").sum()),
            ),
            possession_top_run_count=(
                "id",
                lambda s: int((possession_runs.loc[s.index, "stage"] == "top").sum()),
            ),
        )
    )
    possession_level["has_top_run"] = (possession_level["possession_top_run_count"] > 0).astype(int)
    possession_level["has_sprint"] = (possession_level["possession_sprint_count"] > 0).astype(int)
    possession_level["has_sprint_and_hsr"] = (
        (possession_level["possession_sprint_count"] > 0) & (possession_level["possession_hsr_count"] > 0)
    ).astype(int)
    possession_level["has_2plus_runs"] = (possession_level["possession_run_count"] >= 2).astype(int)
    possession_level["has_2plus_top_runs"] = (possession_level["possession_top_run_count"] >= 2).astype(int)

    possession_grouped = (
        possession_level.groupby(
            ["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
            as_index=False,
        )
        .agg(
            last15_unique_run_possessions=("possession", "nunique"),
            possessions_with_2plus_runs=("has_2plus_runs", "sum"),
            possessions_with_2plus_top_runs=("has_2plus_top_runs", "sum"),
            possessions_with_sprint_and_hsr=("has_sprint_and_hsr", "sum"),
            possessions_with_top_run=("has_top_run", "sum"),
            possessions_with_sprint=("has_sprint", "sum"),
        )
    )
    possession_first_seen = (
        possession_level.sort_values(
            ["player_appearance_id", "checkpoint_period_order", "checkpoint_bucket", "possession"]
        )
        .drop_duplicates(subset=["player_appearance_id", "possession"], keep="first")
        .groupby(
            ["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
            as_index=False,
        )
        .agg(new_run_possessions=("possession", "size"))
    )
    grouped = grouped.merge(
        safe_grouped,
        on=["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
        how="left",
    )
    grouped = grouped.merge(
        possession_grouped,
        on=["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
        how="left",
    )
    grouped = grouped.merge(
        possession_first_seen,
        on=["player_appearance_id", "checkpoint", "checkpoint_period_order", "checkpoint_bucket"],
        how="left",
    )
    safe_fill_cols = [
        "top_sprint_distance",
        "top_hsr_distance",
        "safe_total_distance",
        "safe_run_count",
        "safe_total_sprint_distance",
        "safe_top_total_distance",
        "safe_middle_total_distance",
        "safe_bottom_total_distance",
        "safe_top_sprint_count",
        "safe_unique_possessions",
        "last15_unique_run_possessions",
        "possessions_with_2plus_runs",
        "possessions_with_2plus_top_runs",
        "possessions_with_sprint_and_hsr",
        "possessions_with_top_run",
        "possessions_with_sprint",
        "new_run_possessions",
    ]
    grouped[safe_fill_cols] = grouped[safe_fill_cols].fillna(0.0)

    grouped = grouped.sort_values(
        ["player_appearance_id", "checkpoint_period_order", "checkpoint_bucket"]
    )
    grouped["cumul_top_sprint_count"] = grouped.groupby("player_appearance_id")["last15_top_sprint_count"].cumsum()
    grouped["cumul_top_hsr_count"] = grouped.groupby("player_appearance_id")["last15_top_hsr_count"].cumsum()
    grouped["cumul_middle_sprint_count"] = grouped.groupby("player_appearance_id")[
        "last15_middle_sprint_count"
    ].cumsum()
    grouped["cumul_middle_hsr_count"] = grouped.groupby("player_appearance_id")[
        "last15_middle_hsr_count"
    ].cumsum()
    grouped["cumul_bottom_sprint_count"] = grouped.groupby("player_appearance_id")[
        "last15_bottom_sprint_count"
    ].cumsum()
    grouped["cumul_bottom_hsr_count"] = grouped.groupby("player_appearance_id")[
        "last15_bottom_hsr_count"
    ].cumsum()
    grouped["cumul_unique_run_possessions"] = grouped.groupby("player_appearance_id")[
        "new_run_possessions"
    ].cumsum()

    grouped["last15_total_run_count"] = (
        grouped["last15_top_sprint_count"]
        + grouped["last15_top_hsr_count"]
        + grouped["last15_middle_sprint_count"]
        + grouped["last15_middle_hsr_count"]
        + grouped["last15_bottom_sprint_count"]
        + grouped["last15_bottom_hsr_count"]
    )
    grouped["cumul_total_run_count"] = (
        grouped["cumul_top_sprint_count"]
        + grouped["cumul_top_hsr_count"]
        + grouped["cumul_middle_sprint_count"]
        + grouped["cumul_middle_hsr_count"]
        + grouped["cumul_bottom_sprint_count"]
        + grouped["cumul_bottom_hsr_count"]
    )
    grouped["cumul_total_sprint_count"] = (
        grouped["cumul_top_sprint_count"]
        + grouped["cumul_middle_sprint_count"]
        + grouped["cumul_bottom_sprint_count"]
    )
    grouped["cumul_total_hsr_count"] = (
        grouped["cumul_top_hsr_count"]
        + grouped["cumul_middle_hsr_count"]
        + grouped["cumul_bottom_hsr_count"]
    )

    grouped["cumul_top_run_share"] = (
        (grouped["cumul_top_sprint_count"] + grouped["cumul_top_hsr_count"])
        / grouped["cumul_total_run_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["cumul_middle_run_share"] = (
        (grouped["cumul_middle_sprint_count"] + grouped["cumul_middle_hsr_count"])
        / grouped["cumul_total_run_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["cumul_bottom_run_share"] = (
        (grouped["cumul_bottom_sprint_count"] + grouped["cumul_bottom_hsr_count"])
        / grouped["cumul_total_run_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["cumul_top_sprint_share"] = (
        grouped["cumul_top_sprint_count"] / grouped["cumul_total_sprint_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["cumul_top_hsr_share"] = (
        grouped["cumul_top_hsr_count"] / grouped["cumul_total_hsr_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["cumul_bottom_sprint_share"] = (
        grouped["cumul_bottom_sprint_count"] / grouped["cumul_total_sprint_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["last15_top_run_share"] = (
        (grouped["last15_top_sprint_count"] + grouped["last15_top_hsr_count"])
        / grouped["last15_total_run_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["last15_top_sprint_share"] = (
        grouped["last15_top_sprint_count"]
        / (
            grouped["last15_top_sprint_count"]
            + grouped["last15_middle_sprint_count"]
            + grouped["last15_bottom_sprint_count"]
        ).replace(0, pd.NA)
    ).fillna(0.0)
    grouped["distance_per_run"] = (
        grouped["safe_total_distance"] / grouped["safe_run_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["distance_per_possession"] = (
        grouped["safe_total_distance"] / grouped["safe_unique_possessions"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["top_distance_share"] = (
        grouped["safe_top_total_distance"] / grouped["safe_total_distance"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["middle_distance_share"] = (
        grouped["safe_middle_total_distance"] / grouped["safe_total_distance"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["bottom_distance_share"] = (
        grouped["safe_bottom_total_distance"] / grouped["safe_total_distance"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["sprint_distance_share"] = (
        grouped["safe_total_sprint_distance"] / grouped["safe_total_distance"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["avg_top_sprint_distance"] = (
        grouped["top_sprint_distance"] / grouped["safe_top_sprint_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["runs_per_possession"] = (
        grouped["last15_total_run_count"] / grouped["last15_unique_run_possessions"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["sprints_per_possession"] = (
        (
            grouped["last15_top_sprint_count"]
            + grouped["last15_middle_sprint_count"]
            + grouped["last15_bottom_sprint_count"]
        )
        / grouped["last15_unique_run_possessions"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["top_runs_per_possession"] = (
        (grouped["last15_top_sprint_count"] + grouped["last15_top_hsr_count"])
        / grouped["last15_unique_run_possessions"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["share_of_possessions_with_top_run"] = (
        grouped["possessions_with_top_run"] / grouped["last15_unique_run_possessions"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["share_of_possessions_with_sprint"] = (
        grouped["possessions_with_sprint"] / grouped["last15_unique_run_possessions"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["top_run_repeat_possession_rate"] = (
        grouped["possessions_with_2plus_top_runs"] / grouped["last15_unique_run_possessions"].replace(0, pd.NA)
    ).fillna(0.0)

    return grouped.drop(
        columns=[
            "checkpoint_period_order",
            "checkpoint_bucket",
            "last15_total_run_count",
            "cumul_total_run_count",
            "cumul_total_sprint_count",
            "cumul_total_hsr_count",
            "safe_total_distance",
            "safe_run_count",
            "safe_total_sprint_distance",
            "safe_top_total_distance",
            "safe_middle_total_distance",
            "safe_bottom_total_distance",
            "safe_top_sprint_count",
            "safe_unique_possessions",
            "possessions_with_top_run",
            "possessions_with_sprint",
            "new_run_possessions",
        ]
    )


def build_feature_spine_from_selected_sources(selected_sources: set[str]) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []
    if "passes" in selected_sources:
        feature_frames.append(build_received_pass_features())
    if "pressure" in selected_sources:
        feature_frames.append(build_pressure_features())
    if "runs" in selected_sources:
        feature_frames.append(build_run_features())
    if "shots" in selected_sources:
        feature_frames.append(build_shot_features())

    if not feature_frames:
        raise ValueError(
            "No feature datasets selected. Use --merge-sources with one or more of: "
            "passes, pressure, runs, shots."
        )

    spine = feature_frames[0].copy()
    for frame in feature_frames[1:]:
        spine = spine.merge(frame, on=["player_appearance_id", "checkpoint"], how="outer")
    return spine


def build_label_context_table() -> pd.DataFrame:
    base = pd.read_csv(
        DATA_DIR / "players_quarters_final.csv",
        parse_dates=["date"],
        na_values=CSV_NULL_TOKENS,
    )
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
    base = pd.read_csv(
        DATA_DIR / "players_quarters_final.csv",
        parse_dates=["date"],
        na_values=CSV_NULL_TOKENS,
    )
    base["_checkpoint_order"] = base["checkpoint"].map(CHECKPOINT_ORDER)
    base = base.sort_values(
        ["date", "fixture_id", "player_appearance_id", "_checkpoint_order"]
    ).reset_index(drop=True)
    base = base.drop(columns=["_checkpoint_order"])
    base = split_formation_columns(base)
    base["cumul_in_game_time"] = (
        base["checkpoint"].map(CHECKPOINT_TO_ABS_MINUTE) - base["minute_in"]
    ).clip(lower=0)

    if "passes" in selected_sources:
        received_pass_features = build_received_pass_features()
        base = base.merge(received_pass_features, on=["player_appearance_id", "checkpoint"], how="left")
        base = carry_forward_cumulative_features(
            base,
            [col for col in received_pass_features.columns if is_cumul_feature(col)],
        )
        for col in PASS_FEATURE_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0).astype(int)

    if "pressure" in selected_sources:
        pressure_features = build_pressure_features()
        base = base.merge(pressure_features, on=["player_appearance_id", "checkpoint"], how="left")
        base = carry_forward_cumulative_features(
            base,
            [col for col in pressure_features.columns if is_cumul_feature(col)] + PRESSURE_CARRY_FORWARD_COLS,
        )
        for col in PRESSURE_COUNT_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0).astype(int)
        for col in PRESSURE_CONTEXT_COUNT_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0).astype(int)
        for col in PRESSURE_RATE_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0.0)
        for col in PRESSURE_CONTEXT_VALUE_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0.0)

    if "runs" in selected_sources:
        run_features = build_run_features()
        base = base.merge(run_features, on=["player_appearance_id", "checkpoint"], how="left")
        base = carry_forward_cumulative_features(
            base,
            [col for col in run_features.columns if is_cumul_feature(col)],
        )
        for col in RUN_COUNT_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0).astype(int)
        for col in RUN_SHARE_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0.0)
        for col in RUN_DISTANCE_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0.0)
        for col in RUN_POSSESSION_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0.0)

    if "shots" in selected_sources:
        shot_features = build_shot_features()
        base = base.merge(shot_features, on=["player_appearance_id", "checkpoint"], how="left")
        base = carry_forward_cumulative_features(
            base,
            [col for col in shot_features.columns if is_cumul_feature(col)],
        )
        for col in SHOT_COUNT_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0).astype(int)
        for col in SHOT_RATE_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0.0)

    base = add_team_level_context_features(base)

    filtered_base, removed_rows = apply_row_filters(base)

    fixture_split = build_fixture_split(filtered_base)
    dataset = filtered_base.merge(
        fixture_split[["date", "fixture_id", "split", "fixture_order"]],
        on=["date", "fixture_id"],
        how="left",
    )

    model_columns = build_model_keep_columns(
        dataset_columns=dataset.columns.tolist(),
        selected_sources=selected_sources,
        window_mode=window_mode,
    )
    included_extension_features = get_included_extension_features(model_columns, selected_sources)
    model_dataset = dataset[model_columns + ["split"]].copy()
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
                else "predictor"
                if col in model_columns
                else "split_metadata"
                if col in {"split", "fixture_order"}
                else "excluded_from_model"
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

{chr(10).join([f"- `{c}`" for c in included_extension_features]) if included_extension_features else "- none"}
"""

    (OUTPUT_DIR / "README.md").write_text(summary_md)
    print("## Source")
    print("- players_quarters_final.csv")
    if "passes" in selected_sources:
        print("- player_appearance_pass.csv")
    if "pressure" in selected_sources:
        print("- player_appearance_behaviour_under_pressure.csv")
    if "runs" in selected_sources:
        print("- player_appearance_run.csv")
    if "shots" in selected_sources:
        print(f"- {resolve_shot_file().name}")
    print(f"- feature-window: {window_mode}")
    print(f"- output: {OUTPUT_DIR}")
    print()
    print("## Requested 60 / 20 / 20 split")
    print(split_summary.to_string(index=False))
    print()
    print("## Added baseline extension features")
    for col in included_extension_features:
        print(f"- {col}")
    if not included_extension_features:
        print("- none")


if __name__ == "__main__":
    main()
