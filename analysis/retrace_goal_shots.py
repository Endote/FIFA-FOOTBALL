from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/eda")

CHECKPOINT_ABS = {
    "H1_15": 15,
    "H1_30": 30,
    "H1_45": 45,
    "H2_15": 60,
    "H2_30": 75,
    "H2_45": 90,
    "ET1_15": 105,
}

CHECKPOINT_ORDER = {name: idx for idx, name in enumerate(CHECKPOINT_ABS)}

PERIOD_OFFSET = {
    "half_1": 0,
    "half_2": 45,
    "extra_time_1": 90,
    "extra_time_2": 105,
}


def build_scoring_windows(base: pd.DataFrame) -> pd.DataFrame:
    windows = []
    ordered = base.sort_values(["player_appearance_id", "chk_ord"])
    for player_appearance_id, group in ordered.groupby("player_appearance_id"):
        group = group.sort_values("chk_ord")
        vals = group["scored_after"].tolist()
        if max(vals) == 0:
            continue

        checkpoints = group["checkpoint"].tolist()
        abs_checkpoints = group["abs_checkpoint"].tolist()
        found_transition = False

        for i in range(len(vals) - 1):
            if vals[i] == 1 and vals[i + 1] == 0:
                windows.append(
                    {
                        "player_appearance_id": player_appearance_id,
                        "window_type": "transition",
                        "start_checkpoint": checkpoints[i],
                        "end_checkpoint": checkpoints[i + 1],
                        "start_abs": abs_checkpoints[i],
                        "end_abs": abs_checkpoints[i + 1],
                    }
                )
                found_transition = True
                break

        if not found_transition and vals[-1] == 1:
            windows.append(
                {
                    "player_appearance_id": player_appearance_id,
                    "window_type": "tail",
                    "start_checkpoint": checkpoints[-1],
                    "end_checkpoint": "END",
                    "start_abs": abs_checkpoints[-1],
                    "end_abs": int(group["minute_out"].iloc[-1]),
                }
            )
    return pd.DataFrame(windows)


def add_pre_goal_snapshot(
    exact_goal_shots: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    snapshot_cols = [
        "player_appearance_id",
        "fixture_id",
        "date",
        "checkpoint",
        "position",
        "is_home",
        "formation",
        "minute_in",
        "minute_out",
        "subbed",
        "last15_sprints",
        "last15_hsr",
        "last15_distance",
        "last15_mean_max_speed",
        "last15_peak_speed",
        "last15_shots",
        "last15_shots_on_target",
        "last15_shots_under_press",
        "last15_shots_top_third",
        "cumul_sprints",
        "cumul_hsr",
        "cumul_distance",
        "cumul_mean_max_speed",
        "cumul_peak_speed",
        "cumul_shots",
        "cumul_shots_on_target",
        "cumul_shots_under_press",
        "cumul_shots_top_third",
    ]
    snapshots = base[snapshot_cols].rename(columns={"checkpoint": "pre_goal_checkpoint"})
    return exact_goal_shots.merge(
        snapshots,
        left_on=["player_appearance_id", "start_checkpoint"],
        right_on=["player_appearance_id", "pre_goal_checkpoint"],
        how="left",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(DATA_DIR / "players_quarters_final.csv", parse_dates=["date"])
    base["chk_ord"] = base["checkpoint"].map(CHECKPOINT_ORDER)
    base["abs_checkpoint"] = base["checkpoint"].map(CHECKPOINT_ABS)

    shots = pd.read_csv(DATA_DIR / "player_appearance_shot_limited.csv")
    shots["abs_minute"] = shots["period"].map(PERIOD_OFFSET) + shots["minute"]

    windows = build_scoring_windows(base)
    windows["window_id"] = range(1, len(windows) + 1)

    candidate_shots = windows.merge(shots, on="player_appearance_id", how="left")
    candidate_shots = candidate_shots[
        candidate_shots["abs_minute"].notna()
        & (candidate_shots["abs_minute"] > candidate_shots["start_abs"])
        & (candidate_shots["abs_minute"] <= candidate_shots["end_abs"])
    ].copy()

    shot_counts = candidate_shots.groupby("window_id").size().rename("n_candidate_shots").reset_index()
    windows = windows.merge(shot_counts, on="window_id", how="left").fillna({"n_candidate_shots": 0})
    windows["n_candidate_shots"] = windows["n_candidate_shots"].astype(int)
    windows["inference_status"] = "ambiguous_multiple_shots"
    windows.loc[windows["n_candidate_shots"] == 1, "inference_status"] = "exact_single_shot"
    windows.loc[windows["n_candidate_shots"] == 0, "inference_status"] = "no_shot_found"

    exact_goal_shots = candidate_shots.merge(
        windows.loc[windows["inference_status"] == "exact_single_shot", ["window_id", "inference_status"]],
        on="window_id",
        how="inner",
    ).copy()

    exact_goal_shots = add_pre_goal_snapshot(exact_goal_shots, base)

    joinable_shots = shots[shots["player_appearance_id"].isin(base["player_appearance_id"])].copy()
    exact_ids = set(exact_goal_shots["id"])
    joinable_shots["shot_group"] = joinable_shots["id"].map(
        lambda shot_id: "exact_inferred_goal_shot" if shot_id in exact_ids else "other_joinable_shot"
    )

    compare_rows = []
    for shot_group, group in joinable_shots.groupby("shot_group"):
        compare_rows.append(
            {
                "shot_group": shot_group,
                "n_shots": len(group),
                "under_pressure_rate": group["under_pressure"].mean(),
                "top_stage_rate": (group["stage"] == "top").mean(),
                "header_rate": (group["body_part"] == "head").mean(),
                "left_foot_rate": (group["body_part"] == "left_foot").mean(),
                "right_foot_rate": (group["body_part"] == "right_foot").mean(),
                "regular_play_rate": (group["play_pattern"] == "regular_play").mean(),
                "counter_attack_rate": (group["play_pattern"] == "counter_attack").mean(),
                "set_piece_rate": group["play_pattern"].isin(
                    ["corner_kick", "direct_free_kick", "indirect_free_kick", "throw_in"]
                ).mean(),
                "penalty_rate": (group["play_pattern"] == "penalty").mean(),
                "lob_rate": (group["technique"] == "lob").mean(),
                "volley_rate": (group["technique"] == "volley").mean(),
                "normal_technique_rate": (group["technique"] == "normal").mean(),
            }
        )
    shot_characteristics = pd.DataFrame(compare_rows)

    pre_goal_compare_rows = []
    if not exact_goal_shots.empty:
        exact_pre = exact_goal_shots.copy()
        exact_pre["group"] = "exact_inferred_goal_shot"
        other_base = base.copy()
        other_base["group"] = "all_base_rows"
        compare_cols = [
            "last15_sprints",
            "last15_hsr",
            "last15_distance",
            "last15_mean_max_speed",
            "last15_peak_speed",
            "last15_shots",
            "last15_shots_on_target",
            "last15_shots_under_press",
            "last15_shots_top_third",
            "cumul_sprints",
            "cumul_hsr",
            "cumul_distance",
            "cumul_mean_max_speed",
            "cumul_peak_speed",
            "cumul_shots",
            "cumul_shots_on_target",
            "cumul_shots_under_press",
            "cumul_shots_top_third",
        ]
        for group_name, group in [("exact_inferred_goal_shot", exact_pre), ("all_base_rows", other_base)]:
            row = {"group": group_name, "n_rows": len(group)}
            for col in compare_cols:
                row[f"{col}_mean"] = group[col].mean()
            pre_goal_compare_rows.append(row)
    pre_goal_behavior = pd.DataFrame(pre_goal_compare_rows)

    windows.to_csv(OUTPUT_DIR / "goal_shot_windows.csv", index=False)
    candidate_shots.to_csv(OUTPUT_DIR / "goal_shot_candidates.csv", index=False)
    exact_goal_shots.to_csv(OUTPUT_DIR / "inferred_goal_shots_exact.csv", index=False)
    shot_characteristics.to_csv(OUTPUT_DIR / "inferred_goal_shot_characteristics.csv", index=False)
    pre_goal_behavior.to_csv(OUTPUT_DIR / "inferred_goal_pre_goal_behavior.csv", index=False)

    summary = f"""# Retracing Goal Shots From The Target

## What can be identified exactly

- The target `scored_after` lets us locate the **last scoring window** for each player appearance with at least one future goal.
- It does **not** reveal every goal scored by multi-goal players.
- A shot can be tagged as an **exact inferred goal shot** only when that last scoring window contains exactly one shot by that player.

## Coverage

- Player appearances with at least one future goal signal: {len(windows)}
- Exact single-shot windows: {(windows['inference_status'] == 'exact_single_shot').sum()}
- Ambiguous multi-shot windows: {(windows['inference_status'] == 'ambiguous_multiple_shots').sum()}
- Windows with no shot found: {(windows['inference_status'] == 'no_shot_found').sum()}

## Interpretation

- The `exact_single_shot` subset is the cleanest retraced sample of scoring shots.
- The `ambiguous_multiple_shots` subset almost certainly contains real goals, but the exact shot cannot be identified without an explicit shot outcome field.
- The `no_shot_found` cases suggest either post-checkpoint data coverage issues, non-standard recording gaps, or target/shot table mismatches.

## Files

- [goal_shot_windows.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/goal_shot_windows.csv)
- [goal_shot_candidates.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/goal_shot_candidates.csv)
- [inferred_goal_shots_exact.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/inferred_goal_shots_exact.csv)
- [inferred_goal_shot_characteristics.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/inferred_goal_shot_characteristics.csv)
- [inferred_goal_pre_goal_behavior.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/inferred_goal_pre_goal_behavior.csv)
"""

    (OUTPUT_DIR / "goal_shot_retracing_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
