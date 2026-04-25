from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/datasets/baseline_modeling")

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
]

TARGET_COL = "scored_after"


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(DATA_DIR / "players_quarters_final.csv", parse_dates=["date"])
    base = base.sort_values(["date", "fixture_id", "player_appearance_id", "checkpoint"]).reset_index(drop=True)

    fixture_split = build_fixture_split(base)
    dataset = base.merge(fixture_split[["date", "fixture_id", "split", "fixture_order"]], on=["date", "fixture_id"], how="left")

    model_columns = [col for col in dataset.columns if col not in DROP_FROM_MODEL]
    model_dataset = dataset[model_columns].copy()

    split_summary = summarize_split(dataset)

    dataset.to_csv(OUTPUT_DIR / "baseline_all_with_splits.csv", index=False)
    model_dataset.to_csv(OUTPUT_DIR / "baseline_all_model_ready.csv", index=False)
    fixture_split.to_csv(OUTPUT_DIR / "baseline_fixture_split.csv", index=False)
    split_summary.to_csv(OUTPUT_DIR / "baseline_split_summary.csv", index=False)

    for split_name in ["train", "val", "test"]:
        split_full = dataset[dataset["split"] == split_name].copy()
        split_model = model_dataset[model_dataset["split"] == split_name].copy()
        split_full.to_csv(OUTPUT_DIR / f"baseline_{split_name}_full.csv", index=False)
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
- Output directory: [baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling)

## Split design

- Splits are chronological and fixture-grouped.
- Primary sort order: `date`
- Tie-break inside the same date: `fixture_id`
- This keeps complete fixtures inside one split and prevents future-match leakage.
- Because the source data does not include kickoff timestamps, `fixture_id` is used only as a deterministic within-date ordering rule.

## Requested 60 / 20 / 20 split

- Exact 60 / 20 / 20 is not always achievable with whole-fixture chronological splits.
- The closest deterministic split on this dataset is:

{split_summary.to_string(index=False)}

## Files

- [baseline_all_with_splits.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_all_with_splits.csv)
- [baseline_all_model_ready.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_all_model_ready.csv)
- [baseline_train_full.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_train_full.csv)
- [baseline_val_full.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_val_full.csv)
- [baseline_test_full.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_test_full.csv)
- [baseline_train_model_ready.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_train_model_ready.csv)
- [baseline_val_model_ready.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_val_model_ready.csv)
- [baseline_test_model_ready.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_test_model_ready.csv)
- [baseline_fixture_split.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_fixture_split.csv)
- [baseline_split_summary.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_split_summary.csv)
- [baseline_feature_manifest.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/datasets/baseline_modeling/baseline_feature_manifest.csv)

## What is removed from the model-ready files

- `player_appearance_id`
- `player_id`
- `fixture_id`
- `date`
- `jersey_number`
- `checkpoint_period`
- `checkpoint_min`
- `fixture_order`

These are retained in the `*_full.csv` files for auditability and teammate-side inspection, but removed from the `*_model_ready.csv` files before model fitting.

## What stays in the model-ready files

- `checkpoint`
- `position`
- `is_home`
- `formation`
- `minute_in`
- `minute_out`
- `subbed`
- all baseline `last15_*` and `cumul_*` features
- target `scored_after`
- split label `split`

## Leakage note

- The baseline table already excludes prior goals from the shot aggregates used to form `last15_*` and `cumul_*` shot features.
- No additional helper columns such as absolute-time fields were added to the exported model-ready dataset.
"""

    (OUTPUT_DIR / "README.md").write_text(summary_md)
    print(summary_md)


if __name__ == "__main__":
    main()
