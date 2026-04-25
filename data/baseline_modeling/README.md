# Baseline Modeling Dataset

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

split  rows  fixtures  player_appearances  players  positives  positive_rate  row_share
 test   745         6                 173      111         45       0.060403   0.213712
train  2085        19                 530      280        123       0.058993   0.598107
  val   656         6                 166      155         35       0.053354   0.188181

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
