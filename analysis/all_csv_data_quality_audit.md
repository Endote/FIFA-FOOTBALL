# All Project CSV Data-Quality Audit

Scope:

- Included all project CSVs under `data/` and `output/`
- Excluded `.venv/` sample and package-test CSVs

Important interpretation notes:

- In the event tables, `minute` is local to the match period and can exceed 45 because stoppage time is included. So values like `48`, `50`, or `57` are not errors by themselves.
- Several nullable foreign keys are structurally valid and should not be treated as corruption:
  - `addressee_player_appearance_id` can be null in `player_appearance_pass.csv`
  - `addressee_player_appearance_id` and `pass_angle` can be null in `player_appearance_behaviour_under_pressure.csv`
  - `own_goal_player_appearance_id` and `block_player_appearance_id` can be null in `player_appearance_shot_limited.csv`

## Executive summary

Real corruption is concentrated in the running-distance pipeline:

- `data/player_appearance_run.csv` shows extreme run-distance values for appearance IDs tied to fixtures `1161`, `1167`, and `1201`
- `data/players_quarters_final.csv` inherits that corruption in:
  - `last15_distance`
  - `cumul_distance`
- The same issue propagates into:
  - `data/baseline_modeling/baseline_all_with_splits.csv`
  - `data/baseline_modeling/baseline_train_full.csv`
  - `data/baseline_modeling/baseline_all_model_ready.csv`
  - `data/baseline_modeling/baseline_train_model_ready.csv`
  - `output/eda/inferred_goal_shots_exact.csv`

Outside that, I did not find malformed categorical values or clearly impossible feature values in the raw football tables.

## Per-file audit

### `data/player_appearance_pass.csv`

- Missingness:
  - `addressee_player_appearance_id`: `41`
  - `stage`: `57`
- Illogical values:
  - None found
- Corrupted values:
  - No obvious corruption pattern
- Notes:
  - `stage` blanks are allowed by the dataset description
  - No invalid `period` or `stage` labels found

### `data/player_appearance_behaviour_under_pressure.csv`

- Missingness:
  - `addressee_player_appearance_id`: `649`
  - `pass_angle`: `5793`
  - `stage`: `14`
- Illogical values:
  - None found
- Corrupted values:
  - No obvious corruption pattern
- Notes:
  - `pass_angle` missingness is structurally correct for `ball_carry` and `turnover`
  - `stage` blanks are allowed by the dataset description
  - No invalid `press_induced_outcome` categories found

### `data/player_appearance_run.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - Probable corruption in `distance` for appearance IDs associated with fixtures `1161`, `1167`, and `1201`
- Evidence:
  - Using `players_quarters_final.csv` to identify the affected appearances, runs tied to those bad fixtures have:
    - `3595` rows
    - median `distance = 281.59`
    - mean `distance = 339.67`
    - max `distance = 1694.01`
  - All other run rows have:
    - `31538` rows
    - median `distance = 11.06`
    - mean `distance = 14.67`
    - max `distance = 991.22`
- Interpretation:
  - This is the likely upstream source of the corrupted aggregated distance features

### `data/player_appearance_shot_limited.csv`

- Missingness:
  - `block_player_appearance_id`: `565`
  - `own_goal_player_appearance_id`: `779`
- Illogical values:
  - None found
- Corrupted values:
  - No obvious corruption pattern
- Notes:
  - Both nullable ID fields are structurally allowed because outcome is omitted from the contest version
  - No invalid `body_part`, `technique`, `play_pattern`, or `stage` values found

### `data/players_quarters_final.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - `last15_distance` corrupted for fixtures:
    - `1161` on `2025-06-11`
    - `1167` on `2025-06-14`
    - `1201` on `2025-06-17`
  - `cumul_distance` corrupted for the same fixtures
  - Affected rows: `330`
- Notes:
  - `formation` is clean
  - `date` is clean
  - `checkpoint`, `checkpoint_period`, and `checkpoint_min` are internally consistent

### `data/baseline_modeling/baseline_all_with_splits.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - Inherits corrupted `last15_distance` and `cumul_distance`
  - Affected rows: `330`
- Notes:
  - All corrupted rows are in split `train`

### `data/baseline_modeling/baseline_train_full.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - Inherits corrupted `last15_distance` and `cumul_distance`
  - Affected rows: `330`

### `data/baseline_modeling/baseline_val_full.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `data/baseline_modeling/baseline_test_full.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `data/baseline_modeling/baseline_all_model_ready.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - Inherits corrupted distance features from the full baseline panel

### `data/baseline_modeling/baseline_train_model_ready.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - Inherits corrupted distance features from the training panel

### `data/baseline_modeling/baseline_val_model_ready.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `data/baseline_modeling/baseline_test_model_ready.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `data/baseline_modeling/baseline_feature_manifest.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `data/baseline_modeling/baseline_fixture_split.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `data/baseline_modeling/baseline_split_summary.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/eda/full_feature_logreg_effects.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/eda/full_feature_permutation_importance.csv`

- Missingness:
  - None
- Illogical values:
  - Empty result set
- Corrupted values:
  - Not row-level corruption, but the file is empty (`0` rows), so the permutation-importance export failed or produced nothing

### `output/eda/goal_shot_candidates.csv`

- Missingness:
  - `block_player_appearance_id`: `88`
  - `own_goal_player_appearance_id`: `95`
- Illogical values:
  - None found
- Corrupted values:
  - None detected
- Notes:
  - Missing nullable shot-link fields are expected

### `output/eda/goal_shot_windows.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/eda/inferred_goal_pre_goal_behavior.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/eda/inferred_goal_shot_characteristics.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/eda/inferred_goal_shots_exact.csv`

- Missingness:
  - `block_player_appearance_id`: `50`
  - `own_goal_player_appearance_id`: `51`
- Illogical values:
  - None found
- Corrupted values:
  - Inherits corrupted distance features
  - Affected rows: `6`

### `output/eda/model_benchmarks.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/eda/numeric_feature_screen.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/eda/outfield_only_benchmarks.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/eda/selected_subset_benchmarks.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/eda/split_assignments.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected
- Notes:
  - `split` values are `trainval` and `holdout`, which is valid for this artifact

### `output/model_comparison/catboost_val_predictions.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/model_comparison/catboost_test_predictions.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/model_comparison/xgboost_val_predictions.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/model_comparison/xgboost_test_predictions.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/model_comparison/model_comparison_results.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

### `output/model_comparison/model_failures.csv`

- Missingness:
  - None
- Illogical values:
  - None found
- Corrupted values:
  - None detected

## Files with confirmed or likely corruption

- `data/player_appearance_run.csv`
  - likely upstream corruption in `distance` for appearances belonging to fixtures `1161`, `1167`, `1201`
- `data/players_quarters_final.csv`
  - confirmed corruption in `last15_distance`, `cumul_distance`
- `data/baseline_modeling/baseline_all_with_splits.csv`
  - inherited corruption
- `data/baseline_modeling/baseline_train_full.csv`
  - inherited corruption
- `data/baseline_modeling/baseline_all_model_ready.csv`
  - inherited corruption
- `data/baseline_modeling/baseline_train_model_ready.csv`
  - inherited corruption
- `output/eda/inferred_goal_shots_exact.csv`
  - inherited corruption

## Files with legitimate missingness, not defects

- `data/player_appearance_pass.csv`
- `data/player_appearance_behaviour_under_pressure.csv`
- `data/player_appearance_shot_limited.csv`
- `output/eda/goal_shot_candidates.csv`
- `output/eda/inferred_goal_shots_exact.csv`

## Recommended next action

Rebuild all distance-derived features from `player_appearance_run.csv` after removing or repairing the corrupted appearance IDs linked to fixtures `1161`, `1167`, and `1201`. Until that is done, avoid using any distance-based features in modeling.
