# Data Split And Time Logic

## Observation unit

- The supervised modeling table is [players_quarters_final.csv](/WEC2026/data/players_quarters_final.csv).
- Each row is one `player_appearance_id` at one match checkpoint.
- The target `scored_after` equals `1` if that player scores at any point after the checkpoint in the same match, excluding own goals.
- This means multiple rows from the same player appearance belong to the same match trajectory and are not independent.

## Why random row splitting is wrong

- A single `player_appearance_id` contributes up to 7 checkpoints.
- Those checkpoints share the same player, match, tactical context, and future scoring path.
- If rows from the same `fixture_id` are split randomly across train and validation, the model sees near-duplicate match states from the same game on both sides.
- That creates leakage and gives an inflated estimate of AUROC and balanced accuracy.

## Split logic used

- Final holdout is chronological.
- All fixtures on the latest dates were reserved until both conditions were met:
- holdout row share at least `20%`
- holdout positive-class share at least `20%`
- This produced the holdout dates:
- `2025-06-21`
- `2025-06-22`
- `2025-06-25`
- `2025-06-28`
- Resulting split:
- train/validation rows: `2,635`
- holdout rows: `851`
- train/validation positives: `158`
- holdout positives: `45`

## Validation logic inside the training period

- Model selection on the pre-holdout data uses `StratifiedGroupKFold`.
- Grouping variable: `fixture_id`
- Stratification target: `scored_after`
- This keeps all rows from the same match inside the same fold while still trying to preserve class balance.

## Chronological order logic

- The base table has checkpoints:
- `H1_15`, `H1_30`, `H1_45`
- `H2_15`, `H2_30`, `H2_45`
- `ET1_15`
- Event tables store `period` and local `minute`.
- Local minutes are not enough on their own because stoppage time appears as minutes greater than `45` inside the same half.
- Example:
- an event in `half_1` at minute `46` is first-half stoppage time, not second-half minute `1`

## Absolute time conversion used

- To align event data with checkpoints, each event was mapped to an absolute match minute:
- `half_1`: `0 + minute`
- `half_2`: `45 + minute`
- `extra_time_1`: `90 + minute`
- `extra_time_2`: `105 + minute`
- Checkpoints were mapped as:
- `H1_15 -> 15`
- `H1_30 -> 30`
- `H1_45 -> 45`
- `H2_15 -> 60`
- `H2_30 -> 75`
- `H2_45 -> 90`
- `ET1_15 -> 105`

## Rolling-window logic

- Cumulative features use all events with `abs_minute <= checkpoint_abs_minute`.
- Last-15 features use all events with:
- `checkpoint_abs_minute - 15 < abs_minute <= checkpoint_abs_minute`
- This matters especially for:
- `H2_15`, which must include late first-half stoppage time
- `ET1_15`, which must include late second-half stoppage time

## Why this time logic matters

- A naive period-by-period last-15 calculation is wrong for checkpoints after a period break.
- That would drop valid stoppage-time actions from the preceding period.
- I verified this against the official base features:
- cumulative run features matched exactly
- last-15 run features matched once absolute-minute logic was used

## How all CSV files were used

- [players_quarters_final.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/players_quarters_final.csv)
- base supervised panel, target, context, and official checkpoint-level run/shot aggregates
- [player_appearance_pass.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/player_appearance_pass.csv)
- engineered pass volume, accuracy, field-zone, no-target, and receiver-diversity features
- [player_appearance_behaviour_under_pressure.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/player_appearance_behaviour_under_pressure.csv)
- engineered features from both perspectives:
- player under pressure
- player applying pressure
- [player_appearance_run.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/player_appearance_run.csv)
- validated official run aggregates and added more detailed run-context features
- [player_appearance_shot_limited.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/player_appearance_shot_limited.csv)
- added shot-context features without using any leakage-prone outcome field

## Joinability caveat

- Event tables contain some `player_appearance_id` values not present in the base checkpoint table.
- Those rows cannot be used directly for supervised training because the target exists only in the base table.
- Only event rows joinable to checkpoint rows were used for modeling features.

## Leakage constraints followed

- No target information from future match time was used in feature construction.
- Shot outcome was not used because it would directly leak the target logic.
- `jersey_number` was excluded from modeling even though it showed some signal, because it is likely a competition-specific artifact rather than a stable football mechanism.
- The holdout set was never used for model selection.

## Practical interpretation

- The honest estimate of model quality is the grouped CV on the chronological pre-holdout data, with the holdout used only as a final check.
- If someone reports a better score from random row splitting here, that score is almost certainly contaminated by match-level leakage.
