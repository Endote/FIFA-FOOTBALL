# AGENT.md

## Modeling Pipeline Standards

Use these directives as the default standard for all top-level modeling work in this repository unless a later explicit instruction overrides them.

## Data split and leakage control

- Always use chronological, fixture-grouped splits.
- Never use random row-level splits for checkpoint modeling.
- Keep train, validation, and test separated by whole fixtures.
- Follow the logic documented in [data_split.md](/Users/norbert.jaworski/Documents/small/WEC2026/data_split.md).
- If exact requested proportions are not achievable with whole-fixture chronological splits, use the closest deterministic split and document it.
- Do not leak future information across checkpoints, fixtures, or post-checkpoint events.

## Feature engineering standards

- DO NOT USE scored_after in any feature engineering, because it is the target !!!
- we can only use information until given checkpoint like the last checkpoint or cumulative of already happened checkpoints !
- Keep join keys such as `player_appearance_id` during feature construction when needed.
- Drop join keys before final model fitting unless explicitly required for non-model artifacts.
- Prefer derived football-meaningful variables over raw identifiers or raw bookkeeping columns.
- If a feature is cumulative, validate that it is monotone non-decreasing within a player appearance where appropriate.
- If a cumulative feature is created from sparse event rows, carry values forward across silent checkpoints instead of resetting to zero.
- Validate all event-to-checkpoint mappings against the checkpoint convention used in the base dataset.

## Baseline modeling dataset standards

- Canonical baseline modeling exports live under [data/baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/data/baseline_modeling).
- Use [create_baseline_modeling_dataset.py](/Users/norbert.jaworski/Documents/small/WEC2026/analysis/create_baseline_modeling_dataset.py) as the source of truth for baseline dataset creation.
- Export model-ready datasets without raw identifiers that should not be fitted as predictors.
- Do not export unnecessary duplicate “full” datasets unless explicitly requested.
- Keep split metadata available in separate audit files, not as predictor columns in final model-ready CSVs.

## Current feature policy

- Remove these from model-ready training data unless explicitly overridden:
- `player_appearance_id`
- `player_id`
- `fixture_id`
- `date`
- `jersey_number`
- `checkpoint_period`
- `checkpoint_min`
- `fixture_order`
- `minute_in`
- `minute_out`
- Replace raw pitch-entry timing fields with exposure-style derived features when possible.
- Current exposure feature in baseline modeling data:
- `cumul_in_game_time`

## Current receiver-pass feature policy

- Receiver-side pass features must be built from the addressee perspective.
- Current receiver-pass features in the baseline modeling dataset:
- `last_15_received_succ`
- `last_15_received_unsucc`
- `cumul_received_succ`
- `cumul_received_unsucc`
- `last_15_received_succ_pressure`
- `last_15_received_unsucc_pressure`
- `cumul_received_succ_pressure`
- `cumul_received_unsucc_pressure`
- Restrict these features to the intended event subset only if explicitly documented in the build script and README.

## Data quality filtering standards

- Apply row-quality filters before splitting if those filtered variables are part of the modeling dataset definition.
- Current baseline filters:
- `last15_distance < 1000`
- `cumul_mean_max_speed < 10.3`
- Save removed rows to an audit file whenever quality filters are applied.

## Training and evaluation standards

- Primary optimization metric for this project: balanced accuracy.
- Secondary ranking metric: AUROC.
- Keep model hyperparameters centralized at the top of training scripts.
- Each model family should have its own clearly labeled parameter block.
- Avoid scattering hardcoded tuning values through fit functions.
- If thresholds matter, tune and report them explicitly rather than assuming `0.5` is optimal.

## Documentation standards

- When changing dataset construction logic, update the relevant build script and its exported README or summary.
- When changing split logic, update [data_split.md](/Users/norbert.jaworski/Documents/small/WEC2026/data_split.md) if the documented behavior changes.
- New top-level modeling directives should be added here so pipeline rules stay centralized.