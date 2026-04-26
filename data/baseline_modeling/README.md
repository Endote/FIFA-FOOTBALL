# Baseline Modeling Dataset

## Source

- Source table: [players_quarters_final.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/players_quarters_final.csv)
- Extension source(s): `passes,pressure,runs,shots`
- Feature-window mode: `all`
- Output directory: [baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/data/baseline_modeling)

## Requested 60 / 20 / 20 split

- Split method: `seeded_grouped_fixture_split`
- Split seed: `7`
- Exact 60 / 20 / 20 is not always achievable with whole-fixture grouped splits.
- The closest seeded grouped split on this dataset is:

split  rows  fixtures  player_appearances  players  positives  positive_rate  row_share
 test   765         7                 188      123         43       0.056209   0.238466
train  1876        18                 473      224        105       0.055970   0.584788
  val   567         6                 147      134         34       0.059965   0.176746

## Grouped CV

- CV method: `StratifiedGroupKFold`
- Group variable: `fixture_id`
- Folds: `3`
- Repeats: `2`
- Base seed: `7`

## Added baseline extension features

- `cumul_received_unsucc`
- `cumul_pressured_success_rate`
- `cumul_top_hsr_share`
- `distance_per_run`
- `distance_per_possession`
- `top_distance_share`
- `cumul_unique_run_possessions`
- `cumul_bottom_sprint_share`
- `cumul_pressure_success_rate`
- `cumul_top_run_share`
- `player_share_team_cumul_shots`
- `player_share_team_shots_on_target`
- `cumul_pass_middle_accuracy_rate`
- `cumul_pass_top_accuracy_rate`
- `cumul_pressure_turnover_rate`
- `cumul_pressure_forward_rate`
- `player_z_team_top_distance_share`
