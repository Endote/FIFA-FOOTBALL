# Baseline Modeling Dataset

## Source

- Source table: [players_quarters_final.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/players_quarters_final.csv)
- Extension source(s): `passes,runs,shots`
- Feature-window mode: `cumul`
- Output directory: [baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/data/baseline_modeling)

## Requested 60 / 20 / 20 split

- Exact 60 / 20 / 20 is not always achievable with whole-fixture chronological splits.
- The closest deterministic split on this dataset is:

split  rows  fixtures  player_appearances  players  positives  positive_rate  row_share
 test   635         5                 147       95         29       0.045669   0.197943
train  1917        20                 499      266        109       0.056860   0.597569
  val   656         6                 162      143         44       0.067073   0.204489

## Added baseline extension features

- `cumul_received_succ`
- `cumul_received_unsucc`
- `cumul_top_sprint_count`
- `cumul_top_hsr_count`
- `cumul_middle_sprint_count`
- `cumul_middle_hsr_count`
- `cumul_bottom_sprint_count`
- `cumul_bottom_hsr_count`
- `cumul_top_run_share`
- `cumul_middle_run_share`
- `cumul_bottom_run_share`
- `cumul_top_sprint_share`
- `cumul_top_hsr_share`
- `cumul_bottom_sprint_share`
- `top_sprint_distance`
- `top_hsr_distance`
- `distance_per_run`
- `distance_per_possession`
- `top_distance_share`
- `sprint_distance_share`
- `avg_top_sprint_distance`
- `cumul_unique_run_possessions`
- `runs_per_possession`
- `sprints_per_possession`
- `top_runs_per_possession`
- `share_of_possessions_with_top_run`
- `share_of_possessions_with_sprint`
- `possessions_with_2plus_runs`
- `possessions_with_2plus_top_runs`
- `possessions_with_sprint_and_hsr`
- `top_run_repeat_possession_rate`
- `cumul_shots_total`
- `cumul_shots_special`
- `cumul_shots_set_play`
- `cumul_shots_blocked`
- `cumul_shots_under_pressure`
- `cumul_shots_under_pressure_rate`
