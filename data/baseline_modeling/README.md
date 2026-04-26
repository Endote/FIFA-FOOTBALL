# Baseline Modeling Dataset

## Source

- Source table: [players_quarters_final.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/players_quarters_final.csv)
- Extension source(s): `passes,pressure,runs,shots`
- Feature-window mode: `all`
- Output directory: [baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/data/baseline_modeling)

## Requested 60 / 20 / 20 split

- Exact 60 / 20 / 20 is not always achievable with whole-fixture chronological splits.
- The closest deterministic split on this dataset is:

split  rows  fixtures  player_appearances  players  positives  positive_rate  row_share
 test   635         5                 147       95         29       0.045669   0.197943
train  1917        20                 499      266        109       0.056860   0.597569
  val   656         6                 162      143         44       0.067073   0.204489

## Added baseline extension features

- `cumul_received_unsucc`
- `top_distance_share`
- `avg_top_sprint_distance`
- `possessions_with_2plus_runs`
- `cumul_shots_total`
- `cumul_shots_blocked`
- `cumul_shots_under_pressure`
