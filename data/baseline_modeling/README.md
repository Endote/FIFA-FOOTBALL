# Baseline Modeling Dataset

## Source

- Source table: [players_quarters_final.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/players_quarters_final.csv)
- Extension source(s): `passes,runs`
- Feature-window mode: `cumul`
- Output directory: [baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/data/baseline_modeling)

## Requested 60 / 20 / 20 split

- Exact 60 / 20 / 20 is not always achievable with whole-fixture chronological splits.
- The closest deterministic split on this dataset is:

split  rows  fixtures  player_appearances  players  positives  positive_rate  row_share
 test   643         5                 147       95         29       0.045101   0.199938
train  1917        20                 499      266        109       0.056860   0.596082
  val   656         6                 162      143         44       0.067073   0.203980

## Added baseline extension features

- `last_15_received_succ`
- `last_15_received_unsucc`
- `cumul_received_succ`
- `cumul_received_unsucc`
- `last15_top_sprint_count`
- `last15_top_hsr_count`
- `last15_middle_sprint_count`
- `last15_middle_hsr_count`
- `last15_bottom_sprint_count`
- `last15_bottom_hsr_count`
- `cumul_top_sprint_count`
- `cumul_top_hsr_count`
- `cumul_middle_sprint_count`
- `cumul_middle_hsr_count`
- `cumul_bottom_sprint_count`
- `cumul_bottom_hsr_count`
