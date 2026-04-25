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
 test   741         6                 173      111         45       0.060729   0.216161
train  2043        19                 528      279        123       0.060206   0.595974
  val   644         6                 166      155         35       0.054348   0.187865

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
