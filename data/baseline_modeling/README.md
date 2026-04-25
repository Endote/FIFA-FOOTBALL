# Baseline Modeling Dataset

## Source

- Source table: [players_quarters_final.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/players_quarters_final.csv)
- Extension source(s): `passes,pressure`
- Feature-window mode: `cumul`
- Output directory: [baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/data/baseline_modeling)

## Requested 60 / 20 / 20 split

- Exact 60 / 20 / 20 is not always achievable with whole-fixture chronological splits.
- The closest deterministic split on this dataset is:

split  rows  fixtures  player_appearances  players  positives  positive_rate  row_share
 test   633         5                 147       95         29       0.045814   0.198807
train  1896        20                 493      263        107       0.056435   0.595477
  val   655         6                 162      143         44       0.067176   0.205716

## Added baseline extension features

- `cumul_received_succ`
- `cumul_received_unsucc`
- `cumul_times_pressured`
- `cumul_pressured_succ`
- `cumul_pressured_unsucc`
- `cumul_pressures_applied`
- `cumul_pressures_won`
- `cumul_pressures_lost`
- `cumul_pressured_success_rate`
- `cumul_pressure_success_rate`
- `cumul_in_game_time`
