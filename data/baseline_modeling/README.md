# Baseline Modeling Dataset

## Source

- Source table: [players_quarters_final.csv](/Users/norbert.jaworski/Documents/small/WEC2026/data/players_quarters_final.csv)
- Output directory: [baseline_modeling](/Users/norbert.jaworski/Documents/small/WEC2026/data/baseline_modeling)

## Row quality filters applied before splitting

- `last15_distance < 1000`
- `cumul_distance < 1000`
- `last15_mean_max_speed < 10.3`
- `cumul_mean_max_speed < 10.3`
- Removed rows: 302
- Removed player appearances: 91

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
 test   633         5                 147       95         29       0.045814   0.198807
train  1896        20                 493      263        107       0.056435   0.595477
  val   655         6                 162      143         44       0.067176   0.205716

## What is removed from the model-ready files

- `player_appearance_id`
- `player_id`
- `fixture_id`
- `date`
- `jersey_number`
- `checkpoint_period`
- `checkpoint_min`
- `fixture_order`

## Added baseline extension features

- `last_15_received_succ`
- `last_15_received_unsucc`

