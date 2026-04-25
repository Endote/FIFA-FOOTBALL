# players_quarters_final.csv feature audit

## Dataset shape

- Rows: 3,486
- Columns: 33
- Fixtures: 31
- Unique player appearances: 869

## Confirmed data-quality findings

### 1. `formation` is clean in this file

- All 3,486 values match a valid formation pattern such as `4-2-3-1` or `3-4-2-1`.
- No dates or malformed tokens were found in `formation`.

### 2. `date` is clean in this file

- All values parse correctly as `YYYY-MM-DD`.
- `fixture_id -> date` is deterministic: each fixture maps to exactly one date.

### 3. `last15_distance` is corrupted for three fixtures

The feature description says this is high-intensity distance in meters during the last 15 minutes.

Three fixtures have values that are an order of magnitude above the rest of the dataset:

- `1161` on `2025-06-11`
- `1167` on `2025-06-14`
- `1201` on `2025-06-17`

Evidence:

- Typical fixture median for `last15_distance`: about `100-130`
- Corrupted fixtures median:
  - `1161`: `2648.07`
  - `1167`: `3208.18`
  - `1201`: `2588.67`
- Maximum observed `last15_distance`: `8028.31`

This affects 330 rows indirectly at the fixture level, with 149 obviously extreme rows above `3000`.

### 4. `cumul_distance` is corrupted for the same three fixtures

This feature should be cumulative high-intensity distance in meters up to the checkpoint.

Evidence:

- Typical fixture median for `cumul_distance`: about `240-320`
- Corrupted fixtures median:
  - `1161`: `7487.20`
  - `1167`: `6032.99`
  - `1201`: `5781.35`
- Maximum observed `cumul_distance`: `25667.77`

This is the same pattern as `last15_distance`, so both distance variables appear to share the same upstream issue.

## Checks that passed

- `checkpoint`, `checkpoint_period`, and `checkpoint_min` are internally consistent.
- `last15_*` counts never exceed their cumulative counterparts.
- Shot sub-counts never exceed total shots.
- Speed peaks are never below the corresponding mean max speeds.

## Features most likely safe to leave out

### Drop because they are identifiers or high leakage risk

- `player_appearance_id`
- `player_id`
- `fixture_id`

Reason:

- These are identifiers, not stable football attributes.
- They can inflate apparent signal by memorizing players or matches.
- `player_appearance_id` is especially bad because it uniquely pins down a player-match record and repeats across checkpoints.

### Drop because they are redundant encodings

Keep `checkpoint` and drop:

- `checkpoint_period`
- `checkpoint_min`

Reason:

- `checkpoint` fully determines both of them.
- Using all three adds duplicate time information without new signal.

### Drop one of the date/match proxies

Prefer dropping:

- `date`

Reason:

- `date` adds little standalone signal here and mostly acts as a proxy for match identity.
- If match context is needed, it should be handled explicitly and carefully, not via a raw date token.

### Drop because the values are corrupted unless repaired first

- `last15_distance`
- `cumul_distance`

Reason:

- Both are clearly corrupted for 3 fixtures (`9.47%` of rows).
- If the source can be rebuilt, keep them after repair.
- If not, dropping them is safer than training on broken values.

### Probably drop because it is redundant with `minute_in` and `minute_out`

- `subbed`

Reason:

- `subbed` is largely derivable from entry/exit timing.
- The timing fields carry more detailed information.

## Features worth keeping

- `position`
- `is_home`
- `formation`
- `minute_in`
- `minute_out`
- `jersey_number` only if you are willing to tolerate weak player-specific proxy effects
- `last15_sprints`
- `last15_hsr`
- `last15_mean_max_speed`
- `last15_peak_speed`
- `last15_shots`
- `last15_shots_on_target`
- `last15_shots_under_press`
- `last15_shots_top_third`
- `cumul_sprints`
- `cumul_hsr`
- `cumul_mean_max_speed`
- `cumul_peak_speed`
- `cumul_shots`
- `cumul_shots_on_target`
- `cumul_shots_under_press`
- `cumul_shots_top_third`

## Practical reduced feature set

If the goal is a first clean baseline model, keep:

- `checkpoint`
- `position`
- `is_home`
- `formation`
- `minute_in`
- `minute_out`
- `last15_sprints`
- `last15_hsr`
- `last15_mean_max_speed`
- `last15_peak_speed`
- `last15_shots`
- `last15_shots_on_target`
- `last15_shots_under_press`
- `last15_shots_top_third`
- `cumul_sprints`
- `cumul_hsr`
- `cumul_mean_max_speed`
- `cumul_peak_speed`
- `cumul_shots`
- `cumul_shots_on_target`
- `cumul_shots_under_press`
- `cumul_shots_top_third`

and drop:

- `player_appearance_id`
- `player_id`
- `fixture_id`
- `date`
- `checkpoint_period`
- `checkpoint_min`
- `subbed`
- `last15_distance`
- `cumul_distance`

## Next step

Best next move is to rebuild the two distance features from source tables if possible. If that is not possible, train without them and validate using match-aware splits so player and fixture leakage does not fool you.
