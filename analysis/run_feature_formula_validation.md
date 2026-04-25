# Run-feature formula validation

Goal:

- Recover the exact aggregation formula used to build the run-derived features in `data/players_quarters_final.csv`
- Validate the formula against the published table before changing any data

Files used:

- `data/player_appearance_run.csv`
- `data/players_quarters_final.csv`

## Conclusion

The published run-derived features in `players_quarters_final.csv` are recreated exactly from `player_appearance_run.csv` with the formula below:

- `last15_sprints`
- `last15_hsr`
- `last15_distance`
- `last15_peak_speed`
- `cumul_sprints`
- `cumul_hsr`
- `cumul_distance`
- `cumul_peak_speed`

all match exactly for all `3,486 / 3,486` rows.

The mean-speed features are also reproduced almost exactly:

- `last15_mean_max_speed`: exact for `98.04%` of rows
- `cumul_mean_max_speed`: exact for `98.51%` of rows

Their remaining differences are tiny:

- mean absolute error around `0.00002`
- maximum absolute deviation `0.001`

That residual is a rounding-detail issue, not an aggregation-logic issue.

## Exact formula

### 1. Convert event time to absolute match minute

Use these period offsets:

- `half_1` -> `0`
- `half_2` -> `45`
- `extra_time_1` -> `90`
- `extra_time_2` -> `105`

For each run event:

```text
run_abs = period_offset(period) + minute
```

For each checkpoint in `players_quarters_final.csv`:

- `H1_15` -> `half_1`, minute `15`
- `H1_30` -> `half_1`, minute `30`
- `H1_45` -> `half_1`, minute `45`
- `H2_15` -> `half_2`, minute `15`
- `H2_30` -> `half_2`, minute `30`
- `H2_45` -> `half_2`, minute `45`
- `ET1_15` -> `extra_time_1`, minute `15`

Then:

```text
cp_abs = period_offset(checkpoint_period) + checkpoint_min
```

### 2. Define the cumulative window

Include all runs with:

```text
run_abs <= cp_abs
```

This exactly reproduces:

- `cumul_sprints`
- `cumul_hsr`
- `cumul_distance`
- `cumul_peak_speed`

### 3. Define the last-15 window

Include all runs with:

```text
cp_abs - 15 < run_abs <= cp_abs
```

This is the critical detail.

The last-15 window is based on absolute match minute, not restricted to the same period.

That matters for:

- `H2_15`
- `ET1_15`

because those windows include stoppage-time events from the previous period when they fall into the last 15 absolute match minutes.

This exactly reproduces:

- `last15_sprints`
- `last15_hsr`
- `last15_distance`
- `last15_peak_speed`

## Aggregation rules

Within each window:

- `sprints`:
  - count rows where `run_type == "sprint"`
- `hsr`:
  - count rows where `run_type == "hsr"`
- `distance`:
  - sum `distance` across all run rows in the window
  - this includes both `hsr` and `sprint`
- `peak_speed`:
  - max of `max_speed` across all run rows in the window
- `mean_max_speed`:
  - mean of `max_speed` across all run rows in the window

Rounding observed in the published table:

- `distance` is stored to 2 decimals
- `peak_speed` is stored to 2 decimals
- `mean_max_speed` is stored to 3 decimals

## Validation evidence

### Exact-match validation against the published table

Across all 3,486 rows:

- `last15_sprints`: exact `100.00%`
- `last15_hsr`: exact `100.00%`
- `last15_distance`: exact `100.00%`
- `last15_peak_speed`: exact `100.00%`
- `cumul_sprints`: exact `100.00%`
- `cumul_hsr`: exact `100.00%`
- `cumul_distance`: exact `100.00%`
- `cumul_peak_speed`: exact `100.00%`

Near-exact features:

- `last15_mean_max_speed`: exact `98.04%`, max error `0.001`
- `cumul_mean_max_speed`: exact `98.51%`, max error `0.001`

### Important implication

The formula matches even on the three obviously bad fixtures:

- `1161`
- `1167`
- `1201`

So the inflated values are not caused by a bad aggregation formula.

They are already present upstream in `player_appearance_run.csv`.

## What was ruled out

These candidate formulas did not match:

- restricting `last15` to the same period only
- excluding previous-period stoppage time from `H2_15` and `ET1_15`
- summing distance for `hsr` only
- excluding checkpoint-minute events from cumulative windows

## Practical reconstruction spec

If you want to rebuild `last15_distance` and `cumul_distance` exactly from `player_appearance_run.csv`, use:

```text
run_abs = period_offset(period) + minute
cp_abs  = period_offset(checkpoint_period) + checkpoint_min

last15_distance = round(sum(distance for cp_abs - 15 < run_abs <= cp_abs), 2)
cumul_distance  = round(sum(distance for run_abs <= cp_abs), 2)
```

with period offsets:

```text
half_1 = 0
half_2 = 45
extra_time_1 = 90
extra_time_2 = 105
```

and grouped by:

```text
player_appearance_id, checkpoint
```

## Bottom line

You do not need to search for a different aggregation formula.

The formula is already recovered and validated.

The remaining problem is source-data quality inside `player_appearance_run.csv`, not the way `players_quarters_final.csv` was derived from it.
