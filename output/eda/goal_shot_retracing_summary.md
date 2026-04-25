# Retracing Goal Shots From The Target

## What can be identified exactly

- The target `scored_after` lets us locate the **last scoring window** for each player appearance with at least one future goal.
- It does **not** reveal every goal scored by multi-goal players.
- A shot can be tagged as an **exact inferred goal shot** only when that last scoring window contains exactly one shot by that player.

## Coverage

- Player appearances with at least one future goal signal: 76
- Exact single-shot windows: 51
- Ambiguous multi-shot windows: 20
- Windows with no shot found: 5

## Interpretation

- The `exact_single_shot` subset is the cleanest retraced sample of scoring shots.
- The `ambiguous_multiple_shots` subset almost certainly contains real goals, but the exact shot cannot be identified without an explicit shot outcome field.
- The `no_shot_found` cases suggest either post-checkpoint data coverage issues, non-standard recording gaps, or target/shot table mismatches.

## Files

- [goal_shot_windows.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/goal_shot_windows.csv)
- [goal_shot_candidates.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/goal_shot_candidates.csv)
- [inferred_goal_shots_exact.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/inferred_goal_shots_exact.csv)
- [inferred_goal_shot_characteristics.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/inferred_goal_shot_characteristics.csv)
- [inferred_goal_pre_goal_behavior.csv](/Users/norbert.jaworski/Documents/small/WEC2026/output/eda/inferred_goal_pre_goal_behavior.csv)
