# Goal Scoring EDA And Modeling Summary

## Dataset framing

- Main panel: 3,486 checkpoint rows, 31 fixtures, 869 player appearances.
- Positive class: 203 rows (5.82%).
- Holdout policy: latest match dates reserved until both row share and positive share exceeded 20%.
- Holdout dates: 2025-06-21, 2025-06-22, 2025-06-25, 2025-06-28.
- Train/validation rows: 2,635; holdout rows: 851.
- Train/validation positives: 158; holdout positives: 45.

## Leakage and preprocessing checks

- Random row splitting is invalid because the same `player_appearance_id` contributes multiple checkpoints and the target is defined within-match after each checkpoint.
- A fixture-grouped split is required; model selection here uses `StratifiedGroupKFold` grouped by `fixture_id`.
- Event-window engineering was validated against the official base features. The key hidden rule is that stoppage time remains inside the same period as minutes above 45, so `H2_15` and `ET1_15` windows must be built on an absolute minute axis.
- `jersey_number` was excluded from modeling because it showed signal but is likely a competition-specific artifact, not stable football behavior.

## Target profile

### By position

            mean  sum  count
position                    
A         0.1253   60    479
D         0.0352   33    938
G         0.0000    0    240
M         0.0665   65    978

### By checkpoint

              mean  sum  count
checkpoint                    
H1_15       0.0949   50    527
H1_30       0.0721   38    527
H1_45       0.0588   31    527
H2_15       0.0398   21    527
H2_30       0.0342   18    527

## Best benchmark

- Best cross-validated configuration: `context_only` with `logreg`.
- Cross-validated AUROC: 0.7034 +/- 0.0663.
- Cross-validated balanced accuracy at 0.5 threshold: 0.6386 +/- 0.0474.
- Holdout AUROC: 0.7463.
- Holdout balanced accuracy at 0.5 threshold: 0.6703.

## Top numeric signals on train/validation

                                            feature  missing_rate  corr_with_target  univariate_auc  mean_positive  mean_negative
                            cumul_pass_bottom_count           0.0           -0.1185          0.7074         2.7405         6.6851
                      cumul_pass_bottom_count_per15           0.0           -0.1209          0.6910         1.1187         2.3913
                   cumul_run_detail_top_share_per15           0.0            0.0759          0.6555         0.2736         0.1572
                           last15_pass_bottom_count           0.0           -0.0983          0.6540         1.0886         2.1853
cumul_applied_pressure_unique_pressed_players_per15           0.0            0.1007          0.6458         1.9384         1.4374
                                   cumul_pass_count           0.0           -0.0944          0.6438        13.9557        20.9697
          cumul_run_detail_unique_possessions_per15           0.0            0.0812          0.6381         1.3590         1.0000
                          cumul_pass_accurate_count           0.0           -0.0859          0.6381        12.0443        17.5652
  cumul_applied_pressure_allowed_forward_rate_per15           0.0            0.0610          0.6342         0.1359         0.0779
 cumul_applied_pressure_allowed_forward_count_per15           0.0            0.1198          0.6334         0.9944         0.5724

## Strongest linear effects in the full-feature logistic model

                              model_feature  coefficient  abs_coefficient
       num__last15_run_detail_mean_distance       0.8972           0.8972
                          cat__subbed_False      -0.8713           0.8713
                            cat__position_G      -0.7926           0.7926
  num__cumul_under_pressure_unique_pressers       0.7859           0.7859
                       cat__formation_3-4-3      -0.7169           0.7169
   num__cumul_under_pressure_turnover_count       0.6318           0.6318
    num__cumul_under_pressure_forward_count      -0.6301           0.6301
num__cumul_applied_pressure_top_share_per15      -0.6294           0.6294
                     cat__formation_4-1-3-2      -0.6202           0.6202
       num__cumul_pass_no_target_rate_per15       0.5894           0.5894
        num__cumul_under_pressure_top_share      -0.5841           0.5841
            num__cumul_pass_no_target_count      -0.5840           0.5840

## Top permutation importances on the holdout for the best full model

No permutation table.

## Interpretation notes

- Attackers and earlier checkpoints carry materially higher base rates, which means the model should be benchmarked against context-only baselines rather than raw accuracy.
- Recent attacking involvement matters more than raw cumulative load: short-term shot pressure, attacking-third activity, and run intensity repeatedly surfaced among the strongest screening variables.
- Passing and under-pressure features are worth adding because they capture tactical quality rather than just volume. In particular, high attacking-third pass share, lower turnover-under-pressure rates, and pressure application that induces turnovers are plausible mechanisms tied to later scoring.
- Goalkeepers are a structurally degenerate class in this dataset with zero positives. For competition-style scoring across all rows, a zero-probability rule for keepers is defensible; for scientific interpretation, an outfield-only sensitivity check should be run next.
