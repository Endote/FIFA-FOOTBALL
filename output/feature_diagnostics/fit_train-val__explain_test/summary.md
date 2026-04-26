# Feature Diagnostics

- Fit splits: `train,val`
- Explain split: `test`
- Explain AP: `0.106214`
- Explain AUROC: `0.669039`
- Explain Brier: `0.074993`
- Explain balanced accuracy at threshold 0.5: `0.524351`

## Top SHAP features

- `cumul_received_unsucc`: `0.691421`
- `cumul_bottom_sprint_share`: `0.345142`
- `position`: `0.334606`
- `formation`: `0.303608`
- `cumul_pressured_success_rate`: `0.303268`
- `cumul_top_hsr_share`: `0.283974`
- `distance_per_run`: `0.272855`
- `top_distance_share`: `0.194879`
- `distance_per_possession`: `0.182428`
- `cumul_pressure_success_rate`: `0.165902`
- `cumul_top_run_share`: `0.152847`
- `cumul_sprints`: `0.148675`
- `cumul_in_game_time`: `0.124884`
- `cumul_unique_run_possessions`: `0.055530`
- `checkpoint`: `0.042014`
- `is_home`: `0.015537`

## Strongest numeric/boolean correlations

- `cumul_top_hsr_share` vs `cumul_top_run_share`: corr=`0.9737`, abs=`0.9737`
- `top_distance_share` vs `cumul_top_run_share`: corr=`0.7886`, abs=`0.7886`
- `cumul_top_hsr_share` vs `top_distance_share`: corr=`0.7613`, abs=`0.7613`
- `distance_per_run` vs `distance_per_possession`: corr=`0.5825`, abs=`0.5825`
- `cumul_received_unsucc` vs `cumul_in_game_time`: corr=`0.5214`, abs=`0.5214`
- `cumul_in_game_time` vs `cumul_sprints`: corr=`0.5062`, abs=`0.5062`
- `cumul_unique_run_possessions` vs `cumul_sprints`: corr=`0.4833`, abs=`0.4833`
- `distance_per_possession` vs `cumul_sprints`: corr=`0.4471`, abs=`0.4471`
- `cumul_unique_run_possessions` vs `cumul_top_run_share`: corr=`0.3965`, abs=`0.3965`
- `cumul_top_hsr_share` vs `cumul_unique_run_possessions`: corr=`0.3948`, abs=`0.3948`
- `distance_per_possession` vs `cumul_unique_run_possessions`: corr=`0.3674`, abs=`0.3674`
- `cumul_bottom_sprint_share` vs `cumul_pressure_success_rate`: corr=`0.3616`, abs=`0.3616`
- `distance_per_possession` vs `top_distance_share`: corr=`0.3575`, abs=`0.3575`
- `top_distance_share` vs `cumul_unique_run_possessions`: corr=`0.3543`, abs=`0.3543`
- `cumul_unique_run_possessions` vs `cumul_pressure_success_rate`: corr=`0.3470`, abs=`0.3470`

## Highest VIF features

- `cumul_top_run_share`: vif=`22.9439`, r2=`0.9564`
- `cumul_top_hsr_share`: vif=`19.8989`, r2=`0.9497`
- `distance_per_possession`: vif=`3.1858`, r2=`0.6861`
- `distance_per_run`: vif=`2.9021`, r2=`0.6554`
- `top_distance_share`: vif=`2.4878`, r2=`0.5980`
- `cumul_sprints`: vif=`1.9340`, r2=`0.4829`
- `cumul_in_game_time`: vif=`1.7627`, r2=`0.4327`
- `cumul_unique_run_possessions`: vif=`1.5458`, r2=`0.3531`
- `cumul_bottom_sprint_share`: vif=`1.3228`, r2=`0.2440`
- `cumul_pressure_success_rate`: vif=`1.3051`, r2=`0.2338`
- `cumul_received_unsucc`: vif=`1.1826`, r2=`0.1544`
- `cumul_pressured_success_rate`: vif=`1.1472`, r2=`0.1283`
- `is_home`: vif=`1.0025`, r2=`0.0025`

## Candidate features to review for elimination

- Heuristic only: high correlation is not enough by itself. The table below marks the weaker SHAP side of each highly correlated pair.

- Drop candidate `cumul_top_run_share` over keep `cumul_top_hsr_share`: abs_corr=`0.9737`, drop_shap=`0.152847`, keep_shap=`0.283974`