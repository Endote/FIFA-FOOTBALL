# Feature Diagnostics

- Fit splits: `train,val,test`
- Explain split: `all`
- Explain AP: `0.332050`
- Explain AUROC: `0.849006`
- Explain Brier: `0.061202`
- Explain balanced accuracy at threshold 0.5: `0.553458`

## Top SHAP features

- `cumul_received_unsucc`: `0.388734`
- `cumul_pressured_success_rate`: `0.170704`
- `position`: `0.090437`
- `cumul_bottom_sprint_share`: `0.069088`
- `cumul_in_game_time`: `0.067377`
- `cumul_top_hsr_share`: `0.064748`
- `formation`: `0.054595`
- `cumul_top_run_share`: `0.048959`
- `cumul_pressure_success_rate`: `0.047365`
- `top_distance_share`: `0.041695`
- `formation_attackers`: `0.038428`
- `share_of_possessions_with_top_run`: `0.037164`
- `cumul_unique_run_possessions`: `0.035226`
- `is_home`: `0.034755`
- `distance_per_possession`: `0.034344`

## Strongest numeric/boolean correlations

- `cumul_shots` vs `cumul_shots_total`: corr=`0.9967`, abs=`0.9967`
- `cumul_shots_under_press` vs `cumul_shots_under_pressure`: corr=`0.9963`, abs=`0.9963`
- `cumul_shots_top_third` vs `cumul_shots_total`: corr=`0.9960`, abs=`0.9960`
- `cumul_shots` vs `cumul_shots_top_third`: corr=`0.9927`, abs=`0.9927`
- `possessions_with_2plus_top_runs` vs `top_run_repeat_possession_rate`: corr=`0.9902`, abs=`0.9902`
- `top_sprint_distance` vs `avg_top_sprint_distance`: corr=`0.9893`, abs=`0.9893`
- `cumul_shots_under_pressure` vs `cumul_shots_under_pressure_rate`: corr=`0.9871`, abs=`0.9871`
- `cumul_shots_under_press` vs `cumul_shots_under_pressure_rate`: corr=`0.9833`, abs=`0.9833`
- `cumul_top_run_share` vs `cumul_top_hsr_share`: corr=`0.9755`, abs=`0.9755`
- `cumul_pressures_applied` vs `cumul_pressures_lost`: corr=`0.9413`, abs=`0.9413`
- `cumul_times_pressured` vs `cumul_pressured_succ`: corr=`0.9272`, abs=`0.9272`
- `possessions_with_sprint_and_hsr` vs `share_of_possessions_with_sprint`: corr=`0.9254`, abs=`0.9254`
- `top_hsr_distance` vs `top_runs_per_possession`: corr=`0.9135`, abs=`0.9135`
- `cumul_top_sprint_count` vs `cumul_top_sprint_share`: corr=`0.9029`, abs=`0.9029`
- `cumul_shots_total` vs `cumul_shots_under_pressure`: corr=`0.8968`, abs=`0.8968`

## Highest VIF features

- `cumul_sprints`: vif=`inf`, r2=`1.0000`
- `formation_defenders`: vif=`inf`, r2=`1.0000`
- `formation_midfielders`: vif=`inf`, r2=`1.0000`
- `formation_attackers`: vif=`inf`, r2=`1.0000`
- `formation_striker`: vif=`inf`, r2=`1.0000`
- `cumul_times_pressured`: vif=`inf`, r2=`1.0000`
- `cumul_pressured_succ`: vif=`inf`, r2=`1.0000`
- `cumul_pressured_unsucc`: vif=`inf`, r2=`1.0000`
- `cumul_pressures_applied`: vif=`inf`, r2=`1.0000`
- `cumul_pressures_won`: vif=`inf`, r2=`1.0000`
- `cumul_pressures_lost`: vif=`inf`, r2=`1.0000`
- `cumul_top_sprint_count`: vif=`inf`, r2=`1.0000`
- `cumul_middle_sprint_count`: vif=`inf`, r2=`1.0000`
- `cumul_bottom_sprint_count`: vif=`inf`, r2=`1.0000`
- `cumul_shots_total`: vif=`528.3721`, r2=`0.9981`

## Candidate features to review for elimination

- Heuristic only: high correlation is not enough by itself. The table below marks the weaker SHAP side of each highly correlated pair.

- Drop candidate `cumul_shots` over keep `cumul_shots_total`: abs_corr=`0.9967`, drop_shap=`0.004454`, keep_shap=`0.005275`
- Drop candidate `cumul_shots_under_pressure` over keep `cumul_shots_under_press`: abs_corr=`0.9963`, drop_shap=`0.005629`, keep_shap=`0.006752`
- Drop candidate `cumul_shots_total` over keep `cumul_shots_top_third`: abs_corr=`0.9960`, drop_shap=`0.005275`, keep_shap=`0.008662`
- Drop candidate `cumul_shots` over keep `cumul_shots_top_third`: abs_corr=`0.9927`, drop_shap=`0.004454`, keep_shap=`0.008662`
- Drop candidate `possessions_with_2plus_top_runs` over keep `top_run_repeat_possession_rate`: abs_corr=`0.9902`, drop_shap=`0.010581`, keep_shap=`0.011426`
- Drop candidate `top_sprint_distance` over keep `avg_top_sprint_distance`: abs_corr=`0.9893`, drop_shap=`0.011813`, keep_shap=`0.022204`
- Drop candidate `cumul_shots_under_pressure` over keep `cumul_shots_under_pressure_rate`: abs_corr=`0.9871`, drop_shap=`0.005629`, keep_shap=`0.015907`
- Drop candidate `cumul_shots_under_press` over keep `cumul_shots_under_pressure_rate`: abs_corr=`0.9833`, drop_shap=`0.006752`, keep_shap=`0.015907`
- Drop candidate `cumul_top_run_share` over keep `cumul_top_hsr_share`: abs_corr=`0.9755`, drop_shap=`0.048959`, keep_shap=`0.064748`
- Drop candidate `cumul_pressures_lost` over keep `cumul_pressures_applied`: abs_corr=`0.9413`, drop_shap=`0.004391`, keep_shap=`0.009361`
- Drop candidate `cumul_times_pressured` over keep `cumul_pressured_succ`: abs_corr=`0.9272`, drop_shap=`0.005086`, keep_shap=`0.013124`
- Drop candidate `share_of_possessions_with_sprint` over keep `possessions_with_sprint_and_hsr`: abs_corr=`0.9254`, drop_shap=`0.009136`, keep_shap=`0.010006`
- Drop candidate `top_runs_per_possession` over keep `top_hsr_distance`: abs_corr=`0.9135`, drop_shap=`0.014040`, keep_shap=`0.015772`
- Drop candidate `cumul_top_sprint_count` over keep `cumul_top_sprint_share`: abs_corr=`0.9029`, drop_shap=`0.003645`, keep_shap=`0.012918`
- Drop candidate `cumul_shots_total` over keep `cumul_shots_under_pressure`: abs_corr=`0.8968`, drop_shap=`0.005275`, keep_shap=`0.005629`
- Drop candidate `cumul_shots` over keep `cumul_shots_under_press`: abs_corr=`0.8967`, drop_shap=`0.004454`, keep_shap=`0.006752`
- Drop candidate `cumul_shots_total` over keep `cumul_shots_under_press`: abs_corr=`0.8964`, drop_shap=`0.005275`, keep_shap=`0.006752`
- Drop candidate `runs_per_possession` over keep `distance_per_possession`: abs_corr=`0.8941`, drop_shap=`0.011466`, keep_shap=`0.034344`
- Drop candidate `cumul_shots` over keep `cumul_shots_under_pressure`: abs_corr=`0.8932`, drop_shap=`0.004454`, keep_shap=`0.005629`
- Drop candidate `cumul_shots_under_pressure` over keep `cumul_shots_top_third`: abs_corr=`0.8917`, drop_shap=`0.005629`, keep_shap=`0.008662`