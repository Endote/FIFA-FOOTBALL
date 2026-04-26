# Feature Diagnostics

- Fit splits: `train,val`
- Explain split: `test`
- Explain AP: `0.160930`
- Explain AUROC: `0.814612`
- Explain Brier: `0.067338`
- Explain balanced accuracy at threshold 0.5: `0.568055`

## Top SHAP features

- `cumul_received_unsucc`: `0.366435`
- `position`: `0.273367`
- `formation`: `0.252439`
- `top_distance_share`: `0.118520`
- `checkpoint`: `0.102265`
- `avg_top_sprint_distance`: `0.072228`
- `cumul_in_game_time`: `0.055394`
- `possessions_with_2plus_runs`: `0.054596`
- `cumul_shots_blocked`: `0.026875`
- `cumul_shots_total`: `0.013405`
- `cumul_shots_under_pressure`: `0.011683`
- `cumul_shots_on_target`: `0.009101`

## Strongest numeric/boolean correlations

- `cumul_shots_total` vs `cumul_shots_under_pressure`: corr=`0.8888`, abs=`0.8888`
- `cumul_shots_on_target` vs `cumul_shots_total`: corr=`0.6330`, abs=`0.6330`
- `cumul_shots_on_target` vs `cumul_shots_under_pressure`: corr=`0.6012`, abs=`0.6012`
- `cumul_shots_total` vs `cumul_shots_blocked`: corr=`0.5964`, abs=`0.5964`
- `top_distance_share` vs `avg_top_sprint_distance`: corr=`0.5697`, abs=`0.5697`
- `cumul_shots_blocked` vs `cumul_shots_under_pressure`: corr=`0.5136`, abs=`0.5136`
- `cumul_received_unsucc` vs `cumul_in_game_time`: corr=`0.5005`, abs=`0.5005`
- `top_distance_share` vs `possessions_with_2plus_runs`: corr=`0.4515`, abs=`0.4515`
- `avg_top_sprint_distance` vs `possessions_with_2plus_runs`: corr=`0.3876`, abs=`0.3876`
- `top_distance_share` vs `cumul_shots_total`: corr=`0.2571`, abs=`0.2571`
- `cumul_in_game_time` vs `cumul_shots_total`: corr=`0.2458`, abs=`0.2458`
- `top_distance_share` vs `cumul_shots_under_pressure`: corr=`0.2364`, abs=`0.2364`
- `cumul_received_unsucc` vs `top_distance_share`: corr=`-0.2320`, abs=`0.2320`
- `avg_top_sprint_distance` vs `cumul_shots_total`: corr=`0.2137`, abs=`0.2137`
- `possessions_with_2plus_runs` vs `cumul_shots_total`: corr=`0.2127`, abs=`0.2127`

## Highest VIF features

- `cumul_shots_total`: vif=`7.8501`, r2=`0.8726`
- `cumul_shots_under_pressure`: vif=`5.0594`, r2=`0.8023`
- `cumul_shots_on_target`: vif=`2.4805`, r2=`0.5969`
- `cumul_shots_blocked`: vif=`2.3134`, r2=`0.5677`
- `top_distance_share`: vif=`1.4303`, r2=`0.3009`
- `avg_top_sprint_distance`: vif=`1.3611`, r2=`0.2653`
- `cumul_in_game_time`: vif=`1.2616`, r2=`0.2073`
- `possessions_with_2plus_runs`: vif=`1.2169`, r2=`0.1782`
- `cumul_received_unsucc`: vif=`1.1624`, r2=`0.1397`

## Candidate features to review for elimination

- Heuristic only: high correlation is not enough by itself. The table below marks the weaker SHAP side of each highly correlated pair.

- Drop candidate `cumul_shots_under_pressure` over keep `cumul_shots_total`: abs_corr=`0.8888`, drop_shap=`0.011683`, keep_shap=`0.013405`