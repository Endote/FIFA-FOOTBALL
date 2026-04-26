# Feature Diagnostics

- Fit splits: `train,val`
- Explain split: `test`
- Explain AP: `0.130042`
- Explain AUROC: `0.791169`
- Explain Brier: `0.068924`
- Explain balanced accuracy at threshold 0.5: `0.569705`

## Top SHAP features

- `cumul_received_unsucc`: `0.361433`
- `formation`: `0.207213`
- `cumul_pressure_turnover_rate`: `0.188718`
- `position`: `0.180677`
- `cumul_pressure_forward_rate`: `0.125353`
- `mean_abs_pass_angle_under_pressure`: `0.072745`
- `checkpoint`: `0.054064`
- `top_distance_share`: `0.053180`
- `avg_top_sprint_distance`: `0.051093`
- `player_z_team_top_distance_share`: `0.043341`
- `top_third_pressure_turnover_rate`: `0.042900`
- `player_share_team_top_distance`: `0.040994`
- `pressure_forward_minus_backward`: `0.040145`
- `cumul_in_game_time`: `0.037989`
- `pressure_escape_score`: `0.032505`
- `last15_pressure_turnover_rate`: `0.030500`
- `possessions_with_2plus_runs`: `0.025378`
- `player_rank_team_top_distance_share`: `0.022540`
- `cumul_shots_blocked`: `0.019980`
- `last15_pressure_events`: `0.018801`
- `player_z_team_shots_total`: `0.016351`
- `top_third_pressure_count`: `0.016203`
- `player_rank_team_cumul_shots`: `0.015769`
- `last15_pressure_forward_rate`: `0.012181`
- `player_share_team_shots_on_target`: `0.011618`
- `player_share_team_cumul_shots`: `0.009855`
- `cumul_shots_under_pressure`: `0.007048`
- `cumul_pressure_events`: `0.007024`
- `cumul_shots_on_target`: `0.005457`
- `cumul_shots_total`: `0.004859`

## Strongest numeric/boolean correlations

- `cumul_shots_on_target` vs `player_share_team_shots_on_target`: corr=`0.9966`, abs=`0.9966`
- `cumul_shots_total` vs `player_share_team_cumul_shots`: corr=`0.9818`, abs=`0.9818`
- `player_share_team_top_distance` vs `player_z_team_top_distance_share`: corr=`0.9670`, abs=`0.9670`
- `player_rank_team_top_distance_share` vs `player_z_team_top_distance_share`: corr=`-0.9331`, abs=`0.9331`
- `top_distance_share` vs `player_share_team_top_distance`: corr=`0.9249`, abs=`0.9249`
- `top_distance_share` vs `player_z_team_top_distance_share`: corr=`0.9059`, abs=`0.9059`
- `cumul_shots_total` vs `cumul_shots_under_pressure`: corr=`0.8888`, abs=`0.8888`
- `cumul_shots_under_pressure` vs `player_share_team_cumul_shots`: corr=`0.8664`, abs=`0.8664`
- `player_share_team_top_distance` vs `player_rank_team_top_distance_share`: corr=`-0.8636`, abs=`0.8636`
- `player_share_team_cumul_shots` vs `player_z_team_shots_total`: corr=`0.7911`, abs=`0.7911`
- `top_distance_share` vs `player_rank_team_top_distance_share`: corr=`-0.7849`, abs=`0.7849`
- `cumul_shots_total` vs `player_z_team_shots_total`: corr=`0.7797`, abs=`0.7797`
- `pressure_forward_minus_backward` vs `pressure_escape_score`: corr=`0.7224`, abs=`0.7224`
- `player_rank_team_cumul_shots` vs `player_z_team_shots_total`: corr=`-0.7110`, abs=`0.7110`
- `cumul_pressure_events` vs `top_third_pressure_count`: corr=`0.7030`, abs=`0.7030`

## Highest VIF features

- `player_z_team_top_distance_share`: vif=`34.9361`, r2=`0.9714`
- `player_z_team_shots_total`: vif=`25.8662`, r2=`0.9613`
- `cumul_shots_total`: vif=`12.9345`, r2=`0.9227`
- `player_share_team_cumul_shots`: vif=`11.7233`, r2=`0.9147`
- `player_share_team_top_distance`: vif=`11.1704`, r2=`0.9105`
- `player_rank_team_top_distance_share`: vif=`9.0828`, r2=`0.8899`
- `pressure_escape_score`: vif=`7.4965`, r2=`0.8666`
- `cumul_pressure_events`: vif=`5.7788`, r2=`0.8270`
- `cumul_shots_under_pressure`: vif=`5.1750`, r2=`0.8068`
- `cumul_shots_on_target`: vif=`5.1157`, r2=`0.8045`
- `top_distance_share`: vif=`4.7701`, r2=`0.7904`
- `pressure_forward_minus_backward`: vif=`4.3095`, r2=`0.7680`
- `player_share_team_shots_on_target`: vif=`3.7107`, r2=`0.7305`
- `player_rank_team_cumul_shots`: vif=`2.8004`, r2=`0.6429`
- `top_third_pressure_count`: vif=`2.7595`, r2=`0.6376`

## Candidate features to review for elimination

- Heuristic only: high correlation is not enough by itself. The table below marks the weaker SHAP side of each highly correlated pair.

- Drop candidate `cumul_shots_on_target` over keep `player_share_team_shots_on_target`: abs_corr=`0.9966`, drop_shap=`0.005457`, keep_shap=`0.011618`
- Drop candidate `cumul_shots_total` over keep `player_share_team_cumul_shots`: abs_corr=`0.9818`, drop_shap=`0.004859`, keep_shap=`0.009855`
- Drop candidate `player_share_team_top_distance` over keep `player_z_team_top_distance_share`: abs_corr=`0.9670`, drop_shap=`0.040994`, keep_shap=`0.043341`
- Drop candidate `player_rank_team_top_distance_share` over keep `player_z_team_top_distance_share`: abs_corr=`0.9331`, drop_shap=`0.022540`, keep_shap=`0.043341`
- Drop candidate `player_share_team_top_distance` over keep `top_distance_share`: abs_corr=`0.9249`, drop_shap=`0.040994`, keep_shap=`0.053180`
- Drop candidate `player_z_team_top_distance_share` over keep `top_distance_share`: abs_corr=`0.9059`, drop_shap=`0.043341`, keep_shap=`0.053180`
- Drop candidate `cumul_shots_total` over keep `cumul_shots_under_pressure`: abs_corr=`0.8888`, drop_shap=`0.004859`, keep_shap=`0.007048`
- Drop candidate `cumul_shots_under_pressure` over keep `player_share_team_cumul_shots`: abs_corr=`0.8664`, drop_shap=`0.007048`, keep_shap=`0.009855`
- Drop candidate `player_rank_team_top_distance_share` over keep `player_share_team_top_distance`: abs_corr=`0.8636`, drop_shap=`0.022540`, keep_shap=`0.040994`