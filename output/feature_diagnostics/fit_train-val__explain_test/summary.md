# Feature Diagnostics

- Fit splits: `train,val`
- Explain split: `test`
- Explain AP: `0.133793`
- Explain AUROC: `0.816945`
- Explain Brier: `0.066398`
- Explain balanced accuracy at threshold 0.5: `0.551639`

## Top SHAP features

- `cumul_received_unsucc`: `0.377293`
- `position`: `0.225701`
- `formation`: `0.215542`
- `player_z_team_top_distance_share`: `0.077065`
- `avg_top_sprint_distance`: `0.063614`
- `team_total_top_runs`: `0.062834`
- `checkpoint`: `0.061919`
- `top_distance_share`: `0.052256`
- `team_minus_opponent_shot_load`: `0.050590`
- `player_rank_team_top_distance_share`: `0.040031`
- `player_share_team_top_distance`: `0.038319`
- `opponent_total_cumul_shots`: `0.035676`
- `player_z_team_shots_total`: `0.033242`
- `cumul_in_game_time`: `0.030887`
- `possessions_with_2plus_runs`: `0.026659`
- `cumul_shots_blocked`: `0.022849`
- `team_total_cumul_shots`: `0.013949`
- `player_share_team_shots_on_target`: `0.012602`
- `cumul_shots_under_pressure`: `0.009801`
- `player_rank_team_cumul_shots`: `0.009675`
- `player_share_team_cumul_shots`: `0.009397`
- `cumul_shots_total`: `0.007045`
- `cumul_shots_on_target`: `0.006328`

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
- `player_rank_team_cumul_shots` vs `player_z_team_shots_total`: corr=`-0.7110`, abs=`0.7110`
- `cumul_shots_under_pressure` vs `player_z_team_shots_total`: corr=`0.6905`, abs=`0.6905`
- `team_total_cumul_shots` vs `team_total_top_runs`: corr=`0.6850`, abs=`0.6850`

## Highest VIF features

- `team_total_cumul_shots`: vif=`inf`, r2=`1.0000`
- `opponent_total_cumul_shots`: vif=`inf`, r2=`1.0000`
- `team_minus_opponent_shot_load`: vif=`inf`, r2=`1.0000`
- `player_z_team_top_distance_share`: vif=`34.7873`, r2=`0.9713`
- `player_z_team_shots_total`: vif=`28.0460`, r2=`0.9643`
- `cumul_shots_total`: vif=`17.6630`, r2=`0.9434`
- `player_share_team_cumul_shots`: vif=`11.9338`, r2=`0.9162`
- `player_share_team_top_distance`: vif=`11.1239`, r2=`0.9101`
- `player_rank_team_top_distance_share`: vif=`9.3803`, r2=`0.8934`
- `top_distance_share`: vif=`5.2549`, r2=`0.8097`
- `cumul_shots_under_pressure`: vif=`5.1262`, r2=`0.8049`
- `cumul_shots_on_target`: vif=`5.0598`, r2=`0.8024`
- `player_rank_team_cumul_shots`: vif=`4.5770`, r2=`0.7815`
- `player_share_team_shots_on_target`: vif=`3.6872`, r2=`0.7288`
- `team_total_top_runs`: vif=`2.6037`, r2=`0.6159`

## Candidate features to review for elimination

- Heuristic only: high correlation is not enough by itself. The table below marks the weaker SHAP side of each highly correlated pair.

- Drop candidate `cumul_shots_on_target` over keep `player_share_team_shots_on_target`: abs_corr=`0.9966`, drop_shap=`0.006328`, keep_shap=`0.012602`
- Drop candidate `cumul_shots_total` over keep `player_share_team_cumul_shots`: abs_corr=`0.9818`, drop_shap=`0.007045`, keep_shap=`0.009397`
- Drop candidate `player_share_team_top_distance` over keep `player_z_team_top_distance_share`: abs_corr=`0.9670`, drop_shap=`0.038319`, keep_shap=`0.077065`
- Drop candidate `player_rank_team_top_distance_share` over keep `player_z_team_top_distance_share`: abs_corr=`0.9331`, drop_shap=`0.040031`, keep_shap=`0.077065`
- Drop candidate `player_share_team_top_distance` over keep `top_distance_share`: abs_corr=`0.9249`, drop_shap=`0.038319`, keep_shap=`0.052256`
- Drop candidate `top_distance_share` over keep `player_z_team_top_distance_share`: abs_corr=`0.9059`, drop_shap=`0.052256`, keep_shap=`0.077065`
- Drop candidate `cumul_shots_total` over keep `cumul_shots_under_pressure`: abs_corr=`0.8888`, drop_shap=`0.007045`, keep_shap=`0.009801`
- Drop candidate `player_share_team_cumul_shots` over keep `cumul_shots_under_pressure`: abs_corr=`0.8664`, drop_shap=`0.009397`, keep_shap=`0.009801`
- Drop candidate `player_share_team_top_distance` over keep `player_rank_team_top_distance_share`: abs_corr=`0.8636`, drop_shap=`0.038319`, keep_shap=`0.040031`