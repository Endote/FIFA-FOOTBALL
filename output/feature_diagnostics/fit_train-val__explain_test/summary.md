# Feature Diagnostics

- Fit splits: `train,val`
- Explain split: `test`
- Explain AP: `0.107149`
- Explain AUROC: `0.691651`
- Explain Brier: `0.073093`
- Explain balanced accuracy at threshold 0.5: `0.550377`

## Top SHAP features

- `cumul_received_unsucc`: `0.675767`
- `position`: `0.329254`
- `cumul_pressure_forward_rate`: `0.314862`
- `cumul_bottom_sprint_share`: `0.279983`
- `formation`: `0.266903`
- `cumul_pressure_turnover_rate`: `0.265706`
- `cumul_top_hsr_share`: `0.254877`
- `distance_per_run`: `0.232912`
- `cumul_top_run_share`: `0.185107`
- `top_distance_share`: `0.164474`
- `cumul_sprints`: `0.164376`
- `player_z_team_top_distance_share`: `0.149469`
- `cumul_pressured_success_rate`: `0.128785`
- `cumul_pressure_success_rate`: `0.114902`
- `distance_per_possession`: `0.109888`
- `cumul_in_game_time`: `0.093927`
- `player_share_team_cumul_shots`: `0.067629`
- `cumul_pass_middle_accuracy_rate`: `0.047194`
- `checkpoint`: `0.039542`
- `cumul_unique_run_possessions`: `0.038450`
- `cumul_pass_top_accuracy_rate`: `0.036603`
- `player_share_team_shots_on_target`: `0.023739`
- `is_home`: `0.007854`

## Strongest numeric/boolean correlations

- `cumul_top_hsr_share` vs `cumul_top_run_share`: corr=`0.9737`, abs=`0.9737`
- `top_distance_share` vs `player_z_team_top_distance_share`: corr=`0.9152`, abs=`0.9152`
- `top_distance_share` vs `cumul_top_run_share`: corr=`0.7886`, abs=`0.7886`
- `cumul_top_hsr_share` vs `top_distance_share`: corr=`0.7613`, abs=`0.7613`
- `cumul_top_run_share` vs `player_z_team_top_distance_share`: corr=`0.7558`, abs=`0.7558`
- `cumul_top_hsr_share` vs `player_z_team_top_distance_share`: corr=`0.7273`, abs=`0.7273`
- `player_share_team_cumul_shots` vs `player_share_team_shots_on_target`: corr=`0.6269`, abs=`0.6269`
- `cumul_pressured_success_rate` vs `cumul_pressure_turnover_rate`: corr=`-0.5875`, abs=`0.5875`
- `distance_per_run` vs `distance_per_possession`: corr=`0.5825`, abs=`0.5825`
- `cumul_pressured_success_rate` vs `cumul_pressure_forward_rate`: corr=`0.5369`, abs=`0.5369`
- `cumul_received_unsucc` vs `cumul_in_game_time`: corr=`0.5214`, abs=`0.5214`
- `cumul_in_game_time` vs `cumul_sprints`: corr=`0.5062`, abs=`0.5062`
- `cumul_unique_run_possessions` vs `cumul_sprints`: corr=`0.4833`, abs=`0.4833`
- `distance_per_possession` vs `cumul_sprints`: corr=`0.4471`, abs=`0.4471`
- `cumul_unique_run_possessions` vs `cumul_top_run_share`: corr=`0.3965`, abs=`0.3965`

## Highest VIF features

- `cumul_top_run_share`: vif=`23.3444`, r2=`0.9572`
- `cumul_top_hsr_share`: vif=`20.0595`, r2=`0.9501`
- `top_distance_share`: vif=`5.8193`, r2=`0.8282`
- `player_z_team_top_distance_share`: vif=`5.1222`, r2=`0.8048`
- `distance_per_possession`: vif=`3.2236`, r2=`0.6898`
- `distance_per_run`: vif=`2.9365`, r2=`0.6595`
- `cumul_pressured_success_rate`: vif=`2.2630`, r2=`0.5581`
- `cumul_sprints`: vif=`1.9850`, r2=`0.4962`
- `cumul_in_game_time`: vif=`1.8457`, r2=`0.4582`
- `cumul_unique_run_possessions`: vif=`1.7045`, r2=`0.4133`
- `cumul_pressure_turnover_rate`: vif=`1.6250`, r2=`0.3846`
- `player_share_team_cumul_shots`: vif=`1.5748`, r2=`0.3650`
- `cumul_pressure_forward_rate`: vif=`1.5174`, r2=`0.3410`
- `player_share_team_shots_on_target`: vif=`1.5013`, r2=`0.3339`
- `cumul_bottom_sprint_share`: vif=`1.3362`, r2=`0.2516`

## Candidate features to review for elimination

- Heuristic only: high correlation is not enough by itself. The table below marks the weaker SHAP side of each highly correlated pair.

- Drop candidate `cumul_top_run_share` over keep `cumul_top_hsr_share`: abs_corr=`0.9737`, drop_shap=`0.185107`, keep_shap=`0.254877`
- Drop candidate `player_z_team_top_distance_share` over keep `top_distance_share`: abs_corr=`0.9152`, drop_shap=`0.149469`, keep_shap=`0.164474`