# Full Feature Signal Audit

- Rows after quality filters: `3208`
- Rows removed by quality filters: `278`
- Predictors audited: `126`

## Top 15 By SHAP

- `cumul_received_unsucc`: shap=`0.345666`, spearman=`-0.140908`
- `formation`: shap=`0.172129`, cramers_v=`0.078174`
- `cumul_pass_middle_count`: shap=`0.162069`, spearman=`-0.077056`
- `position`: shap=`0.146008`, cramers_v=`0.166420`
- `cumul_pass_middle_accuracy_rate`: shap=`0.114653`, spearman=`0.054642`
- `cumul_pressure_forward_rate`: shap=`0.110173`, spearman=`-0.082927`
- `team_minus_opponent_shot_load`: shap=`0.106969`, spearman=`0.022891`
- `cumul_pressured_success_rate`: shap=`0.106871`, spearman=`-0.079502`
- `last_15_received_unsucc`: shap=`0.105558`, spearman=`-0.095253`
- `mean_abs_pass_angle_under_pressure`: shap=`0.105016`, spearman=`-0.053446`
- `cumul_pressure_turnover_rate`: shap=`0.103322`, spearman=`0.091770`
- `cumul_pass_top_accuracy_rate`: shap=`0.101951`, spearman=`-0.002935`
- `distance_per_run`: shap=`0.096557`, spearman=`0.041007`
- `last15_mean_max_speed`: shap=`0.095475`, spearman=`0.025812`
- `cumul_peak_speed`: shap=`0.085090`, spearman=`0.025962`

## Top 15 By Target Association

- `position`: cramers_v=`0.166420`, shap=`0.146008`
- `cumul_received_unsucc`: spearman=`-0.140908`, shap=`0.345666`
- `cumul_top_hsr_share`: spearman=`0.129127`, shap=`0.075210`
- `cumul_top_run_share`: spearman=`0.127709`, shap=`0.037377`
- `player_z_team_top_distance_share`: spearman=`0.125877`, shap=`0.082573`
- `top_distance_share`: spearman=`0.121415`, shap=`0.035120`
- `player_share_team_top_distance`: spearman=`0.117066`, shap=`0.047315`
- `player_rank_team_top_distance_share`: spearman=`-0.116209`, shap=`0.012813`
- `last15_top_run_share`: spearman=`0.111803`, shap=`0.051153`
- `share_of_possessions_with_top_run`: spearman=`0.101520`, shap=`0.048976`
- `top_runs_per_possession`: spearman=`0.100923`, shap=`0.025301`
- `top_hsr_distance`: spearman=`0.100665`, shap=`0.040273`
- `last15_top_hsr_count`: spearman=`0.099422`, shap=`0.004583`
- `last_15_received_unsucc`: spearman=`-0.095253`, shap=`0.105558`
- `cumul_in_game_time`: spearman=`-0.092350`, shap=`0.035318`

## Bottom 15 By SHAP

- `cumul_shots_under_pressure`: shap=`0.002697`, spearman=`0.067573`
- `cumul_shots`: shap=`0.002694`, spearman=`0.065974`
- `cumul_shots_top_third`: shap=`0.002559`, spearman=`0.067426`
- `last_15_times_pressured`: shap=`0.002493`, spearman=`0.067552`
- `last15_middle_sprint_count`: shap=`0.002489`, spearman=`0.002134`
- `last_15_shots_under_pressure_rate`: shap=`0.002317`, spearman=`0.065006`
- `last15_pressure_events`: shap=`0.001376`, spearman=`0.047498`
- `last15_shots_top_third`: shap=`0.000817`, spearman=`0.068797`
- `last_15_shots_special`: shap=`0.000433`, spearman=`0.042104`
- `last_15_shots_blocked`: shap=`0.000258`, spearman=`0.011148`
- `share_of_possessions_with_sprint`: shap=`0.000139`, spearman=`0.042858`
- `last_15_shots_total`: shap=`0.000133`, spearman=`0.069666`
- `last15_shots_on_target`: shap=`0.000065`, spearman=`0.050934`
- `last15_bottom_sprint_count`: shap=`0.000046`, spearman=`0.006448`
- `cumul_top_sprint_count`: shap=`0.000004`, spearman=`0.046843`