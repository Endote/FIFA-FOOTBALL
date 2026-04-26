# Full Feature Signal Audit

- Rows after quality filters: `3208`
- Rows removed by quality filters: `278`
- Predictors audited: `98`

## Top 15 By SHAP

- `cumul_received_unsucc`: shap=`0.434139`, spearman=`-0.140908`
- `cumul_pressured_success_rate`: shap=`0.273946`, spearman=`-0.079502`
- `position`: shap=`0.206140`, cramers_v=`0.166420`
- `formation`: shap=`0.148576`, cramers_v=`0.078174`
- `cumul_peak_speed`: shap=`0.141151`, spearman=`0.025962`
- `cumul_top_hsr_share`: shap=`0.132577`, spearman=`0.129127`
- `cumul_pressure_success_rate`: shap=`0.106168`, spearman=`0.015219`
- `last15_mean_max_speed`: shap=`0.096653`, spearman=`0.025812`
- `is_home`: shap=`0.089931`, spearman=`-0.029951`
- `last_15_pressured_success_rate`: shap=`0.087880`, spearman=`-0.007285`
- `cumul_bottom_sprint_share`: shap=`0.085661`, spearman=`-0.024360`
- `cumul_mean_max_speed`: shap=`0.081602`, spearman=`0.007734`
- `distance_per_run`: shap=`0.079209`, spearman=`0.041007`
- `cumul_pressured_succ`: shap=`0.077533`, spearman=`-0.047071`
- `last_15_received_unsucc`: shap=`0.075201`, spearman=`-0.095253`

## Top 15 By Target Association

- `position`: cramers_v=`0.166420`, shap=`0.206140`
- `cumul_received_unsucc`: spearman=`-0.140908`, shap=`0.434139`
- `cumul_top_hsr_share`: spearman=`0.129127`, shap=`0.132577`
- `cumul_top_run_share`: spearman=`0.127709`, shap=`0.042676`
- `top_distance_share`: spearman=`0.121415`, shap=`0.043522`
- `last15_top_run_share`: spearman=`0.111803`, shap=`0.038055`
- `share_of_possessions_with_top_run`: spearman=`0.101520`, shap=`0.054902`
- `top_runs_per_possession`: spearman=`0.100923`, shap=`0.019436`
- `top_hsr_distance`: spearman=`0.100665`, shap=`0.040215`
- `last15_top_hsr_count`: spearman=`0.099422`, shap=`0.003575`
- `last_15_received_unsucc`: spearman=`-0.095253`, shap=`0.075201`
- `cumul_in_game_time`: spearman=`-0.092350`, shap=`0.042225`
- `last_15_pressures_applied`: spearman=`0.090243`, shap=`0.054232`
- `last_15_pressured_unsucc`: spearman=`0.088541`, shap=`0.026090`
- `possessions_with_2plus_top_runs`: spearman=`0.085967`, shap=`0.009803`

## Bottom 15 By SHAP

- `last15_bottom_sprint_count`: shap=`0.003875`, spearman=`0.006448`
- `last15_top_sprint_count`: shap=`0.003840`, spearman=`0.075209`
- `last15_top_hsr_count`: shap=`0.003575`, spearman=`0.099422`
- `last15_shots_under_press`: shap=`0.003469`, spearman=`0.064436`
- `last_15_shots_under_pressure_rate`: shap=`0.003430`, spearman=`0.065006`
- `cumul_shots_total`: shap=`0.003001`, spearman=`0.067083`
- `last15_shots_on_target`: shap=`0.002726`, spearman=`0.050934`
- `share_of_possessions_with_sprint`: shap=`0.002422`, spearman=`0.042858`
- `last15_shots_top_third`: shap=`0.001981`, spearman=`0.068797`
- `cumul_top_sprint_count`: shap=`0.001745`, spearman=`0.046843`
- `cumul_shots`: shap=`0.001623`, spearman=`0.065974`
- `last_15_shots_special`: shap=`0.000672`, spearman=`0.042104`
- `last_15_shots_blocked`: shap=`0.000241`, spearman=`0.011148`
- `last_15_shots_total`: shap=`0.000189`, spearman=`0.069666`
- `last_15_shots_set_play`: shap=`0.000000`, spearman=`0.024056`