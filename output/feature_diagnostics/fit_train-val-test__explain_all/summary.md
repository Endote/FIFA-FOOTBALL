# Feature Diagnostics

- Fit splits: `train,val,test`
- Explain split: `all`
- Explain AP: `0.253539`
- Explain AUROC: `0.828772`
- Explain Brier: `0.061006`
- Explain balanced accuracy at threshold 0.5: `0.533247`

## Top SHAP features

- `cumul_received_unsucc`: `0.393652`
- `position`: `0.207554`
- `cumul_top_hsr_share`: `0.123238`
- `cumul_in_game_time`: `0.095260`
- `cumul_bottom_sprint_share`: `0.085136`
- `cumul_top_run_share`: `0.070623`
- `last15_top_run_share`: `0.060968`
- `formation`: `0.057443`
- `formation_attackers`: `0.041307`
- `cumul_bottom_run_share`: `0.036223`
- `formation_striker`: `0.032391`
- `last_15_received_unsucc`: `0.029310`
- `last15_top_sprint_share`: `0.028864`
- `formation_midfielders`: `0.021982`
- `last15_middle_hsr_count`: `0.021216`

## Strongest numeric/boolean correlations

- `last_15_shots_under_pressure` vs `last_15_shots_under_pressure_rate`: corr=`0.9990`, abs=`0.9990`
- `last_15_shots_total` vs `cumul_shots_total`: corr=`0.9975`, abs=`0.9975`
- `cumul_shots_under_pressure` vs `cumul_shots_under_pressure_rate`: corr=`0.9966`, abs=`0.9966`
- `cumul_shots` vs `cumul_shots_top_third`: corr=`0.9928`, abs=`0.9928`
- `last15_shots` vs `last15_shots_top_third`: corr=`0.9817`, abs=`0.9817`
- `last_15_shots_under_pressure_rate` vs `cumul_shots_under_pressure_rate`: corr=`0.9787`, abs=`0.9787`
- `last_15_shots_under_pressure` vs `cumul_shots_under_pressure_rate`: corr=`0.9777`, abs=`0.9777`
- `cumul_top_run_share` vs `cumul_top_hsr_share`: corr=`0.9757`, abs=`0.9757`
- `last_15_shots_under_pressure` vs `cumul_shots_under_pressure`: corr=`0.9728`, abs=`0.9728`
- `cumul_shots_under_pressure` vs `last_15_shots_under_pressure_rate`: corr=`0.9720`, abs=`0.9720`
- `last15_top_sprint_count` vs `last15_top_sprint_share`: corr=`0.9711`, abs=`0.9711`
- `last15_shots` vs `last_15_shots_total`: corr=`0.9603`, abs=`0.9603`
- `last15_shots_top_third` vs `last_15_shots_total`: corr=`0.9589`, abs=`0.9589`
- `last15_shots_under_press` vs `last_15_shots_under_pressure`: corr=`0.9587`, abs=`0.9587`
- `last15_shots` vs `cumul_shots_total`: corr=`0.9577`, abs=`0.9577`

## Highest VIF features

- `formation_defenders`: vif=`inf`, r2=`1.0000`
- `formation_midfielders`: vif=`inf`, r2=`1.0000`
- `formation_attackers`: vif=`inf`, r2=`1.0000`
- `formation_striker`: vif=`inf`, r2=`1.0000`
- `cumul_sprints`: vif=`518.3316`, r2=`0.9981`
- `last15_shots`: vif=`155.2585`, r2=`0.9936`
- `cumul_shots`: vif=`144.0140`, r2=`0.9931`
- `cumul_middle_sprint_count`: vif=`136.9525`, r2=`0.9927`
- `last_15_shots_total`: vif=`135.5705`, r2=`0.9926`
- `cumul_shots_top_third`: vif=`135.4188`, r2=`0.9926`
- `last_15_shots_under_pressure`: vif=`127.8954`, r2=`0.9922`
- `cumul_top_sprint_count`: vif=`119.1844`, r2=`0.9916`
- `last15_shots_under_press`: vif=`93.1365`, r2=`0.9893`
- `cumul_bottom_sprint_count`: vif=`80.3733`, r2=`0.9876`
- `cumul_top_run_share`: vif=`72.6653`, r2=`0.9862`

## Candidate features to review for elimination

- Heuristic only: high correlation is not enough by itself. The table below marks the weaker SHAP side of each highly correlated pair.

- Drop candidate `last_15_shots_under_pressure` over keep `last_15_shots_under_pressure_rate`: abs_corr=`0.9990`, drop_shap=`0.003448`, keep_shap=`0.013978`
- Drop candidate `cumul_shots_total` over keep `last_15_shots_total`: abs_corr=`0.9975`, drop_shap=`0.004817`, keep_shap=`0.006171`
- Drop candidate `cumul_shots_under_pressure` over keep `cumul_shots_under_pressure_rate`: abs_corr=`0.9966`, drop_shap=`0.005931`, keep_shap=`0.012806`
- Drop candidate `cumul_shots` over keep `cumul_shots_top_third`: abs_corr=`0.9928`, drop_shap=`0.006226`, keep_shap=`0.007586`
- Drop candidate `last15_shots_top_third` over keep `last15_shots`: abs_corr=`0.9817`, drop_shap=`0.004082`, keep_shap=`0.005082`
- Drop candidate `cumul_shots_under_pressure_rate` over keep `last_15_shots_under_pressure_rate`: abs_corr=`0.9787`, drop_shap=`0.012806`, keep_shap=`0.013978`
- Drop candidate `last_15_shots_under_pressure` over keep `cumul_shots_under_pressure_rate`: abs_corr=`0.9777`, drop_shap=`0.003448`, keep_shap=`0.012806`
- Drop candidate `cumul_top_run_share` over keep `cumul_top_hsr_share`: abs_corr=`0.9757`, drop_shap=`0.070623`, keep_shap=`0.123238`
- Drop candidate `last_15_shots_under_pressure` over keep `cumul_shots_under_pressure`: abs_corr=`0.9728`, drop_shap=`0.003448`, keep_shap=`0.005931`
- Drop candidate `cumul_shots_under_pressure` over keep `last_15_shots_under_pressure_rate`: abs_corr=`0.9720`, drop_shap=`0.005931`, keep_shap=`0.013978`
- Drop candidate `last15_top_sprint_count` over keep `last15_top_sprint_share`: abs_corr=`0.9711`, drop_shap=`0.004637`, keep_shap=`0.028864`
- Drop candidate `last15_shots` over keep `last_15_shots_total`: abs_corr=`0.9603`, drop_shap=`0.005082`, keep_shap=`0.006171`
- Drop candidate `last15_shots_top_third` over keep `last_15_shots_total`: abs_corr=`0.9589`, drop_shap=`0.004082`, keep_shap=`0.006171`
- Drop candidate `last_15_shots_under_pressure` over keep `last15_shots_under_press`: abs_corr=`0.9587`, drop_shap=`0.003448`, keep_shap=`0.004487`
- Drop candidate `cumul_shots_total` over keep `last15_shots`: abs_corr=`0.9577`, drop_shap=`0.004817`, keep_shap=`0.005082`
- Drop candidate `last15_shots_under_press` over keep `last_15_shots_under_pressure_rate`: abs_corr=`0.9574`, drop_shap=`0.004487`, keep_shap=`0.013978`
- Drop candidate `last15_shots_top_third` over keep `cumul_shots_total`: abs_corr=`0.9561`, drop_shap=`0.004082`, keep_shap=`0.004817`
- Drop candidate `last15_shots_under_press` over keep `cumul_shots_under_pressure_rate`: abs_corr=`0.9370`, drop_shap=`0.004487`, keep_shap=`0.012806`
- Drop candidate `last15_shots_under_press` over keep `cumul_shots_under_pressure`: abs_corr=`0.9319`, drop_shap=`0.004487`, keep_shap=`0.005931`
- Drop candidate `cumul_shots_total` over keep `cumul_shots_under_pressure`: abs_corr=`0.9105`, drop_shap=`0.004817`, keep_shap=`0.005931`