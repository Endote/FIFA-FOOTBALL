# Feature Diagnostics

- Fit splits: `train,val,test`
- Explain split: `all`
- Explain AP: `0.219585`
- Explain AUROC: `0.803985`
- Explain Brier: `0.049677`
- Explain balanced accuracy at threshold 0.5: `0.500000`

## Top SHAP features

- `cumul_received_unsucc`: `0.279028`
- `position`: `0.194307`
- `cumul_in_game_time`: `0.110005`
- `cumul_pressured_success_rate`: `0.096971`
- `formation`: `0.085917`
- `cumul_pressured_unsucc`: `0.040987`
- `cumul_pressure_success_rate`: `0.039443`
- `is_home`: `0.030019`
- `cumul_pressures_applied`: `0.025795`
- `cumul_peak_speed`: `0.023424`
- `cumul_shots_under_press`: `0.023311`
- `cumul_pressures_won`: `0.017617`
- `cumul_sprints`: `0.017081`
- `cumul_shots_top_third`: `0.015540`
- `cumul_shots_on_target`: `0.014239`

## Strongest numeric/boolean correlations

- `cumul_shots` vs `cumul_shots_top_third`: corr=`0.9926`, abs=`0.9926`
- `cumul_pressures_applied` vs `cumul_pressures_lost`: corr=`0.9539`, abs=`0.9539`
- `cumul_times_pressured` vs `cumul_pressured_succ`: corr=`0.9457`, abs=`0.9457`
- `cumul_hsr` vs `cumul_distance`: corr=`0.9446`, abs=`0.9446`
- `cumul_pressures_applied` vs `cumul_pressures_won`: corr=`0.9029`, abs=`0.9029`
- `cumul_times_pressured` vs `cumul_pressured_unsucc`: corr=`0.8967`, abs=`0.8967`
- `cumul_shots` vs `cumul_shots_under_press`: corr=`0.8961`, abs=`0.8961`
- `cumul_shots_under_press` vs `cumul_shots_top_third`: corr=`0.8906`, abs=`0.8906`
- `cumul_sprints` vs `cumul_distance`: corr=`0.8455`, abs=`0.8455`
- `cumul_sprints` vs `cumul_hsr`: corr=`0.8182`, abs=`0.8182`
- `cumul_pressures_won` vs `cumul_pressures_lost`: corr=`0.7472`, abs=`0.7472`
- `cumul_pressured_succ` vs `cumul_pressured_unsucc`: corr=`0.7197`, abs=`0.7197`
- `cumul_pressures_won` vs `cumul_pressure_success_rate`: corr=`0.6783`, abs=`0.6783`
- `cumul_sprints` vs `cumul_peak_speed`: corr=`0.6698`, abs=`0.6698`
- `cumul_hsr` vs `cumul_pressures_applied`: corr=`0.6557`, abs=`0.6557`

## Highest VIF features

- `cumul_times_pressured`: vif=`inf`, r2=`1.0000`
- `cumul_pressured_succ`: vif=`inf`, r2=`1.0000`
- `cumul_pressured_unsucc`: vif=`inf`, r2=`1.0000`
- `cumul_pressures_applied`: vif=`inf`, r2=`1.0000`
- `cumul_pressures_won`: vif=`inf`, r2=`1.0000`
- `cumul_pressures_lost`: vif=`inf`, r2=`1.0000`
- `cumul_shots`: vif=`61.9618`, r2=`0.9839`
- `cumul_shots_top_third`: vif=`57.2836`, r2=`0.9825`
- `cumul_hsr`: vif=`10.1173`, r2=`0.9012`
- `cumul_distance`: vif=`9.9260`, r2=`0.8993`
- `cumul_peak_speed`: vif=`8.1601`, r2=`0.8775`
- `cumul_mean_max_speed`: vif=`6.9546`, r2=`0.8562`
- `cumul_shots_under_press`: vif=`5.5973`, r2=`0.8213`
- `cumul_sprints`: vif=`3.5187`, r2=`0.7158`
- `cumul_in_game_time`: vif=`2.5461`, r2=`0.6072`

## Candidate features to review for elimination

- Heuristic only: high correlation is not enough by itself. The table below marks the weaker SHAP side of each highly correlated pair.

- Drop candidate `cumul_shots` over keep `cumul_shots_top_third`: abs_corr=`0.9926`, drop_shap=`0.011893`, keep_shap=`0.015540`
- Drop candidate `cumul_pressures_lost` over keep `cumul_pressures_applied`: abs_corr=`0.9539`, drop_shap=`0.013602`, keep_shap=`0.025795`
- Drop candidate `cumul_pressured_succ` over keep `cumul_times_pressured`: abs_corr=`0.9457`, drop_shap=`0.006907`, keep_shap=`0.007128`
- Drop candidate `cumul_hsr` over keep `cumul_distance`: abs_corr=`0.9446`, drop_shap=`0.008640`, keep_shap=`0.009359`
- Drop candidate `cumul_pressures_won` over keep `cumul_pressures_applied`: abs_corr=`0.9029`, drop_shap=`0.017617`, keep_shap=`0.025795`
- Drop candidate `cumul_times_pressured` over keep `cumul_pressured_unsucc`: abs_corr=`0.8967`, drop_shap=`0.007128`, keep_shap=`0.040987`
- Drop candidate `cumul_shots` over keep `cumul_shots_under_press`: abs_corr=`0.8961`, drop_shap=`0.011893`, keep_shap=`0.023311`
- Drop candidate `cumul_shots_top_third` over keep `cumul_shots_under_press`: abs_corr=`0.8906`, drop_shap=`0.015540`, keep_shap=`0.023311`