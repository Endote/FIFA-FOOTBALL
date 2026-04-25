# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| xgboost                | 0.170754              | 0.70072   | 0.060681        | 0.090458               | 0.699943   | 0.042725         |
| logistic_regression    | 0.158594              | 0.676322  | 0.222458        | 0.078576               | 0.570944   | 0.187186         |
| catboost               | 0.1419                | 0.714201  | 0.077462        | 0.096718               | 0.715546   | 0.067128         |
| decision_tree          | 0.140744              | 0.734384  | 0.214594        | 0.0947                 | 0.68355    | 0.205399         |
| hist_gradient_boosting | 0.100314              | 0.621212  | 0.107723        | 0.060737               | 0.599039   | 0.08184          |
