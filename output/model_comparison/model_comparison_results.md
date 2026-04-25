# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| xgboost                | 0.201005              | 0.73752   | 0.059946        | 0.144713               | 0.75274    | 0.041878         |
| logistic_regression    | 0.185701              | 0.669636  | 0.235338        | 0.206351               | 0.663165   | 0.243952         |
| catboost               | 0.178467              | 0.753032  | 0.060517        | 0.15956                | 0.820222   | 0.041869         |
| hist_gradient_boosting | 0.147416              | 0.663555  | 0.158878        | 0.150155               | 0.678066   | 0.175221         |
| decision_tree          | 0.132787              | 0.715054  | 0.227191        | 0.087042               | 0.714861   | 0.22712          |
