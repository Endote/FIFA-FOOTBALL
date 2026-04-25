# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| logistic_regression    | 0.185872              | 0.682895  | 0.233796        | 0.178223               | 0.659725   | 0.240545         |
| xgboost                | 0.167034              | 0.740605  | 0.060068        | 0.146291               | 0.72539    | 0.041907         |
| decision_tree          | 0.140981              | 0.698177  | 0.235064        | 0.084529               | 0.66146    | 0.238789         |
| hist_gradient_boosting | 0.13843               | 0.680741  | 0.124291        | 0.135523               | 0.604643   | 0.136577         |
| catboost               | 0.123262              | 0.7028    | 0.061423        | 0.165312               | 0.796973   | 0.042155         |
