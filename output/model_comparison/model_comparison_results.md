# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| catboost               | 0.17032               | 0.737799  | 0.060297        | 0.199348               | 0.802638   | 0.041519         |
| logistic_regression    | 0.170302              | 0.672035  | 0.239774        | 0.164111               | 0.669388   | 0.244696         |
| xgboost                | 0.159839              | 0.733782  | 0.060304        | 0.147903               | 0.722254   | 0.041963         |
| decision_tree          | 0.141033              | 0.698278  | 0.234093        | 0.084609               | 0.662509   | 0.234958         |
| hist_gradient_boosting | 0.119179              | 0.666456  | 0.123023        | 0.143625               | 0.607045   | 0.135811         |
