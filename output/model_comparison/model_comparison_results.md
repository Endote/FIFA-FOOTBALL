# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name    | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| xgboost       | 0.149037              | 0.755682  | 0.060238        | 0.125633               | 0.807158   | 0.041915         |
| catboost      | 0.146252              | 0.752562  | 0.082908        | 0.177054               | 0.842096   | 0.071687         |
| decision_tree | 0.106128              | 0.65042   | 0.238307        | 0.07639                | 0.635456   | 0.274147         |
