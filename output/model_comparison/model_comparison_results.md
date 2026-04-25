# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| xgboost                | 0.213183              | 0.745135  | 0.059724        | 0.127307               | 0.762833   | 0.041361         |
| catboost               | 0.190734              | 0.753769  | 0.078753        | 0.206373               | 0.808772   | 0.072028         |
| hist_gradient_boosting | 0.163194              | 0.679887  | 0.166538        | 0.14969                | 0.653965   | 0.169034         |
| decision_tree          | 0.148691              | 0.739899  | 0.227422        | 0.09256                | 0.739498   | 0.229289         |
| logistic_regression    | 0.139498              | 0.644366  | 0.255506        | 0.201955               | 0.640739   | 0.230037         |
