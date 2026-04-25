# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| catboost   | 0.134253              | 0.679438  | 0.066192        | 0.13955                | 0.655344   | 0.04627          |
| xgboost    | 0.118823              | 0.614603  | 0.065303        | 0.108735               | 0.613496   | 0.047585         |
