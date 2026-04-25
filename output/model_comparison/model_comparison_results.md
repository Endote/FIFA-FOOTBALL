# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| catboost               | 0.189321              | 0.717676  | 0.060897        | 0.210111               | 0.778717   | 0.042156         |
| xgboost                | 0.180043              | 0.69573   | 0.06052         | 0.146688               | 0.703528   | 0.04223          |
| logistic_regression    | 0.157089              | 0.639935  | 0.219167        | 0.244915               | 0.718828   | 0.20967          |
| hist_gradient_boosting | 0.118654              | 0.601213  | 0.113529        | 0.114924               | 0.667104   | 0.100149         |
| decision_tree          | 0.104364              | 0.639786  | 0.233849        | 0.086134               | 0.67975    | 0.220671         |
