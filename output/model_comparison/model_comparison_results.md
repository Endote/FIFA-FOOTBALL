# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| decision_tree          | 0.1306                | 0.700417  | 0.24697         | 0.092806               | 0.721512   | 0.234317         |
| catboost               | 0.130575              | 0.676313  | 0.065802        | 0.184349               | 0.67544    | 0.044831         |
| hist_gradient_boosting | 0.100765              | 0.635545  | 0.113325        | 0.134784               | 0.624572   | 0.110714         |
| xgboost                | 0.096899              | 0.627325  | 0.066927        | 0.085219               | 0.619319   | 0.046445         |
| logistic_regression    | 0.093213              | 0.578597  | 0.249309        | 0.142601               | 0.616522   | 0.245426         |
