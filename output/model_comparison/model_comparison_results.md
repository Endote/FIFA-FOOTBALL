# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| catboost               | 0.114265              | 0.638189  | 0.07277         | 0.090287               | 0.637548   | 0.078564         |
| logistic_regression    | 0.095156              | 0.652827  | 0.215057        | 0.087795               | 0.611814   | 0.226503         |
| decision_tree          | 0.088416              | 0.655196  | 0.224801        | 0.082229               | 0.621153   | 0.228459         |
| hist_gradient_boosting | 0.086218              | 0.664978  | 0.117621        | 0.070232               | 0.532375   | 0.132261         |
| xgboost                | 0.08572               | 0.663054  | 0.051043        | 0.095485               | 0.664304   | 0.056287         |
