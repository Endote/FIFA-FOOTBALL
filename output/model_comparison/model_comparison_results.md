# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| catboost               | 0.173014              | 0.728201  | 0.061882        | 0.174023               | 0.676568   | 0.043385         |
| logistic_regression    | 0.10087               | 0.597631  | 0.256366        | 0.133616               | 0.650165   | 0.252233         |
| hist_gradient_boosting | 0.098795              | 0.621472  | 0.112942        | 0.113627               | 0.62729    | 0.111604         |
| decision_tree          | 0.097437              | 0.638907  | 0.233613        | 0.067537               | 0.623393   | 0.245989         |
| xgboost                | 0.091431              | 0.607917  | 0.067286        | 0.088519               | 0.633094   | 0.046416         |
