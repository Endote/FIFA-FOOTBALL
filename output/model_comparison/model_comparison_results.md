# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| decision_tree          | 0.130119              | 0.696325  | 0.246393        | 0.093573               | 0.72979    | 0.232146         |
| catboost               | 0.128107              | 0.677801  | 0.06564         | 0.136666               | 0.655743   | 0.044682         |
| hist_gradient_boosting | 0.104551              | 0.633537  | 0.112798        | 0.146505               | 0.638331   | 0.109814         |
| xgboost                | 0.096751              | 0.62301   | 0.06699         | 0.086145               | 0.614981   | 0.04649          |
| logistic_regression    | 0.094882              | 0.583879  | 0.247633        | 0.139787               | 0.619377   | 0.244975         |
