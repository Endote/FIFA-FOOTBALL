# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name             | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ---------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| logistic_regression    | 0.225756              | 0.69359   | 0.20094         | 0.075557               | 0.504609   | 0.202881         |
| xgboost                | 0.203369              | 0.743687  | 0.059967        | 0.092246               | 0.729828   | 0.042802         |
| catboost               | 0.169366              | 0.742387  | 0.075604        | 0.095927               | 0.743598   | 0.06804          |
| hist_gradient_boosting | 0.145036              | 0.688354  | 0.094521        | 0.066035               | 0.624559   | 0.088193         |
| decision_tree          | 0.115042              | 0.68581   | 0.220956        | 0.058059               | 0.571099   | 0.236232         |
