# Model Comparison

Primary ranking metric: Average Precision / PR-AUC.
Secondary ranking metric: ROC-AUC.
Calibration metric: Brier score (lower is better).

| model_name    | val_average_precision | val_auroc | val_brier_score | test_average_precision | test_auroc | test_brier_score |
| ------------- | --------------------- | --------- | --------------- | ---------------------- | ---------- | ---------------- |
| xgboost       | 0.195668              | 0.779078  | 0.059415        | 0.132145               | 0.797826   | 0.041704         |
| catboost      | 0.178337              | 0.759544  | 0.08224         | 0.210278               | 0.834699   | 0.07551          |
| decision_tree | 0.133352              | 0.703524  | 0.242447        | 0.097613               | 0.67745    | 0.273284         |
