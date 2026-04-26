# Model Comparison

Primary ranking metric: PR-AUC.
Co-primary threshold metric: Balanced Accuracy, with threshold selected on each training fold only.
Secondary diagnostics: ROC-AUC and Brier score (lower is better).

| model_name    | cv_pr_auc_mean | cv_pr_auc_std | cv_balanced_accuracy_mean | cv_balanced_accuracy_std | cv_auroc_mean | cv_auroc_std | cv_brier_score_mean | cv_brier_score_std | val_average_precision | val_balanced_accuracy | val_auroc | val_brier_score | test_average_precision | test_balanced_accuracy | test_auroc | test_brier_score |
| ------------- | -------------- | ------------- | ------------------------- | ------------------------ | ------------- | ------------ | ------------------- | ------------------ | --------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------------------- | ---------- | ---------------- |
| catboost      | 0.139516       | 0.046902      | 0.657353                  | 0.04064                  | 0.700781      | 0.036997     | 0.124808            | 0.069199           | 0.100801              | 0.696005              | 0.704365  | 0.18126         | 0.188829               | 0.657959               | 0.730867   | 0.180899         |
| xgboost       | 0.118738       | 0.034074      | 0.6142                    | 0.044074                 | 0.683243      | 0.02599      | 0.053429            | 0.006815           | 0.097291              | 0.66124               | 0.678899  | 0.055921        | 0.135229               | 0.656606               | 0.676238   | 0.052648         |
| decision_tree | 0.084044       | 0.01022       | 0.588751                  | 0.037865                 | 0.632689      | 0.030042     | 0.220699            | 0.013488           | 0.075644              | 0.557389              | 0.593919  | 0.22031         | 0.101829               | 0.69645                | 0.697707   | 0.222319         |
