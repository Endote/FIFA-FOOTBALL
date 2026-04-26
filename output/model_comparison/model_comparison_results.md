# Model Comparison

Primary ranking metric: PR-AUC.
Co-primary threshold metric: Balanced Accuracy, with threshold selected on each training fold only.
Secondary diagnostics: ROC-AUC and Brier score (lower is better).

| model_name    | cv_pr_auc_mean | cv_pr_auc_std | cv_balanced_accuracy_mean | cv_balanced_accuracy_std | cv_auroc_mean | cv_auroc_std | cv_brier_score_mean | cv_brier_score_std | val_average_precision | val_balanced_accuracy | val_auroc | val_brier_score | test_average_precision | test_balanced_accuracy | test_auroc | test_brier_score |
| ------------- | -------------- | ------------- | ------------------------- | ------------------------ | ------------- | ------------ | ------------------- | ------------------ | --------------------- | --------------------- | --------- | --------------- | ---------------------- | ---------------------- | ---------- | ---------------- |
| catboost      | 0.149818       | 0.038709      | 0.625805                  | 0.049751                 | 0.684739      | 0.053396     | 0.164862            | 0.071255           | 0.142915              | 0.695039              | 0.740205  | 0.210121        | 0.237418               | 0.719449               | 0.759711   | 0.209576         |
| xgboost       | 0.124073       | 0.035904      | 0.624733                  | 0.033613                 | 0.695911      | 0.032558     | 0.053401            | 0.00681            | 0.109175              | 0.586773              | 0.681961  | 0.05593         | 0.121701               | 0.669619               | 0.684178   | 0.052594         |
| decision_tree | 0.09124        | 0.014665      | 0.632614                  | 0.04596                  | 0.656886      | 0.036627     | 0.218728            | 0.016491           | 0.07491               | 0.561445              | 0.58727   | 0.219295        | 0.102712               | 0.712233               | 0.704696   | 0.222853         |
