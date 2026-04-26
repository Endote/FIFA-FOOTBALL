# Feature Pruning

- Start timestamp: `20260426_023936`
- Max accepted drops: `5`
- Target PRAUC stop: `None`
- Candidate top-K per cycle: `12`
- Inner folds: `4`
- Accepted drops: `4`
- Final feature count: `57`

## Initial CV and holdout

- Mean CV PRAUC: `0.229351`
- Mean CV balanced accuracy: `0.513204`
- Mean CV AUROC: `0.699705`
- Mean CV Brier: `0.110825`
- Test PRAUC: `0.137071`
- Test balanced accuracy: `0.500000`
- Test AUROC: `0.764083`
- Test Brier: `0.062895`

## Final CV and holdout

- Mean CV PRAUC: `0.231791`
- Mean CV balanced accuracy: `0.504198`
- Mean CV AUROC: `0.705419`
- Mean CV Brier: `0.076667`
- Test PRAUC: `0.110918`
- Test balanced accuracy: `0.500000`
- Test AUROC: `0.676454`
- Test Brier: `0.237525`

## Accepted drops

- `last_15_times_pressured` accepted in cycle `1`: delta PRAUC=`-0.016620`, delta balanced accuracy=`-0.000610`, delta AUROC=`0.005345`, delta Brier=`-0.027065`
- `checkpoint_period` accepted in cycle `2`: delta PRAUC=`0.000412`, delta balanced accuracy=`-0.012594`, delta AUROC=`0.000397`, delta Brier=`0.047797`
- `last_15_pressures_lost` accepted in cycle `3`: delta PRAUC=`0.000504`, delta balanced accuracy=`0.000000`, delta AUROC=`0.004416`, delta Brier=`0.016057`
- `cumul_pressured_unsucc` accepted in cycle `4`: delta PRAUC=`0.018144`, delta balanced accuracy=`0.004198`, delta AUROC=`-0.004444`, delta Brier=`-0.070947`