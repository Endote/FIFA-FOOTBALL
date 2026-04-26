from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_DIR = Path("data/baseline_modeling")
OUTPUT_DIR = Path("output/model_comparison")

TRAIN_PATH = DATA_DIR / "baseline_train_model_ready.csv"
VAL_PATH = DATA_DIR / "baseline_val_model_ready.csv"
TEST_PATH = DATA_DIR / "baseline_test_model_ready.csv"
CV_DATA_PATH = DATA_DIR / "baseline_all_model_cv_ready.csv"
CV_ASSIGNMENTS_PATH = DATA_DIR / "baseline_cv_fixture_assignments.csv"
SINGLE_RUN_DIR = OUTPUT_DIR / "_single_runs"

TARGET_COL = "scored_after"
SPLIT_COL = "split"
RANDOM_STATE = 42

# Models to train in this run
MODELS_TO_RUN = [
    # "logistic_regression",
    "decision_tree",
    # "hist_gradient_boosting",
    "xgboost",
    "catboost",
]

# Logistic Regression hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 5000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

# Decision Tree hyperparameters
DECISION_TREE_PARAMS = {
    "max_depth": 4,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# HistGradientBoosting hyperparameters
HIST_GRADIENT_BOOSTING_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "max_iter": 300,
    "min_samples_leaf": 20,
    "l2_regularization": 5.0,
    "random_state": RANDOM_STATE,
}

# TabPFN hyperparameters
TABPFN_PARAMS = {
    "random_state": RANDOM_STATE,
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 2,
    "learning_rate": 0.002,
    "subsample": 0.75,
    "colsample_bytree": 0.80,
    "min_child_weight": 8,
    "reg_lambda": 10.0,
    "reg_alpha": 0.8,
    "gamma": 0.1,
    "max_delta_step": 1,
    "objective": "binary:logistic",
    "early_stopping_rounds": 150,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

# CatBoost hyperparameters
CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC:type=Classic;use_weights=false",
    "custom_metric": [
        "PRAUC:type=Classic;use_weights=false",
        "BalancedAccuracy",
        "AUC:type=Classic;use_weights=false",
        "BrierScore:use_weights=false",
        "Logloss",
    ],
    "iterations": 5000,
    "learning_rate": 0.01,
    "depth": 4,
    "l2_leaf_reg": 250,
    "random_strength": 1,
    "early_stopping_rounds": 150,
    "random_seed": RANDOM_STATE,
    "verbose": 100,
    "boosting_type": "Ordered",
    "bootstrap_type": "MVS",
    "subsample": 0.8,
    "leaf_estimation_method": "Newton",
    "leaf_estimation_backtracking": "AnyImprovement",
    "auto_class_weights": "SqrtBalanced",
}

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str((OUTPUT_DIR / "tabpfn_cache").resolve()))
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")


@dataclass
class ModelResult:
    model_name: str
    cv_pr_auc_mean: float = float("nan")
    cv_pr_auc_std: float = float("nan")
    cv_balanced_accuracy_mean: float = float("nan")
    cv_balanced_accuracy_std: float = float("nan")
    cv_auroc_mean: float = float("nan")
    cv_auroc_std: float = float("nan")
    cv_brier_score_mean: float = float("nan")
    cv_brier_score_std: float = float("nan")
    val_average_precision: float = float("nan")
    val_balanced_accuracy: float = float("nan")
    val_auroc: float = float("nan")
    val_brier_score: float = float("nan")
    test_average_precision: float = float("nan")
    test_balanced_accuracy: float = float("nan")
    test_auroc: float = float("nan")
    test_brier_score: float = float("nan")

def is_tabpfn_mps_oom(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "TabPFNMPSOutOfMemoryError" or "MPS out of memory" in str(exc)


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    raw_lines = path.read_text().splitlines()
    unnamed_values: list[str] = []

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
        else:
            unnamed_values.append(line)

    # Backward-compatible fallback for a badly formatted .env containing only the raw token.
    if "TABPFN_TOKEN" not in os.environ and unnamed_values:
        os.environ["TABPFN_TOKEN"] = unnamed_values[0]


def detect_accelerator() -> dict[str, Any]:
    accelerator = {
        "preferred": "cpu",
        "tabpfn_device": "cpu",
        "xgboost_device": "cpu",
        "xgboost_tree_method": "hist",
        "catboost_task_type": "CPU",
        "catboost_devices": None,
    }

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.backends.mps.is_available():
        accelerator["preferred"] = "mps"
        accelerator["tabpfn_device"] = "mps"
        return accelerator

    if torch is not None and torch.cuda.is_available():
        accelerator["preferred"] = "cuda"
        accelerator["tabpfn_device"] = "cuda"
        accelerator["xgboost_device"] = "cuda"
        accelerator["xgboost_tree_method"] = "hist"
        accelerator["catboost_task_type"] = "GPU"
        accelerator["catboost_devices"] = "0"

    return accelerator


def require_package(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing required package '{module_name}'. Install it with: {install_hint}"
        ) from exc


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{path} is missing target column '{TARGET_COL}'")
    return df


def load_cv_inputs() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not CV_DATA_PATH.exists() or not CV_ASSIGNMENTS_PATH.exists():
        return None, None

    cv_df = pd.read_csv(CV_DATA_PATH)
    cv_assignments = pd.read_csv(CV_ASSIGNMENTS_PATH)
    if TARGET_COL not in cv_df.columns:
        raise ValueError(f"{CV_DATA_PATH} is missing target column '{TARGET_COL}'")
    required_assignment_cols = {"repeat", "fold", "fixture_id", "cv_split", "seed"}
    missing = required_assignment_cols - set(cv_assignments.columns)
    if missing:
        raise ValueError(f"{CV_ASSIGNMENTS_PATH} is missing columns: {sorted(missing)}")
    return cv_df, cv_assignments


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [TARGET_COL]
    if SPLIT_COL in df.columns:
        drop_cols.append(SPLIT_COL)
    x = df.drop(columns=drop_cols)
    y = df[TARGET_COL].astype(int)
    return x, y


def infer_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols = []
    for col in x.columns:
        if (
            pd.api.types.is_bool_dtype(x[col])
            or pd.api.types.is_object_dtype(x[col])
            or pd.api.types.is_string_dtype(x[col])
            or isinstance(x[col].dtype, pd.CategoricalDtype)
        ):
            categorical_cols.append(col)
    numeric_cols = [col for col in x.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def build_one_hot_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_cols),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
    )


def build_scaled_one_hot_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
    )


def impute_splits(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_filled = x_train.copy()
    val_filled = x_val.copy()
    test_filled = x_test.copy()

    for col in numeric_cols:
        fill_value = train_filled[col].median()
        train_filled[col] = train_filled[col].fillna(fill_value)
        val_filled[col] = val_filled[col].fillna(fill_value)
        test_filled[col] = test_filled[col].fillna(fill_value)

    for col in categorical_cols:
        mode = train_filled[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "missing"
        train_filled[col] = train_filled[col].fillna(fill_value)
        val_filled[col] = val_filled[col].fillna(fill_value)
        test_filled[col] = test_filled[col].fillna(fill_value)

    return train_filled, val_filled, test_filled


def to_catboost_frame(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    converted = df.copy()
    for col in categorical_cols:
        converted[col] = converted[col].astype(str)
    return converted


def positive_class_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        raise ValueError("Training split has no positive examples.")
    return negatives / positives


def balanced_sample_weight(y: pd.Series) -> np.ndarray:
    pos_weight = positive_class_weight(y)
    return np.where(y.to_numpy() == 1, pos_weight, 1.0)


def select_balanced_accuracy_threshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
    thresholds = np.unique(np.concatenate(([0.0], y_proba, [1.0])))
    best_threshold = 0.5
    best_score = -np.inf
    y_true_np = y_true.to_numpy()
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = balanced_accuracy_score(y_true_np, y_pred)
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float]:
    average_precision = average_precision_score(y_true, y_proba)
    balanced_accuracy = balanced_accuracy_score(y_true, (y_proba >= threshold).astype(int))
    auroc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    return float(average_precision), float(balanced_accuracy), float(auroc), float(brier)


def predict_proba_for_model(
    model_name: str,
    fitted_model: Any,
    x: pd.DataFrame,
    categorical_cols: list[str],
) -> np.ndarray:
    if model_name == "catboost":
        inputs = to_catboost_frame(x, categorical_cols)
        return fitted_model.predict_proba(inputs)[:, 1]
    if model_name in {"xgboost", "hist_gradient_boosting"}:
        inputs = fitted_model["preprocessor"].transform(x)
        if model_name == "xgboost":
            return np.asarray(
                fitted_model["model"].get_booster().inplace_predict(inputs, predict_type="value")
            )
        return fitted_model["model"].predict_proba(inputs)[:, 1]
    if model_name == "tabpfn":
        return batched_predict_proba(fitted_model, x, batch_size=64)
    return fitted_model.predict_proba(x)[:, 1]


def fit_isotonic_calibrator(
    fitted_model: Any,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> CalibratedClassifierCV:
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(fitted_model),
        method="isotonic",
        cv=None,
    )
    calibrator.fit(x_calibration, y_calibration)
    return calibrator


def batched_predict_proba(pipeline: Any, x: pd.DataFrame, batch_size: int = 128) -> np.ndarray:
    outputs = []
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        outputs.append(pipeline.predict_proba(x.iloc[start:stop])[:, 1])
    return np.concatenate(outputs)


def build_tabpfn_pipeline(
    device: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    tabpfn = require_package("tabpfn", ".venv/bin/pip install tabpfn")
    return Pipeline(
        steps=[
            ("preprocess", build_one_hot_preprocessor(numeric_cols, categorical_cols)),
            ("model", tabpfn.TabPFNClassifier(device=device, **TABPFN_PARAMS)),
        ]
    )


def save_predictions(
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
) -> None:
    out = pd.DataFrame(
        {
            "y_true": y_true.to_numpy(),
            "y_proba": y_proba,
        }
    )
    out.to_csv(OUTPUT_DIR / f"{model_name}_{split_name}_predictions.csv", index=False)


def fit_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    accelerator: dict[str, Any],
):
    xgboost = require_package("xgboost", ".venv/bin/pip install xgboost")

    def xgboost_balanced_accuracy_eval(y_true: np.ndarray, y_score: np.ndarray) -> float:
        return balanced_accuracy_score(y_true, (y_score >= 0.5).astype(int))

    preprocessor = build_one_hot_preprocessor(numeric_cols, categorical_cols)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_val_transformed = preprocessor.transform(x_val)
    model = xgboost.XGBClassifier(
        **XGBOOST_PARAMS,
        eval_metric=[xgboost_balanced_accuracy_eval, "aucpr"],
        tree_method=accelerator["xgboost_tree_method"],
        device=accelerator["xgboost_device"],
    )
    model.fit(
        x_train_transformed,
        y_train,
        eval_set=[(x_val_transformed, y_val)],
        verbose=False,
    )
    val_proba = np.asarray(model.get_booster().inplace_predict(x_val_transformed, predict_type="value"))
    return {"preprocessor": preprocessor, "model": model}, val_proba


def fit_tabpfn(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    accelerator: dict[str, Any],
):
    preferred_device = accelerator["tabpfn_device"]
    attempted_devices = [preferred_device]
    if preferred_device != "cpu":
        attempted_devices.append("cpu")

    last_error: Exception | None = None
    for device in attempted_devices:
        pipeline = build_tabpfn_pipeline(device, numeric_cols, categorical_cols)
        try:
            pipeline.fit(x_train, y_train)
            if device == "mps":
                _ = pipeline.predict_proba(x_val.iloc[:1])[:, 1]
            val_batch_size = 1 if device == "mps" else 64
            val_proba = batched_predict_proba(pipeline, x_val, batch_size=val_batch_size)
            return pipeline, val_proba
        except Exception as exc:
            last_error = exc
            if device == "mps" and is_tabpfn_mps_oom(exc):
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("TabPFN failed without a captured exception.")


def fit_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_scaled_one_hot_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    val_proba = pipeline.predict_proba(x_val)[:, 1]
    return pipeline, val_proba


def fit_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_one_hot_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                DecisionTreeClassifier(**DECISION_TREE_PARAMS),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    val_proba = pipeline.predict_proba(x_val)[:, 1]
    return pipeline, val_proba


def fit_hist_gradient_boosting(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    preprocessor = build_one_hot_preprocessor(numeric_cols, categorical_cols)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_val_transformed = preprocessor.transform(x_val)
    model = HistGradientBoostingClassifier(**HIST_GRADIENT_BOOSTING_PARAMS)
    model.fit(x_train_transformed, y_train, sample_weight=balanced_sample_weight(y_train))
    val_proba = model.predict_proba(x_val_transformed)[:, 1]
    return {"preprocessor": preprocessor, "model": model}, val_proba


def fit_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_cols: list[str],
    accelerator: dict[str, Any],
):
    catboost = require_package("catboost", ".venv/bin/pip install catboost")
    train_frame = to_catboost_frame(x_train, categorical_cols)
    val_frame = to_catboost_frame(x_val, categorical_cols)
    catboost_params = {
        **CATBOOST_PARAMS,
        "task_type": accelerator["catboost_task_type"],
    }
    if accelerator["catboost_devices"] is not None:
        catboost_params["devices"] = accelerator["catboost_devices"]
    model = catboost.CatBoostClassifier(**catboost_params)
    model.fit(
        train_frame,
        y_train,
        cat_features=categorical_cols,
        eval_set=(val_frame, y_val),
        use_best_model=True,
        verbose=False,
    )
    val_proba = model.predict_proba(val_frame)[:, 1]
    return model, val_proba


def evaluate_model(
    model_name: str,
    fitted_model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
    val_proba: np.ndarray,
    catboost_calibration: str,
) -> ModelResult:
    train_proba = predict_proba_for_model(model_name, fitted_model, x_train, categorical_cols)
    selected_threshold = select_balanced_accuracy_threshold(y_train, train_proba)
    val_average_precision, val_balanced_accuracy, val_auroc, val_brier_score = compute_metrics(
        y_val,
        val_proba,
        selected_threshold,
    )

    if model_name == "catboost":
        test_inputs = to_catboost_frame(x_test, categorical_cols)
        test_proba = fitted_model.predict_proba(test_inputs)[:, 1]
        if catboost_calibration == "isotonic":
            val_inputs = to_catboost_frame(x_val, categorical_cols)
            calibrator = fit_isotonic_calibrator(fitted_model, val_inputs, y_val)
            save_predictions(model_name, "test_raw", y_test, test_proba)
            test_proba = calibrator.predict_proba(test_inputs)[:, 1]
    else:
        test_proba = predict_proba_for_model(model_name, fitted_model, x_test, categorical_cols)

    test_average_precision, test_balanced_accuracy, test_auroc, test_brier_score = compute_metrics(
        y_test,
        test_proba,
        selected_threshold,
    )
    save_predictions(model_name, "val", y_val, val_proba)
    save_predictions(model_name, "test", y_test, test_proba)

    return ModelResult(
        model_name=model_name,
        val_average_precision=float(val_average_precision),
        val_balanced_accuracy=float(val_balanced_accuracy),
        val_auroc=float(val_auroc),
        val_brier_score=float(val_brier_score),
        test_average_precision=float(test_average_precision),
        test_balanced_accuracy=float(test_balanced_accuracy),
        test_auroc=float(test_auroc),
        test_brier_score=float(test_brier_score),
    )


def evaluate_grouped_cv(
    model_name: str,
    cv_df: pd.DataFrame,
    cv_assignments: pd.DataFrame,
    accelerator: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    fold_records: list[dict[str, float | int | str]] = []

    for (repeat, fold), assignment_df in cv_assignments.groupby(["repeat", "fold"], sort=True):
        train_fixture_ids = assignment_df.loc[assignment_df["cv_split"] == "train", "fixture_id"]
        val_fixture_ids = assignment_df.loc[assignment_df["cv_split"] == "val", "fixture_id"]

        train_df = cv_df[cv_df["fixture_id"].isin(train_fixture_ids)].copy()
        val_df = cv_df[cv_df["fixture_id"].isin(val_fixture_ids)].copy()
        if train_df.empty or val_df.empty:
            raise ValueError(
                f"CV fold repeat={repeat}, fold={fold} produced an empty train or validation split."
            )

        x_train, y_train = split_xy(train_df)
        x_val, y_val = split_xy(val_df)
        numeric_cols, categorical_cols = infer_feature_types(x_train)
        x_train, x_val, _ = impute_splits(x_train, x_val, x_val.copy(), numeric_cols, categorical_cols)

        trainer = get_trainers(x_train, y_train, x_val, y_val, numeric_cols, categorical_cols, accelerator)[model_name]
        fitted_model, val_proba = trainer()
        train_proba = predict_proba_for_model(model_name, fitted_model, x_train, categorical_cols)
        selected_threshold = select_balanced_accuracy_threshold(y_train, train_proba)
        average_precision, balanced_accuracy, auroc, brier_score = compute_metrics(
            y_val,
            val_proba,
            selected_threshold,
        )
        fold_records.append(
            {
                "model_name": model_name,
                "repeat": int(repeat),
                "fold": int(fold),
                "seed": int(assignment_df["seed"].iloc[0]),
                "average_precision": float(average_precision),
                "balanced_accuracy": float(balanced_accuracy),
                "auroc": float(auroc),
                "brier_score": float(brier_score),
                "selected_threshold": float(selected_threshold),
                "rows": int(len(val_df)),
                "fixtures": int(val_df["fixture_id"].nunique()),
                "positives": int(y_val.sum()),
            }
        )

    fold_df = pd.DataFrame(fold_records)
    summary = {
        "cv_pr_auc_mean": float(fold_df["average_precision"].mean()),
        "cv_pr_auc_std": float(fold_df["average_precision"].std(ddof=0)),
        "cv_balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=0)),
        "cv_auroc_mean": float(fold_df["auroc"].mean()),
        "cv_auroc_std": float(fold_df["auroc"].std(ddof=0)),
        "cv_brier_score_mean": float(fold_df["brier_score"].mean()),
        "cv_brier_score_std": float(fold_df["brier_score"].std(ddof=0)),
    }
    return summary, fold_df


def write_summary(results_df: pd.DataFrame, catboost_calibration: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)

    headers = list(results_df.columns)
    rows = [headers] + results_df.round(6).astype(str).values.tolist()
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]
    md_lines = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(rows[0])) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |",
    ]
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |")

    markdown = [
        "# Model Comparison",
        "",
        "Primary ranking metric: PR-AUC.",
        "Co-primary threshold metric: Balanced Accuracy, with threshold selected on each training fold only.",
        "Secondary diagnostics: ROC-AUC and Brier score (lower is better).",
        "",
        *md_lines,
        "",
    ]
    if catboost_calibration == "isotonic":
        markdown.insert(
            5,
            "CatBoost test probabilities are isotonic-calibrated on the validation split; validation metrics remain on raw probabilities.",
        )
    (OUTPUT_DIR / "model_comparison_results.md").write_text("\n".join(markdown))

    metadata = {
        "train_path": str(TRAIN_PATH),
        "val_path": str(VAL_PATH),
        "test_path": str(TEST_PATH),
        "target_col": TARGET_COL,
        "selection_metric": "pr_auc",
        "cv_selection_metric": "cv_pr_auc_mean",
        "cv_tie_break_metric": "cv_balanced_accuracy_mean",
        "secondary_metrics": ["auroc", "brier_score"],
        "balanced_accuracy_threshold_policy": "selected on training data only within each split/fold",
        "catboost_calibration": catboost_calibration,
    }
    if catboost_calibration == "isotonic":
        metadata["catboost_test_probability_postprocessing"] = (
            "isotonic calibration fit on validation split with sklearn.frozen.FrozenEstimator"
        )
        metadata["catboost_validation_metrics_use_raw_probabilities"] = True
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


def write_failures(failures: list[dict[str, str]]) -> None:
    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT_DIR / "model_failures.csv", index=False)
        (OUTPUT_DIR / "model_failures.json").write_text(json.dumps(failures, indent=2))


def get_trainers(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    accelerator: dict[str, Any],
) -> dict[str, Any]:
    trainers = {
        "logistic_regression": lambda: fit_logistic_regression(x_train, y_train, x_val, numeric_cols, categorical_cols),
        "decision_tree": lambda: fit_decision_tree(x_train, y_train, x_val, numeric_cols, categorical_cols),
        "hist_gradient_boosting": lambda: fit_hist_gradient_boosting(x_train, y_train, x_val, numeric_cols, categorical_cols),
        # "tabpfn": lambda: fit_tabpfn(x_train, y_train, x_val, numeric_cols, categorical_cols, accelerator),
        "xgboost": lambda: fit_xgboost(x_train, y_train, x_val, y_val, numeric_cols, categorical_cols, accelerator),
        "catboost": lambda: fit_catboost(x_train, y_train, x_val, y_val, categorical_cols, accelerator),
    }
    unknown_models = [model_name for model_name in MODELS_TO_RUN if model_name not in trainers]
    if unknown_models:
        raise ValueError(f"MODELS_TO_RUN contains unsupported models: {unknown_models}")
    return trainers


def run_single_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    accelerator: dict[str, Any],
    catboost_calibration: str,
    cv_df: pd.DataFrame | None,
    cv_assignments: pd.DataFrame | None,
) -> None:
    SINGLE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    result_path = SINGLE_RUN_DIR / f"{model_name}.json"
    try:
        trainer = get_trainers(x_train, y_train, x_val, y_val, numeric_cols, categorical_cols, accelerator)[model_name]
        fitted_model, val_proba = trainer()
        result = evaluate_model(
            model_name,
            fitted_model,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            categorical_cols,
            val_proba,
            catboost_calibration,
        )
        cv_fold_df = pd.DataFrame()
        if cv_df is not None and cv_assignments is not None:
            cv_summary, cv_fold_df = evaluate_grouped_cv(model_name, cv_df, cv_assignments, accelerator)
            for key, value in cv_summary.items():
                setattr(result, key, value)
        payload = {
            "status": "ok",
            "result": result.__dict__,
            "accelerator": accelerator,
            "cv_fold_results": cv_fold_df.to_dict(orient="records"),
        }
        result_path.write_text(json.dumps(payload, indent=2))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as exc:
        payload = {"status": "error", "error": "".join(traceback.format_exception(exc))}
        result_path.write_text(json.dumps(payload, indent=2))
        raise


def run_models_in_subprocesses(
    model_names: list[str],
    catboost_calibration: str,
) -> tuple[list[ModelResult], list[dict[str, str]], pd.DataFrame]:
    SINGLE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    results: list[ModelResult] = []
    failures: list[dict[str, str]] = []
    cv_fold_records: list[dict[str, Any]] = []

    for model_name in model_names:
        result_path = SINGLE_RUN_DIR / f"{model_name}.json"
        result_path.unlink(missing_ok=True)

        completed = subprocess.run(
            [sys.executable, __file__, "--single-model", model_name, "--catboost-calibration", catboost_calibration],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            if result_path.exists():
                payload = json.loads(result_path.read_text())
                if payload.get("status") == "ok":
                    results.append(ModelResult(**payload["result"]))
                    cv_fold_records.extend(payload.get("cv_fold_results", []))
                    continue
                if payload.get("status") == "error":
                    failures.append({"model_name": model_name, "error": payload.get("error", "Unknown error")})
                    continue
            failures.append(
                {
                    "model_name": model_name,
                    "error": stderr or stdout or f"Subprocess exited with code {completed.returncode}",
                }
            )
            continue

        if not result_path.exists():
            failures.append(
                {
                    "model_name": model_name,
                    "error": "Subprocess completed without producing a result file.",
                }
            )
            continue

        payload = json.loads(result_path.read_text())
        if payload.get("status") != "ok":
            failures.append({"model_name": model_name, "error": payload.get("error", "Unknown error")})
            continue

        results.append(ModelResult(**payload["result"]))
        cv_fold_records.extend(payload.get("cv_fold_results", []))

    return results, failures, pd.DataFrame(cv_fold_records)


def main(single_model: str | None = None, catboost_calibration: str = "none") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_env_file()

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)

    x_train, y_train = split_xy(train_df)
    x_val, y_val = split_xy(val_df)
    x_test, y_test = split_xy(test_df)

    numeric_cols, categorical_cols = infer_feature_types(x_train)
    x_train, x_val, x_test = impute_splits(x_train, x_val, x_test, numeric_cols, categorical_cols)
    accelerator = detect_accelerator()
    cv_df, cv_assignments = load_cv_inputs()

    if single_model is not None:
        run_single_model(
            single_model,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            numeric_cols,
            categorical_cols,
            accelerator,
            catboost_calibration,
            cv_df,
            cv_assignments,
        )
        return

    results, failures, cv_fold_df = run_models_in_subprocesses(MODELS_TO_RUN, catboost_calibration)

    if not results:
        write_failures(failures)
        raise RuntimeError("All model fits failed. See output/model_comparison/model_failures.json")

    results_df = pd.DataFrame([result.__dict__ for result in results])
    sort_cols = ["val_average_precision", "val_balanced_accuracy", "val_auroc", "val_brier_score"]
    ascending = [False, False, False, True]
    if "cv_pr_auc_mean" in results_df.columns and results_df["cv_pr_auc_mean"].notna().any():
        sort_cols = [
            "cv_pr_auc_mean",
            "cv_balanced_accuracy_mean",
            "cv_auroc_mean",
            "cv_brier_score_mean",
        ]
        ascending = [False, False, False, True]
    results_df = results_df.sort_values(sort_cols, ascending=ascending)
    write_summary(results_df, catboost_calibration)
    write_failures(failures)
    if not cv_fold_df.empty:
        cv_fold_df.to_csv(OUTPUT_DIR / "model_comparison_cv_fold_results.csv", index=False)
    print(
        results_df[
            [
                "model_name",
                "cv_pr_auc_mean",
                "cv_pr_auc_std",
                "cv_balanced_accuracy_mean",
                "cv_balanced_accuracy_std",
                "cv_auroc_mean",
                "cv_auroc_std",
                "cv_brier_score_mean",
                "cv_brier_score_std",
                "test_average_precision",
                "test_balanced_accuracy",
                "test_auroc",
                "test_brier_score",
            ]
        ].round(6).to_string(index=False)
    )
    print("\nAccelerator selection:")
    print(json.dumps(accelerator, indent=2))
    if failures:
        print("\nFailed models:")
        print(pd.DataFrame(failures).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-model",
        choices=MODELS_TO_RUN,
        default=None,
    )
    parser.add_argument(
        "--catboost-calibration",
        choices=["none", "isotonic"],
        default="none",
    )
    args = parser.parse_args()
    main(single_model=args.single_model, catboost_calibration=args.catboost_calibration)
