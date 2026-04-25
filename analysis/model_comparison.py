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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_DIR = Path("data/baseline_modeling")
OUTPUT_DIR = Path("output/model_comparison")

TRAIN_PATH = DATA_DIR / "baseline_train_model_ready.csv"
VAL_PATH = DATA_DIR / "baseline_val_model_ready.csv"
TEST_PATH = DATA_DIR / "baseline_test_model_ready.csv"
SINGLE_RUN_DIR = OUTPUT_DIR / "_single_runs"

TARGET_COL = "scored_after"
SPLIT_COL = "split"
RANDOM_STATE = 42

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str((OUTPUT_DIR / "tabpfn_cache").resolve()))
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")


@dataclass
class ModelResult:
    model_name: str
    val_average_precision: float
    val_auroc: float
    val_brier_score: float
    test_average_precision: float
    test_auroc: float
    test_brier_score: float

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


def compute_metrics(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float, float]:
    average_precision = average_precision_score(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    return float(average_precision), float(auroc), float(brier)


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
            ("model", tabpfn.TabPFNClassifier(device=device, random_state=RANDOM_STATE)),
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
    numeric_cols: list[str],
    categorical_cols: list[str],
    accelerator: dict[str, Any],
):
    xgboost = require_package("xgboost", ".venv/bin/pip install xgboost")
    preprocessor = build_one_hot_preprocessor(numeric_cols, categorical_cols)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_val_transformed = preprocessor.transform(x_val)
    model = xgboost.XGBClassifier(
        n_estimators=2000,
        max_depth=2,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.80,
        min_child_weight=8,
        reg_lambda=10.0,
        reg_alpha=0.5,
        gamma=0.1,
        max_delta_step=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        tree_method=accelerator["xgboost_tree_method"],
        device=accelerator["xgboost_device"],
        n_jobs=1,
    )
    model.fit(x_train_transformed, y_train)
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
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
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
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
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
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train_transformed, y_train, sample_weight=balanced_sample_weight(y_train))
    val_proba = model.predict_proba(x_val_transformed)[:, 1]
    return {"preprocessor": preprocessor, "model": model}, val_proba


def fit_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    categorical_cols: list[str],
    accelerator: dict[str, Any],
):
    catboost = require_package("catboost", ".venv/bin/pip install catboost")
    train_frame = to_catboost_frame(x_train, categorical_cols)
    val_frame = to_catboost_frame(x_val, categorical_cols)
    model = catboost.CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=2000,
        learning_rate=0.03,
        depth=3,
        l2_leaf_reg=10,
        random_strength=1.0,
        early_stopping_rounds=150,
        random_seed=42,
        verbose=100
    )
    model.fit(train_frame, y_train, cat_features=categorical_cols)
    val_proba = model.predict_proba(val_frame)[:, 1]
    return model, val_proba


def evaluate_model(
    model_name: str,
    fitted_model: Any,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
    val_proba: np.ndarray,
) -> ModelResult:
    val_average_precision, val_auroc, val_brier_score = compute_metrics(y_val, val_proba)

    if model_name == "catboost":
        test_inputs = to_catboost_frame(x_test, categorical_cols)
        test_proba = fitted_model.predict_proba(test_inputs)[:, 1]
    elif model_name in {"xgboost", "hist_gradient_boosting"}:
        test_inputs = fitted_model["preprocessor"].transform(x_test)
        if model_name == "xgboost":
            test_proba = np.asarray(
                fitted_model["model"].get_booster().inplace_predict(test_inputs, predict_type="value")
            )
        else:
            test_proba = fitted_model["model"].predict_proba(test_inputs)[:, 1]
    elif model_name == "tabpfn":
        test_proba = batched_predict_proba(fitted_model, x_test, batch_size=64)
    else:
        test_proba = fitted_model.predict_proba(x_test)[:, 1]

    test_average_precision, test_auroc, test_brier_score = compute_metrics(y_test, test_proba)
    save_predictions(model_name, "val", y_val, val_proba)
    save_predictions(model_name, "test", y_test, test_proba)

    return ModelResult(
        model_name=model_name,
        val_average_precision=float(val_average_precision),
        val_auroc=float(val_auroc),
        val_brier_score=float(val_brier_score),
        test_average_precision=float(test_average_precision),
        test_auroc=float(test_auroc),
        test_brier_score=float(test_brier_score),
    )


def write_summary(results_df: pd.DataFrame) -> None:
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
        "Primary ranking metric: Average Precision / PR-AUC.",
        "Secondary ranking metric: ROC-AUC.",
        "Calibration metric: Brier score (lower is better).",
        "",
        *md_lines,
        "",
    ]
    (OUTPUT_DIR / "model_comparison_results.md").write_text("\n".join(markdown))

    metadata = {
        "train_path": str(TRAIN_PATH),
        "val_path": str(VAL_PATH),
        "test_path": str(TEST_PATH),
        "target_col": TARGET_COL,
        "selection_metric": "average_precision",
        "secondary_metric": "auroc",
        "calibration_metric": "brier_score",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


def write_failures(failures: list[dict[str, str]]) -> None:
    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT_DIR / "model_failures.csv", index=False)
        (OUTPUT_DIR / "model_failures.json").write_text(json.dumps(failures, indent=2))


def get_trainers(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    accelerator: dict[str, Any],
) -> dict[str, Any]:
    return {
        "logistic_regression": lambda: fit_logistic_regression(x_train, y_train, x_val, numeric_cols, categorical_cols),
        "decision_tree": lambda: fit_decision_tree(x_train, y_train, x_val, numeric_cols, categorical_cols),
        "hist_gradient_boosting": lambda: fit_hist_gradient_boosting(x_train, y_train, x_val, numeric_cols, categorical_cols),
        # "tabpfn": lambda: fit_tabpfn(x_train, y_train, x_val, numeric_cols, categorical_cols, accelerator),
        "xgboost": lambda: fit_xgboost(x_train, y_train, x_val, numeric_cols, categorical_cols, accelerator),
        "catboost": lambda: fit_catboost(x_train, y_train, x_val, categorical_cols, accelerator),
    }


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
) -> None:
    SINGLE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    result_path = SINGLE_RUN_DIR / f"{model_name}.json"
    try:
        trainer = get_trainers(x_train, y_train, x_val, numeric_cols, categorical_cols, accelerator)[model_name]
        fitted_model, val_proba = trainer()
        result = evaluate_model(
            model_name,
            fitted_model,
            x_val,
            y_val,
            x_test,
            y_test,
            categorical_cols,
            val_proba,
        )
        payload = {"status": "ok", "result": result.__dict__, "accelerator": accelerator}
        result_path.write_text(json.dumps(payload, indent=2))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as exc:
        payload = {"status": "error", "error": "".join(traceback.format_exception(exc))}
        result_path.write_text(json.dumps(payload, indent=2))
        raise


def run_models_in_subprocesses(model_names: list[str]) -> tuple[list[ModelResult], list[dict[str, str]]]:
    SINGLE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    results: list[ModelResult] = []
    failures: list[dict[str, str]] = []

    for model_name in model_names:
        result_path = SINGLE_RUN_DIR / f"{model_name}.json"
        result_path.unlink(missing_ok=True)

        completed = subprocess.run(
            [sys.executable, __file__, "--single-model", model_name],
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

    return results, failures


def main(single_model: str | None = None) -> None:
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
        )
        return

    results, failures = run_models_in_subprocesses(
        [
            "logistic_regression",
            "decision_tree",
            "hist_gradient_boosting",
            # "tabpfn",
            "xgboost",
            "catboost",
        ]
    )

    if not results:
        write_failures(failures)
        raise RuntimeError("All model fits failed. See output/model_comparison/model_failures.json")

    results_df = pd.DataFrame([result.__dict__ for result in results]).sort_values(
        ["val_average_precision", "val_auroc", "val_brier_score"], ascending=[False, False, True]
    )
    write_summary(results_df)
    write_failures(failures)
    print(
        results_df[
            [
                "model_name",
                "val_average_precision",
                "val_auroc",
                "val_brier_score",
                "test_average_precision",
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
        choices=[
            "logistic_regression",
            "decision_tree",
            "hist_gradient_boosting",
            # "tabpfn",
            "xgboost",
            "catboost",
        ],
        default=None,
    )
    args = parser.parse_args()
    main(single_model=args.single_model)
