from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.create_baseline_modeling_dataset import (
    CHECKPOINT_ORDER,
    DATA_DIR as SOURCE_DATA_DIR,
    CSV_NULL_TOKENS,
    apply_row_filters,
)
from analysis.model_comparison import CATBOOST_PARAMS, TARGET_COL, require_package


DATA_DIR = Path("data/baseline_modeling")
OUTPUT_ROOT = Path("output/feature_pruning")
RANDOM_STATE = 42

MODEL_READY_PATH = DATA_DIR / "baseline_all_model_ready.csv"
FIXTURE_SPLIT_PATH = DATA_DIR / "baseline_fixture_split.csv"
SOURCE_BASE_PATH = SOURCE_DATA_DIR / "players_quarters_final.csv"

DEFAULT_FIT_SPLITS = ("train", "val")
DEFAULT_HOLDOUT_SPLIT = "test"

DEFAULT_CATBOOST_PRUNING_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "iterations": 1200,
    "learning_rate": 0.01,
    "depth": 4,
    "l2_leaf_reg": 30.0,
    "random_strength": 1.0,
    "early_stopping_rounds": 100,
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "auto_class_weights": "SqrtBalanced",
}


@dataclass
class FoldDefinition:
    fold_id: int
    train_fixture_ids: list[int]
    val_fixture_ids: list[int]


@dataclass
class FoldMetrics:
    average_precision: float
    balanced_accuracy: float
    auroc: float
    brier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-drops", type=int, default=5, help="Maximum number of accepted feature drops.")
    parser.add_argument(
        "--target-prauc",
        type=float,
        default=None,
        help="Stop once incumbent mean CV PRAUC reaches this threshold.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=12,
        help="How many weakest candidates to test per pruning cycle.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=25,
        help="Hard cap on pruning cycles so the script does not run forever.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
        help="Number of chronological fixture-grouped inner folds.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="Absolute Spearman threshold for redundancy clusters.",
    )
    parser.add_argument(
        "--ap-tolerance",
        type=float,
        default=0.002,
        help="Fixed PRAUC tolerance band for accepting neutral drops.",
    )
    parser.add_argument(
        "--se-multiplier",
        type=float,
        default=1.0,
        help="Standard-error multiplier used in the one-standard-error parsimony rule.",
    )
    parser.add_argument(
        "--balacc-tolerance",
        type=float,
        default=0.005,
        help="Maximum allowed balanced-accuracy deterioration for a neutral drop.",
    )
    parser.add_argument(
        "--auroc-tolerance",
        type=float,
        default=0.005,
        help="Maximum allowed AUROC deterioration for a neutral drop.",
    )
    parser.add_argument(
        "--brier-tolerance",
        type=float,
        default=0.002,
        help="Maximum allowed Brier worsening for a neutral drop.",
    )
    parser.add_argument(
        "--max-ap-tolerance",
        type=float,
        default=0.005,
        help="Hard cap on the PRAUC tolerance band for neutral drops.",
    )
    parser.add_argument(
        "--min-nonnegative-ap-fold-share",
        type=float,
        default=0.75,
        help="Minimum share of folds with non-negative PRAUC delta required for a neutral drop.",
    )
    parser.add_argument(
        "--min-train-fixtures",
        type=int,
        default=10,
        help="Minimum number of fixtures in the training side of the first inner fold.",
    )
    parser.add_argument(
        "--min-val-fixtures",
        type=int,
        default=2,
        help="Minimum number of fixtures in each validation block.",
    )
    return parser.parse_args()


def load_model_ready_with_metadata() -> pd.DataFrame:
    model_df = pd.read_csv(MODEL_READY_PATH)

    source_base = pd.read_csv(
        SOURCE_BASE_PATH,
        parse_dates=["date"],
        na_values=CSV_NULL_TOKENS,
    )
    source_base["_checkpoint_order"] = source_base["checkpoint"].map(CHECKPOINT_ORDER)
    source_base = source_base.sort_values(
        ["date", "fixture_id", "player_appearance_id", "_checkpoint_order"]
    ).reset_index(drop=True)
    source_base = source_base.drop(columns=["_checkpoint_order"])
    filtered_base, _ = apply_row_filters(source_base)
    filtered_base = filtered_base.reset_index(drop=True)

    if len(filtered_base) != len(model_df):
        raise ValueError(
            f"Filtered source base has {len(filtered_base)} rows but model-ready export has {len(model_df)} rows."
        )

    for col in ["checkpoint", TARGET_COL]:
        if col in model_df.columns and col in filtered_base.columns:
            if not filtered_base[col].astype(str).equals(model_df[col].astype(str)):
                raise ValueError(f"Row-order alignment failed on shared column '{col}'.")

    metadata = filtered_base[["date", "fixture_id", "player_appearance_id"]].copy()
    fixture_split = pd.read_csv(FIXTURE_SPLIT_PATH, parse_dates=["date"])
    metadata = metadata.merge(
        fixture_split[["date", "fixture_id", "split", "fixture_order"]],
        on=["date", "fixture_id"],
        how="left",
    )
    if metadata["split"].isna().any():
        raise ValueError("Some reconstructed rows could not be matched to fixture split metadata.")

    combined = pd.concat([metadata.reset_index(drop=True), model_df.reset_index(drop=True)], axis=1)
    return combined


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


def impute_from_train(
    x_train: pd.DataFrame,
    x_other: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_filled = x_train.copy()
    other_filled = x_other.copy()

    for col in numeric_cols:
        fill_value = train_filled[col].median()
        train_filled[col] = train_filled[col].fillna(fill_value)
        other_filled[col] = other_filled[col].fillna(fill_value)

    for col in categorical_cols:
        mode = train_filled[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "missing"
        train_filled[col] = train_filled[col].fillna(fill_value)
        other_filled[col] = other_filled[col].fillna(fill_value)

    return train_filled, other_filled


def to_catboost_frame(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        out[col] = out[col].astype(str)
    return out


def fit_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_cols: list[str],
):
    catboost = require_package("catboost", ".venv/bin/pip install catboost")
    params = CATBOOST_PARAMS.copy()
    params.update(DEFAULT_CATBOOST_PRUNING_PARAMS)
    model = catboost.CatBoostClassifier(**params)
    train_frame = to_catboost_frame(x_train, categorical_cols)
    val_frame = to_catboost_frame(x_val, categorical_cols)
    model.fit(
        train_frame,
        y_train,
        eval_set=(val_frame, y_val),
        cat_features=categorical_cols,
        use_best_model=True,
    )
    val_proba = model.predict_proba(val_frame)[:, 1]
    return model, val_proba


def compute_metrics(y_true: pd.Series, y_proba: np.ndarray) -> FoldMetrics:
    prediction = (y_proba >= 0.5).astype(int)
    return FoldMetrics(
        average_precision=float(average_precision_score(y_true, y_proba)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, prediction)),
        auroc=float(roc_auc_score(y_true, y_proba)),
        brier=float(brier_score_loss(y_true, y_proba)),
    )


def build_walk_forward_folds(
    fit_fixture_df: pd.DataFrame,
    inner_folds: int,
    min_train_fixtures: int,
    min_val_fixtures: int,
) -> list[FoldDefinition]:
    ordered = fit_fixture_df.sort_values(["fixture_order", "date", "fixture_id"]).reset_index(drop=True)
    fixture_ids = ordered["fixture_id"].tolist()
    n_fixtures = len(fixture_ids)
    if n_fixtures < (min_train_fixtures + min_val_fixtures):
        raise ValueError("Not enough fit fixtures to create the requested walk-forward folds.")

    remaining = n_fixtures - min_train_fixtures
    val_block = max(min_val_fixtures, math.floor(remaining / max(inner_folds, 1)))
    folds: list[FoldDefinition] = []
    start = min_train_fixtures
    fold_id = 1

    while start < n_fixtures and len(folds) < inner_folds:
        stop = min(n_fixtures, start + val_block)
        val_ids = fixture_ids[start:stop]
        if len(val_ids) < min_val_fixtures:
            break
        train_ids = fixture_ids[:start]
        folds.append(FoldDefinition(fold_id=fold_id, train_fixture_ids=train_ids, val_fixture_ids=val_ids))
        start = stop
        fold_id += 1

    if not folds:
        raise ValueError("Could not build any valid walk-forward inner folds.")
    return folds


def feature_value_frame(x_df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    values = x_df.copy()
    for col in values.columns:
        if col in categorical_cols and not pd.api.types.is_bool_dtype(values[col]):
            values[col] = values[col].astype("category").cat.codes.replace(-1, np.nan)
        elif pd.api.types.is_bool_dtype(values[col]):
            values[col] = values[col].astype(int)
    return values


def compute_corr_matrix(feature_df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    values = feature_value_frame(feature_df, categorical_cols)
    numeric_df = values.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(method="spearman")


def compute_vif_table(feature_df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression

    values = feature_value_frame(feature_df, categorical_cols)
    numeric_df = values.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "vif"])

    filled = numeric_df.copy()
    for col in filled.columns:
        filled[col] = filled[col].fillna(filled[col].median())

    rows = []
    for feature in filled.columns:
        y = filled[feature].to_numpy()
        x = filled.drop(columns=[feature])
        model = LinearRegression()
        model.fit(x, y)
        r2 = float(model.score(x, y))
        vif = np.inf if r2 >= 0.999999 else float(1.0 / (1.0 - r2))
        rows.append({"feature": feature, "vif": vif})
    return pd.DataFrame(rows)


def build_exact_duplicate_map(feature_df: pd.DataFrame) -> dict[str, list[str]]:
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    signatures: dict[tuple, str] = {}
    comparable = feature_df.copy()

    for col in comparable.columns:
        series = comparable[col]
        if pd.api.types.is_numeric_dtype(series):
            signature = tuple(series.fillna(-999999999).round(10).tolist())
        else:
            signature = tuple(series.fillna("__missing__").astype(str).tolist())
        if signature in signatures:
            duplicate_groups[signatures[signature]].append(col)
        else:
            signatures[signature] = col

    return {leader: members for leader, members in duplicate_groups.items() if members}


def build_corr_clusters(corr_matrix: pd.DataFrame, threshold: float) -> list[list[str]]:
    if corr_matrix.empty:
        return []

    adjacency: dict[str, set[str]] = {col: set() for col in corr_matrix.columns}
    cols = list(corr_matrix.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr_matrix.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    visited: set[str] = set()
    clusters: list[list[str]] = []
    for feature in cols:
        if feature in visited or not adjacency[feature]:
            continue
        stack = [feature]
        cluster = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            stack.extend(adjacency[current] - visited)
        if len(cluster) > 1:
            clusters.append(sorted(cluster))
    return clusters


def explicit_algebraic_blocks(features: list[str]) -> list[dict[str, object]]:
    feature_set = set(features)
    candidate_specs = [
        ("checkpoint_block", ["checkpoint", "checkpoint_period", "checkpoint_min"], "checkpoint_encoding"),
        (
            "pressured_count_triplet_last15",
            ["last_15_times_pressured", "last_15_pressured_succ", "last_15_pressured_unsucc"],
            "algebraic_total_equals_parts",
        ),
        (
            "pressured_count_triplet_cumul",
            ["cumul_times_pressured", "cumul_pressured_succ", "cumul_pressured_unsucc"],
            "algebraic_total_equals_parts",
        ),
        (
            "pressure_applied_triplet_last15",
            ["last_15_pressures_applied", "last_15_pressures_won", "last_15_pressures_lost"],
            "algebraic_total_equals_parts",
        ),
        (
            "pressure_applied_triplet_cumul",
            ["cumul_pressures_applied", "cumul_pressures_won", "cumul_pressures_lost"],
            "algebraic_total_equals_parts",
        ),
        (
            "shot_under_pressure_last15",
            ["last_15_shots_total", "last_15_shots_under_pressure", "last_15_shots_under_pressure_rate"],
            "count_rate_family",
        ),
        (
            "shot_under_pressure_cumul",
            ["cumul_shots_total", "cumul_shots_under_pressure", "cumul_shots_under_pressure_rate"],
            "count_rate_family",
        ),
        (
            "top_sprint_distance_family",
            ["top_sprint_distance", "avg_top_sprint_distance", "last15_top_sprint_share"],
            "run_distance_family",
        ),
        (
            "distance_intensity_family",
            ["last15_distance", "distance_per_possession", "distance_per_run"],
            "run_distance_family",
        ),
    ]
    blocks = []
    for block_name, members, reason in candidate_specs:
        present = [feature for feature in members if feature in feature_set]
        if len(present) >= 2:
            blocks.append({"block_name": block_name, "features": present, "reason": reason})
    return blocks


def safe_series(values: dict[str, float], features: list[str], default: float = 0.0) -> pd.Series:
    return pd.Series({feature: float(values.get(feature, default)) for feature in features})


def normalize_series(series: pd.Series, higher_is_weaker: bool) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).fillna(series.replace([np.inf, -np.inf], np.nan).max())
    if clean.nunique(dropna=False) <= 1:
        return pd.Series(0.0, index=clean.index)
    scaled = (clean - clean.min()) / (clean.max() - clean.min())
    return scaled if higher_is_weaker else (1.0 - scaled)


def evaluate_feature_set(
    fit_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    features: list[str],
    folds: list[FoldDefinition],
    *,
    compute_diagnostics: bool,
) -> dict[str, object]:
    catboost = require_package("catboost", ".venv/bin/pip install catboost")

    fold_rows: list[dict[str, object]] = []
    shap_accumulator: dict[str, list[float]] = defaultdict(list)
    perm_accumulator: dict[str, list[float]] = defaultdict(list)

    for fold in folds:
        train_mask = fit_df["fixture_id"].isin(fold.train_fixture_ids)
        val_mask = fit_df["fixture_id"].isin(fold.val_fixture_ids)
        train_df = fit_df.loc[train_mask].copy()
        val_df = fit_df.loc[val_mask].copy()

        x_train = train_df[features].copy()
        y_train = train_df[TARGET_COL].astype(int)
        x_val = val_df[features].copy()
        y_val = val_df[TARGET_COL].astype(int)

        numeric_cols, categorical_cols = infer_feature_types(x_train)
        x_train, x_val = impute_from_train(x_train, x_val, numeric_cols, categorical_cols)
        model, val_proba = fit_catboost(x_train, y_train, x_val, y_val, categorical_cols)
        metrics = compute_metrics(y_val, val_proba)
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "train_fixtures": len(fold.train_fixture_ids),
                "val_fixtures": len(fold.val_fixture_ids),
                "average_precision": metrics.average_precision,
                "balanced_accuracy": metrics.balanced_accuracy,
                "auroc": metrics.auroc,
                "brier": metrics.brier,
            }
        )

        if compute_diagnostics:
            val_frame = to_catboost_frame(x_val, categorical_cols)
            val_pool = catboost.Pool(val_frame, label=y_val, cat_features=categorical_cols)
            contribs = model.get_feature_importance(val_pool, type="ShapValues")
            shap_matrix = pd.DataFrame(contribs[:, :-1], columns=features)
            mean_abs_shap = shap_matrix.abs().mean()
            for feature, value in mean_abs_shap.items():
                shap_accumulator[feature].append(float(value))

            baseline_ap = metrics.average_precision
            for feature in features:
                permuted = x_val.copy()
                rng = np.random.default_rng(RANDOM_STATE + fold.fold_id + len(feature))
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                permuted_frame = to_catboost_frame(permuted, categorical_cols)
                permuted_proba = model.predict_proba(permuted_frame)[:, 1]
                permuted_ap = float(average_precision_score(y_val, permuted_proba))
                perm_accumulator[feature].append(float(baseline_ap - permuted_ap))

    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "mean_average_precision": float(fold_metrics["average_precision"].mean()),
        "mean_balanced_accuracy": float(fold_metrics["balanced_accuracy"].mean()),
        "mean_auroc": float(fold_metrics["auroc"].mean()),
        "mean_brier": float(fold_metrics["brier"].mean()),
    }

    shap_mean = safe_series({k: np.mean(v) for k, v in shap_accumulator.items()}, features)
    shap_std = safe_series({k: np.std(v, ddof=0) for k, v in shap_accumulator.items()}, features)
    perm_mean = safe_series({k: np.mean(v) for k, v in perm_accumulator.items()}, features)
    perm_std = safe_series({k: np.std(v, ddof=0) for k, v in perm_accumulator.items()}, features)

    numeric_cols, categorical_cols = infer_feature_types(fit_df[features])
    corr_matrix = compute_corr_matrix(fit_df[features], categorical_cols) if compute_diagnostics else pd.DataFrame()
    vif_table = compute_vif_table(fit_df[features], categorical_cols) if compute_diagnostics else pd.DataFrame()

    x_train_full = fit_df[features].copy()
    y_train_full = fit_df[TARGET_COL].astype(int)
    x_test = holdout_df[features].copy()
    y_test = holdout_df[TARGET_COL].astype(int)
    numeric_cols, categorical_cols = infer_feature_types(x_train_full)
    x_train_full, x_test = impute_from_train(x_train_full, x_test, numeric_cols, categorical_cols)
    final_model, test_proba = fit_catboost(x_train_full, y_train_full, x_test, y_test, categorical_cols)
    test_metrics = compute_metrics(y_test, test_proba)

    return {
        "fold_metrics": fold_metrics,
        "summary": summary,
        "shap_mean": shap_mean,
        "shap_std": shap_std,
        "perm_mean": perm_mean,
        "perm_std": perm_std,
        "corr_matrix": corr_matrix,
        "vif_table": vif_table,
        "test_metrics": {
            "average_precision": test_metrics.average_precision,
            "balanced_accuracy": test_metrics.balanced_accuracy,
            "auroc": test_metrics.auroc,
            "brier": test_metrics.brier,
        },
    }


def build_candidate_table(
    fit_df: pd.DataFrame,
    features: list[str],
    diagnostics: dict[str, object],
    corr_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shap_mean: pd.Series = diagnostics["shap_mean"]
    shap_std: pd.Series = diagnostics["shap_std"]
    perm_mean: pd.Series = diagnostics["perm_mean"]
    corr_matrix: pd.DataFrame = diagnostics["corr_matrix"]
    vif_table: pd.DataFrame = diagnostics["vif_table"]

    duplicate_map = build_exact_duplicate_map(fit_df[features])
    corr_clusters = build_corr_clusters(corr_matrix, corr_threshold)
    max_abs_corr = pd.Series(0.0, index=features)
    cluster_id = pd.Series("", index=features, dtype=object)
    cluster_reason = pd.Series("", index=features, dtype=object)
    cluster_size = pd.Series(1, index=features, dtype=int)

    for idx, cluster in enumerate(corr_clusters, start=1):
        cluster_name = f"corr_cluster_{idx}"
        cluster_corr = corr_matrix.loc[cluster, cluster].abs().replace(1.0, np.nan)
        cluster_max = cluster_corr.max(axis=1).fillna(0.0)
        for feature in cluster:
            max_abs_corr.loc[feature] = float(max(max_abs_corr.loc[feature], cluster_max.loc[feature]))
            cluster_id.loc[feature] = cluster_name
            cluster_reason.loc[feature] = "high_correlation"
            cluster_size.loc[feature] = max(cluster_size.loc[feature], len(cluster))

    duplicate_flag = pd.Series(0, index=features, dtype=int)
    for leader, members in duplicate_map.items():
        dup_cluster = [leader, *members]
        cluster_name = f"duplicate_{leader}"
        for feature in dup_cluster:
            duplicate_flag.loc[feature] = 1
            cluster_id.loc[feature] = cluster_name
            cluster_reason.loc[feature] = "exact_duplicate"
            cluster_size.loc[feature] = max(cluster_size.loc[feature], len(dup_cluster))
            max_abs_corr.loc[feature] = 1.0

    vif_lookup = vif_table.set_index("feature")["vif"].to_dict() if not vif_table.empty else {}
    vif_series = pd.Series({feature: float(vif_lookup.get(feature, 1.0)) for feature in features})
    instability = shap_std / shap_mean.replace(0.0, np.nan)
    instability = instability.replace([np.inf, -np.inf], np.nan).fillna(instability.max(skipna=True) or 0.0)

    score_df = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": shap_mean.reindex(features).fillna(0.0).to_numpy(),
            "shap_std": shap_std.reindex(features).fillna(0.0).to_numpy(),
            "perm_delta_prauc": perm_mean.reindex(features).fillna(0.0).to_numpy(),
            "max_abs_corr": max_abs_corr.reindex(features).fillna(0.0).to_numpy(),
            "cluster_id": cluster_id.reindex(features).fillna("").to_numpy(),
            "cluster_reason": cluster_reason.reindex(features).fillna("").to_numpy(),
            "cluster_size": cluster_size.reindex(features).fillna(1).to_numpy(),
            "duplicate_flag": duplicate_flag.reindex(features).fillna(0).astype(int).to_numpy(),
            "vif": vif_series.reindex(features).fillna(1.0).to_numpy(),
            "instability": instability.reindex(features).fillna(0.0).to_numpy(),
        }
    )

    score_df["weak_shap"] = normalize_series(score_df["mean_abs_shap"], higher_is_weaker=False)
    score_df["weak_perm"] = normalize_series(score_df["perm_delta_prauc"], higher_is_weaker=False)
    score_df["weak_corr"] = normalize_series(score_df["max_abs_corr"], higher_is_weaker=True)
    score_df["weak_vif"] = normalize_series(np.log1p(score_df["vif"].replace(np.inf, 1e9)), higher_is_weaker=True)
    score_df["weak_instability"] = normalize_series(score_df["instability"], higher_is_weaker=True)
    score_df["weakness_score"] = (
        0.35 * score_df["weak_perm"]
        + 0.30 * score_df["weak_shap"]
        + 0.15 * score_df["weak_corr"]
        + 0.10 * score_df["weak_vif"]
        + 0.10 * score_df["weak_instability"]
        + 0.50 * score_df["duplicate_flag"]
    )

    score_df["stage"] = "B"
    score_df.loc[score_df["duplicate_flag"] == 1, "stage"] = "A"
    score_df.loc[
        (score_df["duplicate_flag"] == 0) & (score_df["cluster_reason"] == "high_correlation"),
        "stage",
    ] = "A"

    score_df = score_df.sort_values(
        ["stage", "weakness_score", "perm_delta_prauc", "mean_abs_shap"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    cluster_summary = score_df.loc[score_df["cluster_id"] != "", ["feature", "cluster_id", "cluster_reason", "cluster_size"]]
    return score_df, cluster_summary


def build_block_table(
    fit_df: pd.DataFrame,
    features: list[str],
    diagnostics: dict[str, object],
    corr_threshold: float,
) -> pd.DataFrame:
    score_df, _ = build_candidate_table(fit_df, features, diagnostics, corr_threshold)
    score_lookup = score_df.set_index("feature")

    blocks: list[dict[str, object]] = []
    seen_feature_sets: set[tuple[str, ...]] = set()

    for block in explicit_algebraic_blocks(features):
        key = tuple(sorted(block["features"]))
        if key in seen_feature_sets:
            continue
        seen_feature_sets.add(key)
        members = block["features"]
        blocks.append(
            {
                "block_name": block["block_name"],
                "block_reason": block["reason"],
                "block_size": len(members),
                "features": members,
                "mean_weakness_score": float(score_lookup.loc[members, "weakness_score"].mean()),
                "mean_abs_shap_sum": float(score_lookup.loc[members, "mean_abs_shap"].sum()),
                "perm_delta_prauc_sum": float(score_lookup.loc[members, "perm_delta_prauc"].sum()),
                "max_abs_corr_max": float(score_lookup.loc[members, "max_abs_corr"].max()),
                "mean_vif": float(score_lookup.loc[members, "vif"].replace(np.inf, np.nan).mean())
                if score_lookup.loc[members, "vif"].replace(np.inf, np.nan).notna().any()
                else np.inf,
            }
        )

    corr_matrix: pd.DataFrame = diagnostics["corr_matrix"]
    corr_clusters = build_corr_clusters(corr_matrix, corr_threshold)
    for idx, cluster in enumerate(corr_clusters, start=1):
        key = tuple(sorted(cluster))
        if key in seen_feature_sets:
            continue
        seen_feature_sets.add(key)
        blocks.append(
            {
                "block_name": f"corr_cluster_{idx}",
                "block_reason": "high_correlation_cluster",
                "block_size": len(cluster),
                "features": cluster,
                "mean_weakness_score": float(score_lookup.loc[cluster, "weakness_score"].mean()),
                "mean_abs_shap_sum": float(score_lookup.loc[cluster, "mean_abs_shap"].sum()),
                "perm_delta_prauc_sum": float(score_lookup.loc[cluster, "perm_delta_prauc"].sum()),
                "max_abs_corr_max": float(score_lookup.loc[cluster, "max_abs_corr"].max()),
                "mean_vif": float(score_lookup.loc[cluster, "vif"].replace(np.inf, np.nan).mean())
                if score_lookup.loc[cluster, "vif"].replace(np.inf, np.nan).notna().any()
                else np.inf,
            }
        )

    if not blocks:
        return pd.DataFrame(
            columns=[
                "block_name",
                "block_reason",
                "block_size",
                "features",
                "mean_weakness_score",
                "mean_abs_shap_sum",
                "perm_delta_prauc_sum",
                "max_abs_corr_max",
                "mean_vif",
            ]
        )

    return pd.DataFrame(blocks).sort_values(
        ["mean_weakness_score", "perm_delta_prauc_sum", "mean_abs_shap_sum"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def acceptance_decision(
    incumbent_fold_metrics: pd.DataFrame,
    candidate_fold_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float]]:
    ap_deltas = candidate_fold_metrics["average_precision"] - incumbent_fold_metrics["average_precision"]
    bal_deltas = candidate_fold_metrics["balanced_accuracy"] - incumbent_fold_metrics["balanced_accuracy"]
    auroc_deltas = candidate_fold_metrics["auroc"] - incumbent_fold_metrics["auroc"]
    brier_deltas = candidate_fold_metrics["brier"] - incumbent_fold_metrics["brier"]

    mean_ap_delta = float(ap_deltas.mean())
    mean_bal_delta = float(bal_deltas.mean())
    mean_auroc_delta = float(auroc_deltas.mean())
    mean_brier_delta = float(brier_deltas.mean())
    nonnegative_ap_fold_share = float((ap_deltas >= 0).mean())

    if len(ap_deltas) > 1:
        ap_se = float(ap_deltas.std(ddof=1) / math.sqrt(len(ap_deltas)))
    else:
        ap_se = 0.0
    tolerance = min(
        float(args.max_ap_tolerance),
        max(float(args.ap_tolerance), float(args.se_multiplier) * ap_se),
    )

    accepted = False
    if mean_ap_delta > 0:
        accepted = True
    elif (
        mean_ap_delta >= -tolerance
        and nonnegative_ap_fold_share >= float(args.min_nonnegative_ap_fold_share)
        and mean_bal_delta >= -float(args.balacc_tolerance)
        and mean_auroc_delta >= -float(args.auroc_tolerance)
        and mean_brier_delta <= float(args.brier_tolerance)
    ):
        accepted = True

    deltas = {
        "delta_average_precision": mean_ap_delta,
        "delta_balanced_accuracy": mean_bal_delta,
        "delta_auroc": mean_auroc_delta,
        "delta_brier": mean_brier_delta,
        "nonnegative_ap_fold_share": nonnegative_ap_fold_share,
        "ap_standard_error": ap_se,
        "ap_acceptance_tolerance": tolerance,
    }
    return accepted, deltas


def select_shortlist(candidate_table: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if candidate_table.empty:
        return candidate_table

    stage_a = candidate_table.loc[candidate_table["stage"] == "A"].copy()
    stage_b = candidate_table.loc[candidate_table["stage"] == "B"].copy()
    n_stage_a = min(len(stage_a), math.ceil(top_k / 2))
    n_stage_b = min(len(stage_b), top_k - n_stage_a)

    shortlist = pd.concat([stage_a.head(n_stage_a), stage_b.head(n_stage_b)], ignore_index=True)
    if len(shortlist) < top_k:
        used = set(shortlist["feature"].tolist())
        remainder = candidate_table.loc[~candidate_table["feature"].isin(used)].head(top_k - len(shortlist))
        shortlist = pd.concat([shortlist, remainder], ignore_index=True)
    return shortlist


def select_block_shortlist(block_table: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if block_table.empty:
        return block_table
    explicit = block_table.loc[block_table["block_reason"] != "high_correlation_cluster"].copy()
    corr = block_table.loc[block_table["block_reason"] == "high_correlation_cluster"].copy()
    n_explicit = min(len(explicit), max(1, math.ceil(top_k / 2)))
    n_corr = min(len(corr), top_k - n_explicit)
    shortlist = pd.concat([explicit.head(n_explicit), corr.head(n_corr)], ignore_index=True)
    if len(shortlist) < top_k:
        used = set(shortlist["block_name"].tolist())
        remainder = block_table.loc[~block_table["block_name"].isin(used)].head(top_k - len(shortlist))
        shortlist = pd.concat([shortlist, remainder], ignore_index=True)
    return shortlist


def write_markdown_summary(
    output_dir: Path,
    args: argparse.Namespace,
    run_summary: dict[str, object],
) -> None:
    initial = run_summary["initial"]
    final = run_summary["final"]
    lines = [
        "# Feature Pruning",
        "",
        f"- Start timestamp: `{run_summary['started_at']}`",
        f"- Max accepted drops: `{args.n_drops}`",
        f"- Target PRAUC stop: `{args.target_prauc}`",
        f"- Candidate top-K per cycle: `{args.candidate_top_k}`",
        f"- Inner folds: `{args.inner_folds}`",
        f"- Accepted drops: `{run_summary['accepted_drop_count']}`",
        f"- Final feature count: `{run_summary['final_feature_count']}`",
        "",
        "## Initial CV and holdout",
        "",
        f"- Mean CV PRAUC: `{initial['cv']['mean_average_precision']:.6f}`",
        f"- Mean CV balanced accuracy: `{initial['cv']['mean_balanced_accuracy']:.6f}`",
        f"- Mean CV AUROC: `{initial['cv']['mean_auroc']:.6f}`",
        f"- Mean CV Brier: `{initial['cv']['mean_brier']:.6f}`",
        f"- Test PRAUC: `{initial['test']['average_precision']:.6f}`",
        f"- Test balanced accuracy: `{initial['test']['balanced_accuracy']:.6f}`",
        f"- Test AUROC: `{initial['test']['auroc']:.6f}`",
        f"- Test Brier: `{initial['test']['brier']:.6f}`",
        "",
        "## Final CV and holdout",
        "",
        f"- Mean CV PRAUC: `{final['cv']['mean_average_precision']:.6f}`",
        f"- Mean CV balanced accuracy: `{final['cv']['mean_balanced_accuracy']:.6f}`",
        f"- Mean CV AUROC: `{final['cv']['mean_auroc']:.6f}`",
        f"- Mean CV Brier: `{final['cv']['mean_brier']:.6f}`",
        f"- Test PRAUC: `{final['test']['average_precision']:.6f}`",
        f"- Test balanced accuracy: `{final['test']['balanced_accuracy']:.6f}`",
        f"- Test AUROC: `{final['test']['auroc']:.6f}`",
        f"- Test Brier: `{final['test']['brier']:.6f}`",
        "",
        "## Accepted drops",
        "",
    ]
    if run_summary.get("accepted_block_drops"):
        for row in run_summary["accepted_block_drops"]:
            lines.append(
                f"- Block `{row['block_name']}` accepted in cycle `{row['cycle']}` "
                f"({row['block_size']} features): delta PRAUC=`{row['delta_average_precision']:.6f}`, "
                f"delta balanced accuracy=`{row['delta_balanced_accuracy']:.6f}`, "
                f"delta AUROC=`{row['delta_auroc']:.6f}`, "
                f"delta Brier=`{row['delta_brier']:.6f}`"
            )
    if not run_summary["accepted_drops"]:
        if not run_summary.get("accepted_block_drops"):
            lines.append("- None")
    else:
        for row in run_summary["accepted_drops"]:
            lines.append(
                f"- `{row['candidate_feature']}` accepted in cycle `{row['cycle']}`: "
                f"delta PRAUC=`{row['delta_average_precision']:.6f}`, "
                f"delta balanced accuracy=`{row['delta_balanced_accuracy']:.6f}`, "
                f"delta AUROC=`{row['delta_auroc']:.6f}`, "
                f"delta Brier=`{row['delta_brier']:.6f}`"
            )
    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = load_model_ready_with_metadata()
    fit_df = combined.loc[combined["split"].isin(DEFAULT_FIT_SPLITS)].copy().reset_index(drop=True)
    holdout_df = combined.loc[combined["split"] == DEFAULT_HOLDOUT_SPLIT].copy().reset_index(drop=True)

    fit_fixture_df = pd.read_csv(FIXTURE_SPLIT_PATH, parse_dates=["date"])
    fit_fixture_df = fit_fixture_df.loc[fit_fixture_df["split"].isin(DEFAULT_FIT_SPLITS)].copy()
    folds = build_walk_forward_folds(
        fit_fixture_df=fit_fixture_df,
        inner_folds=args.inner_folds,
        min_train_fixtures=args.min_train_fixtures,
        min_val_fixtures=args.min_val_fixtures,
    )

    protected_cols = {"date", "fixture_id", "player_appearance_id", "split", "fixture_order", TARGET_COL}
    active_features = [col for col in combined.columns if col not in protected_cols]

    incumbent = evaluate_feature_set(fit_df, holdout_df, active_features, folds, compute_diagnostics=True)
    run_summary: dict[str, object] = {
        "started_at": run_id,
        "accepted_drop_count": 0,
        "accepted_drops": [],
        "accepted_block_drops": [],
        "initial": {
            "cv": incumbent["summary"],
            "test": incumbent["test_metrics"],
        },
    }

    pruning_ledger: list[dict[str, object]] = []
    block_pruning_ledger: list[dict[str, object]] = []
    cycle = 0
    accepted_drops = 0

    while cycle < args.max_cycles and accepted_drops < args.n_drops:
        cycle += 1

        if args.target_prauc is not None and incumbent["summary"]["mean_average_precision"] >= float(args.target_prauc):
            break

        candidate_table, cluster_summary = build_candidate_table(
            fit_df=fit_df,
            features=active_features,
            diagnostics=incumbent,
            corr_threshold=args.corr_threshold,
        )
        block_table = build_block_table(
            fit_df=fit_df,
            features=active_features,
            diagnostics=incumbent,
            corr_threshold=args.corr_threshold,
        )
        candidate_table.to_csv(output_dir / f"cycle_{cycle:02d}_candidate_table.csv", index=False)
        cluster_summary.to_csv(output_dir / f"cycle_{cycle:02d}_cluster_summary.csv", index=False)
        block_table.to_csv(output_dir / f"cycle_{cycle:02d}_block_table.csv", index=False)

        shortlisted_blocks = select_block_shortlist(block_table, args.candidate_top_k)
        shortlisted = select_shortlist(candidate_table, args.candidate_top_k)
        if shortlisted.empty and shortlisted_blocks.empty:
            break

        accepted_this_cycle = False
        for row in shortlisted_blocks.to_dict("records"):
            block_features = [feature for feature in row["features"] if feature in active_features]
            if not block_features:
                continue
            candidate_features = [feature for feature in active_features if feature not in set(block_features)]
            if not candidate_features:
                continue
            candidate_eval = evaluate_feature_set(
                fit_df=fit_df,
                holdout_df=holdout_df,
                features=candidate_features,
                folds=folds,
                compute_diagnostics=False,
            )
            accepted, deltas = acceptance_decision(
                incumbent_fold_metrics=incumbent["fold_metrics"],
                candidate_fold_metrics=candidate_eval["fold_metrics"],
                args=args,
            )
            ledger_row = {
                "cycle": cycle,
                "block_name": row["block_name"],
                "block_reason": row["block_reason"],
                "block_size": int(row["block_size"]),
                "block_features": "|".join(block_features),
                "mean_weakness_score": float(row["mean_weakness_score"]),
                "mean_abs_shap_sum": float(row["mean_abs_shap_sum"]),
                "perm_delta_prauc_sum": float(row["perm_delta_prauc_sum"]),
                "active_feature_count_before": len(active_features),
                "active_feature_count_after": len(candidate_features),
                "mean_average_precision_before": float(incumbent["summary"]["mean_average_precision"]),
                "mean_average_precision_after": float(candidate_eval["summary"]["mean_average_precision"]),
                "mean_balanced_accuracy_before": float(incumbent["summary"]["mean_balanced_accuracy"]),
                "mean_balanced_accuracy_after": float(candidate_eval["summary"]["mean_balanced_accuracy"]),
                "mean_auroc_before": float(incumbent["summary"]["mean_auroc"]),
                "mean_auroc_after": float(candidate_eval["summary"]["mean_auroc"]),
                "mean_brier_before": float(incumbent["summary"]["mean_brier"]),
                "mean_brier_after": float(candidate_eval["summary"]["mean_brier"]),
                "test_average_precision_before": float(incumbent["test_metrics"]["average_precision"]),
                "test_average_precision_after": float(candidate_eval["test_metrics"]["average_precision"]),
                "test_balanced_accuracy_before": float(incumbent["test_metrics"]["balanced_accuracy"]),
                "test_balanced_accuracy_after": float(candidate_eval["test_metrics"]["balanced_accuracy"]),
                "test_auroc_before": float(incumbent["test_metrics"]["auroc"]),
                "test_auroc_after": float(candidate_eval["test_metrics"]["auroc"]),
                "test_brier_before": float(incumbent["test_metrics"]["brier"]),
                "test_brier_after": float(candidate_eval["test_metrics"]["brier"]),
                "accepted": bool(accepted),
                **deltas,
            }
            block_pruning_ledger.append(ledger_row)

            if accepted:
                active_features = candidate_features
                incumbent = evaluate_feature_set(
                    fit_df=fit_df,
                    holdout_df=holdout_df,
                    features=active_features,
                    folds=folds,
                    compute_diagnostics=True,
                )
                accepted_drops += len(block_features)
                run_summary["accepted_drop_count"] = accepted_drops
                run_summary["accepted_block_drops"].append(ledger_row)
                accepted_this_cycle = True
                break

        if accepted_this_cycle:
            pd.DataFrame(block_pruning_ledger).to_csv(output_dir / "block_pruning_ledger.csv", index=False)
            pd.DataFrame(pruning_ledger).to_csv(output_dir / "pruning_ledger.csv", index=False)
            pd.DataFrame({"feature": active_features}).to_csv(output_dir / "active_features_current.csv", index=False)
            continue

        for row in shortlisted.to_dict("records"):
            candidate_feature = str(row["feature"])
            candidate_features = [feature for feature in active_features if feature != candidate_feature]
            candidate_eval = evaluate_feature_set(
                fit_df=fit_df,
                holdout_df=holdout_df,
                features=candidate_features,
                folds=folds,
                compute_diagnostics=False,
            )
            accepted, deltas = acceptance_decision(
                incumbent_fold_metrics=incumbent["fold_metrics"],
                candidate_fold_metrics=candidate_eval["fold_metrics"],
                args=args,
            )
            ledger_row = {
                "cycle": cycle,
                "candidate_feature": candidate_feature,
                "candidate_cluster": row["cluster_id"],
                "candidate_cluster_reason": row["cluster_reason"],
                "candidate_weakness_score": float(row["weakness_score"]),
                "candidate_mean_abs_shap": float(row["mean_abs_shap"]),
                "candidate_perm_delta_prauc": float(row["perm_delta_prauc"]),
                "candidate_max_abs_corr": float(row["max_abs_corr"]),
                "candidate_vif": float(row["vif"]),
                "active_feature_count_before": len(active_features),
                "mean_average_precision_before": float(incumbent["summary"]["mean_average_precision"]),
                "mean_average_precision_after": float(candidate_eval["summary"]["mean_average_precision"]),
                "mean_balanced_accuracy_before": float(incumbent["summary"]["mean_balanced_accuracy"]),
                "mean_balanced_accuracy_after": float(candidate_eval["summary"]["mean_balanced_accuracy"]),
                "mean_auroc_before": float(incumbent["summary"]["mean_auroc"]),
                "mean_auroc_after": float(candidate_eval["summary"]["mean_auroc"]),
                "mean_brier_before": float(incumbent["summary"]["mean_brier"]),
                "mean_brier_after": float(candidate_eval["summary"]["mean_brier"]),
                "test_average_precision_before": float(incumbent["test_metrics"]["average_precision"]),
                "test_average_precision_after": float(candidate_eval["test_metrics"]["average_precision"]),
                "test_balanced_accuracy_before": float(incumbent["test_metrics"]["balanced_accuracy"]),
                "test_balanced_accuracy_after": float(candidate_eval["test_metrics"]["balanced_accuracy"]),
                "test_auroc_before": float(incumbent["test_metrics"]["auroc"]),
                "test_auroc_after": float(candidate_eval["test_metrics"]["auroc"]),
                "test_brier_before": float(incumbent["test_metrics"]["brier"]),
                "test_brier_after": float(candidate_eval["test_metrics"]["brier"]),
                "accepted": bool(accepted),
                **deltas,
            }
            pruning_ledger.append(ledger_row)

            if accepted:
                active_features = candidate_features
                incumbent = evaluate_feature_set(
                    fit_df=fit_df,
                    holdout_df=holdout_df,
                    features=active_features,
                    folds=folds,
                    compute_diagnostics=True,
                )
                accepted_drops += 1
                run_summary["accepted_drop_count"] = accepted_drops
                run_summary["accepted_drops"].append(ledger_row)
                accepted_this_cycle = True
                break

        pd.DataFrame(pruning_ledger).to_csv(output_dir / "pruning_ledger.csv", index=False)
        pd.DataFrame(block_pruning_ledger).to_csv(output_dir / "block_pruning_ledger.csv", index=False)
        pd.DataFrame({"feature": active_features}).to_csv(output_dir / "active_features_current.csv", index=False)
        if not accepted_this_cycle:
            break

    run_summary["final_feature_count"] = len(active_features)
    run_summary["final"] = {
        "cv": incumbent["summary"],
        "test": incumbent["test_metrics"],
    }

    pd.DataFrame({"feature": active_features}).to_csv(output_dir / "final_selected_features.csv", index=False)
    incumbent["fold_metrics"].to_csv(output_dir / "final_fold_metrics.csv", index=False)
    pd.Series(incumbent["shap_mean"], name="mean_abs_shap").sort_values(ascending=False).to_csv(
        output_dir / "final_shap_mean.csv"
    )
    pd.Series(incumbent["perm_mean"], name="perm_delta_prauc").sort_values(ascending=True).to_csv(
        output_dir / "final_permutation_delta_prauc.csv"
    )
    if not incumbent["corr_matrix"].empty:
        incumbent["corr_matrix"].to_csv(output_dir / "final_correlation_matrix.csv")
    if not incumbent["vif_table"].empty:
        incumbent["vif_table"].to_csv(output_dir / "final_vif_table.csv", index=False)
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    write_markdown_summary(output_dir, args, run_summary)

    print(json.dumps({"output_dir": str(output_dir), **run_summary}, indent=2))


if __name__ == "__main__":
    main()
