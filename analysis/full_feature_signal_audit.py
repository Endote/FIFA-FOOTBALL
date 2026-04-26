from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.create_baseline_modeling_dataset import (
    CSV_NULL_TOKENS,
    DATA_DIR,
    PASS_FEATURE_COLS,
    PASS_COUNT_COLS,
    PASS_RATE_COLS,
    PRESSURE_COUNT_COLS,
    PRESSURE_RATE_COLS,
    RUN_COUNT_COLS,
    RUN_DISTANCE_COLS,
    RUN_POSSESSION_COLS,
    RUN_SHARE_COLS,
    SHOT_COUNT_COLS,
    SHOT_RATE_COLS,
    TARGET_COL,
    CHECKPOINT_ORDER,
    CHECKPOINT_TO_ABS_MINUTE,
    apply_row_filters,
    add_team_level_context_features,
    build_pressure_features,
    build_received_pass_features,
    build_run_features,
    build_shot_features,
    carry_forward_cumulative_features,
    is_cumul_feature,
    split_formation_columns,
)


OUTPUT_DIR = Path("output/full_feature_signal_audit")
MODEL_DATASET_PATH = Path("data/baseline_modeling/baseline_all_model_ready.csv")
FEATURE_MANIFEST_PATH = Path("data/baseline_modeling/baseline_feature_manifest.csv")
KNOWN_CATEGORICAL_COLS = {"checkpoint", "position", "formation"}
BASE_EXCLUDE_COLS = {
    TARGET_COL,
    "fixture_id",
    "date",
    "player_id",
    "jersey_number",
    "player_appearance_id",
    "id",
    "minute_in",
    "minute_out",
    "checkpoint_period",
    "checkpoint_min",
    "fixture_order",
    "split",
}
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the current full feature table from raw sources, fit CatBoost on all predictors, "
            "and audit SHAP plus feature-vs-target association."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for CSV and Markdown outputs.",
    )
    return parser.parse_args()


def load_base_table() -> pd.DataFrame:
    base = pd.read_csv(
        DATA_DIR / "players_quarters_final.csv",
        parse_dates=["date"],
        na_values=CSV_NULL_TOKENS,
    )
    base["_checkpoint_order"] = base["checkpoint"].map(CHECKPOINT_ORDER)
    base = base.sort_values(["date", "fixture_id", "player_appearance_id", "_checkpoint_order"]).reset_index(drop=True)
    base = base.drop(columns=["_checkpoint_order"])
    base = split_formation_columns(base)
    base["cumul_in_game_time"] = (
        base["checkpoint"].map(CHECKPOINT_TO_ABS_MINUTE) - base["minute_in"]
    ).clip(lower=0)
    return base


def merge_feature_frame(
    base: pd.DataFrame,
    features: pd.DataFrame,
    count_cols: list[str],
    rate_cols: list[str],
) -> pd.DataFrame:
    merged = base.merge(features, on=["player_appearance_id", "checkpoint"], how="left")
    merged = carry_forward_cumulative_features(
        merged,
        [col for col in features.columns if is_cumul_feature(col)],
    )
    for col in count_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    for col in rate_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    return merged


def assemble_full_feature_dataset() -> pd.DataFrame:
    base = load_base_table()

    received_pass_features = build_received_pass_features()
    base = merge_feature_frame(
        base=base,
        features=received_pass_features,
        count_cols=PASS_COUNT_COLS,
        rate_cols=PASS_RATE_COLS,
    )

    pressure_features = build_pressure_features()
    base = merge_feature_frame(
        base=base,
        features=pressure_features,
        count_cols=PRESSURE_COUNT_COLS,
        rate_cols=PRESSURE_RATE_COLS,
    )

    run_features = build_run_features()
    base = merge_feature_frame(
        base=base,
        features=run_features,
        count_cols=RUN_COUNT_COLS,
        rate_cols=RUN_SHARE_COLS + RUN_DISTANCE_COLS + RUN_POSSESSION_COLS,
    )

    shot_features = build_shot_features()
    base = merge_feature_frame(
        base=base,
        features=shot_features,
        count_cols=SHOT_COUNT_COLS,
        rate_cols=SHOT_RATE_COLS,
    )

    base = add_team_level_context_features(base)
    filtered_base, removed_rows = apply_row_filters(base)
    return filtered_base, removed_rows


def select_predictor_columns(df: pd.DataFrame) -> list[str]:
    keep = [col for col in df.columns if col not in BASE_EXCLUDE_COLS]
    return keep


def normalize_predictor_dtypes(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        if col in KNOWN_CATEGORICAL_COLS:
            out[col] = out[col].astype("string")
            continue
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() == out[col].notna().sum():
                out[col] = converted
    return out


def infer_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols: list[str] = []
    for col in x.columns:
        if col in KNOWN_CATEGORICAL_COLS or isinstance(x[col].dtype, pd.CategoricalDtype):
            categorical_cols.append(col)
    numeric_cols = [col for col in x.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def impute_predictors(
    x: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    out = x.copy()
    for col in numeric_cols:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(int)
        fill_value = out[col].median() if out[col].notna().any() else 0
        out[col] = out[col].fillna(fill_value)
    for col in categorical_cols:
        mode = out[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "missing"
        out[col] = out[col].fillna(fill_value).astype(str)
    return out


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(x, y)
    if contingency.empty:
        return float("nan")
    chi2 = chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    if n <= 1:
        return float("nan")
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return float("nan")
    return float(math.sqrt(phi2corr / denom))


def compute_target_associations(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in x.columns:
        series = x[col]
        if (
            col in KNOWN_CATEGORICAL_COLS or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            assoc = cramers_v(series.astype(str), y)
            rows.append(
                {
                    "feature": col,
                    "association_method": "cramers_v",
                    "target_association": assoc,
                    "abs_target_association": abs(assoc) if pd.notna(assoc) else np.nan,
                }
            )
        else:
            numeric = series.astype(float)
            assoc = spearmanr(numeric, y, nan_policy="omit").statistic
            rows.append(
                {
                    "feature": col,
                    "association_method": "spearman",
                    "target_association": float(assoc),
                    "abs_target_association": float(abs(assoc)),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_target_association", ascending=False).reset_index(drop=True)


def fit_catboost(x: pd.DataFrame, y: pd.Series, categorical_cols: list[str]):
    import catboost

    model = catboost.CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=2000,
        learning_rate=0.03,
        depth=3,
        l2_leaf_reg=10,
        random_strength=1.0,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y, cat_features=categorical_cols)
    return model


def compute_shap_importance(model, x: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    import catboost

    pool = catboost.Pool(x, cat_features=categorical_cols)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    shap_df = pd.DataFrame(shap_values[:, :-1], columns=x.columns, index=x.index)
    importance = (
        shap_df.abs().mean().rename("mean_abs_shap").sort_values(ascending=False).reset_index()
    )
    importance.columns = ["feature", "mean_abs_shap"]
    return importance, shap_df


def build_summary(
    feature_table: pd.DataFrame,
    removed_rows: pd.DataFrame,
    associations: pd.DataFrame,
    shap_importance: pd.DataFrame,
) -> str:
    merged = shap_importance.merge(associations, on="feature", how="left")
    shap_lookup = shap_importance.set_index("feature")["mean_abs_shap"]

    lines = [
        "# Full Feature Signal Audit",
        "",
        f"- Rows after quality filters: `{len(feature_table)}`",
        f"- Rows removed by quality filters: `{len(removed_rows)}`",
        f"- Predictors audited: `{shap_importance['feature'].nunique()}`",
        "",
        "## Top 15 By SHAP",
        "",
    ]
    for row in merged.head(15).to_dict("records"):
        lines.append(
            f"- `{row['feature']}`: shap=`{row['mean_abs_shap']:.6f}`, "
            f"{row['association_method']}=`{row['target_association']:.6f}`"
        )

    lines.extend(["", "## Top 15 By Target Association", ""])
    for row in associations.head(15).to_dict("records"):
        shap_value = float(shap_lookup.get(row["feature"], np.nan))
        lines.append(
            f"- `{row['feature']}`: {row['association_method']}=`{row['target_association']:.6f}`, "
            f"shap=`{shap_value:.6f}`"
        )

    lines.extend(["", "## Bottom 15 By SHAP", ""])
    for row in merged.tail(15).to_dict("records"):
        lines.append(
            f"- `{row['feature']}`: shap=`{row['mean_abs_shap']:.6f}`, "
            f"{row['association_method']}=`{row['target_association']:.6f}`"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_table, removed_rows = assemble_full_feature_dataset()
    predictor_cols = select_predictor_columns(feature_table)

    x = feature_table[predictor_cols].copy()
    y = feature_table[TARGET_COL].astype(int)

    x = normalize_predictor_dtypes(x)
    numeric_cols, categorical_cols = infer_feature_types(x)
    x = impute_predictors(x, numeric_cols, categorical_cols)

    target_associations = compute_target_associations(x, y)
    model = fit_catboost(x, y, categorical_cols)
    shap_importance, shap_values = compute_shap_importance(model, x, categorical_cols)
    combined = shap_importance.merge(target_associations, on="feature", how="left")

    target_associations.to_csv(output_dir / "target_associations.csv", index=False)
    shap_importance.to_csv(output_dir / "catboost_shap_importance.csv", index=False)
    combined.to_csv(output_dir / "catboost_shap_with_target_association.csv", index=False)
    shap_values.to_csv(output_dir / "catboost_shap_values_by_feature.csv", index=False)
    removed_rows.to_csv(output_dir / "removed_rows_quality_filters.csv", index=False)

    audit_predictor_cols = sorted(predictor_cols)
    pd.DataFrame({"feature": audit_predictor_cols}).to_csv(output_dir / "audit_predictor_columns.csv", index=False)

    comparison_payload: dict[str, object] = {}
    if MODEL_DATASET_PATH.exists():
        exported_columns = pd.read_csv(MODEL_DATASET_PATH, nrows=0).columns.tolist()
        exported_predictors = sorted([col for col in exported_columns if col != TARGET_COL])
        comparison_payload["exported_predictor_count"] = len(exported_predictors)
        comparison_payload["audit_only_predictors"] = sorted(set(audit_predictor_cols) - set(exported_predictors))
        comparison_payload["exported_only_predictors"] = sorted(set(exported_predictors) - set(audit_predictor_cols))
        pd.DataFrame({"feature": exported_predictors}).to_csv(
            output_dir / "exported_model_predictor_columns.csv", index=False
        )

    if FEATURE_MANIFEST_PATH.exists():
        manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
        if {"column", "role"}.issubset(manifest.columns):
            manifest_predictors = sorted(manifest.loc[manifest["role"] == "predictor", "column"].astype(str).tolist())
            comparison_payload["manifest_predictor_count"] = len(manifest_predictors)
            comparison_payload["audit_only_vs_manifest"] = sorted(set(audit_predictor_cols) - set(manifest_predictors))
            comparison_payload["manifest_only_vs_audit"] = sorted(set(manifest_predictors) - set(audit_predictor_cols))
            pd.DataFrame({"feature": manifest_predictors}).to_csv(
                output_dir / "manifest_predictor_columns.csv", index=False
            )

    (output_dir / "summary.md").write_text(
        build_summary(feature_table, removed_rows, target_associations, shap_importance)
    )

    metadata = {
        "output_dir": str(output_dir),
        "rows_after_quality_filters": int(len(feature_table)),
        "rows_removed_by_quality_filters": int(len(removed_rows)),
        "predictor_count": int(len(predictor_cols)),
        "categorical_predictors": categorical_cols,
        "numeric_predictors": numeric_cols,
        **comparison_payload,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))
    print("\nTop 10 SHAP features:")
    print(shap_importance.head(50).to_string(index=False))
    print("\nTop 10 target associations:")
    print(target_associations.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
