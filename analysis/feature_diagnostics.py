from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.model_comparison import (
    CATBOOST_PARAMS,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    impute_splits,
    infer_feature_types,
    load_env_file,
    load_split,
    require_package,
    split_xy,
    to_catboost_frame,
)


DATASET_PATHS = {
    "train": TRAIN_PATH,
    "val": VAL_PATH,
    "test": TEST_PATH,
}
VALID_SPLITS = tuple(DATASET_PATHS.keys()) + ("all",)
OUTPUT_ROOT = Path("output/feature_diagnostics")
RANDOM_STATE = 42


def parse_split_list(raw_value: str) -> list[str]:
    splits = [part.strip() for part in raw_value.split(",") if part.strip()]
    invalid = [split for split in splits if split not in VALID_SPLITS]
    if invalid:
        raise ValueError(f"Unknown split(s): {invalid}. Expected one of {sorted(VALID_SPLITS)}.")
    if not splits:
        raise ValueError("At least one split must be provided.")
    if "all" in splits and len(splits) > 1:
        raise ValueError("Use either 'all' or an explicit comma-separated split list, not both.")
    if splits == ["all"]:
        return list(DATASET_PATHS.keys())
    return splits


def load_combined_splits(split_names: list[str]) -> pd.DataFrame:
    return pd.concat([load_split(DATASET_PATHS[name]) for name in split_names], ignore_index=True)


def sanitize_split_label(split_names: list[str]) -> str:
    return "-".join(split_names)


def transformed_to_raw_feature_name(transformed_name: str, categorical_cols: list[str]) -> str:
    if transformed_name.startswith("numeric__"):
        return transformed_name[len("numeric__") :]
    if transformed_name.startswith("categorical__"):
        raw_tail = transformed_name[len("categorical__") :]
        for col in sorted(categorical_cols, key=len, reverse=True):
            prefix = f"{col}_"
            if raw_tail.startswith(prefix):
                return col
        return raw_tail
    return transformed_name


def fit_catboost_for_diagnostics(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_explain: pd.DataFrame,
    categorical_cols: list[str],
):
    catboost = require_package("catboost", ".venv/bin/pip install catboost")
    params = CATBOOST_PARAMS.copy()
    params["verbose"] = False
    params["allow_writing_files"] = False
    model = catboost.CatBoostClassifier(**params)
    train_frame = to_catboost_frame(x_train, categorical_cols)
    explain_frame = to_catboost_frame(x_explain, categorical_cols)
    model.fit(train_frame, y_train, cat_features=categorical_cols)
    explain_proba = model.predict_proba(explain_frame)[:, 1]
    return model, explain_proba


def compute_shap_values(
    fitted_model,
    x_explain: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    catboost = require_package("catboost", ".venv/bin/pip install catboost")
    explain_frame = to_catboost_frame(x_explain, categorical_cols)
    explain_pool = catboost.Pool(explain_frame, cat_features=categorical_cols)
    contribs = fitted_model.get_feature_importance(explain_pool, type="ShapValues")
    shap_matrix = pd.DataFrame(contribs[:, :-1], columns=list(x_explain.columns), index=x_explain.index)
    bias_term = pd.Series(contribs[:, -1], index=x_explain.index, name="bias")
    raw_shap = shap_matrix.copy()
    raw_importance = raw_shap.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
    transformed_importance = pd.DataFrame(
        {
            "mean_abs_shap": shap_matrix.abs().mean().sort_values(ascending=False),
            "raw_feature": shap_matrix.abs().mean().sort_values(ascending=False).index,
        }
    )

    return raw_shap, raw_importance, transformed_importance, bias_term


def feature_value_frame(x_df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    values = x_df.copy()
    for col in values.columns:
        if col in categorical_cols and not pd.api.types.is_bool_dtype(values[col]):
            values[col] = values[col].astype("category").cat.codes.replace(-1, np.nan)
        elif pd.api.types.is_bool_dtype(values[col]):
            values[col] = values[col].astype(int)
    return values


def numeric_feature_frame(x_df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    numeric_df = x_df.copy()
    for col in categorical_cols:
        if pd.api.types.is_bool_dtype(numeric_df[col]):
            numeric_df[col] = numeric_df[col].astype(int)
    return numeric_df.select_dtypes(include=[np.number]).copy()


def compute_correlation_pairs(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    pairs: list[dict[str, float | str]] = []
    columns = list(corr_matrix.columns)
    for idx, left in enumerate(columns):
        for right in columns[idx + 1 :]:
            corr_value = corr_matrix.loc[left, right]
            if pd.isna(corr_value):
                continue
            pairs.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "corr": float(corr_value),
                    "abs_corr": float(abs(corr_value)),
                }
            )
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def compute_vif_table(numeric_df: pd.DataFrame) -> pd.DataFrame:
    vif_rows: list[dict[str, float | str]] = []
    if numeric_df.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "r2_against_others", "vif"])

    filled = numeric_df.copy()
    for col in filled.columns:
        filled[col] = filled[col].fillna(filled[col].median())

    for feature in filled.columns:
        y = filled[feature].to_numpy()
        x = filled.drop(columns=[feature])
        model = LinearRegression()
        model.fit(x, y)
        r2 = float(model.score(x, y))
        if r2 >= 0.999999:
            vif = np.inf
        else:
            vif = float(1.0 / (1.0 - r2))
        vif_rows.append(
            {
                "feature": feature,
                "r2_against_others": r2,
                "vif": vif,
            }
        )

    return pd.DataFrame(vif_rows).sort_values(["vif", "r2_against_others"], ascending=[False, False]).reset_index(
        drop=True
    )


def build_candidate_table(
    correlation_pairs: pd.DataFrame,
    raw_importance: pd.Series,
    vif_table: pd.DataFrame,
    corr_threshold: float,
) -> pd.DataFrame:
    if correlation_pairs.empty:
        return pd.DataFrame(
            columns=[
                "candidate_drop_feature",
                "candidate_keep_feature",
                "abs_corr",
                "drop_mean_abs_shap",
                "keep_mean_abs_shap",
                "drop_vif",
                "keep_vif",
            ]
        )

    high_corr = correlation_pairs.loc[correlation_pairs["abs_corr"] >= corr_threshold].copy()
    if high_corr.empty:
        return pd.DataFrame(
            columns=[
                "candidate_drop_feature",
                "candidate_keep_feature",
                "abs_corr",
                "drop_mean_abs_shap",
                "keep_mean_abs_shap",
                "drop_vif",
                "keep_vif",
            ]
        )

    vif_lookup = vif_table.set_index("feature")["vif"].to_dict()
    rows = []
    for record in high_corr.to_dict("records"):
        left = record["feature_a"]
        right = record["feature_b"]
        left_shap = float(raw_importance.get(left, 0.0))
        right_shap = float(raw_importance.get(right, 0.0))
        if left_shap <= right_shap:
            drop_feature, keep_feature = left, right
            drop_shap, keep_shap = left_shap, right_shap
        else:
            drop_feature, keep_feature = right, left
            drop_shap, keep_shap = right_shap, left_shap
        rows.append(
            {
                "candidate_drop_feature": drop_feature,
                "candidate_keep_feature": keep_feature,
                "abs_corr": float(record["abs_corr"]),
                "drop_mean_abs_shap": drop_shap,
                "keep_mean_abs_shap": keep_shap,
                "drop_vif": float(vif_lookup.get(drop_feature, np.nan)),
                "keep_vif": float(vif_lookup.get(keep_feature, np.nan)),
            }
        )

    candidate_table = pd.DataFrame(rows).sort_values(
        ["abs_corr", "drop_mean_abs_shap"], ascending=[False, True]
    )
    return candidate_table.drop_duplicates(subset=["candidate_drop_feature", "candidate_keep_feature"]).reset_index(
        drop=True
    )


def save_importance_bar(raw_importance: pd.Series, output_path: Path, top_n: int) -> None:
    top = raw_importance.head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, top.shape[0] * 0.35)))
    ax.barh(top.index, top.values, color="#1f77b4")
    ax.set_xlabel("Mean absolute SHAP contribution")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top.shape[0]} Features By SHAP Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_beeswarm_plot(
    raw_shap: pd.DataFrame,
    feature_values: pd.DataFrame,
    raw_importance: pd.Series,
    output_path: Path,
    top_n: int,
) -> None:
    candidate_features = [
        feature
        for feature in raw_importance.index
        if feature in feature_values.columns and pd.api.types.is_numeric_dtype(feature_values[feature])
    ][:top_n]
    if not candidate_features:
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(candidate_features) * 0.45)))
    cmap = colormaps["coolwarm"]

    for row_idx, feature in enumerate(reversed(candidate_features)):
        values = feature_values[feature]
        shap_values = raw_shap[feature]
        valid = values.notna() & shap_values.notna()
        if valid.sum() == 0:
            continue
        vals = values.loc[valid].to_numpy()
        shap_arr = shap_values.loc[valid].to_numpy()
        spread = np.linspace(-0.28, 0.28, num=len(shap_arr))
        rng = np.random.default_rng(RANDOM_STATE + row_idx)
        rng.shuffle(spread)
        if np.nanmin(vals) == np.nanmax(vals):
            colors = np.repeat(0.5, len(vals))
        else:
            colors = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
        ax.scatter(
            shap_arr,
            row_idx + spread,
            c=colors,
            cmap=cmap,
            s=18,
            alpha=0.7,
            edgecolors="none",
        )

    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(candidate_features)))
    ax.set_yticklabels(list(reversed(candidate_features)))
    ax.set_xlabel("SHAP contribution")
    ax.set_title(f"SHAP Beeswarm For Top {len(candidate_features)} Numeric Features")
    norm = Normalize(vmin=0, vmax=1)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    colorbar.set_label("Relative feature value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_dependence_grid(
    raw_shap: pd.DataFrame,
    feature_values: pd.DataFrame,
    raw_importance: pd.Series,
    output_path: Path,
    top_n: int,
) -> None:
    features = [
        feature
        for feature in raw_importance.index
        if feature in feature_values.columns and pd.api.types.is_numeric_dtype(feature_values[feature])
    ][:top_n]
    if not features:
        return

    ncols = 2
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(5, nrows * 4)), squeeze=False)
    cmap = colormaps["viridis"]

    for ax, feature in zip(axes.flatten(), features):
        values = feature_values[feature]
        shap_values = raw_shap[feature]
        valid = values.notna() & shap_values.notna()
        if valid.sum() == 0:
            ax.set_visible(False)
            continue
        vals = values.loc[valid].to_numpy()
        shap_arr = shap_values.loc[valid].to_numpy()
        point_colors = np.abs(shap_arr)
        ax.scatter(vals, shap_arr, c=point_colors, cmap=cmap, s=20, alpha=0.7, edgecolors="none")
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_title(feature)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("SHAP contribution")

    for ax in axes.flatten()[len(features) :]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Path) -> None:
    if corr_matrix.empty:
        return
    labels = list(corr_matrix.columns)
    size = max(10, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(size, size))
    heatmap = ax.imshow(corr_matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Spearman Correlation Heatmap Of Numeric And Boolean Features")
    fig.colorbar(heatmap, ax=ax, fraction=0.025, pad=0.02, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_top_pairs_bar(top_pairs: pd.DataFrame, output_path: Path, top_n: int) -> None:
    if top_pairs.empty:
        return
    plot_df = top_pairs.head(top_n).iloc[::-1].copy()
    labels = plot_df["feature_a"] + " vs " + plot_df["feature_b"]
    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.4)))
    ax.barh(labels, plot_df["abs_corr"], color="#d62728")
    ax.set_xlabel("Absolute Spearman correlation")
    ax.set_ylabel("Feature pair")
    ax.set_title(f"Top {len(plot_df)} Correlated Feature Pairs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_vif_bar(vif_table: pd.DataFrame, output_path: Path, top_n: int) -> None:
    if vif_table.empty:
        return
    plot_df = vif_table.head(top_n).iloc[::-1].copy()
    plot_values = plot_df["vif"].replace(np.inf, np.nan)
    finite_max = plot_values.dropna().max() if plot_values.notna().any() else 1.0
    plot_values = plot_values.fillna(finite_max * 1.1)
    labels = plot_df["feature"]
    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.4)))
    ax.barh(labels, plot_values, color="#ff7f0e")
    ax.set_xlabel("VIF")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {len(plot_df)} Features By Variance Inflation Factor")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    fit_splits: list[str],
    explain_split: str,
    metrics: dict[str, float],
    raw_importance: pd.Series,
    top_pairs: pd.DataFrame,
    vif_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
) -> None:
    lines = [
        "# Feature Diagnostics",
        "",
        f"- Fit splits: `{','.join(fit_splits)}`",
        f"- Explain split: `{explain_split}`",
        f"- Explain AP: `{metrics['average_precision']:.6f}`",
        f"- Explain AUROC: `{metrics['auroc']:.6f}`",
        f"- Explain Brier: `{metrics['brier']:.6f}`",
        f"- Explain balanced accuracy at threshold 0.5: `{metrics['balanced_accuracy_0_5']:.6f}`",
        "",
        "## Top SHAP features",
        "",
    ]
    for feature, value in raw_importance.head(100).items():
        lines.append(f"- `{feature}`: `{value:.6f}`")

    lines.extend(
        [
            "",
            "## Strongest numeric/boolean correlations",
            "",
        ]
    )
    for row in top_pairs.head(15).to_dict("records"):
        lines.append(
            f"- `{row['feature_a']}` vs `{row['feature_b']}`: corr=`{row['corr']:.4f}`, abs=`{row['abs_corr']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Highest VIF features",
            "",
        ]
    )
    for row in vif_table.head(15).to_dict("records"):
        vif_text = "inf" if np.isinf(row["vif"]) else f"{row['vif']:.4f}"
        lines.append(f"- `{row['feature']}`: vif=`{vif_text}`, r2=`{row['r2_against_others']:.4f}`")

    lines.extend(
        [
            "",
            "## Candidate features to review for elimination",
            "",
            "- Heuristic only: high correlation is not enough by itself. The table below marks the weaker SHAP side of each highly correlated pair.",
            "",
        ]
    )
    if candidate_table.empty:
        lines.append("- No feature pairs crossed the configured correlation threshold.")
    else:
        for row in candidate_table.head(20).to_dict("records"):
            lines.append(
                f"- Drop candidate `{row['candidate_drop_feature']}` over keep `{row['candidate_keep_feature']}`: abs_corr=`{row['abs_corr']:.4f}`, drop_shap=`{row['drop_mean_abs_shap']:.6f}`, keep_shap=`{row['keep_mean_abs_shap']:.6f}`"
            )

    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-splits", default="train,val", help="Comma-separated splits to fit on.")
    parser.add_argument(
        "--explain-split",
        default="test",
        choices=sorted(VALID_SPLITS),
        help="Split to explain. Use 'all' to explain the full current model-ready dataset.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help="Absolute Spearman-correlation threshold used to flag redundancy candidates.",
    )
    parser.add_argument("--top-n", type=int, default=100, help="Top-N features/pairs to show in plots.")
    args = parser.parse_args()

    load_env_file()

    fit_splits = parse_split_list(args.fit_splits)
    explain_split = args.explain_split
    output_dir = OUTPUT_ROOT / f"fit_{sanitize_split_label(fit_splits)}__explain_{explain_split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_df = load_combined_splits(fit_splits)
    if explain_split == "all":
        explain_df = load_combined_splits(list(DATASET_PATHS.keys()))
    else:
        explain_df = load_split(DATASET_PATHS[explain_split])

    x_fit, y_fit = split_xy(fit_df)
    x_explain, y_explain = split_xy(explain_df)

    numeric_cols, categorical_cols = infer_feature_types(x_fit)
    x_fit, x_explain, _ = impute_splits(x_fit, x_explain, x_explain.copy(), numeric_cols, categorical_cols)

    fitted_model, explain_proba = fit_catboost_for_diagnostics(
        x_train=x_fit,
        y_train=y_fit,
        x_explain=x_explain,
        categorical_cols=categorical_cols,
    )

    prediction = (explain_proba >= 0.5).astype(int)
    metrics = {
        "average_precision": float(average_precision_score(y_explain, explain_proba)),
        "auroc": float(roc_auc_score(y_explain, explain_proba)),
        "brier": float(brier_score_loss(y_explain, explain_proba)),
        "balanced_accuracy_0_5": float(balanced_accuracy_score(y_explain, prediction)),
    }

    raw_shap, raw_importance, transformed_importance, bias_term = compute_shap_values(
        fitted_model, x_explain, categorical_cols
    )
    feature_values = feature_value_frame(x_explain, categorical_cols)
    numeric_df = numeric_feature_frame(x_fit, categorical_cols)
    corr_matrix = numeric_df.corr(method="spearman")
    correlation_pairs = compute_correlation_pairs(corr_matrix)
    vif_table = compute_vif_table(numeric_df)
    candidate_table = build_candidate_table(correlation_pairs, raw_importance, vif_table, args.corr_threshold)

    raw_importance.to_frame().to_csv(output_dir / "shap_feature_importance.csv")
    raw_shap.to_csv(output_dir / "shap_values_by_feature.csv", index=False)
    transformed_importance.to_csv(output_dir / "shap_transformed_feature_importance.csv", index=True)
    bias_term.to_frame().to_csv(output_dir / "shap_bias_term.csv", index=False)
    corr_matrix.to_csv(output_dir / "feature_correlation_matrix.csv")
    correlation_pairs.to_csv(output_dir / "top_correlated_pairs.csv", index=False)
    vif_table.to_csv(output_dir / "vif_table.csv", index=False)
    candidate_table.to_csv(output_dir / "candidate_feature_drops.csv", index=False)

    save_importance_bar(raw_importance, output_dir / "shap_importance_bar.png", args.top_n)
    save_beeswarm_plot(raw_shap, feature_values, raw_importance, output_dir / "shap_beeswarm.png", args.top_n)
    save_dependence_grid(raw_shap, feature_values, raw_importance, output_dir / "shap_dependence_grid.png", 6)
    save_correlation_heatmap(corr_matrix, output_dir / "feature_correlation_heatmap.png")
    save_top_pairs_bar(correlation_pairs, output_dir / "top_correlated_pairs.png", args.top_n)
    save_vif_bar(vif_table, output_dir / "vif_top_features.png", args.top_n)

    write_summary(output_dir, fit_splits, explain_split, metrics, raw_importance, correlation_pairs, vif_table, candidate_table)

    metadata = {
        "fit_splits": fit_splits,
        "explain_split": explain_split,
        "corr_threshold": args.corr_threshold,
        "top_n": args.top_n,
        "metrics": metrics,
        "model_name": "catboost",
        "output_dir": str(output_dir),
        "target_col": TARGET_COL,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))
    print("\nTop SHAP features:")
    print(raw_importance.head(args.top_n).round(6).to_string())
    print("\nTop correlated pairs:")
    if correlation_pairs.empty:
        print("No numeric/boolean correlation pairs available.")
    else:
        print(correlation_pairs.head(args.top_n).round(6).to_string(index=False))
    print("\nTop VIF features:")
    if vif_table.empty:
        print("VIF could not be computed.")
    else:
        print(vif_table.head(args.top_n).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
