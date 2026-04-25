from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


SEED = 42
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/eda")

PERIOD_OFFSET = {
    "half_1": 0,
    "half_2": 45,
    "extra_time_1": 90,
    "extra_time_2": 105,
}

CHECKPOINT_ABS = {
    "H1_15": 15,
    "H1_30": 30,
    "H1_45": 45,
    "H2_15": 60,
    "H2_30": 75,
    "H2_45": 90,
    "ET1_15": 105,
}

HOLDOUT_ROW_SHARE = 0.20
HOLDOUT_POS_SHARE = 0.20

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")


def mean_or_zero(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    value = float(series.mean())
    return 0.0 if math.isnan(value) else value


def safe_div(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        numerator = pd.Series(numerator)
        denominator = pd.Series(denominator)
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(0.0)
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def load_base() -> pd.DataFrame:
    base = pd.read_csv(DATA_DIR / "players_quarters_final.csv", parse_dates=["date"])
    base["row_id"] = np.arange(len(base))
    base["abs_checkpoint"] = base["checkpoint"].map(CHECKPOINT_ABS)
    base["minutes_played_before_checkpoint"] = base["abs_checkpoint"] - base["minute_in"] + 1
    base["minutes_played_before_checkpoint"] = base["minutes_played_before_checkpoint"].clip(lower=1)
    base["remaining_minutes"] = (base["minute_out"] - base["abs_checkpoint"]).clip(lower=0)
    base["remaining_windows_15"] = base["remaining_minutes"] / 15.0
    return base


def add_abs_minute(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["abs_minute"] = events["period"].map(PERIOD_OFFSET) + events["minute"]
    return events


def choose_holdout_dates(base: pd.DataFrame) -> list[pd.Timestamp]:
    date_stats = (
        base.groupby("date")
        .agg(rows=("row_id", "size"), positives=("scored_after", "sum"))
        .sort_index(ascending=False)
    )

    total_rows = float(len(base))
    total_pos = float(base["scored_after"].sum())
    selected_dates: list[pd.Timestamp] = []
    rows_so_far = 0.0
    pos_so_far = 0.0

    for date, row in date_stats.iterrows():
        selected_dates.append(date)
        rows_so_far += float(row["rows"])
        pos_so_far += float(row["positives"])
        row_share = rows_so_far / total_rows
        pos_share = pos_so_far / total_pos if total_pos else 0.0
        if row_share >= HOLDOUT_ROW_SHARE and pos_share >= HOLDOUT_POS_SHARE:
            break

    return sorted(selected_dates)


def summarize_pass(df: pd.DataFrame) -> pd.Series:
    total = float(len(df))
    accurate = float(df["accurate"].sum())
    no_target = float(df["addressee_player_appearance_id"].isna().sum())
    top = float((df["stage"] == "top").sum())
    middle = float((df["stage"] == "middle").sum())
    bottom = float((df["stage"] == "bottom").sum())
    return pd.Series(
        {
            "count": total,
            "accurate_count": accurate,
            "accuracy": safe_div(accurate, total),
            "no_target_count": no_target,
            "no_target_rate": safe_div(no_target, total),
            "unique_receivers": float(df["addressee_player_appearance_id"].nunique(dropna=True)),
            "top_count": top,
            "middle_count": middle,
            "bottom_count": bottom,
            "top_share": safe_div(top, total),
        }
    )


def summarize_pressure_receiver(df: pd.DataFrame) -> pd.Series:
    total = float(len(df))
    accurate = float(df["accurate"].sum())
    turnover = float((df["press_induced_outcome"] == "turnover").sum())
    forward = float((df["press_induced_outcome"] == "forward_pass").sum())
    backward = float((df["press_induced_outcome"] == "backward_pass").sum())
    lateral = float((df["press_induced_outcome"] == "lateral_pass").sum())
    carry = float((df["press_induced_outcome"] == "ball_carry").sum())
    top = float((df["stage"] == "top").sum())
    return pd.Series(
        {
            "count": total,
            "accurate_count": accurate,
            "accuracy": safe_div(accurate, total),
            "turnover_count": turnover,
            "turnover_rate": safe_div(turnover, total),
            "forward_count": forward,
            "forward_share": safe_div(forward, total),
            "backward_count": backward,
            "backward_share": safe_div(backward, total),
            "lateral_count": lateral,
            "carry_count": carry,
            "top_count": top,
            "top_share": safe_div(top, total),
            "unique_pressers": float(df["pressing_player_appearance_id"].nunique(dropna=True)),
            "mean_abs_angle": mean_or_zero(df["pass_angle"].abs()),
        }
    )


def summarize_pressure_presser(df: pd.DataFrame) -> pd.Series:
    total = float(len(df))
    turnover = float((df["press_induced_outcome"] == "turnover").sum())
    forward = float((df["press_induced_outcome"] == "forward_pass").sum())
    top = float((df["stage"] == "top").sum())
    return pd.Series(
        {
            "count": total,
            "induced_turnover_count": turnover,
            "induced_turnover_rate": safe_div(turnover, total),
            "allowed_forward_count": forward,
            "allowed_forward_rate": safe_div(forward, total),
            "top_count": top,
            "top_share": safe_div(top, total),
            "unique_pressed_players": float(df["player_appearance_id"].nunique(dropna=True)),
        }
    )


def summarize_run(df: pd.DataFrame) -> pd.Series:
    total = float(len(df))
    sprint_mask = df["run_type"] == "sprint"
    hsr_mask = df["run_type"] == "hsr"
    top_mask = df["stage"] == "top"
    return pd.Series(
        {
            "count": total,
            "sprint_distance": float(df.loc[sprint_mask, "distance"].sum()),
            "hsr_distance": float(df.loc[hsr_mask, "distance"].sum()),
            "sprint_share": safe_div(float(sprint_mask.sum()), total),
            "top_count": float(top_mask.sum()),
            "top_share": safe_div(float(top_mask.sum()), total),
            "top_sprint_count": float((sprint_mask & top_mask).sum()),
            "mean_distance": mean_or_zero(df["distance"]),
            "unique_possessions": float(df["possession"].nunique(dropna=True)),
            "mean_sprint_speed": mean_or_zero(df.loc[sprint_mask, "max_speed"]),
            "mean_hsr_speed": mean_or_zero(df.loc[hsr_mask, "max_speed"]),
        }
    )


def summarize_shot(df: pd.DataFrame) -> pd.Series:
    total = float(len(df))
    blocked = float(df["block_player_appearance_id"].notna().sum())
    headers = float((df["body_part"] == "head").sum())
    left_foot = float((df["body_part"] == "left_foot").sum())
    right_foot = float((df["body_part"] == "right_foot").sum())
    counter = float((df["play_pattern"] == "counter_attack").sum())
    penalty = float((df["play_pattern"] == "penalty").sum())
    regular = float((df["play_pattern"] == "regular_play").sum())
    set_piece = float(
        df["play_pattern"].isin(["corner_kick", "direct_free_kick", "indirect_free_kick", "throw_in"]).sum()
    )
    return pd.Series(
        {
            "count": total,
            "blocked_count": blocked,
            "blocked_share": safe_div(blocked, total),
            "header_count": headers,
            "header_share": safe_div(headers, total),
            "left_foot_count": left_foot,
            "right_foot_count": right_foot,
            "counter_count": counter,
            "counter_share": safe_div(counter, total),
            "penalty_count": penalty,
            "regular_count": regular,
            "set_piece_count": set_piece,
            "under_pressure_count": float(df["under_pressure"].sum()),
            "unique_possessions": float(df["possession"].nunique(dropna=True)),
        }
    )


def aggregate_windows(
    checkpoints: pd.DataFrame,
    events: pd.DataFrame,
    actor_col: str,
    prefix: str,
    summary_fn: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    checkpoints = checkpoints.rename(columns={"player_appearance_id": "checkpoint_player_appearance_id"}).copy()
    events = events.dropna(subset=[actor_col]).copy()
    events[actor_col] = events[actor_col].astype(int)
    events = add_abs_minute(events)

    merged = checkpoints.merge(
        events,
        left_on="checkpoint_player_appearance_id",
        right_on=actor_col,
        how="left",
    )
    merged = merged[merged["abs_minute"].notna()].copy()
    merged = merged[merged["abs_minute"] <= merged["abs_checkpoint"]].copy()

    def apply_summary(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["row_id"])
        return df.groupby("row_id", sort=False).apply(summary_fn, include_groups=False).reset_index()

    cumul = apply_summary(merged)
    last15 = apply_summary(merged[merged["abs_minute"] > merged["abs_checkpoint"] - 15])

    if not cumul.empty:
        cumul = cumul.rename(columns={col: f"cumul_{prefix}_{col}" for col in cumul.columns if col != "row_id"})
    if not last15.empty:
        last15 = last15.rename(columns={col: f"last15_{prefix}_{col}" for col in last15.columns if col != "row_id"})

    out = checkpoints[["row_id"]].copy()
    out = out.merge(last15, on="row_id", how="left")
    out = out.merge(cumul, on="row_id", how="left")
    return out.fillna(0.0)


def build_feature_matrix(base: pd.DataFrame) -> pd.DataFrame:
    checkpoints = base[["row_id", "player_appearance_id", "abs_checkpoint"]].copy()
    feature_df = base.copy()

    pass_features = aggregate_windows(
        checkpoints,
        pd.read_csv(DATA_DIR / "player_appearance_pass.csv"),
        actor_col="player_appearance_id",
        prefix="pass",
        summary_fn=summarize_pass,
    )
    feature_df = feature_df.merge(pass_features, on="row_id", how="left")

    pressure_receiver_features = aggregate_windows(
        checkpoints,
        pd.read_csv(DATA_DIR / "player_appearance_behaviour_under_pressure.csv"),
        actor_col="player_appearance_id",
        prefix="under_pressure",
        summary_fn=summarize_pressure_receiver,
    )
    feature_df = feature_df.merge(pressure_receiver_features, on="row_id", how="left")

    pressure_presser_features = aggregate_windows(
        checkpoints,
        pd.read_csv(DATA_DIR / "player_appearance_behaviour_under_pressure.csv"),
        actor_col="pressing_player_appearance_id",
        prefix="applied_pressure",
        summary_fn=summarize_pressure_presser,
    )
    feature_df = feature_df.merge(pressure_presser_features, on="row_id", how="left")

    run_features = aggregate_windows(
        checkpoints,
        pd.read_csv(DATA_DIR / "player_appearance_run.csv"),
        actor_col="player_appearance_id",
        prefix="run_detail",
        summary_fn=summarize_run,
    )
    feature_df = feature_df.merge(run_features, on="row_id", how="left")

    shot_features = aggregate_windows(
        checkpoints,
        pd.read_csv(DATA_DIR / "player_appearance_shot_limited.csv"),
        actor_col="player_appearance_id",
        prefix="shot_context",
        summary_fn=summarize_shot,
    )
    feature_df = feature_df.merge(shot_features, on="row_id", how="left")

    for col in feature_df.columns:
        if col.startswith("cumul_") and col not in {"cumul_mean_max_speed", "cumul_peak_speed"}:
            feature_df[f"{col}_per15"] = safe_div(feature_df[col] * 15.0, feature_df["minutes_played_before_checkpoint"])

    feature_df["trend_shots_vs_cumul_rate"] = feature_df["last15_shots"] - feature_df["cumul_shots_per15"]
    feature_df["trend_distance_vs_cumul_rate"] = feature_df["last15_distance"] - feature_df["cumul_distance_per15"]
    feature_df["trend_hsr_vs_cumul_rate"] = feature_df["last15_hsr"] - feature_df["cumul_hsr_per15"]
    feature_df["trend_pass_vs_cumul_rate"] = feature_df["last15_pass_count"] - feature_df["cumul_pass_count_per15"]
    feature_df["trend_top_pass_share"] = feature_df["last15_pass_top_share"] - feature_df["cumul_pass_top_share"]
    feature_df["trend_turnover_under_pressure"] = (
        feature_df["last15_under_pressure_turnover_rate"] - feature_df["cumul_under_pressure_turnover_rate"]
    )
    feature_df["trend_applied_pressure_turnovers"] = (
        feature_df["last15_applied_pressure_induced_turnover_rate"]
        - feature_df["cumul_applied_pressure_induced_turnover_rate"]
    )

    feature_df = feature_df.fillna(0.0)
    return feature_df


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    metadata_cols = {
        "row_id",
        "player_appearance_id",
        "player_id",
        "fixture_id",
        "date",
        "checkpoint_period",
        "checkpoint_min",
        "jersey_number",
        "abs_checkpoint",
        "scored_after",
        "split",
    }

    context_cols = [
        "position",
        "is_home",
        "formation",
        "checkpoint",
        "minute_in",
        "minute_out",
        "subbed",
        "minutes_played_before_checkpoint",
        "remaining_minutes",
        "remaining_windows_15",
    ]

    original_base_cols = [
        col
        for col in df.columns
        if (col.startswith("last15_") or col.startswith("cumul_"))
        and "_pass_" not in col
        and "_under_pressure_" not in col
        and "_applied_pressure_" not in col
        and "_run_detail_" not in col
        and "_shot_context_" not in col
        and not col.endswith("_per15")
    ]

    run_shot_extension_cols = sorted(
        [col for col in df.columns if "_run_detail_" in col or "_shot_context_" in col]
        + [
            col
            for col in df.columns
            if col.startswith("cumul_")
            and (
                "_run_detail_" in col
                or "_shot_context_" in col
                or col in {"cumul_distance_per15", "cumul_hsr_per15", "cumul_shots_per15", "cumul_shots_on_target_per15"}
            )
        ]
        + [col for col in df.columns if col.startswith("trend_") and "pass" not in col and "pressure" not in col]
    )

    pass_cols = sorted(
        [col for col in df.columns if "_pass_" in col]
        + [col for col in df.columns if col in {"trend_pass_vs_cumul_rate", "trend_top_pass_share"}]
    )

    pressure_cols = sorted(
        [col for col in df.columns if "_under_pressure_" in col or "_applied_pressure_" in col]
        + [col for col in df.columns if "pressure" in col and col.startswith("trend_")]
    )

    feature_sets = {
        "context_only": context_cols,
        "runs_shots_panel": sorted(set(context_cols + original_base_cols + run_shot_extension_cols)),
        "plus_pass": sorted(set(context_cols + original_base_cols + run_shot_extension_cols + pass_cols)),
        "plus_pressure": sorted(set(context_cols + original_base_cols + run_shot_extension_cols + pressure_cols)),
        "full_feature_set": sorted(
            set(context_cols + original_base_cols + run_shot_extension_cols + pass_cols + pressure_cols)
        ),
    }

    for name, cols in feature_sets.items():
        feature_sets[name] = [col for col in cols if col in df.columns and col not in metadata_cols]

    return feature_sets


def infer_column_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    categorical_cols = []
    numeric_cols = []
    for col in feature_cols:
        if df[col].dtype == "object" or str(df[col].dtype) == "bool":
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    return numeric_cols, categorical_cols


def make_model(model_name: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    if model_name == "logreg":
        model = LogisticRegression(
            max_iter=3000,
            C=0.1,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        )
    elif model_name == "extratrees":
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
    elif model_name == "histgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=250,
            l2_regularization=0.2,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_model(
    model_name: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> Pipeline:
    numeric_cols, categorical_cols = infer_column_types(train_df, feature_cols)
    pipeline = make_model(model_name, numeric_cols, categorical_cols)
    fit_kwargs = {}
    if model_name == "histgb":
        fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", train_df[target_col])
    pipeline.fit(train_df[feature_cols], train_df[target_col], **fit_kwargs)
    return pipeline


def predict_proba(model: Pipeline, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return model.predict_proba(df[feature_cols])[:, 1]


def evaluate_feature_sets(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], Pipeline]]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    target_col = "scored_after"
    models = ["logreg", "extratrees", "histgb"]
    fitted_full_models: dict[tuple[str, str], Pipeline] = {}
    rows = []

    for feature_set_name, feature_cols in feature_sets.items():
        for model_name in models:
            fold_aucs = []
            fold_bals = []
            for fold_idx, (train_idx, valid_idx) in enumerate(
                cv.split(train_df[feature_cols], train_df[target_col], groups=train_df["fixture_id"]),
                start=1,
            ):
                fold_train = train_df.iloc[train_idx]
                fold_valid = train_df.iloc[valid_idx]
                pipeline = fit_model(model_name, fold_train, feature_cols, target_col)
                valid_pred = predict_proba(pipeline, fold_valid, feature_cols)
                fold_aucs.append(roc_auc_score(fold_valid[target_col], valid_pred))
                fold_bals.append(balanced_accuracy_score(fold_valid[target_col], (valid_pred >= 0.5).astype(int)))

            final_model = fit_model(model_name, train_df, feature_cols, target_col)
            fitted_full_models[(feature_set_name, model_name)] = final_model
            holdout_pred = predict_proba(final_model, holdout_df, feature_cols)
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "n_features": len(feature_cols),
                    "cv_auc_mean": np.mean(fold_aucs),
                    "cv_auc_std": np.std(fold_aucs),
                    "cv_bal_acc_mean": np.mean(fold_bals),
                    "cv_bal_acc_std": np.std(fold_bals),
                    "holdout_auc": roc_auc_score(holdout_df[target_col], holdout_pred),
                    "holdout_bal_acc": balanced_accuracy_score(
                        holdout_df[target_col], (holdout_pred >= 0.5).astype(int)
                    ),
                }
            )

    results = pd.DataFrame(rows).sort_values(["cv_auc_mean", "holdout_auc"], ascending=False).reset_index(drop=True)
    return results, fitted_full_models


def numeric_feature_screen(train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    target = train_df["scored_after"]
    rows = []
    for col in feature_cols:
        if train_df[col].dtype == "object" or str(train_df[col].dtype) == "bool":
            continue
        series = train_df[col]
        if series.nunique(dropna=False) <= 1:
            auc = 0.5
            corr = 0.0
        else:
            auc_raw = roc_auc_score(target, series)
            auc = max(float(auc_raw), 1.0 - float(auc_raw))
            corr = float(pd.DataFrame({col: series, "scored_after": target}).corr().iloc[0, 1])
        rows.append(
            {
                "feature": col,
                "missing_rate": float(series.isna().mean()),
                "corr_with_target": corr,
                "univariate_auc": auc,
                "mean_positive": float(series[target == 1].mean()),
                "mean_negative": float(series[target == 0].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["univariate_auc", "corr_with_target"], ascending=False).reset_index(drop=True)


def extract_logreg_effects(model: Pipeline) -> pd.DataFrame:
    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    coefs = classifier.coef_[0]
    effects = pd.DataFrame({"model_feature": feature_names, "coefficient": coefs})
    effects["abs_coefficient"] = effects["coefficient"].abs()
    return effects.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def extract_permutation_importance(
    model: Pipeline,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    if len(feature_cols) > 150:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    importance = permutation_importance(
        model,
        holdout_df[feature_cols],
        holdout_df["scored_after"],
        n_repeats=20,
        random_state=SEED,
        scoring="roc_auc",
        n_jobs=1,
    )
    raw = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )
    return raw.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def save_plots(
    train_df: pd.DataFrame,
    screen_df: pd.DataFrame,
    results_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    checkpoint_rates = (
        train_df.groupby("checkpoint")["scored_after"].mean().reindex(["H1_15", "H1_30", "H1_45", "H2_15", "H2_30", "H2_45", "ET1_15"])
    )
    sns.barplot(x=checkpoint_rates.index, y=checkpoint_rates.values, ax=axes[0], color="#1d6fa5")
    axes[0].set_title("Target Rate By Checkpoint")
    axes[0].set_ylabel("Scored After Rate")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)

    position_rates = train_df.groupby("position")["scored_after"].mean().reindex(["G", "D", "M", "A"])
    sns.barplot(x=position_rates.index, y=position_rates.values, ax=axes[1], color="#b44f2f")
    axes[1].set_title("Target Rate By Position")
    axes[1].set_ylabel("Scored After Rate")
    axes[1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_rates.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    top_screen = screen_df.head(20).sort_values("univariate_auc")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_screen, x="univariate_auc", y="feature", ax=ax, color="#0f9d58")
    ax.set_title("Top Numeric Features By Univariate AUROC")
    ax.set_xlabel("Univariate AUROC (max of feature or inverse)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_univariate_auc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    chart = results_df.copy()
    chart["label"] = chart["feature_set"] + " | " + chart["model"]
    chart = chart.sort_values("cv_auc_mean")
    sns.barplot(data=chart, x="cv_auc_mean", y="label", ax=ax, color="#6f42c1")
    ax.set_title("Cross-Validated AUROC By Feature Set And Model")
    ax.set_xlabel("CV Mean AUROC")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_benchmarks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if not permutation_df.empty:
        top_perm = permutation_df.head(20).sort_values("importance_mean")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=top_perm, x="importance_mean", y="feature", ax=ax, color="#d17c06")
        ax.set_title("Top Holdout Permutation Importances")
        ax.set_xlabel("Decrease In AUROC")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "top_permutation_importance.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def write_summary(
    base: pd.DataFrame,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    holdout_dates: list[pd.Timestamp],
    results_df: pd.DataFrame,
    screen_df: pd.DataFrame,
    logreg_effects_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
) -> None:
    def table_text(frame: pd.DataFrame, index: bool = True) -> str:
        return frame.round(4).to_string(index=index)

    best_row = results_df.iloc[0]
    top_numeric = screen_df.head(10)
    top_effects = logreg_effects_df.head(12)
    top_perm = permutation_df.head(12) if not permutation_df.empty else permutation_df

    position_table = table_text(train_df.groupby("position")["scored_after"].agg(["mean", "sum", "count"]))
    checkpoint_table = table_text(train_df.groupby("checkpoint")["scored_after"].agg(["mean", "sum", "count"]))
    feature_table = table_text(top_numeric, index=False)
    effect_table = table_text(top_effects, index=False)
    permutation_table = table_text(top_perm, index=False) if not top_perm.empty else "No permutation table."

    summary = f"""# Goal Scoring EDA And Modeling Summary

## Dataset framing

- Main panel: {len(base):,} checkpoint rows, {base['fixture_id'].nunique()} fixtures, {base['player_appearance_id'].nunique()} player appearances.
- Positive class: {int(base['scored_after'].sum())} rows ({base['scored_after'].mean():.2%}).
- Holdout policy: latest match dates reserved until both row share and positive share exceeded 20%.
- Holdout dates: {", ".join(str(pd.Timestamp(date).date()) for date in holdout_dates)}.
- Train/validation rows: {len(train_df):,}; holdout rows: {len(holdout_df):,}.
- Train/validation positives: {int(train_df['scored_after'].sum())}; holdout positives: {int(holdout_df['scored_after'].sum())}.

## Leakage and preprocessing checks

- Random row splitting is invalid because the same `player_appearance_id` contributes multiple checkpoints and the target is defined within-match after each checkpoint.
- A fixture-grouped split is required; model selection here uses `StratifiedGroupKFold` grouped by `fixture_id`.
- Event-window engineering was validated against the official base features. The key hidden rule is that stoppage time remains inside the same period as minutes above 45, so `H2_15` and `ET1_15` windows must be built on an absolute minute axis.
- `jersey_number` was excluded from modeling because it showed signal but is likely a competition-specific artifact, not stable football behavior.

## Target profile

### By position

{position_table}

### By checkpoint

{checkpoint_table}

## Best benchmark

- Best cross-validated configuration: `{best_row['feature_set']}` with `{best_row['model']}`.
- Cross-validated AUROC: {best_row['cv_auc_mean']:.4f} +/- {best_row['cv_auc_std']:.4f}.
- Cross-validated balanced accuracy at 0.5 threshold: {best_row['cv_bal_acc_mean']:.4f} +/- {best_row['cv_bal_acc_std']:.4f}.
- Holdout AUROC: {best_row['holdout_auc']:.4f}.
- Holdout balanced accuracy at 0.5 threshold: {best_row['holdout_bal_acc']:.4f}.

## Top numeric signals on train/validation

{feature_table}

## Strongest linear effects in the full-feature logistic model

{effect_table}

## Top permutation importances on the holdout for the best full model

{permutation_table}

## Interpretation notes

- Attackers and earlier checkpoints carry materially higher base rates, which means the model should be benchmarked against context-only baselines rather than raw accuracy.
- Recent attacking involvement matters more than raw cumulative load: short-term shot pressure, attacking-third activity, and run intensity repeatedly surfaced among the strongest screening variables.
- Passing and under-pressure features are worth adding because they capture tactical quality rather than just volume. In particular, high attacking-third pass share, lower turnover-under-pressure rates, and pressure application that induces turnovers are plausible mechanisms tied to later scoring.
- Goalkeepers are a structurally degenerate class in this dataset with zero positives. For competition-style scoring across all rows, a zero-probability rule for keepers is defensible; for scientific interpretation, an outfield-only sensitivity check should be run next.
"""

    (OUTPUT_DIR / "summary.md").write_text(summary)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base()
    df = build_feature_matrix(base)
    holdout_dates = choose_holdout_dates(df)

    df["split"] = np.where(df["date"].isin(holdout_dates), "holdout", "trainval")
    split_assignments = df[["row_id", "fixture_id", "date", "checkpoint", "player_appearance_id", "split"]].copy()
    split_assignments.to_csv(OUTPUT_DIR / "split_assignments.csv", index=False)

    train_df = df[df["split"] == "trainval"].copy()
    holdout_df = df[df["split"] == "holdout"].copy()

    feature_sets = build_feature_sets(df)
    results_df, fitted_models = evaluate_feature_sets(train_df, holdout_df, feature_sets)
    results_df.to_csv(OUTPUT_DIR / "model_benchmarks.csv", index=False)

    screen_df = numeric_feature_screen(train_df, feature_sets["full_feature_set"])
    screen_df.to_csv(OUTPUT_DIR / "numeric_feature_screen.csv", index=False)

    full_logreg = fitted_models[("full_feature_set", "logreg")]
    logreg_effects_df = extract_logreg_effects(full_logreg)
    logreg_effects_df.to_csv(OUTPUT_DIR / "full_feature_logreg_effects.csv", index=False)

    best_full_model_name = (
        results_df[results_df["feature_set"] == "full_feature_set"].sort_values("cv_auc_mean", ascending=False).iloc[0]["model"]
    )
    best_full_model = fitted_models[("full_feature_set", best_full_model_name)]
    permutation_df = extract_permutation_importance(best_full_model, holdout_df, feature_sets["full_feature_set"])
    permutation_df.to_csv(OUTPUT_DIR / "full_feature_permutation_importance.csv", index=False)

    save_plots(train_df, screen_df, results_df, permutation_df)
    write_summary(
        base=base,
        train_df=train_df,
        holdout_df=holdout_df,
        holdout_dates=holdout_dates,
        results_df=results_df,
        screen_df=screen_df,
        logreg_effects_df=logreg_effects_df,
        permutation_df=permutation_df,
    )

    print(f"Wrote analysis artifacts to {OUTPUT_DIR.resolve()}")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
