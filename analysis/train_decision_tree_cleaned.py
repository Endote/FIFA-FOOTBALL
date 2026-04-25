#!/usr/bin/env python3
"""
Train an explainable Decision Tree using only the cleaned dataset.

Default input:
    ../data/data_cleaned.csv

Intended run location:
    scripts/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Decision Tree on cleaned checkpoint dataset only."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../data/data_cleaned.csv"),
        help=(
            "Path to cleaned CSV or directory containing one CSV. "
            "Default assumes running from scripts/: ../data/data_cleaned.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../artifacts/decision_tree_cleaned"),
        help="Directory to save model artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Max depth of Decision Tree.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=20,
        help="Minimum samples per leaf.",
    )
    return parser.parse_args()


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def resolve_cleaned_csv(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data path not found: {path}")

    if path.is_file():
        return path

    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in cleaned data directory: {path}")
    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple CSV files found in {path}. Please pass --data-path explicitly to one file."
        )
    return csv_files[0]


def load_cleaned_data(path: Path) -> pd.DataFrame:
    csv_path = resolve_cleaned_csv(path)

    df = pd.read_csv(csv_path, na_values=["NULL", "null", ""])

    # Drop unnamed index-like columns that can appear after CSV exports.
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Some exports save empty first-column header as ''.
    if "" in df.columns:
        df = df.drop(columns=[""])

    if "scored_after" not in df.columns:
        raise ValueError("Expected target column `scored_after` in cleaned dataset.")

    df["scored_after"] = pd.to_numeric(df["scored_after"], errors="coerce").fillna(0).astype(int)
    return df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    fixture_id: pd.Series | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if fixture_id is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=fixture_id.astype(str)))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned data from: {args.data_path}")
    df = load_cleaned_data(args.data_path)
    print(f"Loaded shape: {df.shape}")

    target_col = "scored_after"

    drop_cols = [target_col, "date"]  # avoid leakage from raw date if present
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    # Keep fixture_id for group-aware splitting, but exclude from training features.
    fixture_series = df["fixture_id"] if "fixture_id" in df.columns else None
    feature_drop_cols = existing_drop_cols + [c for c in ["fixture_id"] if c in df.columns]

    X = df.drop(columns=feature_drop_cols).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        fixture_id=fixture_series,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    cat_cols = X_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot()),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = DecisionTreeClassifier(
        random_state=args.random_state,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = np.nan
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print("\n=== Split Summary ===")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows:  {len(X_test):,}")
    print(f"Train positive rate: {y_train.mean():.4f}")
    print(f"Test positive rate:  {y_test.mean():.4f}")

    print("\n=== Metrics (Test) ===")
    print(f"AUC:               {auc:.4f}" if not np.isnan(auc) else "AUC:               undefined")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print("\n=== Classification Report (Test) ===")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    preprocess_fitted: ColumnTransformer = pipeline.named_steps["preprocess"]
    feature_names = preprocess_fitted.get_feature_names_out().tolist()
    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\n=== Top Feature Importances ===")
    print(importance_df.head(25).to_string(index=False))

    model_path = output_dir / "decision_tree_cleaned_pipeline.joblib"
    importance_path = output_dir / "feature_importances.csv"
    rules_path = output_dir / "decision_tree_rules.txt"
    report_path = output_dir / "classification_report.txt"
    plot_path = output_dir / "decision_tree_plot.png"

    joblib.dump(pipeline, model_path)
    importance_df.to_csv(importance_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    tree_text = export_text(
        pipeline.named_steps["model"],
        feature_names=feature_names,
        max_depth=min(6, pipeline.named_steps["model"].get_depth()),
    )
    rules_path.write_text(tree_text, encoding="utf-8")

    plt.figure(figsize=(28, 14))
    plot_tree(
        pipeline.named_steps["model"],
        feature_names=feature_names,
        class_names=["no_goal", "goal"],
        filled=True,
        rounded=True,
        proportion=True,
        max_depth=min(4, pipeline.named_steps["model"].get_depth()),
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=220)
    plt.close()

    metrics_df = pd.DataFrame(
        [
            {
                "auc": float(auc) if not np.isnan(auc) else np.nan,
                "balanced_accuracy": float(bal_acc),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "train_positive_rate": float(y_train.mean()),
                "test_positive_rate": float(y_test.mean()),
            }
        ]
    )
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\n=== Saved Artifacts ===")
    print(f"- Model pipeline:      {model_path}")
    print(f"- Feature importances: {importance_path}")
    print(f"- Tree rules:          {rules_path}")
    print(f"- Tree plot:           {plot_path}")
    print(f"- Classification rep.: {report_path}")
    print(f"- Metrics CSV:         {metrics_path}")


if __name__ == "__main__":
    main()
