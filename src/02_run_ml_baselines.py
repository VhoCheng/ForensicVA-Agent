
import argparse
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

META_COLS = {
    "case_id", "narrative", "source_file",
    "target_fine_34", "target_broad", "target_external",
    "gs_code34", "gs_text34", "va34",
    "gs_code46", "gs_text46", "va46",
    "gs_code55", "gs_text55", "va55",
    "gs_comorbid1", "gs_comorbid2", "gs_level"
}

def clean_X(df):
    X = df.copy()
    for c in X.columns:
        X[c] = X[c].astype(str).replace({"nan": "", "None": ""})
    return X

def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def get_structured_pipeline(model):
    return Pipeline([
        ("clean", FunctionTransformer(clean_X, validate=False)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
        ("clf", model)
    ])

def get_text_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", model)
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, low_memory=False)

    raw_feature_cols = [c for c in df.columns if c not in META_COLS]
    # Remove free-text narrative from structured input and keep all questionnaire features.
    raw_feature_cols = [c for c in raw_feature_cols if c != "narrative"]

    targets = {
        "broad": "target_broad",
        "external": "target_external",
        "fine34": "target_fine_34",
    }

    models = {
        "Structured_LogReg": ("structured", LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)),
        "Structured_LinearSVM": ("structured", LinearSVC(class_weight="balanced")),
        "Structured_RandomForest": ("structured", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        "Narrative_TFIDF_LogReg": ("text", LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)),
        "Narrative_TFIDF_LinearSVM": ("text", LinearSVC(class_weight="balanced")),
    }

    all_metrics = []
    for task_name, target_col in targets.items():
        task_df = df.dropna(subset=[target_col]).copy()
        # Remove very rare classes for stable split in fine-grained task.
        counts = task_df[target_col].value_counts()
        keep = counts[counts >= 5].index
        task_df = task_df[task_df[target_col].isin(keep)].copy()

        y = task_df[target_col].astype(str)
        stratify = y if y.value_counts().min() >= 2 else None
        train_df, test_df, y_train, y_test = train_test_split(
            task_df, y, test_size=args.test_size, random_state=42, stratify=stratify
        )

        for model_name, (input_type, model) in models.items():
            if input_type == "structured":
                X_train = train_df[raw_feature_cols]
                X_test = test_df[raw_feature_cols]
                pipe = get_structured_pipeline(model)
            else:
                X_train = train_df["narrative"].fillna("")
                X_test = test_df["narrative"].fillna("")
                pipe = get_text_pipeline(model)

            print(f"Training {model_name} on task={task_name} ...")
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            metrics = evaluate_predictions(y_test, pred)
            metrics.update({"task": task_name, "method": model_name, "n_test": len(y_test), "n_classes": y.nunique()})
            all_metrics.append(metrics)

            pred_df = pd.DataFrame({
                "case_id": test_df["case_id"].values,
                "task": task_name,
                "method": model_name,
                "y_true": y_test.values,
                "y_pred": pred,
            })
            pred_df.to_csv(out_dir / f"pred_{task_name}_{model_name}.csv", index=False)

            # Save classification report and confusion matrix.
            report = classification_report(y_test, pred, zero_division=0)
            (out_dir / f"report_{task_name}_{model_name}.txt").write_text(report, encoding="utf-8")
            labels = sorted(y.unique())
            cm = confusion_matrix(y_test, pred, labels=labels)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"cm_{task_name}_{model_name}.csv")

    metrics_df = pd.DataFrame(all_metrics).sort_values(["task", "macro_f1"], ascending=[True, False])
    metrics_df.to_csv(out_dir / "all_metrics.csv", index=False)
    print("\n=== Metrics ===")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()
