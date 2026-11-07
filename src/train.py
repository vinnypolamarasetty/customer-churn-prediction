"""Train, compare, tune and evaluate churn classifiers.

Run from the project root:  python -m src.train
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data_prep import CATEGORICAL, NUMERIC, ROOT, TARGET, build_preprocessor, load_dataset

SEED = 42
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    }


def compare_models(X, y):
    """5-fold stratified cross-validation of each baseline model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "f1", "recall", "precision"])
        rows.append({"model": name, **{k.replace("test_", ""): v.mean()
                                       for k, v in scores.items() if k.startswith("test_")}})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def tune_logreg(X, y):
    """Grid-search regularisation strength and class weighting. Logistic Regression is chosen
    because it matches the tree models on ROC-AUC while staying interpretable for the business."""
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    grid = {"model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            "model__class_weight": ["balanced", None]}
    search = GridSearchCV(pipe, grid, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    search.fit(X, y)
    return search


def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-9)
    return float(t[np.argmax(f1[:-1])])


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }, proba, pred


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=SEED)

    print("Cross-validated baselines (train set):")
    comparison = compare_models(X_train, y_train)
    print(comparison.round(4).to_string(index=False))

    search = tune_logreg(X_train, y_train)
    print(f"\nBest params: {search.best_params_}  CV ROC-AUC: {search.best_score_:.4f}")
    model = search.best_estimator_

    # Pick the decision threshold on training-set predictions (never on the test set).
    train_proba = model.predict_proba(X_train)[:, 1]
    thr = best_f1_threshold(y_train, train_proba)
    metrics, proba, pred = evaluate(model, X_test, y_test, thr)
    print("\nHeld-out test metrics:", metrics)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stay", "Churn"],
                                            cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion matrix (test)")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1], name="Tuned LogReg")
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_title("ROC curve (test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "evaluation.png", dpi=150)
    plt.close(fig)

    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                 random_state=SEED, n_jobs=-1)
    importance = (pd.Series(imp.importances_mean, index=X_test.columns)
                  .sort_values(ascending=True).tail(10))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    importance.plot.barh(ax=ax, color="#3b6ea5")
    ax.set_title("Top 10 features (permutation importance, ROC-AUC drop)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    joblib.dump({"model": model, "threshold": thr}, MODEL_DIR / "churn_model.joblib")
    report = {"cv_comparison": comparison.round(4).to_dict(orient="records"),
              "best_params": search.best_params_, "test_metrics": metrics,
              "top_features": importance.sort_values(ascending=False).round(4).to_dict()}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
