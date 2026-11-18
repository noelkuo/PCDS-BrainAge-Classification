#!/usr/bin/env python3
"""Run ILAS nested CV and train the final 53-region RBF-SVM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

N_FEATURES = 53
N_SPLITS = 5
CV_RANDOM_STATE = 42
SVC_RANDOM_STATE = 0
SEARCH_VALUES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
PARAM_GRID = {"C": SEARCH_VALUES, "gamma": SEARCH_VALUES}
MEAN_FPR = np.linspace(0.0, 1.0, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ilas-features", required=True, type=Path)
    parser.add_argument("--ilas-labels", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/roi53_ilas"))
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Model path; defaults to OUTPUT_DIR/roi53_final_model.joblib.",
    )
    return parser.parse_args()


def load_data(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    features = np.loadtxt(features_path, dtype=float)
    labels = np.loadtxt(labels_path, dtype=float)
    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError(f"ILAS features must have 53 columns; received {features.shape}.")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("ILAS labels must contain one value per feature row.")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("Inputs contain missing or non-finite values.")
    if not np.array_equal(np.unique(labels), np.array([0.0, 1.0])):
        raise ValueError("ILAS labels must use binary values 0 and 1.")
    return features, labels


def make_search() -> GridSearchCV:
    inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    classifier = SVC(kernel="rbf", probability=True, random_state=SVC_RANDOM_STATE)
    return GridSearchCV(
        classifier, PARAM_GRID, cv=inner_cv, scoring="accuracy", refit=True, n_jobs=None
    )


def main() -> None:
    args = parse_args()
    features, labels = load_data(args.ilas_features, args.ilas_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_rows: list[list[float]] = []
    interpolated_tprs: list[np.ndarray] = []

    for fold, (train_index, test_index) in enumerate(outer_cv.split(features, labels), start=1):
        search = make_search()
        search.fit(features[train_index], labels[train_index])
        predictions = search.predict(features[test_index])
        probabilities = search.predict_proba(features[test_index])[:, 1]
        fpr, tpr, _ = roc_curve(labels[test_index], probabilities)
        interpolated_tpr = np.interp(MEAN_FPR, fpr, tpr)
        interpolated_tpr[0] = 0.0
        interpolated_tprs.append(interpolated_tpr)
        fold_rows.append([
            float(fold),
            float(search.best_params_["C"]),
            float(search.best_params_["gamma"]),
            float(roc_auc_score(labels[test_index], probabilities)),
            float(accuracy_score(labels[test_index], predictions)),
            float(balanced_accuracy_score(labels[test_index], predictions)),
        ])

    fold_table = np.asarray(fold_rows)
    final_model = make_search()
    final_model.fit(features, labels)
    model_output = args.model_output or args.output_dir / "roi53_final_model.joblib"
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_output)

    results = {
        "analysis": "ILAS 53-region RBF-SVM",
        "n_participants": int(features.shape[0]),
        "n_features": N_FEATURES,
        "search_grid": PARAM_GRID,
        "outer_and_inner_cv": {
            "type": "StratifiedKFold", "n_splits": N_SPLITS,
            "shuffle": True, "random_state": CV_RANDOM_STATE,
        },
        "grid_search_scoring": "accuracy",
        "svc_random_state": SVC_RANDOM_STATE,
        "nested_cv": {
            "auc_mean": float(fold_table[:, 3].mean()),
            "auc_sd": float(fold_table[:, 3].std(ddof=0)),
            "accuracy_mean": float(fold_table[:, 4].mean()),
            "accuracy_sd": float(fold_table[:, 4].std(ddof=0)),
            "balanced_accuracy_mean": float(fold_table[:, 5].mean()),
            "balanced_accuracy_sd": float(fold_table[:, 5].std(ddof=0)),
        },
        "final_model": {
            "best_C": float(final_model.best_params_["C"]),
            "best_gamma": float(final_model.best_params_["gamma"]),
            "best_cv_accuracy": float(final_model.best_score_),
            "saved_to": str(model_output),
        },
    }
    np.savetxt(
        args.output_dir / "roi53_ilas_nested_cv_folds.csv", fold_table, delimiter=",",
        header="fold,best_C,best_gamma,auc,accuracy,balanced_accuracy", comments="",
    )
    np.savetxt(
        args.output_dir / "roi53_ilas_mean_roc.csv",
        np.column_stack([MEAN_FPR, np.mean(interpolated_tprs, axis=0)]), delimiter=",",
        header="false_positive_rate,true_positive_rate", comments="",
    )
    (args.output_dir / "roi53_ilas_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
