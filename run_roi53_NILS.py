#!/usr/bin/env python3
"""Apply the final ILAS-trained 53-region model to NILS-LSA without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

N_FEATURES = 53
MEAN_FPR = np.linspace(0.0, 1.0, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--nils-features", required=True, type=Path)
    parser.add_argument("--nils-labels", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/roi53_nils"))
    return parser.parse_args()


def load_data(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    features = np.loadtxt(features_path, dtype=float)
    labels = np.loadtxt(labels_path, dtype=float)
    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError(f"NILS-LSA features must have 53 columns; received {features.shape}.")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("NILS-LSA labels must contain one value per feature row.")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("Inputs contain missing or non-finite values.")
    if not np.array_equal(np.unique(labels), np.array([0.0, 1.0])):
        raise ValueError("NILS-LSA labels must use binary values 0 and 1.")
    return features, labels


def main() -> None:
    args = parse_args()
    features, labels = load_data(args.nils_features, args.nils_labels)
    # joblib/pickle files can execute code. Load only the controlled model created
    # by run_roi53_ILAS.py, never an untrusted downloaded model.
    model = joblib.load(args.model)
    estimator = getattr(model, "best_estimator_", model)
    if getattr(estimator, "n_features_in_", None) != N_FEATURES:
        raise ValueError("The supplied model was not fitted with 53 features.")

    # External validation only: there is no fit or hyperparameter search here.
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fpr, tpr, _ = roc_curve(labels, probabilities)
    interpolated_tpr = np.interp(MEAN_FPR, fpr, tpr)
    results = {
        "analysis": "NILS-LSA external validation of ILAS-trained 53-region model",
        "n_participants": int(features.shape[0]),
        "n_features": N_FEATURES,
        "model_path": str(args.model),
        "model_refitted": False,
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "roi53_nils_external_roc.csv",
        np.column_stack([MEAN_FPR, interpolated_tpr]), delimiter=",",
        header="false_positive_rate,true_positive_rate", comments="",
    )
    (args.output_dir / "roi53_nils_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
