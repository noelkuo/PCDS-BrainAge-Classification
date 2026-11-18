#!/usr/bin/env python3
"""Explain NILS-LSA predictions from the final ILAS-trained ROI53 model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap


N_FEATURES = 53
POSITIVE_CLASS_INDEX = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--ilas-features", required=True, type=Path)
    parser.add_argument("--nils-features", required=True, type=Path)
    parser.add_argument("--roi-labels", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/roi53_shap"))
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="NumPy seed used by KernelExplainer sampling (default: 0).",
    )
    parser.add_argument(
        "--nsamples",
        default="auto",
        help="KernelExplainer sample count or 'auto' (default: auto).",
    )
    return parser.parse_args()


def load_features(path: Path, cohort: str) -> np.ndarray:
    features = np.loadtxt(path, dtype=float)
    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError(
            f"{cohort} features must have 53 columns; received {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{cohort} features contain missing or non-finite values.")
    return features


def load_roi_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if len(labels) != N_FEATURES:
        raise ValueError(f"ROI label file must contain 53 labels; received {len(labels)}.")
    if len(set(labels)) != N_FEATURES:
        raise ValueError("ROI labels must be unique.")
    return labels


def class_one_shap_values(raw_values: object) -> np.ndarray:
    """Support SHAP's historical list output and current ndarray output."""
    if isinstance(raw_values, list):
        if len(raw_values) <= POSITIVE_CLASS_INDEX:
            raise ValueError("SHAP output does not contain class 1 values.")
        values = np.asarray(raw_values[POSITIVE_CLASS_INDEX], dtype=float)
    else:
        values_array = np.asarray(raw_values, dtype=float)
        if values_array.ndim == 3:
            values = values_array[:, :, POSITIVE_CLASS_INDEX]
        elif values_array.ndim == 2:
            values = values_array
        else:
            raise ValueError(f"Unexpected SHAP output shape: {values_array.shape}.")
    if values.ndim != 2 or values.shape[1] != N_FEATURES:
        raise ValueError(f"Expected an (n, 53) SHAP matrix; received {values.shape}.")
    return values


def class_one_expected_value(explainer: shap.KernelExplainer) -> float:
    expected = np.asarray(explainer.expected_value, dtype=float)
    if expected.ndim == 0:
        return float(expected)
    return float(expected[POSITIVE_CLASS_INDEX])


def write_importance_csv(
    path: Path, labels: list[str], mean_absolute_values: np.ndarray
) -> None:
    order = np.argsort(mean_absolute_values)[::-1]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "feature_index_1based", "roi", "mean_abs_shap"])
        for rank, index in enumerate(order, start=1):
            writer.writerow(
                [rank, int(index + 1), labels[index], float(mean_absolute_values[index])]
            )


def plot_summary(
    shap_values: np.ndarray,
    nils_features: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> None:
    shap.summary_plot(
        shap_values,
        nils_features,
        feature_names=labels,
        max_display=10,
        plot_size=(8, 8),
        show=False,
    )
    figure = plt.gcf()
    figure.savefig(output_dir / "roi53_nils_shap_summary.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "roi53_nils_shap_summary.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_global_importance(
    mean_absolute_values: np.ndarray, labels: list[str], output_dir: Path
) -> None:
    top_indices = np.argsort(mean_absolute_values)[::-1][:10]
    top_labels = [labels[index] for index in top_indices]
    top_values = mean_absolute_values[top_indices]
    figure, axis = plt.subplots(figsize=(5, 8))
    axis.barh(top_labels, top_values, color="salmon")
    axis.set_xlabel("Mean |SHAP value|")
    axis.set_title("Global Feature Importance")
    axis.invert_yaxis()
    figure.savefig(
        output_dir / "roi53_nils_shap_global_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "roi53_nils_shap_global_importance.pdf", bbox_inches="tight"
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ilas_features = load_features(args.ilas_features, "ILAS")
    nils_features = load_features(args.nils_features, "NILS-LSA")
    roi_labels = load_roi_labels(args.roi_labels)

    # joblib/pickle files can execute code. Load only the controlled model created
    # by run_roi53_ILAS.py, never an untrusted downloaded model.
    model = joblib.load(args.model)
    estimator = getattr(model, "best_estimator_", model)
    if getattr(estimator, "n_features_in_", None) != N_FEATURES:
        raise ValueError("The supplied model was not fitted with 53 features.")

    nsamples: str | int
    nsamples = args.nsamples if args.nsamples == "auto" else int(args.nsamples)
    np.random.seed(args.random_seed)

    # This mirrors ROC_Curve_v2.ipynb: full ILAS is the background dataset,
    # NILS-LSA is explained, and the model output is class probability.
    explainer = shap.KernelExplainer(estimator.predict_proba, ilas_features)
    raw_values = explainer.shap_values(nils_features, nsamples=nsamples)
    shap_values = class_one_shap_values(raw_values)
    expected_value = class_one_expected_value(explainer)

    predicted_probability = estimator.predict_proba(nils_features)[:, POSITIVE_CLASS_INDEX]
    reconstructed_probability = expected_value + shap_values.sum(axis=1)
    max_additivity_error = float(
        np.max(np.abs(predicted_probability - reconstructed_probability))
    )
    mean_absolute_values = np.mean(np.abs(shap_values), axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "roi53_nils_shap_values.csv",
        shap_values,
        delimiter=",",
        header=",".join(roi_labels),
        comments="",
    )
    write_importance_csv(
        args.output_dir / "roi53_nils_shap_importance.csv",
        roi_labels,
        mean_absolute_values,
    )
    plot_summary(shap_values, nils_features, roi_labels, args.output_dir)
    plot_global_importance(mean_absolute_values, roi_labels, args.output_dir)

    top_indices = np.argsort(mean_absolute_values)[::-1][:10]
    results = {
        "analysis": "ROI53 SHAP explanation of NILS-LSA class 1 probability",
        "model_refitted": False,
        "background_cohort": "ILAS",
        "background_n": int(ilas_features.shape[0]),
        "explained_cohort": "NILS-LSA",
        "explained_n": int(nils_features.shape[0]),
        "n_features": N_FEATURES,
        "explainer": "shap.KernelExplainer",
        "shap_version": shap.__version__,
        "random_seed": args.random_seed,
        "nsamples": nsamples,
        "class_index": POSITIVE_CLASS_INDEX,
        "expected_value_class_1": expected_value,
        "max_additivity_error": max_additivity_error,
        "top_10_features": [
            {
                "rank": rank,
                "feature_index_1based": int(index + 1),
                "roi": roi_labels[index],
                "mean_abs_shap": float(mean_absolute_values[index]),
            }
            for rank, index in enumerate(top_indices, start=1)
        ],
    }
    (args.output_dir / "roi53_nils_shap_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
