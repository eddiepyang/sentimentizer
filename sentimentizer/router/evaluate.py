"""Evaluation utilities for the SetFit router model.

Provides:
- Cosine similarity matrix (inter-class < 0.65, intra-class > 0.85)
- Tau threshold calibration (where False Positives appear in General category)
- Full evaluation: accuracy, F1, similarity matrix, threshold
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import classification_report

from sentimentizer.router.config import RouteLabels

if TYPE_CHECKING:
    from setfit import SetFitModel

logger = logging.getLogger(__name__)

LABEL_NAMES = RouteLabels.label_names()


def compute_similarity_matrix(
    model: SetFitModel, texts_by_label: dict[int, list[str]]
) -> np.ndarray:
    """Generate inter-class and intra-class cosine similarity heatmap.

    Targets: inter-class similarity < 0.65, intra-class similarity > 0.85.

    Args:
        model: Trained SetFitModel.
        texts_by_label: Dict mapping label int to list of example texts.

    Returns:
        Cosine similarity matrix of shape (num_classes, num_classes).
    """
    # Encode all texts and compute centroids
    centroids = {}
    for label, texts in texts_by_label.items():
        embeddings = model.model_encode(texts)  # (N, D)
        centroids[label] = np.mean(embeddings, axis=0)

    # Compute pairwise cosine similarity between centroids
    labels = sorted(centroids.keys())
    num_classes = len(labels)
    sim_matrix = np.zeros((num_classes, num_classes))

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            vec_i = centroids[label_i]
            vec_j = centroids[label_j]
            dot = np.dot(vec_i, vec_j)
            norm = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
            sim_matrix[i, j] = dot / norm if norm > 0 else 0.0

    # Log results
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if i < j:
                inter_sim = sim_matrix[i, j]
                name_i = LABEL_NAMES.get(label_i, str(label_i))
                name_j = LABEL_NAMES.get(label_j, str(label_j))
                status = "OK" if inter_sim < 0.65 else "HIGH"
                logger.info(
                    f"Inter-class similarity ({name_i} vs {name_j}): {inter_sim:.4f} [{status}]"
                )

    for label in labels:
        name = LABEL_NAMES.get(label, str(label))
        intra_sim = sim_matrix[label, label]
        status = "OK" if intra_sim > 0.85 else "LOW"
        logger.info(f"Intra-class similarity ({name}): {intra_sim:.4f} [{status}]")

    return sim_matrix


def calibrate_threshold(
    model: SetFitModel, eval_dataset: dict, label_names: dict[int, str] | None = None
) -> float:
    """Find tau threshold where False Positives appear in General category.

    The tau threshold is the confidence score below which the router
    should fall back to a default handling strategy.

    Args:
        model: Trained SetFitModel.
        eval_dataset: Evaluation dataset with 'text' and 'label' columns.
        label_names: Optional mapping from label int to name.

    Returns:
        Recommended tau threshold (confidence score).
    """
    if label_names is None:
        label_names = LABEL_NAMES

    texts = eval_dataset["text"]
    true_labels = eval_dataset["label"]

    probs = model.predict_proba(texts)
    preds = model.predict(texts)

    correct_confidences = []
    general_fp_confidences = []
    general_label = [k for k, v in label_names.items() if v == "general"][0]

    for _i, (pred, true, prob_vec) in enumerate(zip(preds, true_labels, probs, strict=True)):
        confidence = max(prob_vec)
        if pred == true:
            correct_confidences.append(confidence)
        elif pred == general_label and true != general_label:
            general_fp_confidences.append(confidence)

    if not correct_confidences:
        logger.warning("No correct predictions found for threshold calibration")
        return 0.5

    tau = min(correct_confidences)
    if general_fp_confidences:
        worst_correct = min(correct_confidences)
        best_fp = max(general_fp_confidences)
        tau = (worst_correct + best_fp) / 2

    logger.info(f"Calibrated tau threshold: {tau:.4f}")
    logger.info(
        f"  Correct predictions: min_conf={min(correct_confidences):.4f}, "
        f"max_conf={max(correct_confidences):.4f}"
    )
    if general_fp_confidences:
        logger.info(
            f"  General false positives: min_conf={min(general_fp_confidences):.4f}, "
            f"max_conf={max(general_fp_confidences):.4f}"
        )

    return float(tau)


def evaluate_router(
    model: SetFitModel, eval_dataset: dict, label_names: dict[int, str] | None = None
) -> dict:
    """Full evaluation: accuracy, F1, similarity matrix, threshold.

    Args:
        model: Trained SetFitModel.
        eval_dataset: Evaluation dataset with 'text' and 'label' columns.
        label_names: Optional mapping from label int to name.

    Returns:
        Dict with 'classification_report', 'similarity_matrix', 'tau_threshold'.
    """
    if label_names is None:
        label_names = LABEL_NAMES

    texts = eval_dataset["text"]
    true_labels = eval_dataset["label"]
    preds = model.predict(texts)

    target_names = [label_names.get(i, str(i)) for i in sorted(label_names.keys())]
    report = classification_report(true_labels, preds, target_names=target_names, output_dict=True)
    logger.info(
        f"Classification report:\n"
        f"{classification_report(true_labels, preds, target_names=target_names)}"
    )

    texts_by_label = defaultdict(list)
    for text, label in zip(texts, true_labels, strict=True):
        texts_by_label[label].append(text)
    sim_matrix = compute_similarity_matrix(model, dict(texts_by_label))

    tau = calibrate_threshold(model, eval_dataset, label_names)

    return {
        "classification_report": report,
        "similarity_matrix": sim_matrix,
        "tau_threshold": tau,
    }
