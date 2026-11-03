"""Evaluation metrics, classification reports, and error analysis for Clinical NER.

This module provides entity-level evaluation using seqeval, ignoring
special/padding tokens with IGNORE_INDEX (-100).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from seqeval.metrics import (
    classification_report as seqeval_classification_report,
    f1_score as seqeval_f1_score,
    precision_score as seqeval_precision_score,
    recall_score as seqeval_recall_score,
)
from seqeval.metrics.sequence_labeling import get_entities

IGNORE_INDEX = -100


def to_serializable(val: Any) -> Any:
    """Convert numpy scalars and collections to standard Python types for JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    return val


def align_predictions_and_labels(
    predictions: Any,
    label_ids: Any,
    id2label: dict[int, str],
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert model prediction logits/IDs and label IDs to lists of tag sequences.

    Tokens where label_id == ignore_index (-100) are excluded from both true
    and predicted tag sequences. Handles both fixed-length arrays (e.g. from Trainer)
    and ragged lists of sequences.

    Parameters
    ----------
    predictions:
        Logits array of shape (batch_size, seq_len, num_labels) or
        argmax token label IDs of shape (batch_size, seq_len), or nested lists.
    label_ids:
        Ground truth label IDs of shape (batch_size, seq_len), or nested lists.
    id2label:
        Mapping from integer label ID to BIO tag string.
    ignore_index:
        Label ID sentinel for tokens to ignore (default -100).

    Returns
    -------
    y_true:
        List of true BIO tag strings per sequence.
    y_pred:
        List of predicted BIO tag strings per sequence.
    """
    # If predictions is a 3D numpy array or has 3 dims, argmax to 2D
    if hasattr(predictions, "ndim") and predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []

    for pred_seq, label_seq in zip(predictions, label_ids):
        true_tags: list[str] = []
        pred_tags: list[str] = []
        for p_id, l_id in zip(pred_seq, label_seq):
            l_int = int(l_id)
            if l_int == ignore_index:
                continue
            # If pred token is still a vector of logits (e.g. in list of lists)
            if isinstance(p_id, (list, tuple, np.ndarray)):
                p_int = int(np.argmax(p_id))
            else:
                p_int = int(p_id)
            true_tags.append(id2label.get(l_int, "O"))
            pred_tags.append(id2label.get(p_int, "O"))
        y_true.append(true_tags)
        y_pred.append(pred_tags)

    return y_true, y_pred


def analyze_entity_errors(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Perform detailed entity-level error and confusion analysis.

    Categorizes entity predictions into:
    - exact_matches: correct entity type and boundary
    - type_mismatches: same exact boundary, but wrong entity type
    - boundary_errors: overlapping span with same type, but boundary mismatch
    - boundary_and_type_errors: overlapping span with both type and boundary mismatch
    - false_negatives: true entity completely missed (no overlap)
    - false_positives: predicted entity completely spurious (no overlap)
    """
    exact_matches = 0
    type_mismatches: list[dict[str, Any]] = []
    boundary_errors: list[dict[str, Any]] = []
    boundary_and_type_errors: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []

    counts_by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"gold": 0, "pred": 0, "exact": 0, "missed": 0, "spurious": 0}
    )
    confusion_pairs: Counter[tuple[str, str]] = Counter()

    for seq_idx, (t_seq, p_seq) in enumerate(zip(y_true, y_pred)):
        true_ents = get_entities(list(t_seq))
        pred_ents = get_entities(list(p_seq))

        for t_type, _, _ in true_ents:
            counts_by_type[t_type]["gold"] += 1
        for p_type, _, _ in pred_ents:
            counts_by_type[p_type]["pred"] += 1

        matched_preds: set[int] = set()

        for t_type, t_start, t_end in true_ents:
            overlapping_preds = [
                (idx, (p_type, p_start, p_end))
                for idx, (p_type, p_start, p_end) in enumerate(pred_ents)
                if max(t_start, p_start) <= min(t_end, p_end)
            ]

            if not overlapping_preds:
                false_negatives.append({
                    "seq_index": seq_idx,
                    "type": t_type,
                    "span": [t_start, t_end],
                })
                counts_by_type[t_type]["missed"] += 1
                continue

            exact_match_found = False
            for p_idx, (p_type, p_start, p_end) in overlapping_preds:
                if t_start == p_start and t_end == p_end:
                    matched_preds.add(p_idx)
                    if t_type == p_type:
                        exact_matches += 1
                        counts_by_type[t_type]["exact"] += 1
                        exact_match_found = True
                        break
                    else:
                        type_mismatches.append({
                            "seq_index": seq_idx,
                            "true_type": t_type,
                            "pred_type": p_type,
                            "span": [t_start, t_end],
                        })
                        confusion_pairs[(t_type, p_type)] += 1
                        exact_match_found = True
                        break

            if exact_match_found:
                continue

            # Overlapping but boundaries differ
            p_idx, (p_type, p_start, p_end) = overlapping_preds[0]
            matched_preds.add(p_idx)
            if t_type == p_type:
                boundary_errors.append({
                    "seq_index": seq_idx,
                    "type": t_type,
                    "true_span": [t_start, t_end],
                    "pred_span": [p_start, p_end],
                })
            else:
                boundary_and_type_errors.append({
                    "seq_index": seq_idx,
                    "true_type": t_type,
                    "pred_type": p_type,
                    "true_span": [t_start, t_end],
                    "pred_span": [p_start, p_end],
                })
                confusion_pairs[(t_type, p_type)] += 1

        for p_idx, (p_type, p_start, p_end) in enumerate(pred_ents):
            if p_idx not in matched_preds:
                false_positives.append({
                    "seq_index": seq_idx,
                    "type": p_type,
                    "span": [p_start, p_end],
                })
                counts_by_type[p_type]["spurious"] += 1

    top_confusions = [
        {"true_label": k[0], "pred_label": k[1], "count": v}
        for k, v in confusion_pairs.most_common(10)
    ]

    return {
        "summary": {
            "exact_matches": exact_matches,
            "type_mismatches": len(type_mismatches),
            "boundary_errors": len(boundary_errors),
            "boundary_and_type_errors": len(boundary_and_type_errors),
            "false_negatives": len(false_negatives),
            "false_positives": len(false_positives),
        },
        "counts_by_type": dict(counts_by_type),
        "top_confusions": top_confusions,
        "samples": {
            "type_mismatches": type_mismatches[:5],
            "boundary_errors": boundary_errors[:5],
            "false_negatives": false_negatives[:5],
            "false_positives": false_positives[:5],
        },
    }


@dataclass
class EvaluationResult:
    """Structured container for NER evaluation metrics."""

    precision: float
    recall: float
    f1: float
    per_class: dict[str, dict[str, float | int]]
    classification_report: str
    classification_report_dict: dict[str, Any]
    error_analysis: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(asdict(self))


def evaluate_sequences(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
) -> EvaluationResult:
    """Compute entity-level metrics, classification report, and error analysis."""
    p = float(seqeval_precision_score(y_true, y_pred, zero_division=0))
    r = float(seqeval_recall_score(y_true, y_pred, zero_division=0))
    f1 = float(seqeval_f1_score(y_true, y_pred, zero_division=0))

    report_str = seqeval_classification_report(y_true, y_pred, zero_division=0)
    report_dict_raw = seqeval_classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    report_dict = to_serializable(report_dict_raw)

    per_class: dict[str, dict[str, float | int]] = {}
    for key, val in report_dict.items():
        if key not in ("micro avg", "macro avg", "weighted avg"):
            per_class[key] = {
                "precision": float(val.get("precision", 0.0)),
                "recall": float(val.get("recall", 0.0)),
                "f1": float(val.get("f1-score", 0.0)),
                "support": int(val.get("support", 0)),
            }

    err_analysis = to_serializable(analyze_entity_errors(y_true, y_pred))

    return EvaluationResult(
        precision=p,
        recall=r,
        f1=f1,
        per_class=per_class,
        classification_report=report_str,
        classification_report_dict=report_dict,
        error_analysis=err_analysis,
    )


def evaluate_predictions(
    predictions: np.ndarray | Sequence[Sequence[Any]],
    label_ids: np.ndarray | Sequence[Sequence[int]],
    id2label: dict[int, str],
    ignore_index: int = IGNORE_INDEX,
) -> EvaluationResult:
    """End-to-end evaluation from model predictions and ground-truth label IDs."""
    y_true, y_pred = align_predictions_and_labels(
        predictions=predictions,
        label_ids=label_ids,
        id2label=id2label,
        ignore_index=ignore_index,
    )
    return evaluate_sequences(y_true, y_pred)


def build_compute_metrics(
    id2label: dict[int, str],
    ignore_index: int = IGNORE_INDEX,
) -> Callable[[Any], dict[str, float]]:
    """Build a compute_metrics function compatible with HuggingFace Trainer.

    Returns a dict with overall 'precision', 'recall', and 'f1' as well as
    per-class F1 scores.
    """

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
        result = evaluate_predictions(
            predictions=predictions,
            label_ids=label_ids,
            id2label=id2label,
            ignore_index=ignore_index,
        )
        metrics: dict[str, float] = {
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
        }
        for cls_name, cls_metrics in result.per_class.items():
            metrics[f"{cls_name}_f1"] = float(cls_metrics["f1"])
        return metrics

    return compute_metrics


def save_evaluation_report(
    result: EvaluationResult,
    output_dir: str | Path,
    prefix: str = "eval",
) -> tuple[Path, Path]:
    """Save evaluation results as both JSON and readable Markdown."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / f"{prefix}_report.json"
    md_path = out_path / f"{prefix}_report.md"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)

    md_lines: list[str] = [
        f"# Evaluation Report: {prefix.upper()}\n",
        "## Overall Entity-Level Metrics\n",
        f"- **Precision**: {result.precision:.4f}",
        f"- **Recall**: {result.recall:.4f}",
        f"- **F1 Score**: {result.f1:.4f}\n",
        "## Per-Entity Classification Report\n",
        "```text",
        result.classification_report,
        "```\n",
        "## Error Analysis Summary\n",
    ]
    summary = result.error_analysis.get("summary", {})
    for k, v in summary.items():
        md_lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")

    top_conf = result.error_analysis.get("top_confusions", [])
    if top_conf:
        md_lines.append("\n### Top Label Confusions\n")
        md_lines.append("| True Label | Predicted Label | Count |")
        md_lines.append("|---|---|---:|")
        for c in top_conf:
            md_lines.append(f"| {c['true_label']} | {c['pred_label']} | {c['count']} |")

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")

    return json_path, md_path
