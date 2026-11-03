"""Unit tests for src/evaluator.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluator import (
    IGNORE_INDEX,
    align_predictions_and_labels,
    analyze_entity_errors,
    build_compute_metrics,
    evaluate_predictions,
    evaluate_sequences,
    save_evaluation_report,
)
from preprocessing import build_label_maps


@pytest.fixture
def label_maps():
    return build_label_maps()


def test_align_predictions_and_labels(label_maps):
    labels, label2id, id2label = label_maps
    # 2 sequences:
    # seq 1: [CLS], "fever", [SEP] -> ignore, B-SYMPTOM, ignore
    # seq 2: [CLS], "cough", "severe", [SEP] -> ignore, B-SYMPTOM, I-SYMPTOM, ignore
    preds = [
        [label2id["O"], label2id["B-SYMPTOM"], label2id["O"]],
        [label2id["O"], label2id["B-SYMPTOM"], label2id["I-SYMPTOM"], label2id["O"]],
    ]
    targets = [
        [IGNORE_INDEX, label2id["B-SYMPTOM"], IGNORE_INDEX],
        [IGNORE_INDEX, label2id["B-SYMPTOM"], label2id["I-SYMPTOM"], IGNORE_INDEX],
    ]
    y_true, y_pred = align_predictions_and_labels(preds, targets, id2label)
    assert y_true == [["B-SYMPTOM"], ["B-SYMPTOM", "I-SYMPTOM"]]
    assert y_pred == [["B-SYMPTOM"], ["B-SYMPTOM", "I-SYMPTOM"]]


def test_align_predictions_with_3d_logits(label_maps):
    labels, label2id, id2label = label_maps
    # Shape: (1, 3, num_labels)
    logits = np.zeros((1, 3, len(labels)), dtype=np.float32)
    logits[0, 1, label2id["B-DISEASE"]] = 10.0  # token 1 predicted as B-DISEASE
    targets = [[IGNORE_INDEX, label2id["B-DISEASE"], IGNORE_INDEX]]

    y_true, y_pred = align_predictions_and_labels(logits, targets, id2label)
    assert y_true == [["B-DISEASE"]]
    assert y_pred == [["B-DISEASE"]]


def test_evaluate_sequences_perfect_match():
    y_true = [["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]]
    y_pred = [["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]]
    res = evaluate_sequences(y_true, y_pred)
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.per_class["DISEASE"]["f1"] == 1.0
    assert res.per_class["DRUG"]["f1"] == 1.0
    assert res.error_analysis["summary"]["exact_matches"] == 2
    assert res.error_analysis["summary"]["false_negatives"] == 0
    assert res.error_analysis["summary"]["false_positives"] == 0


def test_evaluate_sequences_with_errors():
    # Sequence with 1 type mismatch and 1 false negative and 1 false positive
    y_true = [["O", "B-DISEASE", "I-DISEASE", "O", "B-DRUG", "O"]]
    # Pred: predicts SYMPTOM instead of DISEASE, misses DRUG, and predicts PROCEDURE at end
    y_pred = [["O", "B-SYMPTOM", "I-SYMPTOM", "O", "O", "B-PROCEDURE"]]
    res = evaluate_sequences(y_true, y_pred)
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0
    assert res.error_analysis["summary"]["type_mismatches"] == 1
    assert res.error_analysis["summary"]["false_negatives"] == 1
    assert res.error_analysis["summary"]["false_positives"] == 1


def test_build_compute_metrics(label_maps):
    labels, label2id, id2label = label_maps
    compute_fn = build_compute_metrics(id2label)

    class DummyEvalPred:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids

    # 1 sequence with 1 perfect DRUG entity
    preds = np.array([[label2id["O"], label2id["B-DRUG"], label2id["O"]]])
    targets = np.array([[IGNORE_INDEX, label2id["B-DRUG"], IGNORE_INDEX]])

    metrics = compute_fn(DummyEvalPred(preds, targets))
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["DRUG_f1"] == 1.0


def test_save_evaluation_report(tmp_path):
    y_true = [["O", "B-DISEASE", "O"]]
    y_pred = [["O", "B-DISEASE", "O"]]
    res = evaluate_sequences(y_true, y_pred)

    json_p, md_p = save_evaluation_report(res, tmp_path, prefix="test_run")
    assert json_p.exists()
    assert md_p.exists()

    with json_p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["f1"] == 1.0
    assert "DISEASE" in data["per_class"]
