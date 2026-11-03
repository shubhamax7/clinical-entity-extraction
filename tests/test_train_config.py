"""Unit tests for src/train.py configuration, arguments, and Dataset class."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import build_label_maps, load_tokenizer
from train import ClinicalNERDataset, TrainConfig, parse_args


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def label_maps():
    return build_label_maps()


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.max_length == 512
    assert cfg.batch_size == 8
    assert cfg.epochs == 5
    assert cfg.learning_rate == 2e-5
    assert cfg.weight_decay == 0.01
    assert cfg.seed == 42
    assert cfg.early_stopping_patience == 3
    assert cfg.save_total_limit == 2


def test_parse_args_overrides():
    cli_args = [
        "--epochs", "10",
        "--batch_size", "16",
        "--learning_rate", "3e-5",
        "--weight_decay", "0.05",
        "--seed", "123",
        "--max_length", "256",
        "--output_dir", "/tmp/ner_test",
        "--fp16",
        "--early_stopping_patience", "2",
        "--no_test",
    ]
    cfg = parse_args(cli_args)
    assert cfg.epochs == 10
    assert cfg.batch_size == 16
    assert cfg.learning_rate == 3e-5
    assert cfg.weight_decay == 0.05
    assert cfg.seed == 123
    assert cfg.max_length == 256
    assert cfg.output_dir == "/tmp/ner_test"
    assert cfg.fp16 is True
    assert cfg.early_stopping_patience == 2
    assert cfg.do_test is False


def test_clinical_ner_dataset(tokenizer, label_maps):
    labels, label2id, id2label = label_maps
    sample_records = [
        {
            "id": "SYN-TEST-01",
            "text": "Patient has mild headache and nausea.",
            "entities": [
                {"text": "headache", "start": 17, "end": 25, "label": "SYMPTOM", "assertion": "PRESENT"},
                {"text": "nausea", "start": 30, "end": 36, "label": "SYMPTOM", "assertion": "PRESENT"},
            ],
            "split": "train",
            "synthetic": True,
        },
        {
            "id": "SYN-TEST-02",
            "text": "Prescribed aspirin 100 mg daily.",
            "entities": [
                {"text": "aspirin", "start": 11, "end": 18, "label": "DRUG", "assertion": "PRESENT"},
                {"text": "100 mg", "start": 19, "end": 25, "label": "DOSAGE", "assertion": "PRESENT"},
                {"text": "daily", "start": 26, "end": 31, "label": "FREQUENCY", "assertion": "PRESENT"},
            ],
            "split": "train",
            "synthetic": True,
        },
    ]

    dataset = ClinicalNERDataset(
        records=sample_records,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=64,
    )
    assert len(dataset) == 2

    item0 = dataset[0]
    assert "input_ids" in item0
    assert "attention_mask" in item0
    assert "labels" in item0
    assert len(item0["input_ids"]) == len(item0["attention_mask"]) == len(item0["labels"])

    # First token ([CLS]) and last token ([SEP]) must have -100 label
    assert item0["labels"][0] == -100
    assert item0["labels"][-1] == -100

    # Verify B-SYMPTOM is present in labels
    assert label2id["B-SYMPTOM"] in item0["labels"]

    item1 = dataset[1]
    assert label2id["B-DRUG"] in item1["labels"]
    assert label2id["B-DOSAGE"] in item1["labels"]
    assert label2id["B-FREQUENCY"] in item1["labels"]

