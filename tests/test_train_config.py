"""Unit tests for src/train.py configuration, arguments, and Dataset class."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import build_label_maps, load_tokenizer
from train import (
    ClinicalNERDataset,
    TrainConfig,
    build_training_args_dict,
    calculate_warmup_steps,
    count_train_records,
    parse_args,
    resolve_resume_checkpoint,
)


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


class TestResolveResumeCheckpoint:
    """Unit tests for resolve_resume_checkpoint in train.py."""

    def test_none_or_false_returns_none(self, tmp_path):
        assert resolve_resume_checkpoint(None, tmp_path) is None
        assert resolve_resume_checkpoint(False, tmp_path) is None
        assert resolve_resume_checkpoint("", tmp_path) is None

    def test_boolean_true_with_checkpoints(self, tmp_path):
        cp10 = tmp_path / "checkpoint-10"
        cp20 = tmp_path / "checkpoint-20"
        cp10.mkdir()
        cp20.mkdir()
        (cp10 / "trainer_state.json").write_text("{}")
        (cp20 / "trainer_state.json").write_text("{}")

        # Resolves latest (checkpoint-20)
        assert resolve_resume_checkpoint(True, tmp_path) == str(cp20)

    def test_string_alias_true_latest_one(self, tmp_path):
        cp50 = tmp_path / "checkpoint-50"
        cp50.mkdir()
        (cp50 / "trainer_state.json").write_text("{}")

        for alias in ("true", "True", "TRUE", "latest", "1"):
            resolved = resolve_resume_checkpoint(alias, tmp_path)
            assert resolved == str(cp50), f"Failed for alias {alias}"

    def test_boolean_true_with_no_checkpoints_returns_none(self, tmp_path):
        # Empty output directory returns None (starts from scratch)
        assert resolve_resume_checkpoint(True, tmp_path) is None

    def test_nonexistent_output_dir_returns_none(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        assert resolve_resume_checkpoint(True, missing_dir) is None

    def test_explicit_checkpoint_dir_with_state(self, tmp_path):
        cp = tmp_path / "checkpoint-100"
        cp.mkdir()
        (cp / "trainer_state.json").write_text("{}")

        resolved = resolve_resume_checkpoint(str(cp), tmp_path)
        assert resolved == str(cp)

    def test_parent_output_dir_resolves_latest(self, tmp_path):
        out_dir = tmp_path / "models"
        cp1 = out_dir / "checkpoint-100"
        cp2 = out_dir / "checkpoint-200"
        cp1.mkdir(parents=True)
        cp2.mkdir(parents=True)
        (cp1 / "trainer_state.json").write_text("{}")
        (cp2 / "trainer_state.json").write_text("{}")

        # Passing the parent directory as the string path resolves checkpoint-200
        resolved = resolve_resume_checkpoint(str(out_dir), tmp_path)
        assert resolved == str(cp2)

    def test_nonexistent_explicit_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_resume_checkpoint(str(tmp_path / "phantom-checkpoint"), tmp_path)

    def test_invalid_explicit_dir_raises(self, tmp_path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="not a valid checkpoint directory"):
            resolve_resume_checkpoint(str(empty_dir), tmp_path)


class TestWarmupStepsCalculation:
    """Unit tests for dynamic warmup steps and dataset record counting."""

    def test_default_scenario_produces_22_steps(self):
        # 350 records, batch size 8, 5 epochs, warmup ratio 0.1
        # steps_per_epoch = ceil(350 / 8) = 44
        # total_steps = 44 * 5 = 220
        # warmup_steps = ceil(220 * 0.1) = 22
        steps = calculate_warmup_steps(
            num_records=350,
            batch_size=8,
            epochs=5,
            warmup_ratio=0.1,
        )
        assert steps == 22

    def test_different_batch_sizes_and_ratios(self):
        # batch_size=16: ceil(350 / 16) = 22; total = 22 * 3 = 66; warmup = ceil(66 * 0.1) = 7
        assert calculate_warmup_steps(num_records=350, batch_size=16, epochs=3, warmup_ratio=0.1) == 7

        # batch_size=10: ceil(100 / 10) = 10; total = 10 * 5 = 50; warmup = ceil(50 * 0.2) = 10
        assert calculate_warmup_steps(num_records=100, batch_size=10, epochs=5, warmup_ratio=0.2) == 10

        # warmup_ratio=0.0 -> 0 warmup steps
        assert calculate_warmup_steps(num_records=350, batch_size=8, epochs=5, warmup_ratio=0.0) == 0

        # 1 record -> ceil(1/8) = 1; total = 1 * 1 = 1; ceil(1 * 0.1) = 1
        assert calculate_warmup_steps(num_records=1, batch_size=8, epochs=1, warmup_ratio=0.1) == 1

    def test_invalid_parameters_raise_value_error(self):
        with pytest.raises(ValueError, match="Invalid num_records"):
            calculate_warmup_steps(num_records=0, batch_size=8, epochs=5, warmup_ratio=0.1)

        with pytest.raises(ValueError, match="Invalid num_records"):
            calculate_warmup_steps(num_records=-10, batch_size=8, epochs=5, warmup_ratio=0.1)

        with pytest.raises(ValueError, match="Invalid batch_size"):
            calculate_warmup_steps(num_records=350, batch_size=0, epochs=5, warmup_ratio=0.1)

        with pytest.raises(ValueError, match="Invalid epochs"):
            calculate_warmup_steps(num_records=350, batch_size=8, epochs=0, warmup_ratio=0.1)

        with pytest.raises(ValueError, match="Invalid warmup_ratio"):
            calculate_warmup_steps(num_records=350, batch_size=8, epochs=5, warmup_ratio=-0.1)

        with pytest.raises(ValueError, match="Invalid warmup_ratio"):
            calculate_warmup_steps(num_records=350, batch_size=8, epochs=5, warmup_ratio=1.5)

    def test_count_train_records(self, tmp_path):
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"id": 1}\n\n{"id": 2}\n{"id": 3}\n')
        assert count_train_records(train_file) == 3

    def test_count_train_records_empty_raises(self, tmp_path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("\n\n")
        with pytest.raises(ValueError, match="empty or contains no records"):
            count_train_records(empty_file)

    def test_count_train_records_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            count_train_records(tmp_path / "missing.jsonl")

    def test_build_training_args_dict_removes_warmup_ratio(self, tmp_path):
        train_file = tmp_path / "train.jsonl"
        train_file.write_text("\n".join(['{"id": %d}' % i for i in range(350)]))

        cfg = TrainConfig(
            train_file=str(train_file),
            output_dir=str(tmp_path / "out"),
            batch_size=8,
            epochs=5,
            warmup_ratio=0.1,
        )
        args_dict = build_training_args_dict(cfg)

        # Must have warmup_steps set to 22
        assert args_dict["warmup_steps"] == 22
        # Must NOT have warmup_ratio in TrainingArguments kwargs
        assert "warmup_ratio" not in args_dict


class TestMixedPrecisionLogic:
    """Unit tests verifying BF16 is only enabled when CUDA supports it, with FP16 fallback."""

    def test_cuda_with_bf16_support_enables_bf16(self, monkeypatch, tmp_path):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

        cfg = TrainConfig(output_dir=str(tmp_path), bf16=True)
        args_dict = build_training_args_dict(cfg, num_train_records=350)
        assert args_dict.get("bf16") is True
        assert args_dict.get("fp16") is not True

    def test_cuda_without_bf16_support_falls_back_to_fp16(self, monkeypatch, tmp_path):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

        cfg = TrainConfig(output_dir=str(tmp_path), bf16=True)
        args_dict = build_training_args_dict(cfg, num_train_records=350)
        # BF16 requested but unsupported -> falls back to FP16
        assert args_dict.get("fp16") is True
        assert args_dict.get("bf16") is not True

    def test_cuda_default_auto_detection_with_bf16(self, monkeypatch, tmp_path):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

        # Neither bf16 nor fp16 explicitly set -> auto-detects bf16 on supported CUDA
        cfg = TrainConfig(output_dir=str(tmp_path), bf16=False, fp16=False)
        args_dict = build_training_args_dict(cfg, num_train_records=350)
        assert args_dict.get("bf16") is True
        assert args_dict.get("fp16") is not True

    def test_cuda_default_auto_detection_without_bf16(self, monkeypatch, tmp_path):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

        # Neither bf16 nor fp16 explicitly set -> auto-detects fp16 on T4-like CUDA
        cfg = TrainConfig(output_dir=str(tmp_path), bf16=False, fp16=False)
        args_dict = build_training_args_dict(cfg, num_train_records=350)
        assert args_dict.get("fp16") is True
        assert args_dict.get("bf16") is not True

    def test_non_cuda_device_has_no_mixed_precision(self, monkeypatch, tmp_path):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        cfg = TrainConfig(output_dir=str(tmp_path), bf16=True, fp16=True)
        args_dict = build_training_args_dict(cfg, num_train_records=350)
        assert args_dict.get("fp16") is not True
        assert args_dict.get("bf16") is not True



