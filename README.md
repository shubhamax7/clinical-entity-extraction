# Clinical Entity Extraction — Bio_ClinicalBERT NER Pipeline

Fine-tuning [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
for token-level clinical Named Entity Recognition (NER) using the synthetic **SYN-CLINNER-v1** dataset.

> **Important**: SYN-CLINNER-v1 is a fully synthetic, template-generated dataset intended for
> pipeline development and smoke-testing only. It is **not** real patient data, **not** clinically
> validated, and **not** a benchmark. Results on this dataset do not represent clinical-grade NER performance.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset: SYN-CLINNER-v1](#3-dataset-syn-clinner-v1)
4. [Model & Training](#4-model--training)
5. [Quick Start (Local — CPU only)](#5-quick-start-local--cpu-only)
6. [Google Colab GPU Training](#6-google-colab-gpu-training)
7. [Source Code Reference](#7-source-code-reference)
8. [Testing](#8-testing)
9. [Configuration Reference](#9-configuration-reference)
10. [Artifacts & Outputs](#10-artifacts--outputs)
11. [Known Limitations](#11-known-limitations)
12. [Dependency Notes](#12-dependency-notes)

---

## 1. Project Overview

This repository provides a production-quality, fully self-contained pipeline to:

- Load and structurally validate the SYN-CLINNER-v1 clinical NER dataset.
- Align character-level entity spans to WordPiece token boundaries using the Bio_ClinicalBERT fast tokenizer.
- Fine-tune `emilyalsentzer/Bio_ClinicalBERT` for 19-class BIO token classification using Hugging Face Trainer.
- Evaluate with entity-level Precision, Recall, and F1 via `seqeval`.
- Save the best checkpoint to Google Drive with full reproducibility metadata.

**All GPU training must run on Google Colab.** The Mac/local environment is used only for preprocessing, testing, and code development.

---

## 2. Repository Structure

```
clinical-entity-extraction/
│
├── datasets/
│   └── syn_clinner_v1/
│       ├── data/
│       │   ├── train.jsonl            # 350 training records
│       │   ├── validation.jsonl       # 75 validation records
│       │   ├── test.jsonl             # 75 held-out test records
│       │   ├── dataset_metadata.json  # Label set, SHA-256 hashes, metadata
│       │   ├── validation_report.json # Structural validation results
│       │   └── validation_report.md   # Human-readable validation report
│       ├── docs/
│       │   ├── dataset_card.md        # Full dataset documentation
│       │   └── annotation_guidelines.md  # Labelling rules
│       └── validate_dataset.py        # Standalone dataset validator
│
├── notebooks/
│   └── clinicalbert_training.ipynb   # Full Colab training notebook (12 code cells)
│
├── src/
│   ├── preprocessing.py   # Data loading, BIO alignment, tokenization
│   ├── train.py           # Training pipeline, TrainConfig, Trainer setup
│   ├── evaluator.py       # seqeval metrics, error analysis, report saving
│   ├── extractor.py       # (Inference / extraction utilities)
│   ├── normalizer.py      # (Entity normalization utilities)
│   └── utils.py           # (Shared utilities)
│
├── tests/
│   ├── test_preprocessing.py   # 45 tests: labels, loading, validation, alignment
│   ├── test_evaluator.py       # 6 tests: metrics, alignment, reports
│   └── test_train_config.py    # 22 tests: config, warmup, precision, checkpoints
│
├── requirements.txt       # Project dependencies (torch excluded for Colab compatibility)
├── app.py                 # (Placeholder for future Streamlit app)
└── README.md              # This file
```

---

## 3. Dataset: SYN-CLINNER-v1

### Summary

| Property | Value |
|---|---|
| Records | 500 total (train 350 / validation 75 / test 75) |
| Entity annotations | 5,820 |
| Entity types | 9 |
| BIO labels | 19 (O + B-/I- × 9 types) |
| Assertion values | PRESENT, ABSENT, POSSIBLE, HISTORICAL, FAMILY_HISTORY |
| Medical specialties | General medicine, cardiology, neurology, respiratory, endocrinology, gastroenterology, orthopedics, emergency |
| Language | English (US and UK variants) |
| Data type | Fully synthetic — no real patient records |

### Entity Types

| Type | Category | Examples |
|---|---|---|
| `DISEASE` | Primary | `type 2 diabetes mellitus`, `CHF`, `UTI` |
| `SYMPTOM` | Primary | `persistent cough`, `pedal oedema`, `SOB` |
| `DRUG` | Primary | `metformin`, `ASA`, `normal saline` |
| `PROCEDURE` | Primary | `CABG`, `lumbar puncture`, `ORIF` |
| `INVESTIGATION` | Primary | `ECG`, `HbA1c`, `MRI brain` |
| `DOSAGE` | Attribute | `500 mg`, `2 puffs`, `20 units` |
| `ROUTE` | Attribute | `IV`, `orally`, `subcutaneously` |
| `FREQUENCY` | Attribute | `BID`, `twice daily`, `stat` |
| `DURATION` | Attribute | `5 days`, `3 months` |

### Annotation Statistics

| Label | Count |
|---|---:|
| SYMPTOM | 1,629 |
| DISEASE | 1,311 |
| DRUG | 652 |
| DOSAGE | 484 |
| INVESTIGATION | 479 |
| FREQUENCY | 430 |
| PROCEDURE | 407 |
| ROUTE | 279 |
| DURATION | 149 |
| **Total** | **5,820** |

Assertions: PRESENT 4,040 · HISTORICAL 671 · ABSENT 622 · POSSIBLE 268 · FAMILY_HISTORY 219.

### Record Schema

```json
{
  "id": "SYN-0001",
  "text": "Patient presents with chest pain and shortness of breath.",
  "entities": [
    {
      "text": "chest pain",
      "start": 21,
      "end": 31,
      "label": "SYMPTOM",
      "assertion": "PRESENT",
      "normalized_name": "chest pain",
      "normalization_status": "UNMAPPED_DEMO"
    }
  ],
  "tokens": ["Patient", "presents", "with", "chest", "pain", "and", "shortness", "of", "breath", "."],
  "bio_tags": ["O", "O", "O", "B-SYMPTOM", "I-SYMPTOM", "O", "B-SYMPTOM", "I-SYMPTOM", "I-SYMPTOM", "O"],
  "split": "train",
  "synthetic": true,
  "token_offsets": [[0, 7], [8, 16], ...],
  "meta": {"specialty": "cardiology", "length_class": "medium", "template_ids": ["CARD-03"]}
}
```

- Offsets are Python `str` character indices, **half-open** `[start, end)`.
- `meta.template_ids` is generator provenance — **do not use as a model feature**.
- `normalization_status` is always `UNMAPPED_DEMO`. No SNOMED CT / RxNorm / LOINC codes are present.

### Split Hygiene

Templates 0–5 are train-only, 6–7 validation-only, 8–9 test-only.  
Near-duplicate rejection: notes with word-3-gram Jaccard ≥ 0.55 to any accepted note are discarded.  
Maximum cross-split similarity observed: 0.09.

### Dataset Files

| File | SHA-256 |
|---|---|
| `train.jsonl` | `db5a9ac16162d1bff5f34d660bcf2bf476f5fd1509a6c917ab22eedfca3d3473` |
| `validation.jsonl` | `7e616805f03e5323c88ceb3dc5352691b9ed9cfe4e7afcdde6089bba4a3154b0` |
| `test.jsonl` | `ea29554be4eb1ee3cd449d3178286c12f1ed0f5e2ac48c4bf0c8233caa7eaf45` |

---

## 4. Model & Training

### Base Model

[`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — BERT-base pretrained on MIMIC-III clinical notes.

### BIO Label Vocabulary

19 deterministic labels derived from the fixed, alphabetically-sorted entity type list:

```
ID  Label           ID  Label
 0  O               10  B-INVESTIGATION
 1  B-DISEASE       11  I-INVESTIGATION
 2  I-DISEASE       12  B-PROCEDURE
 3  B-DOSAGE        13  I-PROCEDURE
 4  I-DOSAGE        14  B-ROUTE
 5  B-DRUG          15  I-ROUTE
 6  I-DRUG          16  B-SYMPTOM
 7  B-DURATION      17  I-SYMPTOM
 8  I-DURATION      18  (B-FREQUENCY mapped as needed)
 9  B-FREQUENCY     ...
```

The vocabulary is **independent of split contents** — every run produces an identical `label2id` / `id2label`.

### Training Configuration

| Parameter | Value |
|---|---|
| Max sequence length | 512 |
| Batch size | 8 per device |
| Epochs | 5 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup steps | 22 (= `ceil(ceil(350/8) × 5 × 0.1)`) |
| Early stopping patience | 3 epochs |
| Evaluation metric | Entity-level F1 (seqeval) |
| Best-model selection | Highest validation F1 |

### Mixed Precision

Hardware detection uses **CUDA compute capability**, not `torch.cuda.is_bf16_supported()` (which returns incorrect results on T4):

| GPU | Compute Cap. | Precision |
|---|---|---|
| Tesla T4 | 7.5 | **FP16** |
| A100 | 8.0 | **BF16** |
| H100 | 9.0 | **BF16** |
| CPU | — | None |

---

## 5. Quick Start (Local — CPU only)

> **Do not run training locally.** Model weights are large and training requires GPU. Local setup is for code development and running tests only.

### Prerequisites

- Python 3.10+
- macOS / Linux

### Setup

```bash
git clone https://github.com/shubhamax7/clinical-entity-extraction.git
cd clinical-entity-extraction

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (torch excluded — install separately if needed locally)
pip install transformers==5.22.1 accelerate seqeval pytest
```

### Run Tests

```bash
pytest tests/ -v
# Expected: 73 passed
```

### Validate Dataset

```bash
python datasets/syn_clinner_v1/validate_dataset.py
```

### Preprocessing Smoke Test

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from preprocessing import load_jsonl, build_label_maps, load_tokenizer, encode_record

labels, label2id, id2label = build_label_maps()
print(f'{len(labels)} BIO labels')  # → 19

recs = load_jsonl('datasets/syn_clinner_v1/data/train.jsonl')
print(f'{len(recs)} training records')  # → 350

tok = load_tokenizer(local_files_only=False)
enc = encode_record(recs[0], tok, label2id, max_length=512)
print(f'First record: {len(enc.input_ids)} tokens, {len(enc.assertions)} entities')
"
```

---

## 6. Google Colab GPU Training

### Prerequisites

- Google account with Google Drive
- Colab runtime: **Runtime → Change runtime type → T4 GPU**

### Steps

1. Open the notebook:  
   [`notebooks/clinicalbert_training.ipynb`](notebooks/clinicalbert_training.ipynb)

2. **Upload to Colab** or open directly from GitHub via [colab.research.google.com](https://colab.research.google.com).

3. Set runtime to **T4 GPU**: Runtime → Change runtime type → Hardware accelerator: GPU.

4. Run cells **top to bottom** — each cell asserts its own preconditions and stops if they fail:

| Cell | Purpose |
|---|---|
| **1 — GPU Verification** | Confirms CUDA, shows device name, compute capability, VRAM, `nvidia-smi` |
| **2 — Install Dependencies** | Pins `transformers==5.22.1`, `accelerate>=1.1.0`, `seqeval==1.2.2`. Verifies CUDA PyTorch is intact. |
| **3 — Mount Google Drive** | Mounts Drive and verifies the checkpoint directory is writable |
| **4 — Clone Repository** | Clones (or pulls) from GitHub; displays git SHA; verifies required files |
| **5 — Dataset Verification** | Loads all 3 splits; validates counts (350/75/75), entity labels, offsets |
| **6 — BIO Label Verification** | Asserts 19 labels, `O`=0, invertible mapping |
| **7 — Tokenizer Smoke Test** | Downloads tokenizer; verifies alignment, special tokens=-100, valid IDs |
| **8 — Training Preflight** | Builds `TrainingArguments`; verifies warmup=22, FP16 on T4, no `warmup_ratio` error |
| **9 — Load Model & Train** | Downloads model weights, runs `train_model()`, OOM fallback to batch 4 + grad_accum 2 |
| **10 — Test Evaluation** | Displays held-out test set Precision / Recall / F1 and per-entity report |
| **11 — Save Manifest** | Saves reproducibility manifest: git SHA, package versions, dataset hashes |
| **12 — Artifact Verification** | Verifies all required files exist, are non-empty, and are readable from Drive |

### Expected Runtime

~45–75 minutes on T4 GPU for 5 epochs over 350 records.

### Checkpoint Directory

Artifacts are saved to:
```
/content/drive/MyDrive/clinicalbert_checkpoints/syn_clinner_v1/
├── config.json
├── tokenizer_config.json
├── vocab.txt
├── model.safetensors
├── training_metadata.json
├── test_report.json  /  test_report.md
├── validation_report.json  /  validation_report.md
├── reproducibility_manifest.json
└── checkpoint-*/       ← up to save_total_limit=2 intermediate checkpoints
```

### Resuming After Interruption

In Cell 9, set:
```python
RESUME_FROM_CHECKPOINT = True
```
This resolves the latest valid `checkpoint-*` directory in `DRIVE_OUTPUT_DIR` automatically.

---

## 7. Source Code Reference

### [`src/preprocessing.py`](src/preprocessing.py)

| Symbol | Description |
|---|---|
| `ENTITY_TYPES` | Fixed 9-tuple of entity type names (alphabetically sorted) |
| `MODEL_NAME` | `"emilyalsentzer/Bio_ClinicalBERT"` |
| `IGNORE_INDEX` | `-100` (used for special/padding tokens in labels) |
| `build_label_maps()` | Returns `(labels, label2id, id2label)` — 19 deterministic BIO labels |
| `load_tokenizer(local_files_only)` | Loads the Bio_ClinicalBERT fast tokenizer |
| `load_jsonl(path)` | Loads a JSONL file; raises `ValueError` on malformed lines |
| `validate_records(records)` | Structural + offset validation; raises `ValueError` on first problem |
| `align_offsets_to_bio(...)` | Converts character spans to per-token BIO label IDs |
| `encode_record(record, tokenizer, label2id, max_length)` | Returns `EncodedRecord` with aligned labels |

### [`src/train.py`](src/train.py)

| Symbol | Description |
|---|---|
| `TrainConfig` | Dataclass holding all training hyperparameters |
| `count_train_records(path)` | Counts non-empty JSONL lines; raises on missing/empty file |
| `calculate_warmup_steps(n, bs, epochs, ratio)` | `ceil(ceil(n/bs) × epochs × ratio)` |
| `build_training_args_dict(config, num_train_records)` | Builds `TrainingArguments` kwargs; uses `warmup_steps` not `warmup_ratio`; applies compute-capability-based precision |
| `create_training_arguments(config, ...)` | Wraps `build_training_args_dict` into a `TrainingArguments` instance |
| `resolve_resume_checkpoint(resume, output_dir)` | Safely resolves checkpoint path from `True`, `"latest"`, or an explicit path |
| `train_model(config)` | Full training workflow: load → validate → encode → train → evaluate → save |

### [`src/evaluator.py`](src/evaluator.py)

| Symbol | Description |
|---|---|
| `align_predictions_and_labels(preds, labels, id2label)` | Strips `-100` tokens; handles 2D and 3D prediction tensors |
| `evaluate_sequences(y_true, y_pred)` | Entity-level P/R/F1 via seqeval + per-class breakdown + error analysis |
| `evaluate_predictions(predictions, label_ids, id2label)` | End-to-end: logits → BIO strings → metrics |
| `build_compute_metrics(id2label)` | Returns a `compute_metrics` closure for Hugging Face Trainer |
| `save_evaluation_report(result, output_dir, prefix)` | Saves JSON + Markdown report |

---

## 8. Testing

```bash
pytest tests/ -v
```

**73 tests, 0 failures** covering:

| Test File | Coverage |
|---|---|
| `test_preprocessing.py` | Label vocabulary (19 labels, O=0, deterministic), JSONL loading, record validation, BIO alignment, multi-word entities, subword tokens, special token masking, truncation, round-trip encoding of all 500 records |
| `test_evaluator.py` | Prediction alignment (2D/3D), perfect-match metrics, error detection, `compute_metrics` closure, report saving |
| `test_train_config.py` | Default config, CLI argument parsing, `ClinicalNERDataset`, checkpoint resolution (9 cases), warmup calculation (7 cases), mixed precision on T4/Ampere/Hopper/CPU (8 cases), `TrainingArguments` compatibility |

---

## 9. Configuration Reference

### `TrainConfig` fields

| Field | Default | Description |
|---|---|---|
| `train_file` | `datasets/syn_clinner_v1/data/train.jsonl` | Training JSONL path |
| `val_file` | `datasets/syn_clinner_v1/data/validation.jsonl` | Validation JSONL path |
| `test_file` | `datasets/syn_clinner_v1/data/test.jsonl` | Test JSONL path |
| `model_name` | `emilyalsentzer/Bio_ClinicalBERT` | HuggingFace model ID |
| `output_dir` | `artifacts/clinicalbert_ner` | Checkpoint and artifact directory |
| `max_length` | `512` | Maximum token sequence length |
| `batch_size` | `8` | Per-device batch size |
| `epochs` | `5` | Training epochs |
| `learning_rate` | `2e-5` | AdamW learning rate |
| `weight_decay` | `0.01` | Weight decay |
| `warmup_ratio` | `0.1` | Fraction of total steps used for warmup (converted to `warmup_steps` internally) |
| `seed` | `42` | Random seed |
| `early_stopping_patience` | `3` | Epochs without improvement before stopping |
| `fp16` | `False` | Explicitly request FP16 (auto-detected on CUDA if `False`) |
| `bf16` | `False` | Explicitly request BF16 (auto-detected on Ampere+ if `False`) |
| `save_total_limit` | `2` | Maximum number of checkpoints to retain |
| `logging_steps` | `10` | Training log interval in steps |
| `resume_from_checkpoint` | `None` | `None`, `True`, `"latest"`, or explicit path |
| `do_test` | `True` | Run held-out test evaluation after training |
| `local_files_only` | `False` | Load tokenizer from local cache only (offline mode) |

### Warmup Steps Formula

```
steps_per_epoch  = ceil(num_train_records / batch_size)
total_steps      = steps_per_epoch × epochs
warmup_steps     = ceil(total_steps × warmup_ratio)

# Default: ceil(ceil(350/8) × 5 × 0.1) = ceil(44 × 5 × 0.1) = ceil(22) = 22
```

---

## 10. Artifacts & Outputs

After a successful training run the output directory contains:

| File | Contents |
|---|---|
| `config.json` | Model architecture configuration |
| `tokenizer_config.json` | Tokenizer configuration |
| `vocab.txt` | WordPiece vocabulary |
| `model.safetensors` | Fine-tuned model weights (or `pytorch_model.bin` on older transformers) |
| `training_metadata.json` | Model name, labels, config, device, train/val/test metrics |
| `validation_report.json` / `.md` | Entity-level P/R/F1 + per-class + error analysis on validation set |
| `test_report.json` / `.md` | Entity-level P/R/F1 + per-class + error analysis on held-out test set |
| `reproducibility_manifest.json` | Git SHA, package versions, dataset SHA-256 hashes, training config |
| `checkpoint-N/` | Intermediate Trainer checkpoint (up to `save_total_limit`) |

---

## 11. Known Limitations

1. **Synthetic data inflates scores.** Shared entity lexicon between splits means strong F1 here mainly reflects memorisation, not clinical NER generalisation.
2. **Template artefacts.** A model trained here can learn 13 × 10 sentence frames — much simpler than real EHR text.
3. **No human annotation.** Labels are correct by construction but have not been adjudicated.
4. **No real-world noise.** No typos, OCR errors, section headers, or de-identification tokens.
5. **Attribute linking.** DOSAGE/ROUTE/FREQUENCY/DURATION inherit the nearest drug's assertion; no explicit drug–attribute relation is modelled.
6. **Compute capability detection.** `torch.cuda.is_bf16_supported()` returns `True` on T4 in some PyTorch versions; this pipeline uses `torch.cuda.get_device_capability()[0] >= 8` instead.

---

## 12. Dependency Notes

### Colab Installation (Cell 2 of notebook)

```python
# Exact pins verified against this codebase
transformers==5.22.1
accelerate>=1.1.0
seqeval==1.2.2
huggingface_hub>=0.24.0,<3.0.0

# torch is intentionally excluded — Colab provides CUDA builds.
# Installing torch from pip on Colab replaces the CUDA build with a CPU-only build.
```

### Transformers 5.x Compatibility Notes

| Issue | Fix applied |
|---|---|
| `TrainingArguments` removed `warmup_ratio` | Converted to `warmup_steps` in `build_training_args_dict()` |
| `evaluation_strategy` renamed to `eval_strategy` | Auto-detected via `inspect.signature()` |
| `is_bf16_supported()` incorrect on T4 | Replaced with `get_device_capability()[0] >= 8` |

### Local Development

```bash
pip install transformers==5.22.1 accelerate seqeval pytest
# Do NOT install torch from requirements.txt on Colab
```

---

## Git History

| Commit | Description |
|---|---|
| `e77a689` | Rebuild Colab notebook with full preflight, fix deps and mixed-precision detection |
| `884994b` | Fix training arguments compatibility with Transformers 5.x |
| `f2e364d` | Implement dynamic warmup steps and robust mixed precision selection |
| `74bdb8b` | Add local tokenizer loading and checkpoint resolution |
| `79cc23e` | Add Bio_ClinicalBERT fine-tuning pipeline, training, and evaluation modules |
| `0a68102` | Implement tokenization, record validation, and BIO label mapping |
| `125c473` | Add SYN-CLINNER-v1 dataset and validation tools |

---

> **Disclaimer**: This project uses a fully synthetic dataset. No real patient data has been accessed, used, or stored. Drug doses, frequencies, and clinical details in the dataset are illustrative and must not be used as clinical guidance. Do not claim clinical readiness based on results from this pipeline.
