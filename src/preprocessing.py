"""Data loading, annotation validation, BIO alignment, and encoding.

Pipeline
--------
load_jsonl(path) -> list[record]
validate_records(records) -> None           # raises on structural errors
build_label_maps() -> (labels, label2id, id2label)
load_tokenizer() -> PreTrainedTokenizerFast
encode_record(record, tokenizer, label2id, max_length) -> EncodedRecord

Design notes
------------
- Label vocabulary is fixed and deterministic (ENTITY_TYPES sorted, 19 tags).
  It does NOT depend on which labels happen to appear in a split.
- align_offsets_to_bio() operates on tokenizer offset_mapping (WordPiece),
  not on the regex tokens / bio_tags stored in the JSONL.
- Assertion values (PRESENT, ABSENT, …) are preserved in EncodedRecord but
  are NOT part of the NER label vocabulary.
- meta, regex tokens, and precomputed bio_tags are ignored as model inputs.
- Truncated entities (cut by max_length) are detected and reported.
- No model weights are loaded; only the fast tokenizer is used.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical entity type order: sorted alphabetically and fixed at design time.
# This guarantees a deterministic 19-label vocabulary regardless of which
# entity types happen to appear in a given split.
ENTITY_TYPES: tuple[str, ...] = (
    "DISEASE",
    "DOSAGE",
    "DRUG",
    "DURATION",
    "FREQUENCY",
    "INVESTIGATION",
    "PROCEDURE",
    "ROUTE",
    "SYMPTOM",
)

# Model identifier.  Only the tokenizer is loaded on this machine.
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Ignored by HuggingFace loss and metrics.
IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------


def build_label_maps() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Return the fixed, deterministic 19-label BIO vocabulary.

    Layout:  O  B-DISEASE  I-DISEASE  B-DOSAGE  I-DOSAGE  …  B-SYMPTOM  I-SYMPTOM
    Total:   1  + 9 * 2   = 19 labels.

    The vocabulary is independent of the data; every split produces identical
    label2id / id2label mappings.
    """
    labels: list[str] = ["O"]
    for entity_type in ENTITY_TYPES:
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")

    label2id: dict[str, int] = {label: idx for idx, label in enumerate(labels)}
    id2label: dict[int, str] = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def load_tokenizer(local_files_only: bool = True):
    """Load the Bio_ClinicalBERT fast tokenizer (no model weights).

    Parameters
    ----------
    local_files_only:
        If True, only load from local cache without network calls (safe default for local Mac).
        In Google Colab or fresh cloud environments, pass False to download from Hugging Face Hub.
    """
    from transformers import AutoTokenizer  # import here to keep module importable

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=local_files_only,
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records and report malformed lines precisely."""
    path = Path(path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

# Fields every entity must carry (for NER purposes; assertion is checked too).
_REQUIRED_ENTITY_FIELDS = ("text", "start", "end", "label", "assertion")


def validate_records(records: Iterable[dict[str, Any]]) -> None:
    """Validate record structure, entity spans, labels, assertion values, and overlap.

    Raises ValueError on the first structural problem found.
    Does NOT check tokenizer alignment (that happens in encode_record).
    """
    valid_labels = set(ENTITY_TYPES)
    valid_assertions = {"PRESENT", "ABSENT", "POSSIBLE", "HISTORICAL", "FAMILY_HISTORY"}
    seen_ids: set[str] = set()

    for record_index, record in enumerate(records):
        record_id = record.get("id")
        text = record.get("text")
        entities = record.get("entities")

        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Record {record_index}: missing valid id")

        if record_id in seen_ids:
            raise ValueError(f"Duplicate record id: {record_id}")
        seen_ids.add(record_id)

        if not isinstance(text, str) or not text:
            raise ValueError(f"{record_id}: text must be a non-empty string")

        if not isinstance(entities, list):
            raise ValueError(f"{record_id}: entities must be a list")

        spans: list[tuple[int, int, str]] = []

        for entity_index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(
                    f"{record_id}: entity {entity_index} must be an object"
                )

            # Check required fields exist.
            for field_name in _REQUIRED_ENTITY_FIELDS:
                if field_name not in entity:
                    raise ValueError(
                        f"{record_id}: entity {entity_index} missing field '{field_name}'"
                    )

            start = entity["start"]
            end = entity["end"]
            entity_text = entity["text"]
            label = entity["label"]
            assertion = entity["assertion"]

            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
            ):
                raise ValueError(
                    f"{record_id}: entity {entity_index} offsets must be integers"
                )

            if not isinstance(entity_text, str) or not entity_text:
                raise ValueError(
                    f"{record_id}: entity {entity_index} has invalid text"
                )

            if not isinstance(label, str) or label not in valid_labels:
                raise ValueError(
                    f"{record_id}: entity {entity_index} has unknown label {label!r}"
                )

            if not isinstance(assertion, str) or assertion not in valid_assertions:
                raise ValueError(
                    f"{record_id}: entity {entity_index} has invalid assertion {assertion!r}"
                )

            if not (0 <= start < end <= len(text)):
                raise ValueError(
                    f"{record_id}: invalid span ({start}, {end}) "
                    f"for text length {len(text)}"
                )

            actual_text = text[start:end]
            if actual_text != entity_text:
                raise ValueError(
                    f"{record_id}: offset mismatch for {entity_text!r}; "
                    f"text[{start}:{end}] is {actual_text!r}"
                )

            spans.append((start, end, label))

        # Flat BIO tagging: overlapping/nested spans are rejected.
        spans.sort(key=lambda s: (s[0], s[1]))
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"{record_id}: overlapping entities: {previous} and {current}"
                )


# ---------------------------------------------------------------------------
# BIO alignment
# ---------------------------------------------------------------------------


def align_offsets_to_bio(
    offset_mapping: Iterable[tuple[int, int]],
    entities: list[dict[str, Any]],
    label2id: dict[str, int],
    record_id: str = "",
) -> tuple[list[int], list[str]]:
    """Convert character-level entity spans into token-level BIO label IDs.

    Parameters
    ----------
    offset_mapping:
        Sequence of (char_start, char_end) pairs as returned by a HuggingFace
        fast tokenizer (including the special-token sentinels at (0, 0)).
    entities:
        List of entity dicts from the record (each with 'start', 'end', 'label').
        Entities MUST NOT overlap (validated upstream).
    label2id:
        Label-to-int mapping from build_label_maps().
    record_id:
        Used in warning messages only.

    Returns
    -------
    label_ids : list[int]
        Per-token label IDs, length == len(offset_mapping).
        Special tokens and padding (start == end in offset_mapping) get IGNORE_INDEX.
    truncated_entity_texts : list[str]
        Entity texts that were partially or fully cut by max_length truncation.
        Empty list when no truncation occurred.
    """
    offsets = list(offset_mapping)
    n = len(offsets)
    label_ids = [IGNORE_INDEX] * n

    # Mark real content tokens as O; leave special tokens as IGNORE_INDEX.
    for idx, (cs, ce) in enumerate(offsets):
        if cs != ce:                         # non-empty span → real token
            label_ids[idx] = label2id["O"]

    sorted_entities = sorted(entities, key=lambda e: (e["start"], e["end"]))
    truncated: list[str] = []

    for entity in sorted_entities:
        entity_start: int = entity["start"]
        entity_end: int = entity["end"]
        entity_type: str = entity["label"]
        entity_text: str = entity.get("text", "")

        begin_label = f"B-{entity_type}"
        inside_label = f"I-{entity_type}"

        if begin_label not in label2id or inside_label not in label2id:
            raise ValueError(f"{record_id}: Unknown entity type: {entity_type!r}")

        # Collect all real tokens that overlap with this entity span.
        overlapping: list[int] = []
        for token_idx, (ts, te) in enumerate(offsets):
            if ts == te:
                continue                      # special / padding token
            if ts >= entity_end or te <= entity_start:
                continue                      # no overlap
            # A content token must lie fully inside the entity boundary.
            # If it crosses, the tokenization is inconsistent with the annotations.
            if ts < entity_start or te > entity_end:
                raise ValueError(
                    f"{record_id}: Token span ({ts}, {te}) crosses "
                    f"entity boundary ({entity_start}, {entity_end}) "
                    f"for entity {entity_text!r}"
                )
            overlapping.append(token_idx)

        if not overlapping:
            # The entity is entirely beyond the last real token: truncated.
            truncated.append(entity_text)
            warnings.warn(
                f"{record_id}: Entity {entity_text!r} "
                f"[{entity_start}:{entity_end}] not found in token sequence "
                f"(likely truncated by max_length).",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Verify the collected tokens cover the full entity exactly.
        first_ts = offsets[overlapping[0]][0]
        last_te = offsets[overlapping[-1]][1]

        if first_ts != entity_start or last_te != entity_end:
            raise ValueError(
                f"{record_id}: Token coverage mismatch for entity "
                f"{entity_text!r}: entity=({entity_start}, {entity_end}), "
                f"tokens cover ({first_ts}, {last_te}). "
                f"Entity does not fall on token boundaries."
            )

        # Assign BIO tags; detect annotation conflicts (overlapping entities
        # are caught upstream, but guard here as well).
        for position, token_idx in enumerate(overlapping):
            if label_ids[token_idx] != label2id["O"]:
                raise ValueError(
                    f"{record_id}: Conflicting annotation at token index "
                    f"{token_idx} while processing entity {entity_text!r}"
                )
            tag = begin_label if position == 0 else inside_label
            label_ids[token_idx] = label2id[tag]

    return label_ids, truncated


# ---------------------------------------------------------------------------
# Encoded record
# ---------------------------------------------------------------------------


@dataclass
class EncodedRecord:
    """Output of encode_record() — everything needed for model training / eval.

    Fields
    ------
    record_id:        Original record ID (e.g. 'SYN-0001').
    split:            'train', 'validation', or 'test'.
    input_ids:        Token IDs from the tokenizer.
    attention_mask:   1 for real tokens, 0 for padding.
    label_ids:        BIO label IDs; IGNORE_INDEX for special/padding tokens.
    offset_mapping:   (char_start, char_end) per token (for span recovery).
    assertions:       List of (entity_text, label, assertion) for every entity
                      that survived truncation, in document order.
                      Assertions are preserved but NOT part of the NER labels.
    truncated_entities: Entity texts cut off by max_length (empty if none).
    """

    record_id: str
    split: str
    input_ids: list[int]
    attention_mask: list[int]
    label_ids: list[int]
    offset_mapping: list[tuple[int, int]]
    assertions: list[tuple[str, str, str]] = field(default_factory=list)
    truncated_entities: list[str] = field(default_factory=list)


def encode_record(
    record: dict[str, Any],
    tokenizer,
    label2id: dict[str, int],
    max_length: int = 512,
) -> EncodedRecord:
    """Tokenize a record and align character-level entity spans to WordPiece tokens.

    Parameters
    ----------
    record:
        A validated record dict (from load_jsonl / validate_records).
    tokenizer:
        A fast HuggingFace tokenizer (from load_tokenizer()).
    label2id:
        From build_label_maps().
    max_length:
        Maximum total sequence length including special tokens.
        Entities truncated by this limit are reported but not raised.

    Returns
    -------
    EncodedRecord
    """
    text: str = record["text"]
    entities: list[dict[str, Any]] = record.get("entities", [])

    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,          # collation / batching handles padding
        return_offsets_mapping=True,
        return_attention_mask=True,
    )

    offset_mapping: list[tuple[int, int]] = [
        tuple(pair) for pair in encoding["offset_mapping"]  # type: ignore[misc]
    ]

    label_ids, truncated = align_offsets_to_bio(
        offset_mapping=offset_mapping,
        entities=entities,
        label2id=label2id,
        record_id=record.get("id", ""),
    )

    # Build assertion metadata for entities that were NOT truncated.
    truncated_set = set(truncated)
    assertions: list[tuple[str, str, str]] = []
    for entity in sorted(entities, key=lambda e: e["start"]):
        et = entity.get("text", "")
        if et not in truncated_set:
            assertions.append((et, entity["label"], entity["assertion"]))

    return EncodedRecord(
        record_id=record.get("id", ""),
        split=record.get("split", ""),
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        label_ids=label_ids,
        offset_mapping=offset_mapping,
        assertions=assertions,
        truncated_entities=truncated,
    )
