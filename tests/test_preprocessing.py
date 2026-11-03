"""Tests for src/preprocessing.py.

Covers:
  - Deterministic 19-label vocabulary
  - JSONL loading (all three splits)
  - Structural validation (valid + invalid cases)
  - BIO alignment with the real Bio_ClinicalBERT tokenizer
    - Single-token entities
    - Multi-word / multi-subword entities
    - Special-token masking (-100)
    - Assertion metadata preservation
    - Truncation detection
  - round-trip: label_id -> label string -> char span recovery
  - encode_record on real records from all three splits
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

# Module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    ENTITY_TYPES,
    IGNORE_INDEX,
    EncodedRecord,
    align_offsets_to_bio,
    build_label_maps,
    encode_record,
    load_jsonl,
    load_tokenizer,
    validate_records,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).parent.parent
DATA = REPO / "datasets" / "syn_clinner_v1" / "data"
TRAIN = DATA / "train.jsonl"
VALIDATION = DATA / "validation.jsonl"
TEST = DATA / "test.jsonl"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def label_maps():
    return build_label_maps()


@pytest.fixture(scope="module")
def train_records():
    return load_jsonl(TRAIN)


@pytest.fixture(scope="module")
def validation_records():
    return load_jsonl(VALIDATION)


@pytest.fixture(scope="module")
def test_records():
    return load_jsonl(TEST)


# ---------------------------------------------------------------------------
# 1. Label vocabulary
# ---------------------------------------------------------------------------


class TestLabelVocabulary:
    def test_total_labels_is_19(self, label_maps):
        labels, label2id, id2label = label_maps
        assert len(labels) == 19, f"Expected 19 labels, got {len(labels)}: {labels}"

    def test_O_is_index_0(self, label_maps):
        labels, label2id, id2label = label_maps
        assert labels[0] == "O"
        assert label2id["O"] == 0

    def test_all_entity_types_have_B_and_I(self, label_maps):
        labels, label2id, id2label = label_maps
        for et in ENTITY_TYPES:
            assert f"B-{et}" in label2id, f"B-{et} missing"
            assert f"I-{et}" in label2id, f"I-{et} missing"

    def test_label2id_and_id2label_are_inverse(self, label_maps):
        labels, label2id, id2label = label_maps
        for label, idx in label2id.items():
            assert id2label[idx] == label

    def test_vocabulary_is_deterministic(self):
        """Calling build_label_maps() twice yields identical results."""
        a = build_label_maps()
        b = build_label_maps()
        assert a[0] == b[0]
        assert a[1] == b[1]

    def test_entity_types_constant_has_9_types(self):
        assert len(ENTITY_TYPES) == 9
        expected = {
            "DISEASE", "DOSAGE", "DRUG", "DURATION", "FREQUENCY",
            "INVESTIGATION", "PROCEDURE", "ROUTE", "SYMPTOM",
        }
        assert set(ENTITY_TYPES) == expected


# ---------------------------------------------------------------------------
# 2. JSONL loading
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_train_count(self, train_records):
        assert len(train_records) == 350

    def test_validation_count(self, validation_records):
        assert len(validation_records) == 75

    def test_test_count(self, test_records):
        assert len(test_records) == 75

    def test_total_count(self, train_records, validation_records, test_records):
        assert len(train_records) + len(validation_records) + len(test_records) == 500

    def test_all_records_have_required_keys(self, train_records):
        required = {"id", "text", "entities", "split", "synthetic"}
        for r in train_records:
            missing = required - r.keys()
            assert not missing, f"{r['id']}: missing keys {missing}"

    def test_all_records_synthetic_true(self, train_records, validation_records, test_records):
        for r in train_records + validation_records + test_records:
            assert r.get("synthetic") is True, f"{r['id']}: synthetic != True"

    def test_split_field_matches_file(self, train_records, validation_records, test_records):
        for r in train_records:
            assert r["split"] == "train"
        for r in validation_records:
            assert r["split"] == "validation"
        for r in test_records:
            assert r["split"] == "test"

    def test_malformed_json_raises(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"id": "X"}\nnot_json\n')
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_jsonl(bad)


# ---------------------------------------------------------------------------
# 3. Structural validation
# ---------------------------------------------------------------------------


class TestValidateRecords:
    def test_valid_train_passes(self, train_records):
        validate_records(train_records)  # must not raise

    def test_valid_validation_passes(self, validation_records):
        validate_records(validation_records)

    def test_valid_test_passes(self, test_records):
        validate_records(test_records)

    def test_duplicate_id_raises(self):
        rec = {"id": "X-001", "text": "Hello.", "entities": [], "split": "train", "synthetic": True}
        with pytest.raises(ValueError, match="Duplicate"):
            validate_records([rec, rec.copy()])

    def test_missing_id_raises(self):
        rec = {"text": "Hello.", "entities": [], "split": "train", "synthetic": True}
        with pytest.raises(ValueError, match="missing valid id"):
            validate_records([rec])

    def test_bad_offset_raises(self):
        rec = {
            "id": "X-001", "text": "Hello world.", "split": "train", "synthetic": True,
            "entities": [{"text": "Hello", "start": 0, "end": 999,
                          "label": "SYMPTOM", "assertion": "PRESENT",
                          "normalized_name": "Hello", "normalization_status": "UNMAPPED_DEMO"}],
        }
        with pytest.raises(ValueError, match="invalid span"):
            validate_records([rec])

    def test_offset_text_mismatch_raises(self):
        rec = {
            "id": "X-001", "text": "Hello world.", "split": "train", "synthetic": True,
            "entities": [{"text": "Goodbye", "start": 0, "end": 5,
                          "label": "SYMPTOM", "assertion": "PRESENT",
                          "normalized_name": "Hello", "normalization_status": "UNMAPPED_DEMO"}],
        }
        with pytest.raises(ValueError, match="offset mismatch"):
            validate_records([rec])

    def test_overlapping_entities_raises(self):
        rec = {
            "id": "X-001", "text": "chest pain and shortness.", "split": "train", "synthetic": True,
            "entities": [
                {"text": "chest pain", "start": 0, "end": 10,
                 "label": "SYMPTOM", "assertion": "PRESENT",
                 "normalized_name": "chest pain", "normalization_status": "UNMAPPED_DEMO"},
                {"text": "pain and", "start": 6, "end": 14,
                 "label": "SYMPTOM", "assertion": "PRESENT",
                 "normalized_name": "pain", "normalization_status": "UNMAPPED_DEMO"},
            ],
        }
        with pytest.raises(ValueError, match="overlapping"):
            validate_records([rec])

    def test_unknown_label_raises(self):
        rec = {
            "id": "X-001", "text": "Hello.", "split": "train", "synthetic": True,
            "entities": [{"text": "Hello", "start": 0, "end": 5,
                          "label": "UNKNOWN_TYPE", "assertion": "PRESENT",
                          "normalized_name": "Hello", "normalization_status": "UNMAPPED_DEMO"}],
        }
        with pytest.raises(ValueError, match="unknown label"):
            validate_records([rec])

    def test_invalid_assertion_raises(self):
        rec = {
            "id": "X-001", "text": "Hello.", "split": "train", "synthetic": True,
            "entities": [{"text": "Hello", "start": 0, "end": 5,
                          "label": "SYMPTOM", "assertion": "MAYBE",
                          "normalized_name": "Hello", "normalization_status": "UNMAPPED_DEMO"}],
        }
        with pytest.raises(ValueError, match="invalid assertion"):
            validate_records([rec])


# ---------------------------------------------------------------------------
# 4. BIO alignment helpers
# ---------------------------------------------------------------------------


class TestAlignOffsetsToBio:
    """Tests on synthetic records (no tokenizer needed — uses manual offsets)."""

    def _make_label_maps(self):
        return build_label_maps()

    def test_special_tokens_get_ignore_index(self):
        _, label2id, _ = self._make_label_maps()
        # Simulate [CLS] Hello [SEP] with offsets (0,0), (0,5), (0,0)
        offsets = [(0, 0), (0, 5), (0, 0)]
        entities = []
        ids, trunc = align_offsets_to_bio(offsets, entities, label2id)
        assert ids[0] == IGNORE_INDEX   # [CLS]
        assert ids[1] == label2id["O"]  # Hello
        assert ids[2] == IGNORE_INDEX   # [SEP]
        assert trunc == []

    def test_single_token_entity(self):
        _, label2id, id2label = self._make_label_maps()
        # [CLS] "fever" [SEP]  — entity covers (5, 10)
        offsets = [(0, 0), (5, 10), (0, 0)]
        entities = [{"text": "fever", "start": 5, "end": 10, "label": "SYMPTOM",
                     "assertion": "PRESENT"}]
        ids, trunc = align_offsets_to_bio(offsets, entities, label2id)
        assert ids[0] == IGNORE_INDEX
        assert id2label[ids[1]] == "B-SYMPTOM"
        assert ids[2] == IGNORE_INDEX
        assert trunc == []

    def test_multi_token_entity(self):
        _, label2id, id2label = self._make_label_maps()
        # [CLS] "chest"(0,5) "pain"(6,10) [SEP]
        offsets = [(0, 0), (0, 5), (6, 10), (0, 0)]
        entities = [{"text": "chest pain", "start": 0, "end": 10, "label": "SYMPTOM",
                     "assertion": "PRESENT"}]
        ids, trunc = align_offsets_to_bio(offsets, entities, label2id)
        assert id2label[ids[1]] == "B-SYMPTOM"
        assert id2label[ids[2]] == "I-SYMPTOM"
        assert trunc == []

    def test_entity_fully_truncated_warns(self):
        _, label2id, _ = self._make_label_maps()
        # Only [CLS] and [SEP] in the token sequence; entity is beyond it.
        offsets = [(0, 0), (0, 0)]
        entities = [{"text": "fever", "start": 5, "end": 10, "label": "SYMPTOM",
                     "assertion": "PRESENT"}]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ids, trunc = align_offsets_to_bio(offsets, entities, label2id)
        assert "fever" in trunc
        assert any("truncated" in str(w.message).lower() for w in caught)

    def test_token_crossing_boundary_raises(self):
        _, label2id, _ = self._make_label_maps()
        # Token (3, 8) crosses entity boundary (5, 8)
        offsets = [(0, 0), (3, 8), (0, 0)]
        entities = [{"text": "ver", "start": 5, "end": 8, "label": "SYMPTOM",
                     "assertion": "PRESENT"}]
        with pytest.raises(ValueError, match="crosses entity boundary"):
            align_offsets_to_bio(offsets, entities, label2id)


# ---------------------------------------------------------------------------
# 5. encode_record — integration tests using real tokenizer + real data
# ---------------------------------------------------------------------------


class TestEncodeRecord:

    def _check_encoded(self, enc: EncodedRecord, record: dict):
        """Shared assertions for an encoded record."""
        n = len(enc.input_ids)
        assert len(enc.attention_mask) == n
        assert len(enc.label_ids) == n
        assert len(enc.offset_mapping) == n

        # input_ids[0] == [CLS], last non-padding == [SEP]
        # label_ids for [CLS] and [SEP] must be IGNORE_INDEX.
        assert enc.label_ids[0] == IGNORE_INDEX, "CLS token must have IGNORE_INDEX"
        last_real = max(i for i, m in enumerate(enc.attention_mask) if m == 1)
        assert enc.label_ids[last_real] == IGNORE_INDEX, "SEP token must have IGNORE_INDEX"

        # Every non-ignore label_id must correspond to a valid label string.
        labels, label2id, id2label = build_label_maps()
        for lid in enc.label_ids:
            assert lid == IGNORE_INDEX or lid in id2label, f"Invalid label_id: {lid}"

        # Assertions list must not contain NER label names.
        for _, label, assertion in enc.assertions:
            assert label in ENTITY_TYPES
            assert assertion in {"PRESENT", "ABSENT", "POSSIBLE", "HISTORICAL", "FAMILY_HISTORY"}

    def _check_span_recovery(self, enc: EncodedRecord, record: dict):
        """Verify that B- tokens recover correct character spans."""
        text = record["text"]
        labels, label2id, id2label = build_label_maps()
        for token_idx, lid in enumerate(enc.label_ids):
            if lid == IGNORE_INDEX:
                continue
            label_str = id2label[lid]
            if label_str.startswith("B-"):
                cs, ce = enc.offset_mapping[token_idx]
                assert cs < ce, "B- token must have non-empty char span"
                # The character span must be within the original text.
                assert 0 <= cs < ce <= len(text), (
                    f"B-{label_str} span ({cs},{ce}) out of text bounds"
                )

    def test_encode_first_train_record(self, tokenizer, label_maps, train_records):
        _, label2id, _ = label_maps
        rec = train_records[0]
        enc = encode_record(rec, tokenizer, label2id)
        assert enc.record_id == rec["id"]
        assert enc.split == "train"
        self._check_encoded(enc, rec)
        self._check_span_recovery(enc, rec)

    def test_encode_first_validation_record(self, tokenizer, label_maps, validation_records):
        _, label2id, _ = label_maps
        rec = validation_records[0]
        enc = encode_record(rec, tokenizer, label2id)
        assert enc.split == "validation"
        self._check_encoded(enc, rec)
        self._check_span_recovery(enc, rec)

    def test_encode_first_test_record(self, tokenizer, label_maps, test_records):
        _, label2id, _ = label_maps
        rec = test_records[0]
        enc = encode_record(rec, tokenizer, label2id)
        assert enc.split == "test"
        self._check_encoded(enc, rec)
        self._check_span_recovery(enc, rec)

    def test_multi_word_entity_gets_B_then_I(self, tokenizer, label_maps, train_records):
        """Find a multi-word entity and verify B- followed by I- tags."""
        _, label2id, id2label = label_maps
        for rec in train_records:
            multi = [e for e in rec["entities"] if " " in e["text"]]
            if not multi:
                continue
            enc = encode_record(rec, tokenizer, label2id)
            for entity in multi:
                # Find the B- token for this entity type.
                b_tag = f"B-{entity['label']}"
                i_tag = f"I-{entity['label']}"
                b_id = label2id[b_tag]
                i_id = label2id[i_tag]
                # Walk token sequence looking for this B- tag.
                for idx, lid in enumerate(enc.label_ids):
                    if lid == b_id:
                        cs = enc.offset_mapping[idx][0]
                        if cs == entity["start"]:
                            # Multi-token entity: next token(s) should be I-.
                            tokens_in_entity = sum(
                                1 for tok_idx in range(idx, len(enc.label_ids))
                                if enc.label_ids[tok_idx] in (b_id, i_id)
                                and enc.offset_mapping[tok_idx][1] <= entity["end"]
                            )
                            if tokens_in_entity > 1:
                                assert enc.label_ids[idx + 1] == i_id, (
                                    f"Expected I- after B- for {entity['text']!r}"
                                )
                            break
            break  # only need one record

    def test_assertion_metadata_preserved(self, tokenizer, label_maps, train_records):
        """Assertions must be in EncodedRecord but not in label_ids."""
        _, label2id, id2label = label_maps
        rec = next(r for r in train_records if r["entities"])
        enc = encode_record(rec, tokenizer, label2id)
        # Must have assertion tuples for non-truncated entities.
        assert len(enc.assertions) > 0
        # Assertions must NOT appear in label_ids.
        valid_ids = set(label2id.values()) | {IGNORE_INDEX}
        for lid in enc.label_ids:
            assert lid in valid_ids

    def test_no_entity_record_has_all_O(self, tokenizer, label_maps, train_records):
        """A record with no entities should produce all-O (or IGNORE_INDEX) labels."""
        _, label2id, _ = label_maps
        # Build a synthetic empty-entity record.
        rec = {
            "id": "SYNTHETIC-EMPTY",
            "text": "Patient is well.",
            "entities": [],
            "split": "train",
            "synthetic": True,
        }
        enc = encode_record(rec, tokenizer, label2id)
        for lid in enc.label_ids:
            assert lid in (label2id["O"], IGNORE_INDEX), (
                f"Expected O or IGNORE_INDEX, got {lid}"
            )

    def test_max_length_truncation_detected(self, tokenizer, label_maps, train_records):
        """A very small max_length should trigger truncation warnings for long records."""
        _, label2id, _ = label_maps
        # Pick a record with entities and use tiny max_length.
        rec = next(r for r in train_records if len(r["entities"]) >= 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enc = encode_record(rec, tokenizer, label2id, max_length=10)
        # Some entities should be truncated.
        assert enc.truncated_entities or any("truncated" in str(w.message).lower() for w in caught), (
            "Expected truncation with max_length=10 on a long record"
        )

    def test_encode_sample_from_each_split(self, tokenizer, label_maps,
                                           train_records, validation_records, test_records):
        """Smoke-test 5 records from each split."""
        _, label2id, _ = label_maps
        step = lambda recs: [recs[i] for i in range(0, min(len(recs), 5))]
        for split_name, recs in [
            ("train", train_records),
            ("validation", validation_records),
            ("test", test_records),
        ]:
            for rec in step(recs):
                enc = encode_record(rec, tokenizer, label2id)
                assert enc.split == split_name
                self._check_encoded(enc, rec)


# ---------------------------------------------------------------------------
# 6. Round-trip: all records (offset text == recovered span text)
# ---------------------------------------------------------------------------


class TestRoundTripAllRecords:
    """For every B- token in every encoded record, the character span must
    match the start of the corresponding entity text."""

    def _round_trip_split(self, records, tokenizer, label2id, id2label):
        errors = []
        for rec in records:
            enc = encode_record(rec, tokenizer, label2id)
            text = rec["text"]
            entity_starts = {e["start"]: e for e in rec["entities"]}
            for tok_idx, lid in enumerate(enc.label_ids):
                if lid == IGNORE_INDEX:
                    continue
                label_str = id2label[lid]
                if not label_str.startswith("B-"):
                    continue
                cs, ce = enc.offset_mapping[tok_idx]
                if cs not in entity_starts:
                    errors.append(
                        f"{rec['id']}: B-token at chars ({cs},{ce}) "
                        f"has no matching entity start"
                    )
                    continue
                entity = entity_starts[cs]
                recovered = text[cs:ce]
                if not entity["text"].startswith(recovered):
                    errors.append(
                        f"{rec['id']}: B-token text {recovered!r} is not "
                        f"prefix of entity {entity['text']!r}"
                    )
        return errors

    def test_round_trip_train(self, tokenizer, label_maps, train_records):
        _, label2id, id2label = label_maps
        errors = self._round_trip_split(train_records, tokenizer, label2id, id2label)
        assert not errors, f"{len(errors)} round-trip errors:\n" + "\n".join(errors[:5])

    def test_round_trip_validation(self, tokenizer, label_maps, validation_records):
        _, label2id, id2label = label_maps
        errors = self._round_trip_split(validation_records, tokenizer, label2id, id2label)
        assert not errors, f"{len(errors)} round-trip errors:\n" + "\n".join(errors[:5])

    def test_round_trip_test(self, tokenizer, label_maps, test_records):
        _, label2id, id2label = label_maps
        errors = self._round_trip_split(test_records, tokenizer, label2id, id2label)
        assert not errors, f"{len(errors)} round-trip errors:\n" + "\n".join(errors[:5])

