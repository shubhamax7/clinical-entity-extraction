"""Data loading, annotation validation, and BIO alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


BIO_PREFIXES = ("B", "I")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records and report malformed lines precisely."""
    path = Path(path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
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


def validate_records(records: Iterable[dict[str, Any]]) -> None:
    """Validate record structure, entity spans, labels, and overlap."""
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

            start = entity.get("start")
            end = entity.get("end")
            entity_text = entity.get("text")
            label = entity.get("label")

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

            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"{record_id}: entity {entity_index} has invalid label"
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

        # This prototype uses flat BIO tagging, so overlapping/nested
        # entities are rejected instead of being silently corrupted.
        spans.sort(key=lambda span: (span[0], span[1]))

        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"{record_id}: overlapping entities: "
                    f"{previous} and {current}"
                )


def build_label_maps(
    records: Iterable[dict[str, Any]],
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Build a deterministic BIO vocabulary from entity labels."""
    entity_types = sorted(
        {
            entity["label"]
            for record in records
            for entity in record["entities"]
        }
    )

    labels = ["O"]
    for entity_type in entity_types:
        labels.extend(
            f"{prefix}-{entity_type}" for prefix in BIO_PREFIXES
        )

    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    return labels, label2id, id2label


def align_offsets_to_bio(
    offset_mapping: Iterable[tuple[int, int]],
    entities: list[dict[str, Any]],
    label2id: dict[str, int],
) -> list[int]:
    """
    Convert character-level entity spans into token-level BIO label IDs.

    Special tokens and tokens with empty offsets receive -100, so they
    are ignored by the token-classification loss.
    """
    offsets = list(offset_mapping)
    labels = [label2id["O"]] * len(offsets)

    # -100 tells Hugging Face token-classification loss to ignore a token.
    for index, (start, end) in enumerate(offsets):
        if start == end:
            labels[index] = -100

    sorted_entities = sorted(
        entities,
        key=lambda entity: (entity["start"], entity["end"]),
    )

    for entity in sorted_entities:
        entity_start = entity["start"]
        entity_end = entity["end"]
        entity_type = entity["label"]

        begin_label = f"B-{entity_type}"
        inside_label = f"I-{entity_type}"

        if begin_label not in label2id or inside_label not in label2id:
            raise ValueError(f"Unknown entity type: {entity_type}")

        overlapping_tokens: list[int] = []

        for token_index, (token_start, token_end) in enumerate(offsets):
            if token_start == token_end:
                continue

            overlaps = token_start < entity_end and token_end > entity_start
            if overlaps:
                # A token must not cross an entity boundary.
                if token_start < entity_start or token_end > entity_end:
                    raise ValueError(
                        f"Token span ({token_start}, {token_end}) "
                        f"crosses entity boundary ({entity_start}, {entity_end})"
                    )
                overlapping_tokens.append(token_index)

        if not overlapping_tokens:
            raise ValueError(
                f"No tokenizer tokens found for entity "
                f"{entity.get('text')!r} at ({entity_start}, {entity_end})"
            )

        # Ensure the token offsets cover the entire annotated entity.
        first_start = offsets[overlapping_tokens[0]][0]
        last_end = offsets[overlapping_tokens[-1]][1]

        if first_start != entity_start or last_end != entity_end:
            raise ValueError(
                f"Tokenization does not exactly cover entity "
                f"{entity.get('text')!r}: entity=({entity_start}, {entity_end}), "
                f"tokens=({first_start}, {last_end})"
            )

        for position, token_index in enumerate(overlapping_tokens):
            if labels[token_index] != label2id["O"]:
                raise ValueError(
                    f"Conflicting annotation at token index {token_index}"
                )

            tag = begin_label if position == 0 else inside_label
            labels[token_index] = label2id[tag]

    return labels
