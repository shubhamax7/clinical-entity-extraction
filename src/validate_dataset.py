import json
from pathlib import Path
from collections import Counter

DATA_PATH = Path("data/raw/synthetic_clinical_notes.jsonl")

def validate_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    records = [
        json.loads(line)
        for line in DATA_PATH.read_text().splitlines()
        if line.strip()
    ]

    errors = []
    labels = Counter()
    splits = Counter()
    ids = set()

    for record in records:
        text = record["text"]

        if record["id"] in ids:
            errors.append(f"Duplicate ID: {record['id']}")

        ids.add(record["id"])
        splits[record.get("split", "unspecified")] += 1

        for entity in record.get("entities", []):
            start = entity["start"]
            end = entity["end"]
            expected = entity["text"]

            if not (0 <= start < end <= len(text)):
                errors.append(
                    f"{record['id']}: Invalid offsets for {expected}"
                )
                continue

            if text[start:end] != expected:
                errors.append(
                    f"{record['id']}: Offset mismatch for {expected}"
                )

            labels[entity["label"]] += 1

    print("\nDATASET VALIDATION")
    print("------------------")
    print("Total records:", len(records))
    print("Split distribution:", dict(splits))
    print("Entity distribution:", dict(labels))
    print("Total entities:", sum(labels.values()))
    print("Errors:", len(errors))

    for error in errors:
        print("ERROR:", error)

    print(
        "\nDataset validation PASSED."
        if not errors
        else "\nDataset validation FAILED."
    )

if __name__ == "__main__":
    validate_dataset()
