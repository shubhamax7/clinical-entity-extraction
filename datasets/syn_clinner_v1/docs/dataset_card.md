# Dataset Card: SYN-CLINNER-v1

**Fully synthetic. Not real patient data. Not clinically validated. Not a benchmark.**
Intended only for developing and smoke-testing a Clinical NER pipeline until properly licensed, human-annotated clinical data is available.

## Summary

| Item | Value |
|---|---|
| Records | 500 (train 350 / validation 75 / test 75) |
| Entity annotations | 5,820 |
| Primary labels | DISEASE, SYMPTOM, DRUG, PROCEDURE, INVESTIGATION |
| Attribute labels | DOSAGE, ROUTE, FREQUENCY, DURATION |
| Tagging scheme | BIO (`B-`/`I-` + label, `O`) |
| Assertions | PRESENT, ABSENT, POSSIBLE, HISTORICAL, FAMILY_HISTORY |
| Specialties | general medicine, cardiology, neurology, respiratory, endocrinology, gastroenterology, orthopedics, emergency |
| Language | English (US and UK spelling variants) |
| Generation | `generate_dataset.py`, seed `20261102`, re-running gives byte-identical files (verified by SHA-256) |
| Verification | `validate_dataset.py` (independent of the generator), results in `validation_report.md` / `.json` |

## Files

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: one JSON record per line.
- `dataset_metadata.json`: machine-readable metadata, label set, tokenizer regex, file SHA-256 hashes.
- `annotation_guidelines.md`: labelling rules.
- `validation_report.md` / `validation_report.json`: structural checks and distributions.

## Record schema

```json
{
  "id": "SYN-0001",
  "text": "...",
  "entities": [{"text": "...", "start": 0, "end": 5, "label": "DRUG", "assertion": "PRESENT",
                "normalized_name": "...", "normalization_status": "UNMAPPED_DEMO"}],
  "tokens": ["..."],
  "bio_tags": ["O", "B-DRUG"],
  "split": "train",
  "synthetic": true,
  "token_offsets": [[0, 5]],
  "meta": {"specialty": "neurology", "length_class": "short", "template_ids": ["OPEN-02"]}
}
```

- Offsets are Python `str` indices into `text`, half-open `[start, end)`.
- `tokens` / `bio_tags` / `token_offsets` have equal length. Tokenizer regex is stored in `dataset_metadata.json`, and entity boundaries always fall on token boundaries.
- `token_offsets` and `meta` are extra fields beyond the requested schema. Ignore them if not needed. Note that `meta.template_ids` is generator provenance and must not be used as a model feature.
- `normalized_name` is a plain-text canonical string. `normalization_status` is always `UNMAPPED_DEMO`. **No SNOMED CT, RxNorm, LOINC or other codes are present.**

## How it was built

Each note starts with a presentation sentence and then adds 2-11 further sentences drawn from 12 categories (history of present illness, negation, past history, family history, past medication, current medication, examination, investigations, possible diagnosis, diagnosis, procedures, plan). Entities are sampled from per-specialty lexicons with spelling and abbreviation variants (for example HTN / hypertension / high blood pressure; PO / orally; BID / twice daily). Offsets are recorded while the text is assembled, then re-verified in the generator and again by the separate validator.

**Split hygiene.** Each of the 13 sentence categories has 10 templates. Templates 0-5 are used only in train, 6-7 only in validation, 8-9 only in test. Vital-sign string formats are also split-exclusive. Candidate notes whose word-3-gram Jaccard similarity to any accepted note is 0.55 or more are rejected. Measured result: no exact duplicates, no shared templates or entity-masked sentence skeletons, maximum similarity between notes in different splits 0.09.

## Statistics

| Label | Count |
|---|---:|
| DISEASE | 1,311 |
| SYMPTOM | 1,629 |
| DRUG | 652 |
| PROCEDURE | 407 |
| INVESTIGATION | 479 |
| DOSAGE | 484 |
| ROUTE | 279 |
| FREQUENCY | 430 |
| DURATION | 149 |

Assertions: PRESENT 4,040; HISTORICAL 671; ABSENT 622; POSSIBLE 268; FAMILY_HISTORY 219. Per-split breakdowns are in `validation_report.md`.

## Intended use

- Building and debugging a BIO-tagging pipeline (tokenization, label alignment, metrics code, data loaders).
- Testing assertion-classification code paths.

## Not intended for

- Reporting model performance as evidence of clinical NER ability. Scores will be inflated (see below).
- Training models for deployment, or any clinical decision-making.
- Using drug doses, frequencies or durations as medical guidance. They are illustrative.

## Known limitations

1. **Template artefacts.** A model can learn the 13 x 10 sentence frames. Held-out templates make validation/test harder than random splits, but the language is still far simpler than real EHR text.
2. **Shared vocabulary.** The entity lexicon is small and shared by all splits; about 95% of distinct test entity surface forms also appear in train. Strong scores here mainly reflect lexicon memorisation.
3. **Clinical incoherence.** Entities are sampled mostly at random within a specialty (sometimes across specialties), so notes can be medically implausible (for example a chronic neurodegenerative disease described as "cured", or unrelated symptoms and drugs).
4. **Label distribution is generator-driven**, not representative of any patient population or specialty.
5. **No real-world noise**: no typos, OCR errors, tables, section headers, de-identification tokens, or long-range context.
6. **No human annotation or inter-annotator agreement.** Labels are correct by construction relative to the guidelines, not adjudicated.
7. **Attribute linking.** DOSAGE/ROUTE/FREQUENCY/DURATION inherit the assertion of the nearest preceding drug in the same sentence. There is no explicit drug-attribute relation field.
8. **Assertion scope.** Assertion is per entity from the template; real-note scope ambiguity (for example "no fever or chills") is only partly represented.
9. **Minor grammar quirks** (for example "a 85 year old") from templated numbers.

## Privacy

All content is fictional and produced from templates. A regex scan found no email, phone, SSN-like, URL or MRN/DOB patterns. No real patient records were used or copied. Any resemblance to real cases is coincidental.

## Licensing and attribution

Generated by the project owner's scripts. Choose a license before redistribution, and state clearly in any publication that the data is synthetic.
