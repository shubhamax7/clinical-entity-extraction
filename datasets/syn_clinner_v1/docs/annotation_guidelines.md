# Annotation Guidelines: SYN-CLINNER-v1

These rules describe how the synthetic labels are defined, so the data can be extended consistently or replaced by human-annotated data later. They are **not** clinically validated guidelines.

## 1. General rules

1. Annotate the **minimal span** that names the concept. Do not include articles, possessives, or trailing punctuation (`hypertension`, not `the hypertension.`).
2. Offsets are character indices into `text`, half-open `[start, end)`. `text[start:end]` must equal the entity text exactly.
3. **No overlapping or nested entities.** Each character belongs to at most one entity, so each token has exactly one tag.
4. Entities never span a sentence boundary and never start or end with whitespace.
5. Abbreviations and spelling variants are annotated as written (`HTN`, `anaemia`, `PO`, `BID`).
6. Annotate mentions even when the finding is negated, historical, possible or about a relative. The context goes in `assertion`, not in the span.
7. Words indicating the context (`no`, `denies`, `history of`, `suspected`, `family history`) stay **outside** the span.

## 2. Labels

| Label | Annotate | Examples | Do not annotate |
|---|---|---|---|
| DISEASE | Diagnoses and named conditions | `type 2 diabetes mellitus`, `CHF`, `UTI`, `Crohn's disease` | Symptoms, test results |
| SYMPTOM | Patient-reported or observed complaints and signs | `persistent cough`, `pedal oedema`, `jaundice`, `SOB` | Vital-sign values, lab values |
| DRUG | Drug or fluid name (generic) | `metformin`, `ASA`, `normal saline` | Dose, route, frequency, duration, drug classes |
| PROCEDURE | Surgical, interventional or therapeutic procedures | `CABG`, `lumbar puncture`, `ORIF`, `intubation` | Diagnostic tests (those are INVESTIGATION) |
| INVESTIGATION | The test or imaging study **name** | `ECG`, `HbA1c`, `MRI brain`, `urine R/M` | The result text (`ST elevation`, `a value of 9.2%`) |
| DOSAGE | Amount per administration | `500 mg`, `2 puffs`, `20 units` | Total course duration |
| ROUTE | Administration route | `IV`, `orally`, `subcutaneously`, `via nebuliser` | |
| FREQUENCY | How often (including timing instructions) | `BID`, `twice daily`, `at bedtime`, `stat`, `once weekly` | |
| DURATION | Length of treatment course | `5 days`, `3 months`, `12 months` | Onset durations such as "for two weeks" in symptom history |

Notes:
- Investigations: the investigation name is annotated; the finding or value that follows is plain `O` text in this dataset.
- Vital signs (`BP 120/80 mmHg`, `SpO2 95%`) are not annotated.
- Attribute labels (DOSAGE, ROUTE, FREQUENCY, DURATION) are annotated only where they belong to a medication in the same sentence.

## 3. Assertion values

| Value | Use when | Cue examples |
|---|---|---|
| PRESENT | The finding, diagnosis, drug or procedure applies to the patient now or is current/planned (default) | `presented with`, `diagnosis:`, `started on`, `planned for` |
| ABSENT | Explicitly negated, not performed, not taken, or resolved | `no`, `denies`, `negative for`, `not performed`, `resolved` |
| POSSIBLE | Suspected, differential, unconfirmed | `suspected`, `rule out`, `r/o`, `?`, `likely`, `differential diagnoses include` |
| HISTORICAL | Occurred in the past and is not the current problem | `past history of`, `previously treated`, `discontinued last year` |
| FAMILY_HISTORY | Experienced by a relative, not the patient | `family history of`, `mother had`, `father had` |

Rules:
- A mention is **not** a confirmed current diagnosis unless its assertion is PRESENT.
- `No family history of X` is **ABSENT** (the negation takes precedence over the family context).
- A past medication keeps its attributes with the same assertion (for example HISTORICAL drug and HISTORICAL dosage).
- Attribute spans inherit the assertion of their drug.
- One assertion per entity. No multi-label assertions.

## 4. BIO tagging rules

1. Tokenize with the regex stored in `dataset_metadata.json` (`tokenizer_regex`). Hyphenated, slashed and apostrophe words stay as one token (`high-grade`, `NS1`, `Parkinson's`).
2. The first token of an entity gets `B-<LABEL>`, the remaining tokens `I-<LABEL>`; all other tokens get `O`.
3. Every entity must start and end exactly on token boundaries.
4. `I-X` must directly follow `B-X` or `I-X` of the same label.
5. Two adjacent entities of different labels are two separate `B-` spans.

## 5. Normalization fields

- `normalized_name` is a plain-text canonical form chosen by the generator (the first spelling variant of the concept). For attribute labels it equals the surface text.
- `normalization_status` is always `UNMAPPED_DEMO`. Do **not** add SNOMED CT, RxNorm, LOINC or similar identifiers unless they are verified against the authoritative terminology source.

## 6. Edge cases

| Case | Decision |
|---|---|
| `Rx: Tab. metformin 500 mg BID x 3 months` | `Tab.` and `x` are outside spans. Annotate drug, dosage, frequency, duration. |
| `Inj. ceftriaxone 1 g IV given stat` | `ceftriaxone` DRUG, `1 g` DOSAGE, `IV` ROUTE, `stat` FREQUENCY. |
| `? pneumonia` / `r/o PE` | Disease span only (`pneumonia`, `PE`), assertion POSSIBLE. |
| `CT head showed no acute abnormality` | `CT head` INVESTIGATION only. |
| `Denies chest pain, SOB or palpitations` | Three SYMPTOM spans, all ABSENT. |
| `Symptomatic improvement noted; nausea resolved` | `nausea` SYMPTOM, ABSENT. |
| Abbreviation shared by two concepts (`PE`, `MI`, `LP`) | Annotate as written. The concept is determined by the generator's lexicon entry, so real data would need context-aware adjudication. |

## 7. Quality checks applied

See `validation_report.md`. In short: offset/text equality, BIO-to-offset round trip, token coverage, overlap check, assertion/label vocabulary check, duplicate and leakage checks.

## 8. Limits of these guidelines

The labels were produced programmatically from templates, with no human adjudication. Real clinical text contains scope ambiguity, discontiguous entities, nested concepts and implicit negation that these rules do not cover. Extend this document with adjudicated examples before using it to annotate real notes.
