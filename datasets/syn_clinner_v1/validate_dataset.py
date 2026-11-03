#!/usr/bin/env python3
"""Independent validator for the synthetic clinical NER dataset.

Does NOT import the generator. Re-derives everything from the JSONL files.
Writes data/validation_report.json and data/validation_report.md.
"""
import collections
import json
import re
import sys

DATA = "data"
SPLITS = {"train": 350, "validation": 75, "test": 75}
PRIMARY = ["DISEASE", "SYMPTOM", "DRUG", "PROCEDURE", "INVESTIGATION"]
ATTRS = ["DOSAGE", "ROUTE", "FREQUENCY", "DURATION"]
LABELS = PRIMARY + ATTRS
ASSERT = {"PRESENT", "ABSENT", "POSSIBLE", "HISTORICAL", "FAMILY_HISTORY"}
VALID_TAGS = {"O"} | {p + l for l in LABELS for p in ("B-", "I-")}
NEAR_DUP = 0.70
PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "url": re.compile(r"https?://|www\."),
    "mrn_like": re.compile(r"\b(MRN|UHID|Aadhaar|DOB)\b", re.I),
}


def load(split):
    with open(f"{DATA}/{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sh(text):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def bio_to_spans(tags, offs):
    spans, cur = [], None
    for tag, (a, b) in zip(tags, offs):
        if tag == "O":
            if cur:
                spans.append(cur)
            cur = None
        elif tag.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = [tag[2:], a, b]
        else:
            lab = tag[2:]
            if cur is None or cur[0] != lab:
                return None  # inconsistent
            cur[2] = b
    if cur:
        spans.append(cur)
    return [tuple(s) for s in spans]


def masked_sentences(r):
    """Split into sentences and mask entity spans/digits -> template-like skeletons."""
    text, ents = r["text"], sorted(r["entities"], key=lambda e: e["start"])
    out, pos = [], 0
    for e in ents:
        out.append(text[pos:e["start"]])
        out.append(f"<{e['label']}>")
        pos = e["end"]
    out.append(text[pos:])
    masked = re.sub(r"\d+", "#", "".join(out))
    sents = [s.strip() for s in re.split(r"(?<=[.;])\s+|\n", masked) if len(s.strip()) > 12]
    return sents


def main():
    errors = collections.Counter()
    samples = collections.defaultdict(list)

    def err(kind, rid, detail=""):
        errors[kind] += 1
        if len(samples[kind]) < 3:
            samples[kind].append(f"{rid}: {detail}")

    data = {s: load(s) for s in SPLITS}
    all_recs = [r for s in SPLITS for r in data[s]]

    # ---- split sizes, ids, flags
    split_counts = {s: len(data[s]) for s in SPLITS}
    for s, n in SPLITS.items():
        if split_counts[s] != n:
            err("split_size", s, f"expected {n}, got {split_counts[s]}")
    ids = [r["id"] for r in all_recs]
    if len(set(ids)) != len(ids):
        err("duplicate_id", "-", "ids not unique")
    if ids != [f"SYN-{i:04d}" for i in range(1, len(ids) + 1)]:
        err("id_format", "-", "ids not sequential SYN-0001..")

    ent_counts = collections.Counter()
    asr_counts = collections.Counter()
    asr_by_label = collections.defaultdict(collections.Counter)
    per_split_ent = {s: collections.Counter() for s in SPLITS}
    len_class = collections.Counter()
    spec_count = collections.Counter()
    tok_total = 0

    for r in all_recs:
        rid, text = r["id"], r["text"]
        if r.get("synthetic") is not True:
            err("not_marked_synthetic", rid)
        if r.get("split") not in SPLITS or r["id"] not in {x["id"] for x in data[r["split"]]}:
            err("split_field_mismatch", rid)
        for k in ("id", "text", "entities", "tokens", "bio_tags", "split", "synthetic"):
            if k not in r:
                err("missing_field", rid, k)
        for name, rx in PII.items():
            if rx.search(text):
                err(f"pii_pattern_{name}", rid, rx.search(text).group())
        len_class[r["meta"]["length_class"]] += 1
        spec_count[r["meta"]["specialty"]] += 1

        # ---- entity offsets
        ents = r["entities"]
        prev_end = -1
        for e in sorted(ents, key=lambda x: (x["start"], x["end"])):
            for k in ("text", "start", "end", "label", "assertion", "normalized_name", "normalization_status"):
                if k not in e:
                    err("entity_missing_field", rid, k)
            if not (isinstance(e["start"], int) and isinstance(e["end"], int)) or not (0 <= e["start"] < e["end"] <= len(text)):
                err("offset_out_of_range", rid, str(e))
                continue
            if text[e["start"]:e["end"]] != e["text"]:
                err("offset_text_mismatch", rid, f"{e['text']!r} vs {text[e['start']:e['end']]!r}")
            if e["text"] != e["text"].strip():
                err("entity_whitespace_boundary", rid, repr(e["text"]))
            if e["label"] not in LABELS:
                err("bad_label", rid, e["label"])
            if e["assertion"] not in ASSERT:
                err("bad_assertion", rid, str(e["assertion"]))
            if e["normalization_status"] != "UNMAPPED_DEMO":
                err("bad_norm_status", rid, e["normalization_status"])
            if re.search(r"\b(SNOMED|RxNorm|LOINC)\b|\bC\d{7}\b", e["normalized_name"], re.I):
                err("terminology_code_present", rid, e["normalized_name"])
            if e["start"] < prev_end:
                err("overlapping_entities", rid, str(e))
            prev_end = max(prev_end, e["end"])
            ent_counts[e["label"]] += 1
            per_split_ent[r["split"]][e["label"]] += 1
            asr_counts[e["assertion"]] += 1
            asr_by_label[e["label"]][e["assertion"]] += 1

        # ---- tokens / BIO
        toks, tags, offs = r["tokens"], r["bio_tags"], r.get("token_offsets", [])
        tok_total += len(toks)
        if not (len(toks) == len(tags) == len(offs)):
            err("length_mismatch_tokens_tags_offsets", rid, f"{len(toks)}/{len(tags)}/{len(offs)}")
            continue
        for t, (a, b) in zip(toks, offs):
            if text[a:b] != t:
                err("token_offset_mismatch", rid, f"{t!r} vs {text[a:b]!r}")
        for i, tg in enumerate(tags):
            if tg not in VALID_TAGS:
                err("invalid_tag", rid, tg)
        for i, tg in enumerate(tags):  # I- must follow same-label B-/I-
            if tg.startswith("I-") and (i == 0 or tags[i - 1][2:] != tg[2:] or tags[i - 1] == "O"):
                err("orphan_I_tag", rid, f"{i}:{tg}")
        spans = bio_to_spans(tags, offs)
        if spans is None:
            err("bio_inconsistent", rid)
            continue
        gold = sorted((e["label"], e["start"], e["end"]) for e in ents)
        if sorted(spans) != gold:
            err("bio_vs_entity_offsets_mismatch", rid, f"bio={sorted(spans)[:3]} gold={gold[:3]}")
        # tokens fully covered by entity must be tagged, uncovered must be O
        covered = set()
        for e in ents:
            for i, (a, b) in enumerate(offs):
                if a >= e["start"] and b <= e["end"]:
                    covered.add(i)
        for i, tg in enumerate(tags):
            if (tg == "O") == (i in covered):
                err("tag_coverage_mismatch", rid, f"token {i}")

    # ---- duplicates
    seen, exact = {}, 0
    for r in all_recs:
        if r["text"] in seen:
            exact += 1
        seen[r["text"]] = r["id"]
    sigs = [(r["id"], r["split"], sh(r["text"])) for r in all_recs]
    near, near_cross, max_j = [], 0, 0.0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            v = jac(sigs[i][2], sigs[j][2])
            max_j = max(max_j, v)
            if v >= NEAR_DUP:
                near.append((sigs[i][0], sigs[j][0], round(v, 3)))
                near_cross += sigs[i][1] != sigs[j][1]
    cross_max = 0.0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            if sigs[i][1] != sigs[j][1]:
                cross_max = max(cross_max, jac(sigs[i][2], sigs[j][2]))

    # ---- template / sentence-skeleton leakage across splits
    tpl = {s: {t for r in data[s] for t in r["meta"]["template_ids"]} for s in SPLITS}
    tpl_overlap = {
        "train&validation": sorted(tpl["train"] & tpl["validation"]),
        "train&test": sorted(tpl["train"] & tpl["test"]),
        "validation&test": sorted(tpl["validation"] & tpl["test"]),
    }
    skel = {s: {x for r in data[s] for x in masked_sentences(r)} for s in SPLITS}
    skel_overlap = {
        "train&validation": sorted(skel["train"] & skel["validation"]),
        "train&test": sorted(skel["train"] & skel["test"]),
        "validation&test": sorted(skel["validation"] & skel["test"]),
    }
    # raw exact sentence overlap (surface, unmasked)
    def raw_sents(s):
        return {x.strip() for r in data[s] for x in re.split(r"(?<=[.;])\s+|\n", r["text"]) if len(x.strip()) > 25}
    raw = {s: raw_sents(s) for s in SPLITS}
    raw_overlap = {
        "train&validation": len(raw["train"] & raw["validation"]),
        "train&test": len(raw["train"] & raw["test"]),
        "validation&test": len(raw["validation"] & raw["test"]),
    }
    entity_surface = {s: {(e["label"], e["text"].lower()) for r in data[s] for e in r["entities"] if e["label"] in PRIMARY} for s in SPLITS}
    surface_overlap_test_in_train = len(entity_surface["test"] & entity_surface["train"]) / max(1, len(entity_surface["test"]))

    # ---- coverage checks
    coverage = {
        "records_with_negation(ABSENT)": sum(any(e["assertion"] == "ABSENT" for e in r["entities"]) for r in all_recs),
        "records_with_HISTORICAL": sum(any(e["assertion"] == "HISTORICAL" for e in r["entities"]) for r in all_recs),
        "records_with_POSSIBLE": sum(any(e["assertion"] == "POSSIBLE" for e in r["entities"]) for r in all_recs),
        "records_with_FAMILY_HISTORY": sum(any(e["assertion"] == "FAMILY_HISTORY" for e in r["entities"]) for r in all_recs),
        "records_with_drug+dosage+route": sum(
            any(e["label"] == "DRUG" for e in r["entities"]) and any(e["label"] == "DOSAGE" for e in r["entities"]) and any(e["label"] == "ROUTE" for e in r["entities"]) for r in all_recs),
        "records_with_>=2_DISEASE": sum(sum(e["label"] == "DISEASE" for e in r["entities"]) >= 2 for r in all_recs),
        "records_with_>=2_SYMPTOM": sum(sum(e["label"] == "SYMPTOM" for e in r["entities"]) >= 2 for r in all_recs),
        "records_with_>=2_INVESTIGATION": sum(sum(e["label"] == "INVESTIGATION" for e in r["entities"]) >= 2 for r in all_recs),
        "records_with_PROCEDURE": sum(any(e["label"] == "PROCEDURE" for e in r["entities"]) for r in all_recs),
    }
    abbr = re.compile(r"\b(HTN|T2DM|BID|TDS|QID|OD|PO|IV|IM|SC|PRN|CBC|ECG|CXR|MRI|CT|CRP|COPD|CHF|PCI|CABG|DKA|UTI)\b")
    coverage["records_with_medical_abbreviations"] = sum(bool(abbr.search(r["text"])) for r in all_recs)
    lens = [len(r["text"]) for r in all_recs]

    total_entities = sum(ent_counts.values())
    annotation_errors = sum(errors.values())
    leakage_problems = sum(len(v) for v in tpl_overlap.values()) + sum(len(v) for v in skel_overlap.values()) + sum(raw_overlap.values())

    report = {
        "total_records": len(all_recs),
        "records_per_split": split_counts,
        "total_entities": total_entities,
        "entities_by_category": {l: ent_counts[l] for l in LABELS},
        "entities_by_category_per_split": {s: {l: per_split_ent[s][l] for l in LABELS} for s in SPLITS},
        "category_distribution_pct": {l: round(100 * ent_counts[l] / total_entities, 2) for l in LABELS},
        "assertion_counts": dict(asr_counts),
        "assertion_by_label": {l: dict(asr_by_label[l]) for l in PRIMARY},
        "annotation_error_count": annotation_errors,
        "annotation_errors_by_type": dict(errors),
        "annotation_error_samples": dict(samples),
        "duplicate_count": {"exact_duplicate_texts": exact, f"near_duplicate_pairs_jaccard>={NEAR_DUP}": len(near),
                            "near_duplicate_pairs_across_splits": near_cross,
                            "max_pairwise_3gram_jaccard_all": round(max_j, 3),
                            "max_pairwise_3gram_jaccard_across_splits": round(cross_max, 3)},
        "leakage": {"template_id_overlap": tpl_overlap, "masked_sentence_skeleton_overlap": skel_overlap,
                    "exact_sentence_surface_overlap_counts": raw_overlap,
                    "note": "Entity VALUES (e.g. 'metformin') are intentionally shared across splits; sentence TEMPLATES are not.",
                    "pct_of_test_entity_surface_forms_also_in_train": round(100 * surface_overlap_test_in_train, 1)},
        "leakage_problem_count": leakage_problems,
        "length_class_counts": dict(len_class),
        "specialty_counts": dict(spec_count),
        "char_length_min_mean_max": [min(lens), round(sum(lens) / len(lens), 1), max(lens)],
        "total_tokens": tok_total,
        "clinical_complexity_coverage": coverage,
        "all_records_synthetic": all(r.get("synthetic") is True for r in all_recs),
        "pass": annotation_errors == 0 and exact == 0 and near_cross == 0 and leakage_problems == 0,
    }
    with open(f"{DATA}/validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    L = []
    L.append("# Validation Report: SYN-CLINNER-v1\n")
    L.append("> Synthetic data. These checks verify **structural consistency only** (offsets, BIO, leakage). "
             "They say nothing about clinical validity or real-world model performance.\n")
    L.append(f"**Overall structural check: {'PASS' if report['pass'] else 'FAIL'}**\n")
    L.append("## Counts\n")
    L.append(f"- Total records: **{report['total_records']}**")
    L.append("- Records per split: " + ", ".join(f"{s}={n}" for s, n in split_counts.items()))
    L.append(f"- Total entity annotations: **{total_entities}** (tokens: {tok_total})")
    L.append(f"- Annotation error count: **{annotation_errors}**")
    L.append(f"- Exact duplicate texts: **{exact}**; near-duplicate pairs (3-gram Jaccard >= {NEAR_DUP}): **{len(near)}** "
             f"(across splits: {near_cross}); max pairwise Jaccard overall {max_j:.3f}, across splits {cross_max:.3f}")
    L.append(f"- All records marked `synthetic: true`: **{report['all_records_synthetic']}**\n")
    L.append("## Entities by category\n")
    L.append("| Label | Total | % | train | validation | test |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for l in LABELS:
        L.append(f"| {l} | {ent_counts[l]} | {report['category_distribution_pct'][l]} | "
                 f"{per_split_ent['train'][l]} | {per_split_ent['validation'][l]} | {per_split_ent['test'][l]} |")
    L.append("\n## Assertion distribution (all entities)\n")
    L.append("| Assertion | Count |")
    L.append("|---|---:|")
    for a in sorted(ASSERT):
        L.append(f"| {a} | {asr_counts[a]} |")
    L.append("\n## Clinical-complexity coverage (records)\n")
    for k, v in coverage.items():
        L.append(f"- {k}: {v}")
    L.append("\n## Composition\n")
    L.append("- Length classes: " + ", ".join(f"{k}={v}" for k, v in sorted(len_class.items())))
    L.append("- Specialties: " + ", ".join(f"{k}={v}" for k, v in sorted(spec_count.items())))
    L.append(f"- Characters per note (min / mean / max): {report['char_length_min_mean_max']}\n")
    L.append("## Leakage checks\n")
    L.append(f"- Template-ID overlap between splits: {sum(len(v) for v in tpl_overlap.values())}")
    L.append(f"- Entity-masked sentence skeleton overlap: {sum(len(v) for v in skel_overlap.values())}")
    L.append(f"- Exact sentence overlap (surface): {sum(raw_overlap.values())}")
    L.append(f"- {report['leakage']['pct_of_test_entity_surface_forms_also_in_train']}% of distinct test entity surface forms also occur in train "
             "(expected: the same small vocabulary is used in every split; see limitations).\n")
    L.append("## Error details\n")
    if errors:
        for k, v in errors.items():
            L.append(f"- {k}: {v}  e.g. {samples[k]}")
    else:
        L.append("- No annotation errors detected by the checks listed in the checks section.")
    L.append("\n## Checks performed\n")
    L.append("1. Every entity: 0 <= start < end <= len(text); `text[start:end] == entity.text`; no boundary whitespace; no overlaps; valid label and assertion; `normalization_status == UNMAPPED_DEMO`; no terminology codes.")
    L.append("2. Tokens: `text[tok_start:tok_end] == token`; lengths of tokens / bio_tags / token_offsets equal.")
    L.append("3. BIO: tags valid; no orphan `I-`; spans decoded from BIO equal the character-offset entities exactly; every token inside an entity is tagged, every other token is `O`.")
    L.append("4. Duplicates: exact text, and word-3-gram Jaccard near-duplicates (all pairs, 500 notes).")
    L.append("5. Leakage: template IDs, entity-masked sentence skeletons, and exact sentences compared across splits.")
    L.append("6. Split sizes 350 / 75 / 75; sequential unique IDs; `synthetic: true` on every record.")
    L.append("7. Regex scan for identifier-like patterns (emails, phone numbers, SSN-like, URLs, MRN/DOB keywords): none expected.\n")
    L.append("## Limitations of this synthetic dataset\n")
    for s in [
        "Notes are assembled from a small set of hand-written sentence templates (13 categories x 10 templates; 6 train / 2 validation / 2 test per category). Language diversity is far below real EHR text; a model can learn template structure.",
        "Because templates are split-exclusive, validation/test measure generalisation to *new sentence frames*, but entity vocabulary is shared across all splits (small closed lexicon), so test entities are mostly seen in training. Scores on this data will be inflated and must not be reported as benchmark results.",
        "Entity-label balance is an artefact of the generator, not of any clinical population. Disease/symptom/drug co-occurrence is partly random, so some notes are clinically incoherent (e.g. a symptom unrelated to the drug listed).",
        "Doses, frequencies and durations are illustrative only and must not be used as clinical guidance.",
        "No real abbreviations noise, typos, copy-paste artefacts, tables, de-identification placeholders, or long-range discourse. Spelling variants are limited to listed US/UK forms.",
        "Annotation is generated by construction, not by human annotators; there is no inter-annotator agreement. Assertion labels follow the template, which cannot capture scope ambiguity found in real notes.",
        "Attribute spans (DOSAGE/ROUTE/FREQUENCY/DURATION) inherit the assertion of their drug. They are not linked to drugs by explicit relation IDs.",
        "`normalized_name` is a plain-text label chosen by the generator. No SNOMED CT, RxNorm or LOINC mapping was performed.",
        "Not clinically validated. Not real patient data. Not a licensed benchmark. Replace with properly licensed, human-annotated data before drawing scientific conclusions.",
    ]:
        L.append(f"- {s}")
    with open(f"{DATA}/validation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(json.dumps({k: report[k] for k in ("total_records", "records_per_split", "total_entities", "entities_by_category",
                                              "annotation_error_count", "duplicate_count", "leakage_problem_count", "pass")}, indent=1))
    print("assertions:", report["assertion_counts"])
    print("coverage:", coverage)
    print("lengths:", report["length_class_counts"], report["char_length_min_mean_max"])
    if errors:
        print("ERRORS:", dict(errors))
        for k, v in samples.items():
            print(" ", k, v)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
