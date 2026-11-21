"""Freeze natural evaluation cohorts and strictly parse unconstrained answers."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import string
import sys
import unicodedata
from urllib.request import Request, urlopen


SALT = "ouro-native-state-20261121"
ARC_REVISION = "fdb98c79578d61bdf502e2ade3253c71154dd11f"
ARC_TRAIN_SHA256 = "0f84cb4e393a729371c3dfd334fce14215a99dcff8095bf31271787f1e305f20"
BOOLQ_VALIDATION_URL = "https://storage.googleapis.com/boolq/dev.jsonl"
BOOLQ_VALIDATION_SHA256 = "ebc29ea3808c5c611672384b3de56e83349fe38fc1fe876fd29b674d81d0a80a"
GSM8K_TEST_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
OPENBOOKQA_TRAIN_SHA256 = "98148f8a54e62eb862346a75192d5fb824d6cbb68f2f59aecd793d39ecb5cd8b"
OPENBOOKQA_REVISION = "388097ea7776314e93a529163e0fea805b8a6454"
COHORTS = ("dev", "pilot", "confirm_arc", "confirm_external", "controlled")
LEADING_ANSWER = re.compile(r"^(?:answer\s*:\s*|the answer is\s*:?[ \t]+)", re.IGNORECASE)
NUMERIC = r"[-+]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
EXACT_NUMBER = re.compile(rf"^\s*\$?({NUMERIC})\s*[.!]?\s*$")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    dig = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            dig.update(data)
    return dig.hexdigest()


def stable_key(*pieces):
    return sha("\0".join((SALT, *pieces)).encode("utf-8"))


def normalized_answer(text):
    clean = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return clean.rstrip(".!?").rstrip()


def normalized_stem(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized))


def parse_generation(text: str, choices: list[str], family: str | None = None) -> int | None:
    """Identify one exact full-choice match from the first nonblank output line."""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if family == "gsm8k":
        # Accept a bare first-line number or a sole explicit final answer line.
        marked = [line.strip()[4:].strip() for line in text.splitlines() if line.strip().startswith("####")]
        if len(marked) > 1:
            return None
        candidate = marked[0] if marked else LEADING_ANSWER.sub("", first, count=1)
        match = EXACT_NUMBER.fullmatch(candidate)
        if not match:
            return None
        value = Decimal(match.group(1).replace(",", ""))
        matches = [index for index, option in enumerate(choices) if Decimal(option) == value]
        return matches[0] if len(matches) == 1 else None
    value = normalized_answer(LEADING_ANSWER.sub("", first, count=1))
    if not value:
        return None
    candidates = [i for i, choice in enumerate(choices) if normalized_answer(choice) == value]
    return candidates[0] if len(candidates) == 1 else None


parse_natural_generation = parse_generation


def read_json_records(path):
    if path.suffix.casefold() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return value
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(row) for row in stream if row.strip()]


def read_columns(path, columns):
    """Project Parquet columns before materializing rows. JSON ignores omitted fields."""
    if path.suffix.casefold() == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path, columns=columns).to_pylist()
    return [{name: row[name] for name in columns} for row in read_json_records(path)]


def text_fields(question, choices, passage=None):
    prefix = f"Question: {question}\nAnswer:"
    rewrite = f"Query: {question}\nAnswer:"
    labels = "".join(f"\n{letter}. {c}" for letter, c in zip(string.ascii_uppercase, choices))
    letter_prompt = f"Question: {question}{labels}\nAnswer:"
    if passage is not None:
        header = f"Passage: {passage}\n"
        prefix = header + prefix
        rewrite = header + rewrite
        letter_prompt = header + letter_prompt
    return prefix, rewrite, letter_prompt


@dataclass(frozen=True)
class Item:
    """Selection record. Answer data are absent from the type."""

    identifier: str
    stem: str
    passage: str | None
    choices: list[str]
    family: str
    source_index: int
    source_labels: list[str] | None = None
    controlled_prompt: str | None = None

    @property
    def stem_key(self):
        return normalized_stem(self.stem)

    @property
    def passage_key(self):
        return normalized_stem(self.passage) if self.passage is not None else None


def multiple_choice_rows(path, family):
    result = []
    stem_column = "question_stem" if family == "openbookqa" else "question"
    for i, row in enumerate(read_columns(path, ["id", stem_column, "choices"])):
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        if len(choices) != len(labels) or len(set(labels)) != len(labels):
            raise ValueError(f"Malformed choices for {row['id']}")
        result.append(Item(row["id"], row[stem_column], None, choices, family, i, labels))
    return result


def boolean_rows(path):
    result = []
    for i, row in enumerate(read_columns(path, ["question", "passage"])):
        stem = row["question"].strip()
        if not stem.endswith("?"):
            stem += "?"
        passage = row["passage"].strip()
        identifier = "boolq:" + stable_key(normalized_stem(stem), normalized_stem(passage))[:24]
        result.append(Item(identifier, stem, passage, ["yes", "no"], "boolq", i))
    return result


def arithmetic_rows(path):
    return [Item(f"gsm8k/test/{i:04}", row["question"], None, [], "gsm8k", i)
            for i, row in enumerate(read_columns(path, ["question"]))]


def arithmetic_diagnostic(item, solution):
    marked = solution.rsplit("####", 1)
    match = EXACT_NUMBER.fullmatch(marked[-1]) if len(marked) == 2 else None
    if not match:
        raise ValueError(f"Missing official final number for {item.identifier}")
    target = Decimal(match.group(1).replace(",", ""))
    offsets = (-3, -1, 1, 3) if target >= 3 else (1, 2, 3, 5)
    values = [target, *(target + delta for delta in offsets[:3])]
    values.sort(key=lambda x: stable_key("numeric-diagnostic", item.identifier, str(x)))
    return replace(item, choices=[format(x, "f") for x in values]), values.index(target)


def rewrite_controlled(prompt):
    if prompt.startswith("Question:"):
        return "Query:" + prompt[len("Question:"):]
    if prompt.rstrip().endswith("Answer:"):
        stop = prompt.rfind("Answer:")
        return prompt[:stop] + "Response:" + prompt[stop + len("Answer:"):]
    return "Query: " + prompt


def controlled_rows(path):
    source = read_json_records(path)
    result = []
    for i, row in enumerate(source):
        # Previously held-out structural groups only unless the input already
        # consists of normalized evaluator items with no split field.
        split = row.get("split")
        if split is not None and split not in {"confirm", "transfer", "independent", "holdout", "test", "fresh_structure"}:
            continue
        task = row.get("task") or row.get("family") or "controlled"
        if "worlds" in row:
            prompt = row["worlds"]["r"]["prompt"]
            choices = row["options"]
        else:
            prompt = row["prompt"]
            choices = row.get("choices", row.get("options"))
        result.append(Item("controlled:" + row["id"], prompt, None, choices, f"controlled:{task}",
                           i, controlled_prompt=prompt))
    return result, source


def exclusion_lists(exclude_paths, demo_path):
    ids, stems = set(), set()
    for path in exclude_paths:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            value = value["exclude_ids"]
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"Expected string IDs or exclude_ids in {path}")
        ids.update(value)
    if demo_path is not None:
        rows = read_json_records(demo_path)
        stems.update(normalized_stem(row["stem"]) for row in rows)
    return ids, stems


def choose(rows, scope, claimed_stems, excluded_ids, max_chars):
    admitted = []
    seen_ids, seen_passages = set(), set()
    rejections = Counter()
    order = sorted(rows, key=lambda item: (stable_key(scope, item.identifier), item.identifier, item.source_index))
    for item in order:
        if item.identifier in excluded_ids:
            rejections["excluded_id"] += 1
            continue
        if not item.stem_key or (item.passage is not None and not item.passage_key):
            rejections["empty_text"] += 1
            continue
        if len(item.stem) + len(item.passage or "") > max_chars:
            rejections["predeclared_length_limit"] += 1
            continue
        if (not isinstance(item.choices, list)
                or (item.family != "gsm8k" and not 2 <= len(item.choices) <= 8)):
            rejections["invalid_choices"] += 1
            continue
        if any(not isinstance(x, str) for x in item.choices):
            rejections["invalid_choice_text"] += 1
            continue
        options = [normalized_answer(x) for x in item.choices]
        if not all(options) or len(set(options)) != len(options):
            rejections["ambiguous_full_answer"] += 1
            continue
        if item.identifier in seen_ids or item.stem_key in claimed_stems or (
            item.passage_key is not None and item.passage_key in seen_passages
        ):
            rejections["duplicate_id_stem_or_passage"] += 1
            continue
        admitted.append(item)
        seen_ids.add(item.identifier)
        claimed_stems.add(item.stem_key)
        if item.passage_key is not None:
            seen_passages.add(item.passage_key)
    return admitted, dict(rejections)


def collect_gold(path, items, column):
    if not items:
        return {}
    selected = {item.source_index for item in items}
    if path.suffix.casefold() == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        # Select row indices only after the ID/stem-based cohort is fixed.
        source = pq.read_table(path, columns=[column])
        indices = sorted(selected)
        values = source.take(pa.array(indices))[column].to_pylist()
        raw = dict(zip(indices, values))
    elif column == "arithmetic":
        raw = {i: row["answer"] for i, row in enumerate(read_json_records(path)) if i in selected}
        return {item.identifier: arithmetic_diagnostic(item, raw[item.source_index]) for item in items}
    else:
        raw = {i: row[column] for i, row in enumerate(read_json_records(path)) if i in selected}
    result = {}
    for item in items:
        answer = raw[item.source_index]
        if column == "answer":
            if type(answer) is not bool:
                raise ValueError(f"Expected Boolean target for {item.identifier}")
            result[item.identifier] = 0 if answer else 1
        else:
            result[item.identifier] = item.source_labels.index(answer)
    return result


def serialize(item, gold):
    if item.controlled_prompt is None:
        prompt, rewrite, letter_prompt = text_fields(item.stem, item.choices, item.passage)
    else:
        prompt, rewrite = item.controlled_prompt, rewrite_controlled(item.controlled_prompt)
        letter_prompt = None
    if not (type(gold) is int and 0 <= gold < len(item.choices)):
        raise ValueError(f"Invalid evaluator target for {item.identifier}")
    return {
        "id": item.identifier,
        "family": item.family,
        "prompt": prompt,
        "rewrite": rewrite,
        "choices": item.choices,
        "gold": gold,
        "letter_prompt": letter_prompt,
    }


def gold_free(manifest):
    output = dict(manifest)
    for cohort in COHORTS:
        output[cohort] = [
            {k: v for k, v in item.items() if k not in ({"gold", "choices", "letter_prompt"}
                                                       if item["family"] == "gsm8k" else {"gold"})}
            for item in manifest[cohort]
        ]
    return output


def min_rows(population, count, name):
    if len(population) < count:
        raise ValueError(f"{name} needs {count} distinct eligible items, found {len(population)}")


def build(
    arc_train: Path,
    external: Path | None = None,
    *,
    external_family: str = "boolq",
    controlled: Path | None = None,
    exclude=(),
    demo_stems: Path | None = None,
    pilot_count: int = 24,
    dev_count: int = 96,
    confirm_arc_count: int = 96,
    external_count: int = 96,
    controlled_count: int = 24,
    max_arc_chars: int = 800,
    max_external_chars: int = 950,
):
    for name, value in (
        ("pilot", pilot_count), ("dev", dev_count), ("confirm_arc", confirm_arc_count),
        ("controlled", controlled_count),
    ):
        if value < 1:
            raise ValueError(f"{name} count must be positive")
    if external_count < 0 or (external_count and external is None):
        raise ValueError("Supply --external for a positive --external-count")
    ids, used_stems = exclusion_lists(exclude, demo_stems)
    used_stems = set(used_stems)
    arc_rows = multiple_choice_rows(arc_train, "arc_challenge")
    arc, arc_rejections = choose(arc_rows, "arc", used_stems, ids, max_arc_chars)
    n_arc = pilot_count + dev_count + confirm_arc_count
    min_rows(arc, n_arc, "ARC")
    if external_count == 0:
        ext_rows = []
        column = None
    elif external_family == "boolq":
        ext_rows = boolean_rows(external)
        column = "answer"
    elif external_family == "openbookqa":
        ext_rows = multiple_choice_rows(external, "openbookqa")
        column = "answerKey"
    elif external_family == "gsm8k":
        if file_sha(external) != GSM8K_TEST_SHA256:
            raise ValueError("GSM8K requires the exact official main test JSONL")
        ext_rows = arithmetic_rows(external)
        column = "arithmetic"
    else:
        raise ValueError(f"Unknown external family {external_family!r}")
    ext, ext_rejections = choose(ext_rows, external_family, used_stems, ids, max_external_chars)
    min_rows(ext, external_count, external_family)
    pools = {
        "pilot": arc[:pilot_count],
        "dev": arc[pilot_count:pilot_count + dev_count],
        "confirm_arc": arc[pilot_count + dev_count:n_arc],
        "confirm_external": ext[:external_count],
        "controlled": [],
    }
    # Locks all choice-independent cohort assignments before opening label columns.
    controlled_meta = {}
    if controlled is not None:
        candidates, source = controlled_rows(controlled)
        admitted, rejections = choose(candidates, "controlled", used_stems, ids, max_arc_chars * 3)
        min_rows(admitted, controlled_count, "controlled")
        by_family = {name: [x for x in admitted if x.family == name]
                     for name in sorted({x.family for x in admitted})}
        balanced = [items[i] for i in range(max(map(len, by_family.values())))
                    for items in by_family.values() if i < len(items)]
        pools["controlled"] = balanced[:controlled_count]
        controlled_meta = {
            "sha256": file_sha(controlled), "raw": len(source), "eligible": len(admitted),
            "rejections": rejections, "selected_families": dict(Counter(x.family for x in pools["controlled"])),
            "scope": "Prior controlled structural groups. One receiver per group, round-robin across families.",
        }
    fixed_ids = {key: [item.identifier for item in pool] for key, pool in pools.items()}
    if len(set().union(*map(set, fixed_ids.values()))) != sum(map(len, fixed_ids.values())):
        raise ValueError("Duplicate identity across frozen cohorts")
    arc_gold = collect_gold(arc_train, arc[:n_arc], "answerKey")
    ext_gold = collect_gold(external, ext[:external_count], column)
    if column == "arithmetic":
        pools["confirm_external"] = [ext_gold[item.identifier][0] for item in pools["confirm_external"]]
        ext_gold = {key: value[1] for key, value in ext_gold.items()}
    controlled_gold = {}
    if controlled is not None:
        for item in pools["controlled"]:
            row = source[item.source_index]
            target = row["worlds"]["r"]["gold"] if "worlds" in row else row["gold"]
            controlled_gold[item.identifier] = item.choices.index(target) if isinstance(target, str) else target
    tags = {"pilot": arc_gold, "dev": arc_gold, "confirm_arc": arc_gold, "confirm_external": ext_gold,
            "controlled": controlled_gold}
    assigned = {cohort: [serialize(item, tags[cohort][item.identifier]) for item in pools[cohort]]
                for cohort in COHORTS}
    arc_hash = file_sha(arc_train)
    ext_hash = file_sha(external) if external_count else None
    assigned["provenance"] = {
        "salt": SALT,
        "selection": "Fixed hash order of source IDs. One normalized stem per cohort and across cohorts. "
                     "One normalized passage per external cohort. Input validity and fixed character caps only.",
        "cohort_ids_sha256": sha(json.dumps(fixed_ids, sort_keys=True).encode()),
        "source_hashes": {
            "arc": {
                "sha256": arc_hash, "standard_train_snapshot": arc_hash == ARC_TRAIN_SHA256,
                "standard_revision": ARC_REVISION,
                "raw": len(arc_rows), "eligible": len(arc), "rejections": arc_rejections,
                "scope": "ARC historically exposed. Confirm ARC is disjoint within this run.",
            },
            "external": {
                "family": external_family, "sha256": ext_hash,
                "known_boolq_validation_original": ext_hash == BOOLQ_VALIDATION_SHA256,
                "known_gsm8k_test_original": ext_hash == GSM8K_TEST_SHA256,
                "known_openbookqa_train_original": ext_hash == OPENBOOKQA_TRAIN_SHA256,
                "openbookqa_train_revision": OPENBOOKQA_REVISION if external_family == "openbookqa" else None,
                "raw": len(ext_rows), "eligible": len(ext), "rejections": ext_rejections,
                "scope": "Dataset transfer selected before behavior. Pretraining exposure unverified.",
            } if external_count else {"scope": "External cohort deferred. ARC cohort order is invariant."},
            "controlled": controlled_meta,
        },
        "excluded_input_files": [{"sha256": file_sha(path)} for path in exclude],
        "demo_stems_file_sha256": file_sha(demo_stems) if demo_stems else None,
        "max_stem_and_passage_characters": {"arc": max_arc_chars, "external": max_external_chars},
        "prompt_rewrite": "Replace only the leading question cue on natural tasks. Stem and passage are unchanged.",
        "primary_protocol": "Open unrestricted generation with exact first-line full-choice parsing.",
        "other_protocols": "Teacher-forced full content and optional displayed-letter readout are separate.",
        "gsm8k_diagnostics": "Numeric alternatives depend on the target and are restricted to the evaluator. "
                             "They appear only in diagnostic multiple-choice prompts, never in natural generation."
                             if external_family == "gsm8k" and external_count else None,
    }
    return assigned


def write_new(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def download_boolq(path):
    request = Request(BOOLQ_VALIDATION_URL, headers={"User-Agent": "ouro-native-state/1.0"})
    with urlopen(request, timeout=30) as response:
        body = response.read(4 << 20)
    if sha(body) != BOOLQ_VALIDATION_SHA256:
        raise ValueError("Downloaded BoolQ bytes do not match the official validation digest")
    with path.open("xb") as stream:
        stream.write(body)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    b = sub.add_parser("build")
    b.add_argument("--arc-train", type=Path, required=True)
    b.add_argument("--external", type=Path)
    b.add_argument("--external-family", choices=("boolq", "openbookqa", "gsm8k"), default="boolq")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--discovery-out", type=Path)
    b.add_argument("--controlled", type=Path)
    b.add_argument("--exclude", type=Path, action="append", default=[])
    b.add_argument("--demo-stems", type=Path)
    for name, default in (
        ("pilot-count", 24), ("dev-count", 96), ("confirm-arc-count", 96), ("external-count", 96),
        ("controlled-count", 24), ("max-arc-chars", 800), ("max-external-chars", 950),
    ):
        b.add_argument("--" + name, type=int, default=default)
    fetch = sub.add_parser("fetch-boolq")
    fetch.add_argument("out", type=Path)
    a = parser.parse_args(argv)
    try:
        if a.operation == "fetch-boolq":
            download_boolq(a.out)
            print(BOOLQ_VALIDATION_SHA256)
            return 0
        fields = vars(a).copy()
        for key in ("out", "discovery_out", "operation"):
            fields.pop(key)
        result = build(**fields)
        if a.discovery_out is not None:
            if a.out.resolve() == a.discovery_out.resolve():
                raise ValueError("Evaluation and discovery require different output paths")
            if a.out.exists() or a.discovery_out.exists():
                raise FileExistsError("One or both manifest destinations already exist")
            write_new(a.discovery_out, gold_free(result))
        write_new(a.out, result)
        print(json.dumps({key: len(result[key]) for key in COHORTS}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Data build rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
