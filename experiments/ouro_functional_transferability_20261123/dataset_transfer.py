"""Freeze matched ARC/CommonsenseQA training and independently scoped evaluation cohorts."""
import argparse
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
ARC_HASHES = {
    ("ARC-Challenge", "train"): (1119, "0f84cb4e393a729371c3dfd334fce14215a99dcff8095bf31271787f1e305f20"),
    ("ARC-Challenge", "validation"): (299, "a0796ca9c64d4633b595dae2b350485de9cc4f45c4adf9b59b371268f945984d"),
    ("ARC-Challenge", "test"): (1172, "b8e71916ead552c4e3bb05e3506bfbe4b5caa93b155e680fabb74dcb1ba14ec0"),
    ("ARC-Easy", "train"): (2251, "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5"),
    ("ARC-Easy", "validation"): (570, "ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74"),
    ("ARC-Easy", "test"): (2376, "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177"),
}
CSQA_HASHES = {
    "train": (9741, "58ffa3c8472410e24b8c43f423d89c8a003d8284698a6ed7874355dedd09a2fb", "train_rand_split.jsonl"),
    "validation": (1221, "3210497fdaae614ac085d9eb873dd7f4d49b6f965a93adadc803e1229fd8a02a",
                   "dev_rand_split.jsonl"),
}
EARLIER_DIRS = {
    "minimal": "ouro-minimal-functional-feedback-20261123",
    "control": "ouro-functional-controllability-20261123",
    "unified": "ouro-unified-rank1-20261123",
    "readout": "ouro-functional-readout-feedback-20261122",
}
SEED = "functional-transferability-20261123-v1"
TRAIN_N = 1119
DISCOVERY_N = 200
CONFIRM_N = 299


def canonical_prompt(row):
    return f'Question: {row["question"]}\nAnswer:'


def canonical_letter_prompt(row):
    labels = [chr(65 + index) for index in range(len(row["choices"]["text"]))]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, row["choices"]["text"], strict=True))
    target = labels[row["choices"]["label"].index(row["answerKey"])]
    return f'Question: {row["question"]}\n{options}\nAnswer with the option letter.\nAnswer:', labels, target


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_values(values):
    payload = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write(path, value, jsonl=False):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    temporary.replace(path)


def raw_check(path, count, checksum, reader):
    require(digest(path) == checksum, f"Official input checksum mismatch in {path}")
    rows = reader(path)
    require(len(rows) == count == len({row["id"] for row in rows}), f"Wrong official coverage in {path}")
    return rows


def pq_rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def canonical(row, dataset, source_config, source_split):
    question = row.get("question", row.get("question_stem"))
    if isinstance(question, dict):
        options = question["choices"]
        question = question["stem"]
        labels, choices = [item["label"] for item in options], [item["text"] for item in options]
    else:
        labels, choices = row["choices"]["label"], row["choices"]["text"]
    require(isinstance(row["id"], str) and isinstance(question, str) and bool(question.strip()),
            f"Missing row ID or question: {row.get('id')}")
    require(2 <= len(choices) == len(labels) == len(set(labels)) and row["answerKey"] in labels
            and all(isinstance(option, str) and option.strip() for option in choices),
            f"Malformed options or gold for {row['id']}")
    return dict(id=row["id"], question=question, choices=dict(label=labels, text=choices), answerKey=row["answerKey"],
                dataset=dataset, source_config=source_config, source_split=source_split)


def normalized(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", value))


def signatures(row):
    stem = normalized(row["question"])
    options = sorted(normalized(c) for c in row["choices"]["text"])
    return row["id"], stem, json.dumps([stem, options], ensure_ascii=False, separators=(",", ":"))


def overlap(left, right):
    left_columns = identity_sets(left)
    right_columns = identity_sets(right)
    return {name: len(a & b) for name, a, b in zip(("id", "normalized_stem", "normalized_stem_and_options"),
                                                  left_columns, right_columns, strict=True)}


def identity_sets(rows):
    columns = set(), set(), set()
    for row in rows:
        for column, token in zip(columns, signatures(row), strict=True):
            column.add(token)
    return columns


def order(rows, namespace):
    return sorted(rows, key=lambda row: (hashlib.sha256(
        f"{SEED}:{namespace}:{row['id']}".encode("utf-8")).hexdigest(), row["id"]))


def heldout_pool(rows, excluded, namespace):
    history = identity_sets(excluded)
    seen = identity_sets([])
    pool, counted = [], Counter()
    for row in order(rows, namespace):
        ident, stem, full = signatures(row)
        if ident in history[0]:
            counted["excluded_historical_or_other_source_id"] += 1
        elif stem in history[1]:
            counted["excluded_historical_or_other_source_normalized_stem"] += 1
        elif stem in seen[1]:
            counted["excluded_within_pool_duplicate_normalized_stem"] += 1
        else:
            pool.append(row)
            for column, token in zip(seen, (ident, stem, full), strict=True):
                column.add(token)
    return pool, dict(counted)


def pick(data, count, name):
    require(len(data) >= count, f"{name} only has {len(data)} audited questions, needs {count}")
    return data[:count]


def historical_ids(roots, arc, easy, csqa):
    records = {}

    def check_ids(path, expected, relation="equal"):
        previous = json_lines(path)
        observed = {row["id"] for row in previous}
        needed = {row["id"] for row in expected}
        require(len(previous) == len(observed) and
                (observed == needed if relation == "equal" else observed <= needed),
                f"Historical eligible population is incompatible in {path}")
        records[path.name] = dict(file=str(path), sha256=digest(path), n_ids=len(observed),
                                  id_sha256=digest_values(sorted(observed)), relation=relation)
        return observed

    control = roots["control"] / "results"
    check_ids(control / "CSQA_TRANSFER/PREDICTIONS.jsonl", csqa["validation"])
    check_ids(control / "BENCHMARK_RESULTS/RESULTS_ARC.jsonl", arc["test"])
    dev = roots["unified"] / "results/dev"
    coh = json_read(dev / "COHORTS.json")["cohorts"]["arc_train_dev"]
    # The input lock stores the original IDs. Cross-check against actual model outputs.
    ids = check_ids(dev / "arc_train_dev__native__k4__rank.jsonl", arc["train"], "subset")
    require(ids == set(coh["screen"] + coh["lock"]) and len(ids) == 256,
            "ARC development lock and actual saved model outcomes differ")
    records["unified_arc_dev_lock"] = dict(file=str(dev / "COHORTS.json"), sha256=digest(dev / "COHORTS.json"),
                                         n_ids=len(ids), id_sha256=digest_values(sorted(ids)))
    stage2 = roots["minimal"] / "stage2/inputs/arc_easy.jsonl"
    receipt_path = stage2.with_suffix(".jsonl.manifest.json")
    receipt = json_read(receipt_path)
    scope = receipt["tasks"][0]
    require(receipt["jsonl_sha256"] == digest(stage2) and len(receipt["tasks"]) == 1
            and (scope["task"], scope["dataset_config"], scope["dataset_revision"], scope["eval_split"],
                 scope["fewshot_split"], scope["shots"]) == ("arc_easy", "ARC-Easy", ARC_REVISION, "test", "train", 8),
            "Prior ARC-Easy membership or few-shot split metadata mismatch")
    previous = json_lines(stage2)
    require(len(previous) == len(easy["test"]) == len({row["doc_id"] for row in previous})
            and {row["doc_id"] for row in previous} == set(range(len(easy["test"]))),
            "Archived ARC-Easy results must map to every original official Test document")
    for row in previous:
        item = easy["test"][row["doc_id"]]
        require(row["choices"] == item["choices"]["text"]
                and row["target"] == item["choices"]["label"].index(item["answerKey"]),
                f"Archived ARC-Easy Test index does not match official ID {item['id']}")
    records["stage2_arc_easy_test_ids"] = dict(
        file=str(stage2), sha256=receipt["jsonl_sha256"], lock_sha256=digest(receipt_path),
        n_ids=len(previous), id_sha256=digest_values(sorted(row["id"] for row in easy["test"])),
        direct_mapping="doc_id indexes official Test at the same revision; choices and gold matched per row",
        earlier_demonstration_scope="ARC-Easy official Train, eight examples per prompt; exclude its entire source split",
    )
    return records


def cohort_meta(rows, path, scope):
    return dict(
        file=path.name, sha256=digest(path), n=len(rows), id_sha256=digest_values(row["id"] for row in rows),
        sorted_id_sha256=digest_values(sorted(row["id"] for row in rows)),
        normalized_stem_sha256=digest_values(sorted(normalized(row["question"]) for row in rows)),
        duplicate_normalized_stems=len(rows) - len(identity_sets(rows)[1]),
        source_counts=dict(sorted(Counter(f'{r["source_config"]}/{r["source_split"]}' for r in rows).items())),
        choices_per_question=dict(sorted(Counter(len(r["choices"]["text"]) for r in rows).items())),
        gold_option_index=dict(sorted(Counter(r["choices"]["label"].index(r["answerKey"]) for r in rows).items())),
        role=scope,
    )


def load_cohort(path, name):
    """Load one immutable model-input cohort from the shared manifest."""
    path = Path(path)
    manifest = json_read(path)
    item = manifest["cohorts"][name]
    cohort = path.parent / item["file"]
    rows = raw_check(cohort, item["n"], item["sha256"], json_lines)
    require(digest_values(row["id"] for row in rows) == item["id_sha256"], f"ID order mismatch in {cohort}")
    return rows


def tokenize(a, cohorts):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer.resolve(), trust_remote_code=True, local_files_only=True)
    output = {}
    for name, rows in cohorts.items():
        stats = Counter()
        for row in rows:
            prefix = canonical_prompt(row)
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            for text in row["choices"]["text"]:
                full = tokenizer.encode(prefix + " " + text, add_special_tokens=False)
                require(full[:len(prefix_tokens)] == prefix_tokens and len(full) > len(prefix_tokens),
                        f"Prompt boundary differs from legacy scorer for {row['id']}")
                stats["candidate_input_tokens"] += len(full) - 1
                stats["candidate_answer_tokens"] += len(full) - len(prefix_tokens)
                stats["candidates"] += 1
            letter, _, gold = canonical_letter_prompt(row)
            prompt_ids = tokenizer.encode(letter, add_special_tokens=False)
            full = tokenizer.encode(letter + " " + gold, add_special_tokens=False)
            require(full[:len(prompt_ids)] == prompt_ids and len(full) > len(prompt_ids),
                    f"Generation label token boundary does not preserve legacy scorer for {row['id']}")
            stats["generation_input_tokens"] += len(prompt_ids)
            stats["generation_gold_answer_tokens"] += len(full) - len(prompt_ids)
        output[name] = dict(stats)
    return output


def prepare(a):
    arcs = {}
    sources = {}
    for (config, split), (count, checksum) in ARC_HASHES.items():
        path = (a.arc_corpus / f"{split}.parquet" if config == "ARC-Challenge"
                else a.inputs / f"arc_easy_{split}.parquet")
        raw = raw_check(path, count, checksum, pq_rows)
        arcs.setdefault(config, {})[split] = [canonical(row, "arc", config, split) for row in raw]
        sources[f"{config}/{split}"] = dict(
            n=count, sha256=checksum, file=str(path), license="CC BY-SA 4.0",
            source=f"https://huggingface.co/datasets/allenai/ai2_arc/resolve/{ARC_REVISION}/{config}/{split}-00000-of-00001.parquet",
        )
    roots = {key: a.historical_root / directory for key, directory in EARLIER_DIRS.items()}
    author_val = a.csqa_validation or roots["control"] / "results/CSQA_TRANSFER/commonsenseqa_validation.jsonl"
    csqa = {}
    for split, (count, checksum, filename) in CSQA_HASHES.items():
        path = a.inputs / "csqa_train.jsonl" if split == "train" else author_val
        raw = raw_check(path, count, checksum, json_lines)
        csqa[split] = [canonical(row, "csqa", "CSQA-author-random", split) for row in raw]
        sources[f"CSQA-author-random/{split}"] = dict(
            n=count, sha256=checksum, file=str(path), license_status="Author dataset metadata license field is blank",
            source=f"https://s3.amazonaws.com/commensenseqa/{filename}",
        )

    arc, easy = arcs["ARC-Challenge"], arcs["ARC-Easy"]
    audit = historical_ids(roots, arc, easy, csqa)
    historical_mcq = [
        ("historical_obqa_test", roots["readout"] / "external/openbookqa.parquet"),
        ("historical_obqa_validation", roots["unified"] / "ob_validation.parquet"),
    ] + [(f"extra_history_{i}", path) for i, path in enumerate(a.extra_historical_parquet)]
    extra = []
    for key, path in historical_mcq:
        require(path.is_file(), f"Missing audited comparison dataset {path}")
        rows = [canonical(row, "historical", key, "unknown") for row in pq_rows(path)]
        extra.extend(rows)
        audit[key] = dict(file=str(path), sha256=digest(path), n_rows=len(rows))

    past_arc = arc["train"] + arc["validation"] + arc["test"] + easy["train"] + easy["test"]
    outside_csqa = past_arc + easy["validation"] + csqa["validation"] + extra
    csqa_pool, csqa_filter = heldout_pool(csqa["train"], outside_csqa, "csqa-author-train")
    clean_train = pick(csqa_pool, TRAIN_N, "CSQA Train")
    csqa_rest = csqa_pool[TRAIN_N:]
    csqa_discovery = pick(csqa_rest, DISCOVERY_N, "CSQA discovery")
    csqa_confirm = pick(csqa_rest[DISCOVERY_N:], CONFIRM_N, "CSQA confirmation")
    easy_excluded = past_arc + csqa["validation"] + clean_train + csqa_discovery + csqa_confirm + extra
    easy_pool, easy_filter = heldout_pool(easy["validation"], easy_excluded, "arc-easy-validation")
    arc_discovery = pick(easy_pool, DISCOVERY_N, "ARC-Easy discovery")
    arc_confirm = pick(easy_pool[DISCOVERY_N:], CONFIRM_N, "ARC-Easy confirmation")
    arc_train = order(arc["train"], "arc-challenge-train")
    mixed = order(order(arc_train, "mixed-arc")[:560] + order(clean_train, "mixed-csqa")[:559], "mixed")

    outputs = dict(
        arc_train=arc_train, csqa_train=clean_train, arc_csqa_mixed_train=mixed,
        arc_discovery=arc_discovery, arc_confirm=arc_confirm,
        csqa_discovery=csqa_discovery, csqa_confirm=csqa_confirm,
        arc_challenge_validation=arc["validation"], arc_challenge_test=arc["test"],
        csqa_validation=csqa["validation"],
    )
    scope = {
        "arc_train": "Official ARC-Challenge Train, all 1119; used by historical and current fitting",
        "csqa_train": "Official CSQA Train-derived, 1119 unique stems sampled under frozen rule",
        "arc_csqa_mixed_train": "560 ARC-Challenge and 559 CSQA from the named fitting cohorts, total 1119",
        "arc_discovery": "ARC-Easy Validation-derived discovery subcohort, disjoint from historical audited inputs",
        "arc_confirm": "ARC-Easy Validation-derived independent confirmation, subject to recorded audit scope",
        "csqa_discovery": "CSQA official Train-derived discovery held out from current fitting and historical reported Val",
        "csqa_confirm": "CSQA official Train-derived independent confirmation, subject to recorded audit scope",
        "arc_challenge_validation": "Historical ARC-Challenge Validation previously used in selection; secondary only",
        "arc_challenge_test": "Historical ARC-Challenge Test, previously scored; secondary only",
        "csqa_validation": "Author official CSQA Validation in full; historically scored, separate secondary cohort",
    }
    fresh = ("arc_discovery", "arc_confirm", "csqa_discovery", "csqa_confirm")
    checks = {}
    for first, second in combinations(outputs, 2):
        if first not in fresh and second not in fresh:
            continue
        check = overlap(outputs[first], outputs[second])
        require(not any(check.values()), f"Data overlap from {first} to {second}: {check}")
        checks[f"{first}__{second}"] = check
    for key in fresh:
        check = overlap(outputs[key], past_arc + csqa["validation"] + extra)
        require(not any(check.values()), f"Historical exact-normalized match survived in {key}: {check}")
        checks[f"{key}__audited_history"] = check
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema=1, seed=SEED, code_sha256=digest(Path(__file__)), sources=sources, historical_id_artifacts=audit,
        sampling={
            "train_per_source": TRAIN_N, "heldout_per_source": {"discovery": DISCOVERY_N, "confirm": CONFIRM_N},
            "order": "SHA-256(seed:namespace:official_id), with official_id tie-break, independent of labels/scores",
            "dedup": "Unicode NFKC casefold and regex word runs; exclude ID or normalized-stem matches",
            "arc_easy_eligible": len(easy_pool), "arc_easy_filters": easy_filter,
            "csqa_train_eligible": len(csqa_pool), "csqa_train_filters": csqa_filter,
        },
        independence_scope=[
            "No ID or normalized-stem overlap with the named prior corpora or new fitting/discovery cohorts.",
            "ARC primary uses ARC-Easy, a difficulty-shifted sister configuration, while fitted ARC is ARC-Challenge.",
            "ARC-Easy training/test and all ARC-Challenge rows are conservatively removed as a historical exclusion set.",
            "CSQA primary confirmation is a holdout from official Train. Official author Validation was evaluated historically.",
            "This audit cannot exclude paraphrastic overlap or foundation-model exposure.",
        ],
        parity_audits=checks,
        rendering={
            "candidate_prefix": "Question: {question}\nAnswer:",
            "candidate_continuation": "one leading ASCII space followed by the original option text",
            "strict_zero_shot_letter_prompt": "same rendering as historical minimal.label_prompt, contiguous A–E labels",
        },
        cohorts={},
    )
    if a.tokenizer:
        manifest["token_accounting"] = tokenize(a, outputs)
    for name, rows in outputs.items():
        path = a.out / f"{name}.jsonl"
        write(path, rows, jsonl=True)
        manifest["cohorts"][name] = cohort_meta(rows, path, scope[name])
    write(a.out / "MANIFEST.json", manifest)
    print("FROZEN", a.out / "MANIFEST.json", digest(a.out / "MANIFEST.json"),
          {name: item["n"] for name, item in manifest["cohorts"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("arc-corpus", "inputs", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("csqa-validation", "tokenizer"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--historical-root", type=Path, default=Path("/root"))
    parser.add_argument("--extra-historical-parquet", type=Path, action="append", default=[])
    prepare(parser.parse_args())
