"""Freeze optional, equal-question OpenBookQA and QASC training sources."""
import argparse
from pathlib import Path

from dataset_transfer import (
    TRAIN_N, canonical, cohort_meta, digest, digest_values, heldout_pool, json_read,
    load_cohort, overlap, pick, require, tokenize, write,
)


SOURCES = {
    "obqa": dict(
        repository="allenai/openbookqa", config="main", revision="8c4888010de9b23a2d136ebd31d5f5ea9f3d90fa",
        stem="question_stem", license="Apache-2.0",
        splits=dict(
            train=(4957, "98148f8a54e62eb862346a75192d5fb824d6cbb68f2f59aecd793d39ecb5cd8b"),
            validation=(500, "35370b9cfee8c1ff325ccc74adc434d12c47ca0ac3244aa87f3fa77069285206"),
            test=(500, "cd5483e366daa230c1c87bbdc512d8b7229f14f6dd04d19fc8b1a3855aaaa8a3"),
        ),
    ),
    "qasc": dict(
        repository="allenai/qasc", config="data", revision="a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",
        stem="question", license="CC BY 4.0",
        splits=dict(
            train=(8134, "b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1"),
            validation=(926, "d1ae34ae13c5fce2c55305372c203c6cdb789728d0d7e5ea2956d55bc33f40ae"),
            test=(920, "495cfbd17abc1720b54785cdce68825104aaf60d7b8bdb2acac62157a31eb517"),
        ),
    ),
}
PRIOR_QASC_SELECTED_SHA = "342110fc6e459dcee6d780732f9bed2cb230e1e5c621254ec9fd50702ff4904c"
PRIOR_QASC_AUDIT_SHA = "7f03a6a4cdb21de9b587054d6b5b2ccb7465dfaf371a55e44cada64f402d6ad0"


def official(inputs):
    import pyarrow.parquet as pq

    sets, metadata = {}, {}
    for name, info in SOURCES.items():
        sets[name] = {}
        for split, (count, checksum) in info["splits"].items():
            path = inputs / f"{name}_{split}.parquet"
            require(digest(path) == checksum, f"Official checksum mismatch in {path}")
            columns = ["id", info["stem"], "choices"]
            raw = pq.read_table(path, columns=columns).to_pylist()
            require(len(raw) == len({row["id"] for row in raw}) == count, f"Official coverage mismatch in {path}")
            sets[name][split] = [
                {"id": row["id"], "question": row[info["stem"]], "choices": row["choices"]} for row in raw
            ]
            source = (f"https://huggingface.co/datasets/{info['repository']}/resolve/{info['revision']}/"
                      f"{info['config']}/{split}-00000-of-00001.parquet")
            metadata[f"{name}/{split}"] = dict(n=count, sha256=checksum, columns=columns, source=source,
                                              license=info["license"])
    return sets, metadata


def attach_train_labels(inputs, name, rows):
    import pyarrow.parquet as pq

    allowed = [row["id"] for row in rows]
    labels = pq.read_table(inputs / f"{name}_train.parquet", columns=["id", "answerKey"],
                           filters=[("id", "in", allowed)]).to_pylist()
    labels = {row["id"]: row["answerKey"] for row in labels}
    require(set(labels) == set(allowed), "Training label join changed the frozen row set")
    config = "OpenBookQA-main" if name == "obqa" else "QASC"
    return [canonical(dict(row, answerKey=labels[row["id"]]), name, config, "train") for row in rows]


def qasc_previous(sets, sealed, audit_path):
    require(digest(sealed) == PRIOR_QASC_SELECTED_SHA and digest(audit_path) == PRIOR_QASC_AUDIT_SHA,
            "Historical QASC provenance digest mismatch")
    audit = json_read(audit_path)
    previous = json_read(sealed)
    qasc_sha = SOURCES["qasc"]["splits"]["train"][1]
    require(audit["parquet_sha256"] == previous["parquet_sha256"] == qasc_sha, "QASC revision changed")
    selected = {r["id"] for r in previous["cohort"]}
    require(len(selected) == 384, "Historical QASC confirmation count is invalid")
    train = sets["qasc"]["train"]
    by_id = {r["id"]: r for r in train}
    components = [set(ids) for ids in audit["qasc_duplicate_components"]]
    prior_excluded = set(audit["exclude_ids"])
    require(selected | prior_excluded <= by_id.keys() and all(ids <= by_id.keys() for ids in components),
            "Historical QASC IDs fail to map to the pinned training split")
    excluded = selected | prior_excluded
    for group in components:
        if excluded & group:
            excluded.update(group)
    return [by_id[k] for k in sorted(excluded)], components, dict(
        historical_confirmation_sha256=PRIOR_QASC_SELECTED_SHA, historical_exclusions_sha256=PRIOR_QASC_AUDIT_SHA,
        historical_confirmation_n=len(selected), historical_overlap_ids=len(prior_excluded),
        n_excluded_after_component_propagation=len(excluded), excluded_id_sha256=digest_values(sorted(excluded)),
        historical_label_or_outcome_access=False,
    )


def decluster(rows, components):
    group_by_id = {member: min(group) for group in components for member in group}
    seen, retained = set(), []
    for row in rows:
        group = group_by_id.get(row["id"], row["id"])
        if group not in seen:
            retained.append(row)
            seen.add(group)
    return retained, len(rows) - len(retained)


def prepare(args):
    sets, metadata = official(args.inputs)
    history, components, audit = qasc_previous(sets, args.qasc_sealed, args.qasc_audit)
    if args.archived_obqa_train:
        path = args.archived_obqa_train
        require(digest(path) == SOURCES["obqa"]["splits"]["train"][1],
                "The prior OpenBookQA train file differs from the pinned official release")
        audit["previous_architecture_obqa_train_reused_exact_sha"] = digest(path)
    main = json_read(args.manifest)
    reference = {key: load_cohort(args.manifest, key) for key in main["cohorts"]}
    evals = {
        f"{name}_{split}": sets[name][split] for name in SOURCES for split in ("validation", "test")
    }
    all_named = {**reference, **evals, "historical_qasc_evaluation_and_components": history}
    exclusions = [row for rows in all_named.values() for row in rows]
    datasets, filters, pool_sizes = {}, {}, {}
    for name in SOURCES:
        other = "qasc" if name == "obqa" else "obqa"
        pool, counted = heldout_pool(sets[name]["train"], exclusions + sets[other]["train"], f"optional-fit-{name}")
        if name == "qasc":
            pool, grouped = decluster(pool, components)
            counted["excluded_historical_near_duplicate_component_members"] = grouped
        pool_sizes[name] = len(pool)
        filters[name] = counted
        keep = pick(pool, TRAIN_N, f"{name} official Train")
        datasets[f"{name}_train"] = attach_train_labels(args.inputs, name, keep)
    results = {}
    for name, rows in datasets.items():
        other_train = sets["qasc" if name == "obqa_train" else "obqa"]["train"]
        for label, reference_set in {**all_named, "other_optional_official_train": other_train}.items():
            match = overlap(rows, reference_set)
            require(not any(match.values()), f"Unexpected overlap {name} with {label}: {match}")
            results[f"{name}__{label}"] = match
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema=1, main_manifest_sha256=digest(args.manifest), code_sha256=digest(Path(__file__)),
        sources=metadata, historical_qasc_and_obqa=audit, pool_sizes=pool_sizes, filters=filters,
        parity_audits=results, matching_scope="1119 source-specific questions under main SHA256 ordering and dedup rules",
        scorer_columns=["id", "question", "choices", "answerKey", "dataset", "source_config", "source_split"],
        label_read="Only id and answerKey for the 1119 already selected official Train IDs per source",
        qasc_excluded_columns=["fact1", "fact2", "combinedfact", "formatted_question"],
        optional_training_only=True, cohorts={},
    )
    if args.tokenizer:
        manifest["token_accounting"] = tokenize(args, datasets)
    for name, rows in datasets.items():
        output = args.out / f"{name}.jsonl"
        write(output, rows, jsonl=True)
        manifest["cohorts"][name] = cohort_meta(rows, output, "Optional held-back official Train fit source")
    write(args.out / "MANIFEST.json", manifest)
    print("FROZEN", args.out / "MANIFEST.json", digest(args.out / "MANIFEST.json"),
          {key: value["n"] for key, value in manifest["cohorts"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "inputs", "out", "qasc-sealed", "qasc-audit"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--archived-obqa-train", type=Path)
    prepare(parser.parse_args())
