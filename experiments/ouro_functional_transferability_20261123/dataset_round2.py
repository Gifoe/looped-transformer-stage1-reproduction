"""Freeze independent follow-up confirmations without rewriting either data receipt."""
import argparse
from pathlib import Path

from dataset_optional import official, qasc_previous
from dataset_transfer import (
    ARC_HASHES, CONFIRM_N, CSQA_HASHES, canonical, cohort_meta, digest, heldout_pool, json_lines, json_read,
    load_cohort, overlap, pick, pq_rows, raw_check, require, tokenize, write,
)


def read_cohorts(manifest):
    return {name: load_cohort(manifest, name) for name in json_read(manifest)["cohorts"]}


def prepare(args):
    main, optional = read_cohorts(args.main), read_cohorts(args.optional)
    require(json_read(args.optional)["main_manifest_sha256"] == digest(args.main),
            "The optional fit manifest refers to a different main freeze")
    archived = {}
    for (config, split), (count, checksum) in ARC_HASHES.items():
        path = (args.arc_corpus / f"{split}.parquet" if config == "ARC-Challenge"
                else args.inputs / f"arc_easy_{split}.parquet")
        key = f"official:{config}/{split}"
        archived[key] = [canonical(row, "arc", config, split)
                         for row in raw_check(path, count, checksum, pq_rows)]
    old_external, _ = official(args.inputs)
    for source in ("obqa", "qasc"):
        for split in ("validation", "test"):
            archived[f"official:{source}/{split}"] = old_external[source][split]
    qasc, _, past_provenance = qasc_previous(old_external, args.qasc_sealed, args.qasc_audit)
    archived["past_qasc_confirmation_or_components"] = qasc

    pool_arc = archived.pop("official:ARC-Easy/validation")
    # Existing Easy discovery and confirmation remain in the main exclusions.
    existing = {**archived, **main, **optional}
    used = [row for rows in existing.values() for row in rows]
    holdout_arc, arc_filters = heldout_pool(pool_arc, used, "arc-easy-validation")
    require(holdout_arc, "No clean, previously unused ARC-Easy Validation prompts remain")

    count, csqa_sha, _ = CSQA_HASHES["train"]
    chosen_csqa_file = args.inputs / "csqa_train.jsonl"
    data_csqa = [canonical(row, "csqa", "CSQA-author-random", "train")
                 for row in raw_check(chosen_csqa_file, count, csqa_sha, json_lines)]
    clean_csqa, csqa_filters = heldout_pool(data_csqa, used + pool_arc, "csqa-author-train-round2")
    holdout_csqa = pick(clean_csqa, CONFIRM_N, "Fresh CSQA Round 2 confirmation")

    output = dict(arc_round2_confirm=holdout_arc, csqa_round2_confirm=holdout_csqa)
    checks = {}
    for key, rows in output.items():
        other = "arc_round2_confirm" if key == "csqa_round2_confirm" else "csqa_round2_confirm"
        for exclusion, previous in {**existing, other: output[other]}.items():
            passed = overlap(rows, previous)
            require(not any(passed.values()), f"Round 2 overlap {key} / {exclusion}: {passed}")
            checks[f"{key}__{exclusion}"] = passed

    counts = dict(arc=dict(official_validation=len(pool_arc), eligible_used_all=len(holdout_arc), excluded=arc_filters),
                  csqa=dict(official_train=count, eligible=len(clean_csqa), selected=len(holdout_csqa),
                            excluded=csqa_filters))
    report = dict(schema=1, code_sha256=digest(Path(__file__)), main_manifest_sha256=digest(args.main),
                  optional_fit_manifest_sha256=digest(args.optional), sources_histories=list(existing),
                  audited_prior_qasc=past_provenance, counts=counts, parity_audits=checks,
                  rule="Take all clean unused ARC-Easy Val by original hash order, then 299 CSQA Train by separate hash order",
                  independent_scope="No matched ID or normalized stem within named cohorts. Upstream demos and pretraining are unverified.",
                  cohorts={})
    require(args.out / "ROUND2_MANIFEST.json" != args.optional, "Refusing to replace an existing data freeze")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.tokenizer:
        report["token_accounting"] = tokenize(args, output)
    for key, rows in output.items():
        path = args.out / f"{key}.jsonl"
        write(path, rows, jsonl=True)
        report["cohorts"][key] = cohort_meta(rows, path, "Held back from all named stages and fitting sources")
    destination = args.out / "ROUND2_MANIFEST.json"
    write(destination, report)
    print("FROZEN", destination, digest(destination), counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("main", "optional", "arc-corpus", "inputs", "out", "qasc-sealed", "qasc-audit"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    prepare(parser.parse_args())
