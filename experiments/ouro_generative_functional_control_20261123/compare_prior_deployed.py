"""Audit the released T1/T2 deployment against the new controller without refitting."""

import argparse
from pathlib import Path
import json

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from dataset_transfer import canonical
from evaluate_generation import data_mode, evaluate
from paired_analysis import compare, read
from task_data import load
import minimal


def indexed(rows, key="id"):
    table = {row[key]: row for row in rows}
    if len(table) != len(rows):
        raise ValueError(f"Repeated {key} in archived source")
    return table


def categorical(record, key):
    values = np.asarray(record["scores"][key])
    normalized = values / np.asarray(record["characters"])
    return dict(acc=int(values.argmax()) == record["gold"], acc_norm=int(normalized.argmax()) == record["gold"])


def candidate_comparison(old_source, archive, current_data, new_rows, native_rows, t1_key):
    previous_scores = indexed(load(archive))
    raw_source = load(old_source) if old_source.suffix == ".jsonl" else pq.read_table(old_source).to_pylist()
    previous_data = indexed([canonical(row, "historical", "historical", "evaluation") for row in raw_source])
    source = indexed(current_data, "sample_id")
    reference, own, original_native, current_native = [], [], [], []
    excluded = []
    native = indexed(native_rows)
    maximum_native_score_diff = 0.
    for model in new_rows:
        qid = model["id"]
        archived = previous_data.get(qid)
        archived_scores = previous_scores.get(qid)
        fresh = source[qid]["candidate"]
        if not archived or not archived_scores:
            excluded.append(qid)
            continue
        signatures_equal = (archived["question"] == fresh["question"] and
                            archived["choices"]["text"] == fresh["choices"]["text"] and
                            archived["choices"]["label"].index(archived["answerKey"]) == model["gold"] == archived_scores["gold"] and
                            archived_scores["characters"] == model["chars"])
        if not signatures_equal:
            excluded.append(qid)
            continue
        reference.append(dict(id=qid, metrics=categorical(archived_scores, t1_key)))
        own.append(model)
        original_native.append(dict(id=qid, metrics=categorical(archived_scores, "native_k4")))
        current_native.append(native[qid])
        before = np.asarray(archived_scores["scores"]["native_k4"])
        after = np.asarray(native[qid]["scores"])
        maximum_native_score_diff = max(maximum_native_score_diff, float(abs(before-after).max()))
    return dict(source_sha256=minimal.sha(old_source), historical_scores_sha256=minimal.sha(archive),
                historical_method=t1_key, exact_stem_ordered_option_and_gold_matches=len(own), excluded_ids=excluded,
                deployed_vs_new=compare(reference, own), native_across_runs=compare(original_native, current_native),
                max_native_candidate_logprob_difference=maximum_native_score_diff)


def prior_generation(archive, current_source, new_rows, tok):
    prior = indexed(load(archive))
    formatted = indexed(data_mode(current_source, "obqa", "letter"), "sample_id")
    older = []
    newer = []
    for row in new_rows:
        key = row["id"]
        x = formatted[key]
        y = prior[key]
        if y["prompt"] != x["prompt"] or y["gold"] != x["label"]:
            raise ValueError(f"Different deployed prompt or target {key}")
        if len(y["tokens"]) > 6:
            raise ValueError("Historical cap differs from six tokens")
        former = evaluate(x, "letter", y["text"], y["stop"])
        if former["correct"] != y["correct"] or former["strict"] != (y["correct"] and y["strict_format"]):
            raise RuntimeError(f"Historical outcome differs under the current parser {key}")
        prefix = tok.decode(row["generated_tokens"][:6], skip_special_tokens=True)
        older.append(dict(id=key, metrics=former))
        newer.append(dict(id=key, metrics=evaluate(x, "letter", prefix, row["stop"])))
    return dict(source_sha256=minimal.sha(archive), matched=len(newer), max_new=6,
                procedure="Archive T2 and new greedy prefix with matched prompt, full vocabulary and six tokens. Batches were 16 and 48.",
                deployed_vs_new=compare(older, newer))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("model", "arc", "obqa", "csqa", "old", "results", "data"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    a = parser.parse_args()
    cfg = read(a.results / "FINAL_PROTOCOL.json")
    chosen = cfg["selected_arm"]
    tests = a.results / "confirmation"
    output = {}
    for dataset, source, key, name in (("arc_challenge", a.arc, "rank1_t1_k4", "ARC"),
                                      ("csqa", a.csqa, "frozen_k4", "COMMONSENSEQA"),
                                      ("obqa", a.obqa, "rank1_t1_k4", "OPENBOOKQA")):
        data = load(a.data / f"test_{dataset}.jsonl")
        new = load(tests / f"{dataset}__{chosen}__k4__candidate.jsonl")
        native = load(tests / f"{dataset}__native__k4__candidate.jsonl")
        output[dataset] = candidate_comparison(source, a.old / f"INHERITED_{name}.jsonl", data, new, native, key)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=True)
    output["obqa_six_token_generation"] = prior_generation(
        a.old / "OPTION_GENERATION/openbookqa_test__locked__k4.jsonl", load(a.data / "test_obqa.jsonl"),
        load(tests / f"obqa__{chosen}__k4__letter.jsonl"), tokenizer,
    )
    (a.results / "PAIRED_OLD_DEPLOYMENT.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
