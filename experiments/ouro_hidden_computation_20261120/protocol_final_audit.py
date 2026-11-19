"""Check the standalone endpoint and measurement records against the grid."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    a = p.parse_args()
    o = a.out
    conf = read(o/"GENERATION_COST_CONFIG.json")
    measured = read(o/"GENERATION_COST_METRICS.json")
    assert set(measured) == set("ABCDEF")
    ids = conf["timed_ids"]
    assert len(ids) == len(set(ids)) == 16 and conf["warmup_id"] not in ids
    agreement_points, generation_records = {}, {}
    for protocol, methods in measured.items():
        expected = {f"native_k{k}" for k in range(1,5)}
        expected |= {f"loopcd_{v}_k{k}" for k in range(2,5) for v in ("fixed","adaptive","hidden")}
        assert set(methods) == expected
        phase_refs = {}
        for phase in ("native","loopcd"):
            data = o/f"{phase}.validation.{protocol}.jsonl"
            phase_refs[phase] = {r["id"]: r for r in map(json.loads,data.read_text().splitlines())}
        maxima, different = {}, {}
        for name, aggregate in methods.items():
            path = o/f"generation.{protocol}.{name}.json"
            assert aggregate["records_sha256"] == sha(path)
            rows = read(path)
            generation_records[(protocol,name)] = rows
            assert [r["id"] for r in rows] == ids
            assert sum(r["exact"]==r["gold"] for r in rows) == aggregate["exact_correct"]
            assert math.isclose(1000*statistics.median(r["candidate_seconds"] for r in rows),
                                aggregate["median_candidate_ms"])
            assert math.isclose(1000*statistics.median(r["prefill_seconds"] for r in rows),
                                aggregate["median_prefill_ms"])
            diffs, disagreed = [], 0
            for rec in rows:
                reference = phase_refs["native" if name.startswith("native") else "loopcd"][rec["id"]]
                before, after = reference["scores"][name],rec["candidate_scores"]
                assert len(before)==len(after)
                assert all(math.isfinite(x) for x in after)
                assert reference["prompt_sha256"] == rec["prompt_sha256"]
                assert reference["gold"] == rec["gold"]
                assert reference["context_tokens"] == rec["context_tokens"]
                assert len(rec["generated_ids"]) <= conf["budget_new_tokens"]
                assert rec["stop"] in ("newline","eos","token_limit")
                assert rec["candidate_seconds"] > 0 and rec["prefill_seconds"] > 0
                assert all(t > 0 for t in rec["decode_seconds"])
                length = reference["characters"] if protocol in "ABCD" else [1]*len(before)
                pred = lambda scores: max(reference["order"],key=lambda i:scores[i]/length[i])
                disagreed += pred(before) != pred(after)
                diffs.extend(abs(x-y) for x,y in zip(before,after))
            maxima[name], different[name] = max(diffs), disagreed
        agreement_points[protocol] = {
            "standalone_vs_batched_max_raw_score_deviation": maxima,
            "standalone_vs_batched_character_normalized_prediction_disagreements": different,
        }
    equal = Counter()
    for method in measured["C"]:
        left, right = (generation_records[(c,method)] for c in "CE")
        for x,y in zip(left,right):
            assert (x["id"],x["prompt_sha256"],x["gold"])==(y["id"],y["prompt_sha256"],y["gold"])
            equal["identical"] += x["generated_ids"]==y["generated_ids"]
            equal["total"] += 1
    checks = {
        "generation_method_protocol_cells": sum(map(len,measured.values())),
        "unique_generation_questions": len(ids),
        "generation_rows": sum(map(len,generation_records.values())),
        "prompt_identical_C_E_generated_token_sequence_agreement": dict(equal),
        "candidate_score_agreement": agreement_points,
        "audit_implementation_sha256": sha(Path(__file__)),
    }
    (o/"PROTOCOL_GENERATION_AUDIT.json").write_text(json.dumps(checks,indent=2)+"\n")
    print("protocol generation audit passed", checks["generation_rows"], dict(equal))


if __name__ == "__main__":
    main()
