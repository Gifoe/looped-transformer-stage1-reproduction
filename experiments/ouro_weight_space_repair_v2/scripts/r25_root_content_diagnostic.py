"""Exploratory CPU-only postrun target-support diagnostic, never an estimator."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main():
    verification = read("results/r25_ROOT_VERIFICATION.json")
    assert verification["status"] == "passed_independent_postrun_review"
    tokenizer_path = ROOT / "models/Ouro-1.4B/tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = ROOT / "artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json"
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    assert tokenizer_hash == "fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d"
    tok = Tokenizer.from_file(str(tokenizer_path))
    memory = read("configs/r25_MEMORY_MANIFEST.json")
    values = np.asarray(memory["values"], dtype=int)
    manifest = {r["id"]: r for r in read("configs/r24_INPUT_MANIFEST.json")["rows"]}
    gold = {r["id"]: r["gold"] for r in read("results/r25_ANALYSIS.json")["question_results"]}
    rows = [json.loads(s) for s in (ROOT / "results/r25_pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    candidates = []; position_records = []; direction_errors = []
    for row in rows:
        md = manifest[row["id"]]
        for j, c in enumerate(row["candidates"]):
            context, continuation = md["request_pairs"][j]
            prefix = tok.encode(context, add_special_tokens=False).ids
            whole = tok.encode(context + continuation, add_special_tokens=False).ids
            assert whole[:len(prefix)] == prefix
            target = whole[len(prefix):]
            assert digest(target) == c["target_ids_sha256"] and len(target) == c["tokens"]
            retrieved_values = values[np.asarray([r["neighbor_occurrences"] for r in c["retrieval"]], dtype=int)]
            direct = []; ideal_direction = []
            for t, token in enumerate(target):
                weighted = {}
                for value, weight in zip(retrieved_values[:, t], [3, -1, -1, -1]):
                    weighted[int(value)] = weighted.get(int(value), 0) + weight
                sparse = {v: n/3 for v, n in weighted.items() if n}
                a = sparse.get(token, 0.0)
                direct.append(a)
                log_z = math.log(memory["vocabulary_size"] - len(sparse) + sum(math.exp(x) for x in sparse.values()))
                ideal_direction.append(a-log_z)
                position_records.append(dict(gold_candidate=j == gold[row["id"]], target_supported=token in sparse,
                                             positive=a > 0, negative=a < 0,
                                             query_value_hits_target=int(retrieved_values[0, t]) == token,
                                             reference_value_hits_target=int((retrieved_values[1:, t] == token).sum()),
                                             zero_direction=not sparse, direct_logit=a))
            ideal_sum = math.fsum(ideal_direction)
            direction_errors.append(abs(ideal_sum-c["sums"]["memory_direction_only"]))
            observed_delta = c["sums"]["primary"] - c["sums"]["M3"]
            direct_sum = math.fsum(direct)
            candidates.append(dict(id=row["id"], candidate=j, gold_candidate=j == gold[row["id"]],
                                   tokens=len(target), characters=c["characters"], supported_positions=sum(x != 0 for x in direct),
                                   direct_ideal_logit_sum=direct_sum,
                                   observed_primary_minus_M3_LL=observed_delta,
                                   partition_and_FP32_residual=direct_sum-observed_delta,
                                   normalized_direct_ideal_logit_sum=direct_sum/c["characters"],
                                   normalized_observed_primary_minus_M3=observed_delta/c["characters"]))
    summaries = {}
    for group in ["all", "gold", "incorrect"]:
        pos = [p for p in position_records if group == "all" or p["gold_candidate"] == (group == "gold")]
        sub = [c for c in candidates if group == "all" or c["gold_candidate"] == (group == "gold")]
        summaries[group] = dict(candidates=len(sub), positions=len(pos),
                               target_supported_positions=sum(p["target_supported"] for p in pos),
                               positive_target_positions=sum(p["positive"] for p in pos),
                               negative_target_positions=sum(p["negative"] for p in pos),
                               query_value_target_hits=sum(p["query_value_hits_target"] for p in pos),
                               reference_value_target_hits=sum(p["reference_value_hits_target"] for p in pos),
                               zero_direction_positions=sum(p["zero_direction"] for p in pos),
                               candidates_without_direct_target_change=sum(c["supported_positions"] == 0 for c in sub),
                               direct_ideal_logit_sum=math.fsum(p["direct_logit"] for p in pos),
                               mean_normalized_observed_primary_minus_M3=float(np.mean([c["normalized_observed_primary_minus_M3"] for c in sub])))
    out = dict(status="complete_exploratory_CPU_content_diagnostic", n=24, GPU_worker_seconds=0,
               tokenizer_sha256=tokenizer_hash, all_96_targets_retokenized_and_hashed=True,
               summary=summaries, candidates=candidates,
               memory_direction_only_ideal_double_vs_worker_FP32_max_candidate_sum_error=max(direction_errors),
               limitation="Postrun reused-development description only. Gold and actual next tokens are used retrospectively for this diagnostic, never for inference or selecting a method. Direct logit sums reconstruct ideal c; partition residual includes the actual FP32 cast/logsoftmax/sum error and is not an exact isolated partition effect. No semantic efficacy claim, no hyperparameter choice, no GPU job.")
    path = ROOT / "results/r25_ROOT_CONTENT_DIAGNOSTIC.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps(dict(status=out["status"], summary=summaries, direction_sum_max_error=max(direction_errors))))


if __name__ == "__main__":
    main()
