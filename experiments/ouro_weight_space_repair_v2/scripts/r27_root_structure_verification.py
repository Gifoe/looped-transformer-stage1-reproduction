"""Independent Rust-tokenizer regeneration of the bounded R27 CPU structure."""
from pathlib import Path
import hashlib
import json
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def ids_digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    m = read("configs/r27_CPU_STRUCTURE_MANIFEST.json")
    receipt = read("results/r27_CPU_STRUCTURE_RECEIPT.json")
    assert m["status"] == receipt["status"] == "passed_CPU_structural_admission_only" and not m["errors"] and not receipt["errors"]
    assert sha("configs/r27_CPU_STRUCTURE_MANIFEST.json") == receipt["manifest_sha256"] == "c7e59ff12fc7b70fe74e16b1f49361cd96f390066f168a19f3c9d6cb9cdf942a"
    assert sha("scripts/r27_cpu_structure.py") == receipt["source_sha256"] == m["source_sha256"]
    assert sha("r27_FUNCTIONAL_ADMISSION_PROPOSAL.md") == m["proposal_sha256"] == "5f2b13c148c58ce38a34196a6cccd04e0204ea2056b64bd98401fd36a1791df7"
    assert sha("r27_CPU_STRUCTURE_AUTHORIZATION.md") == m["authorization_sha256"] == "8f0e542b26197b842ca0eeab0d4439eb30134f2daf5ef4f0c07edebc73faad43"
    assert receipt["GPU_worker_seconds"] == 0 and not receipt["model_loaded"] and receipt["retries"] == 0
    tokenizer_path = ROOT / "artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json"
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == m["tokenizer_metadata_sha256"]["tokenizer.json"]
    tok = Tokenizer.from_file(str(tokenizer_path))
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    # A single vectorized integer draw, independently of the generator's string-choice loops.
    digits = np.random.default_rng(2021).integers(0, len(alphabet), size=(2, 32, 6))
    pools = [["zx" + "".join(alphabet[int(x)] for x in word) for word in table] for table in digits]
    assert pools == m["pools"] and len(set(sum(pools, []))) == 64
    key_text = read("configs/r25_MEMORY_MANIFEST.json")["key_text"]
    assert key_text.endswith("Question:")
    prefix = key_text[:-len("Question:")]
    assert hashlib.sha256(prefix.encode()).hexdigest() == m["fixed25_prefix_utf8_sha256"]
    suffix = "For code {nonce}, what is its label? Reply with only A, B, C, or D.\nAnswer:"
    intro = "Use the lookup table below. Each code maps to exactly one label.\n"
    shape_summary = []; checked_prompts = 0
    for table_index, (pool, table) in enumerate(zip(pools, m["tables"])):
        lengths = [len(tok.encode(suffix.format(nonce=k), add_special_tokens=False).ids) for k in pool]
        pair = min((i, j) for i in range(32) for j in range(i+1, 32) if lengths[i] == lengths[j])
        other = [i for i in range(32) if i not in pair][:2]
        order = list(pair) + other if table_index == 0 else other + list(pair)
        query_rows = [0, 1] if table_index == 0 else [2, 3]
        keys = [pool[i] for i in order]
        assert table["pool_suffix_lengths"] == lengths and table["query_pair_indices"] == list(pair)
        assert table["row_pool_indices"] == order and table["row_keys"] == keys and table["query_rows"] == query_rows
        previous_by_version = {}
        for query_row, instance in zip(query_rows, table["instances"]):
            assert instance["query_row"] == query_row and instance["query_key"] == keys[query_row]
            independently_encoded = []
            for version, variant in zip(["A", "B"], instance["variants"]):
                labels = list("ABCD")
                if version == "B":
                    i, j = query_rows; labels[i], labels[j] = labels[j], labels[i]
                before_rows = prefix + "Question: " + intro
                body = ""; fields = []
                for key, label in zip(keys, labels):
                    start = len(before_rows + body + key + " ->")
                    fields.append((start, start+2))
                    body += key + " -> " + label + "\n"
                query_start = len(before_rows + body)
                prompt = before_rows + body + "For code " + keys[query_row] + ", what is its label? Reply with only A, B, C, or D.\nAnswer:"
                assert variant["version"] == version and variant["prompt"] == prompt
                assert variant["prompt_utf8_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
                encoded = tok.encode(prompt, add_special_tokens=False)
                assert variant["input_ids"] == encoded.ids and variant["input_ids_sha256"] == ids_digest(encoded.ids)
                assert variant["input_shape"] == [1, len(encoded.ids)] and variant["query_start"] == query_start
                positions = []
                for i, ((a, b), label, source) in enumerate(zip(fields, labels, variant["sources"])):
                    assert len(prefix) < a < b <= query_start and prompt[a:b] == " " + label
                    overlapping = [j for j, (x, y) in enumerate(encoded.offsets) if x < y and x < b and y > a]
                    assert len(overlapping) == 1
                    position = overlapping[0]; x, y = encoded.offsets[position]
                    assert a <= x < y <= b
                    whole = tok.encode(prompt + " " + label, add_special_tokens=False).ids
                    assert whole[:len(encoded.ids)] == encoded.ids and len(whole) == len(encoded.ids)+1
                    assert whole[-1] == encoded.ids[position]
                    assert source == dict(row=i, label=label, field_start=a, field_end=b, field_text=prompt[a:b], position=position, offset=[x, y], token_id=encoded.ids[position], continuation_ids=[whole[-1]], continuation_prefix_ids_sha256=ids_digest(encoded.ids))
                    positions.append(position)
                assert len(set(positions)) == 4 and variant["source_positions"] == positions
                assert variant["expected_label"] == labels[query_row] and variant["expected_label_metadata_only"] and variant["no_expected_answer_appended"]
                if version in previous_by_version:
                    previous = previous_by_version[version]
                    assert previous["input_shape"] == variant["input_shape"] and previous["source_positions"] == positions
                    assert previous["input_ids"][:positions[-1]+1] == encoded.ids[:positions[-1]+1]
                previous_by_version[version] = variant
                independently_encoded.append(encoded.ids); checked_prompts += 1
            a, b = independently_encoded
            changed = [j for j, (x, y) in enumerate(zip(a, b)) if x != y]
            expected_changed = [instance["variants"][0]["source_positions"][i] for i in query_rows]
            assert len(a) == len(b) and changed == sorted(expected_changed) == instance["changed_positions"]
        shape_summary.append(dict(table=table_index+1, selected_pool_indices=list(pair), row_pool_indices=order, full_input_tokens=len(a)))
    assert checked_prompts == 8
    result = dict(status="passed_independent_CPU_tokenizer_structure_review", prompts=8, generated_nonce_count=64,
                  all_pools_choices_prompts_tokens_offsets_labels_recomputed=True, summary=shape_summary,
                  manifest_sha256=receipt["manifest_sha256"], tokenizer_sha256=m["tokenizer_metadata_sha256"]["tokenizer.json"],
                  GPU_worker_seconds=0, model_loaded=False, functional_evidence=False, model_execution_authorized=False,
                  limitation="Only a tokenizer/metadata feasibility check. No source activations, mapping-following behavior, identity patch, ARC accuracy, or deployment method was tested. Private tokenizer is excluded from publication.")
    with (ROOT / "results/r27_ROOT_STRUCTURE_VERIFICATION.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
