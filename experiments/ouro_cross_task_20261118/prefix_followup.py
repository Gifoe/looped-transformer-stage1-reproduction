"""Test source-region transport after the single-anchor confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from analyze import clustered, digest, interval, load_lines, write_csv
from design import aligned_group, fixed_sample, PROOF_SOURCE, proof_data, TASKS
from run import (aligned_inputs, dose_matched, dump_line, encode, fast_forward,
                 load_runtime, margin, norm_delta, predictions, write_json, DIRECTIONS)

SELECTION = {"multihop": 5, "prontoqa_derived": 11, "arithmetic": 5}
CONTRASTS = {
    "full_minus_natural": ("same_answer_full", "natural"),
    "full_minus_irrelevant": ("same_answer_full", "irrelevant_matched"),
    "full_minus_random": ("same_answer_full", "random_matched"),
    "map_only_minus_natural": ("map_only", "natural"),
    "changed_tokens_minus_natural": ("changed_tokens", "natural"),
    "unchanged_tokens_minus_natural": ("unchanged_tokens", "natural"),
}
CONFIDENCE = 1 - 0.05 / 3


def prepare(study, source, tok, path):
    existing = load_lines(study / "semantic_interventions.jsonl")
    used = {g["source"] for g in existing if g["task"] == "prontoqa_derived"}
    fresh = fixed_sample([entry for entry in proof_data(source, PROOF_SOURCE) if entry[0] not in used],
                         "prontoqa_prefix_followup", 30)
    assert len(fresh) == 30
    with path.open("x", encoding="utf-8") as file:
        for task in TASKS:
            for i in range(30):
                variant = fresh[i] if task == "prontoqa_derived" else None
                g = aligned_group(task, i+100, variant, lambda row: aligned_inputs(tok, row))
                g["split"] = "fresh_prefix_followup"
                dump_line(file, g)


def patch_rows(model, tok, group):
    worlds, options = group["worlds"], group["options"]
    encs = {name: encode(tok, world, options) for name, world in worlds.items()}
    sizes = {e["anchor"] + 1 for e in encs.values()}
    assert len(sizes) == 1
    positions = list(range(sizes.pop()))
    lifted = {name: {**e, "anchor": positions} for name, e in encs.items()}
    layer = SELECTION[group["task"]]
    sites = {(1, layer), (2, layer)}
    baseline = {name: fast_forward(model, lifted[name], sites) for name in lifted}
    for rx, same, alt, map_only, noise in DIRECTIONS:
        enc = lifted[rx]
        goal, original = worlds[alt]["answer"], worlds[rx]["answer"]
        source_ids = enc["ids"][:len(positions)]
        same_ids = lifted[same]["ids"][:len(positions)]
        assert goal != original == worlds[same]["answer"]
        assert lifted[map_only]["ids"][:len(positions)] == source_ids
        assert lifted[alt]["ids"][:len(positions)] == same_ids
        changed = torch.tensor([a != b for a, b in zip(source_ids, same_ids)], device="cuda")
        assert bool(changed.any()) and not changed[-1]
        scores, bank, _, _, native_full = baseline[rx]
        for site in sorted(sites):
            base = bank[site]
            donor = baseline[same][1][site]
            diff = donor.float() - base.float()
            dose = diff.norm().item()
            assert torch.equal(baseline[map_only][1][site], base)
            nuisance = baseline[noise][1][site].float() - base.float()
            assert nuisance.norm().item() > 0
            key = f"prefix/{group['id']}/{rx}/{site}"
            rng = torch.Generator(device="cuda").manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
            rnd = torch.randn(base.shape, device=base.device, generator=rng)
            rnd *= diff.norm(dim=-1, keepdim=True) / rnd.norm(dim=-1, keepdim=True)
            variants = {
                "same_answer_full": donor,
                "irrelevant_matched": dose_matched(base, nuisance, dose),
                "random_matched": (base.float() + rnd).to(base.dtype),
                "map_only": baseline[map_only][1][site],
                "self": base,
                "changed_tokens": torch.where(changed[:, None], donor, base),
                "unchanged_tokens": torch.where(changed[:, None], base, donor),
            }
            conditions = {}
            for name, value in variants.items():
                s, _, _, _, full = fast_forward(model, enc, edits={site: (positions, value)})
                if name in ("self", "map_only"):
                    assert torch.equal(full, native_full)
                if site[0] == 2:
                    assert torch.equal(full[0], native_full[0])
                conditions[name] = {"scores": s, "margin": margin(s, goal, original),
                                    "pred": predictions(s), "dose": norm_delta(value, base)}
            yield {"id": group["id"], "task": group["task"], "receive": rx,
                   "site": site, "first_stage_tokens": len(positions),
                   "changed_positions": [i for i, (a, b) in enumerate(zip(source_ids,same_ids)) if a != b],
                   "counterfactual": goal, "original": original, "same_answer_donor": same,
                   "same_answer_donor_native_pred": predictions(baseline[same][0]),
                   "natural": {"scores": scores, "margin": margin(scores, goal, original),
                               "pred": predictions(scores)}, "conditions": conditions}


def summarize(out):
    rows = load_lines(out / "rows.jsonl")
    assert len(rows) == 360
    effects, cross = [], []
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        all_comparisons = {}
        for k in (1, 2):
            subset = [r for r in task_rows if r["site"][0] == k]
            for depth in range(4):
                for name, arms in CONTRASTS.items():
                    if k == 2 and depth == 3 and name.startswith("full_minus"):
                        confidence = CONFIDENCE
                    else:
                        confidence = .95
                    val, lo, hi = interval(clustered(subset, *arms, depth), confidence)
                    record = {"task": task, "patch_K": k, "layer_zero_index": SELECTION[task],
                              "readout_K": depth+1, "comparison": name, "groups": 30,
                              "effect_nats": val, "lower": lo, "upper": hi, "interval_level": confidence}
                    effects.append(record)
                    all_comparisons[(k, depth+1, name)] = record
        k1, k2 = ([r for r in task_rows if r["site"][0] == k] for k in (1, 2))
        main = [all_comparisons[(2,4,name)] for name in CONTRASTS if name.startswith("full_minus")]
        # k1 and k2 share the exact group order.
        k2_k1 = clustered(k2, *CONTRASTS["full_minus_natural"], 3) - clustered(k1, *CONTRASTS["full_minus_natural"], 3)
        depth_diff, depth_lo, depth_hi = interval(k2_k1, .95)
        original = [r for r in k2 if r["natural"]["pred"][3] == r["original"] ==
                    r["same_answer_donor_native_pred"][3]]
        flips = [r for r in original if r["conditions"]["same_answer_full"]["pred"][3] == r["counterfactual"]]
        errors = [abs(r["conditions"][c]["dose"] - r["conditions"]["same_answer_full"]["dose"]) /
                  r["conditions"]["same_answer_full"]["dose"]
                  for r in k2 for c in ("random_matched", "irrelevant_matched")]
        cross.append({"task": task, "groups": 30, "prev_frozen_layer_zero_index": SELECTION[task],
                      "K2_full_vs_native_nats": main[0]["effect_nats"],
                      "K2_full_vs_nuisance_nats": main[1]["effect_nats"],
                      "K2_full_vs_random_nats": main[2]["effect_nats"],
                      "K2_passes_three_primary_intervals": all(r["lower"] > 0 for r in main),
                      "K2_changed_only_vs_native": all_comparisons[(2,4,"changed_tokens_minus_natural")]["effect_nats"],
                      "K2_unchanged_only_vs_native": all_comparisons[(2,4,"unchanged_tokens_minus_natural")]["effect_nats"],
                      "K2_minus_K1_full_patch_nats": depth_diff, "K2_minus_K1_lower95": depth_lo,
                      "K2_minus_K1_upper95": depth_hi,
                      "correct_native_directions": len(original), "correct_native_adopting_counterfactual": len(flips),
                      "max_dose_relative_error": max(errors)})
    write_csv(out / "causal_effects.csv", effects)
    write_csv(out / "cross_task_mechanisms.csv", cross)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("study", type=Path)
    cli.add_argument("--model", type=Path)
    cli.add_argument("--prontoqa-zip", type=Path)
    cli.add_argument("--analyze-only", action="store_true")
    args = cli.parse_args()
    out = args.study / "prefix_followup"
    if args.analyze_only:
        summarize(out)
        return
    freeze = json.loads((args.study / "site_freeze.json").read_text())
    assert freeze["selected"] == {name: [2,value] for name,value in SELECTION.items()}
    out.mkdir()
    manifest = out / "semantic_interventions.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    prepare(args.study, args.prontoqa_zip, tokenizer, manifest)
    groups = load_lines(manifest)
    model, tokenizer, runtime = load_runtime(args.model)
    write_json(out / "runtime.json", runtime)
    clock = time.perf_counter()
    count = 0
    with torch.inference_mode(), (out / "rows.jsonl").open("w", encoding="utf-8") as file:
        for group in groups:
            for r in patch_rows(model, tokenizer, group):
                dump_line(file, r)
                count += 1
            if count % 40 == 0:
                file.flush()
                print(f"Checked {count} patch rows", flush=True)
    torch.cuda.synchronize()
    write_json(out / "completed.json", {"rows": count, "groups": len(groups),
               "gpu_work_seconds": time.perf_counter()-clock, "inputs_sha256": digest(manifest),
               "script_sha256": digest(Path(__file__)),
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()})
    summarize(out)


if __name__ == "__main__":
    main()

