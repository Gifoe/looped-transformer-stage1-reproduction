"""Wave 2C: Standardized 840-step Cold-Start Rank-1 Training on Unseen Prospective Pools.
STRICT CRYPTOGRAPHIC LOCK ENFORCEMENT:
Refuses to run unless `FORECAST_LOCK_V4_PROSPECTIVE.json` and `.sha256` already exist and match!
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import minimal
from v4_common import (
    DEPTHS,
    RECIPE,
    UNLABELED_AUDIT,
    FastUnifiedDevEvaluator,
    cloze_option_logits_differentiable,
    evaluate_unlabeled_manifold,
    extract_q_opts_gold,
    load_base_runtime,
    load_jsonl,
    load_minimal_controller,
    make_candidate_row,
    save_json,
    save_rank1_controller,
    sha256_file,
)


def train_standardized_cold(archive, model, tokenizer, items, dataset, seed, steps=840):
    torch.manual_seed(seed)
    net = minimal.Minimal("linear", 1, 0.45).to(model.device)
    net.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
    rng = random.Random(f"unseen_cold/{dataset}/{seed}")
    order = list(range(len(items)))
    rng.shuffle(order)
    ptr = 0
    t0 = time.perf_counter()
    losses = []
    for step in range(1, steps + 1):
        depth = DEPTHS[(step - 1) % len(DEPTHS)]
        mb = []
        for _ in range(RECIPE["effective_batch"]):
            if ptr >= len(order):
                rng.shuffle(order)
                ptr = 0
            mb.append(items[order[ptr]])
            ptr += 1
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        # Adaptive microbatching: exact effective_batch=4 at maximum GPU util without OOM on long texts!
        max_len = max(len(it["question"]) + max(len(o) for o in it["options"]) for it in mb)
        max_opts = max(len(it["options"]) for it in mb)
        sub_bs = 1 if (max_len > 1400 or max_len * max_opts > 5200) else (2 if (max_len > 650 or max_len * max_opts > 2600) else len(mb))
        for j in range(0, len(mb), sub_bs):
            sub = mb[j:j + sub_bs]
            crows = [make_candidate_row(it["id"], it["question"], it["options"], it["gold"], dataset=dataset) for it in sub]
            with torch.enable_grad():
                opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, net)
                loss_sub = sum(F.cross_entropy(lg.unsqueeze(0), torch.tensor([it["gold"]], device=lg.device))
                               for lg, it in zip(opt_logits, sub)) / len(mb)
                loss_sub.backward()
                step_loss += float(loss_sub.detach())
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE["clip"])
        assert grad.isfinite().item()
        opt.step()
        losses.append(step_loss)
        if step % 280 == 0 or step == steps:
            print(f"    [{dataset} s={seed}] step={step}/{steps} loss={sum(losses[-70:])/min(len(losses),70):.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return net, {"final_loss_avg70": sum(losses[-70:]) / 70.0, "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--pools-dir", type=Path, required=True)
    ap.add_argument("--lock-file", type=Path, required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--steps", type=int, default=840)
    args = ap.parse_args()

    # Enforce pre-registered cryptographic lock!
    sha_file = args.lock_file.with_suffix(".sha256")
    assert args.lock_file.exists() and sha_file.exists(), "REFUSING TO TRAIN: FORECAST_LOCK_V4_PROSPECTIVE not frozen!"
    expected_sha = sha_file.read_text(encoding="utf-8").strip().split()[0]
    actual_sha = sha256_file(args.lock_file)
    assert actual_sha == expected_sha, f"Lock SHA mismatch: {actual_sha} != {expected_sha}"
    print(f"VERIFIED CRYPTOGRAPHIC FORECAST LOCK: {actual_sha}", flush=True)

    torch.set_num_threads(4)
    out_dir = args.out_root / "prospective_unseen" / "trained"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, archive, model, tokenizer = load_base_runtime()
    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    native_ev = evaluator.evaluate(None)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    seeds = (26112721, 26112722)

    for src in sources:
        res_path = out_dir / f"TRAIN_{src}.json"
        results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {"source": src, "lock_sha256": actual_sha, "native": native_ev, "arms": {}}
        raw = load_jsonl(args.pools_dir / f"{src}.jsonl")
        items = []
        for r in raw:
            q, opts, g = extract_q_opts_gold(r)
            items.append({"id": r["sample_id"], "question": q, "options": opts, "gold": g})
        for seed in seeds:
            arm_id = f"{src}.cold840.s{seed}"
            if arm_id in results["arms"]:
                continue
            ckpt = out_dir / f"{arm_id}.pt"
            if ckpt.exists():
                net = load_minimal_controller(ckpt, device=model.device)
                meta = {"reused": True}
            else:
                net, meta = train_standardized_cold(archive, model, tokenizer, items, src, seed, steps=args.steps)
                meta["sha256"] = save_rank1_controller(ckpt, net, {"arm": arm_id, "source": src, "seed": seed, "steps": args.steps, "lock_sha256": actual_sha})
            ev = evaluator.evaluate(net)
            unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
            d_mac5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_clz3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
            results["arms"][arm_id] = {
                "source": src, "seed": seed, "train": meta, "eval": ev, "unlabeled": unlab,
                "delta_macro5_pp": d_mac5, "delta_cloze3_pp": d_clz3,
            }
            print(f"  => [UNSEEN TRAIN] {arm_id:28s}: d_mac5={d_mac5:+6.2f} d_clz3={d_clz3:+6.2f} dNLL={unlab['delta_nll']:+.3f} eos={unlab['eos_abs_shift']:.4f}", flush=True)
            save_json(res_path, results)
        sub = list(results["arms"].values())
        results["mean_delta_macro5_pp"] = sum(x["delta_macro5_pp"] for x in sub) / len(sub)
        results["mean_delta_cloze3_pp"] = sum(x["delta_cloze3_pp"] for x in sub) / len(sub)
        save_json(res_path, results)
        print(f"  ===> {src:14s} MEAN: d_mac5={results['mean_delta_macro5_pp']:+6.2f} d_clz3={results['mean_delta_cloze3_pp']:+6.2f}", flush=True)
    print("UNSEEN_TRAIN_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
