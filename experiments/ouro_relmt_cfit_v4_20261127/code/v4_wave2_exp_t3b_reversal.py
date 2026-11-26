"""Wave 2B: Exp T3-B — Within-Source Regime III vs Regime II Causal Reversal.
On sources {medmcqa, qasc, obqa, race} using pre-scored 2,048-sample banks from cfit_14_features/regime3_samples/:
Constructs 5 strictly equal-N (N=512) cohorts per source:
  1. Top_RegimeIII   : highest w_III = (1 - p_cloze) * p_lat_mc   (high readout error + high latent knowledge)
  2. Top_RegimeII    : highest w_II  = (1 - p_cloze) * (1 - p_lat_mc) (HIGHEST error & gradient, LOW latent knowledge!)
  3. Top_RegimeI     : highest p_cloze                            (already mastered)
  4. Top_Unsigned_Grad: highest e_gap = 1 - p_cloze               (classic hard-example / gradient-norm mining)
  5. Random_512      : uniform random N=512
Trains cold-start 4,096-param Rank-1 (560 steps, 2 seeds, exact E0 rank recipe) and evaluates on UNIFIED_EXT_DEV.
"""

import argparse
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
)


def train_cohort(archive, model, tokenizer, items, dataset, cohort, seed, steps=560):
    torch.manual_seed(seed)
    net = minimal.Minimal("linear", 1, 0.45).to(model.device)
    net.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
    rng = random.Random(f"t3b/{dataset}/{cohort}/{seed}")
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
        # Microbatch=1 gradient accumulation gives exact effective_batch=4 without OOM on long RACE passages!
        for it in mb:
            crows = [make_candidate_row(it["id"], it["question"], it["options"], it["gold"], dataset=dataset)]
            with torch.enable_grad():
                opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, net)
                loss_i = F.cross_entropy(opt_logits[0].unsqueeze(0), torch.tensor([it["gold"]], device=opt_logits[0].device)) / len(mb)
                loss_i.backward()
                step_loss += float(loss_i.detach())
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE["clip"])
        assert grad.isfinite().item()
        opt.step()
        losses.append(step_loss)
    return net, {"final_loss_avg70": sum(losses[-70:]) / 70.0, "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--samples-dir", type=Path, required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--n-cohort", type=int, default=512)
    ap.add_argument("--steps", type=int, default=560)
    args = ap.parse_args()

    torch.set_num_threads(4)
    out_dir = args.out_root / "exp_t3b_reversal"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, archive, model, tokenizer = load_base_runtime()
    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    native_ev = evaluator.evaluate(None)
    print(f"[T3B] Native baseline: macro5={native_ev['macro5']*100:.2f}% cloze3={native_ev['cloze3']*100:.2f}%", flush=True)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    seeds = (26112711, 26112712)

    for src in sources:
        res_path = out_dir / f"T3B_{src}.json"
        results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {"source": src, "native": native_ev, "cohorts": {}, "arms": {}}
        p1 = Path("/root/ouro-loop-aware-functional-identification-20261125/data/pools_complete") / f"{src}.jsonl"
        p2 = Path("/root/ouro-v3-controllability-20261126/stage0_env/PROSPECTIVE_TRAIN_DRAFT") / f"{src}.jsonl"
        raw_rows = {r["sample_id"]: r for r in load_jsonl(p1 if p1.exists() else p2)}
        scored = load_jsonl(args.samples_dir / f"{src}.jsonl")
        N = min(args.n_cohort, len(scored) // 2)

        def to_items(sub_recs):
            out = []
            for rec in sub_recs:
                r = raw_rows[rec["sample_id"]]
                q, opts, g = extract_q_opts_gold(r)
                out.append({"id": rec["sample_id"], "question": q, "options": opts, "gold": g})
            return out

        rng_rand = random.Random(f"t3b_rand/{src}")
        rand_sub = list(scored)
        rng_rand.shuffle(rand_sub)
        cohorts = {
            "Top_RegimeIII": sorted(scored, key=lambda r: r["w_iii_mc"], reverse=True)[:N],
            "Top_RegimeII": sorted(scored, key=lambda r: r["w_ii"], reverse=True)[:N],
            "Top_RegimeI": sorted(scored, key=lambda r: r["p_cloze_k4"], reverse=True)[:N],
            "Top_Unsigned_Grad": sorted(scored, key=lambda r: r["e_gap"], reverse=True)[:N],
            "Random_512": rand_sub[:N],
        }
        for cname, sub in cohorts.items():
            results["cohorts"][cname] = {
                "n": len(sub),
                "mean_egap": sum(r["e_gap"] for r in sub) / len(sub),
                "mean_p_lat_mc": sum(r["p_lat_mc"] for r in sub) / len(sub),
                "mean_w_iii": sum(r["w_iii_mc"] for r in sub) / len(sub),
                "mean_w_ii": sum(r["w_ii"] for r in sub) / len(sub),
                "clz_acc": sum(r["clz_correct"] for r in sub) / len(sub),
                "mc_acc": sum(r["mc_correct"] for r in sub) / len(sub),
            }
            print(f"  [Cohort {src}/{cname:17s}] egap={results['cohorts'][cname]['mean_egap']:.3f} p_lat={results['cohorts'][cname]['mean_p_lat_mc']:.3f} w_III={results['cohorts'][cname]['mean_w_iii']:.3f} w_II={results['cohorts'][cname]['mean_w_ii']:.3f}", flush=True)

        for cname, sub in cohorts.items():
            items = to_items(sub)
            for seed in seeds:
                arm_id = f"{src}.{cname}.s{seed}"
                if arm_id in results["arms"]:
                    continue
                ckpt = out_dir / f"{arm_id}.pt"
                if ckpt.exists():
                    net = load_minimal_controller(ckpt, device=model.device)
                    meta = {"reused": True}
                else:
                    net, meta = train_cohort(archive, model, tokenizer, items, src, cname, seed, steps=args.steps)
                    meta["sha256"] = save_rank1_controller(ckpt, net, {"arm": arm_id, "source": src, "cohort": cname, "n": N, "steps": args.steps, "seed": seed})
                ev = evaluator.evaluate(net)
                unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
                d_mac5 = (ev["macro5"] - native_ev["macro5"]) * 100
                d_clz3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
                results["arms"][arm_id] = {
                    "source": src, "cohort": cname, "seed": seed, "train": meta, "eval": ev, "unlabeled": unlab,
                    "delta_macro5_pp": d_mac5, "delta_cloze3_pp": d_clz3,
                }
                print(f"  => {arm_id:34s}: d_mac5={d_mac5:+6.2f} d_clz3={d_clz3:+6.2f} dNLL={unlab['delta_nll']:+.3f} eos={unlab['eos_abs_shift']:.4f}", flush=True)
                save_json(res_path, results)

        # Cohort summary
        summary = {}
        for cname in cohorts:
            sub_arms = [v for v in results["arms"].values() if v["cohort"] == cname]
            summary[cname] = {
                "delta_macro5_pp": sum(x["delta_macro5_pp"] for x in sub_arms) / len(sub_arms),
                "delta_cloze3_pp": sum(x["delta_cloze3_pp"] for x in sub_arms) / len(sub_arms),
                "delta_nll_unlab": sum(x["unlabeled"]["delta_nll"] for x in sub_arms) / len(sub_arms),
            }
        results["summary"] = summary
        save_json(res_path, results)
        print(f"\n=== T3-B {src} SUMMARY ===\n{json.dumps(summary, indent=2)}", flush=True)
    print("EXP_T3B_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
