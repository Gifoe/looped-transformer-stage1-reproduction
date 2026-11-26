"""Matched BoolQ + MedMCQA poison control for the V4 T5 development protocol."""
import argparse
from pathlib import Path

import torch

import v4_t5_locked as t5
from v4_common import (
    FastUnifiedDevEvaluator, UNLABELED_AUDIT, evaluate_unlabeled_manifold,
    load_base_runtime, load_jsonl, load_minimal_controller, save_json,
    save_rank1_controller, sha256_file,
)

SOURCES = {"arc_challenge": .25, "csqa": .25, "boolq": .25, "medmcqa": .25}
ARMS = ("raw", "regime3")
T5_LOCK_SHA = "8424d838570e0c40981f4fd701be673e8e2e164895aa9ca6c3e378c0fe59d24f"


def cohort(root, receipt, arm):
    bundle, manifest = [], {}
    for source, weight in SOURCES.items():
        pool = t5.original.resolve_pool_path(source)
        fingerprint = sha256_file(pool)
        if fingerprint != receipt["sources"][source]["pool_sha256"]:
            raise ValueError(f"The historical training pool changed for {source}")
        items = {}
        for raw in load_jsonl(pool):
            item = t5.raw_item(raw, source)
            if 2 <= len(item["options"]) <= 10:
                if item["id"] in items:
                    raise ValueError(f"The source pool has a duplicate ID: {source}")
                items[item["id"]] = item
        regime_hash = None
        if arm == "regime3" and source in ("boolq", "medmcqa"):
            regime = root / f"cfit_14_features/regime3_samples/{source}.jsonl"
            regime_hash = sha256_file(regime)
            if regime_hash != receipt["sources"][source]["regime_sha256"]:
                raise ValueError(f"The archived Regime III measurements changed for {source}")
            eligible = [row for row in load_jsonl(regime)
                        if row["w_iii_mc"] > 0 and row["sample_id"] in items]
            eligible.sort(key=lambda row: (-row["w_iii_mc"], row["sample_id"]))
            selected = [(items[row["sample_id"]], row["w_iii_mc"]) for row in eligible[:512]]
        else:
            selected = [(item, 1.) for item in items.values()]
        if not selected or len({item["id"] for item, _ in selected}) != len(selected):
            raise ValueError(f"Invalid retained cohort in {arm}/{source}")
        bundle.append((source, weight, selected))
        manifest[source] = dict(eligible_n=len(items), selected_n=len(selected),
                                pool_sha256=fingerprint, regime3_sha256=regime_hash, proportion=weight)
    return bundle, manifest


def atomic_protocol(root):
    out = root / "exp_t5_mixing/dual_source_poison"
    out.mkdir(parents=True, exist_ok=True)
    base = root / "handoff_codex/T5_TRAINING_LOCK.json"
    if sha256_file(base) != T5_LOCK_SHA:
        raise ValueError("The registered T5 lock changed")
    receipt = t5.read(base)
    if sha256_file(Path(t5.__file__)) != receipt["runner_sha256"]:
        raise ValueError("The shared training runner differs from its lock")
    lock = out / "PROTOCOL.json"
    prepared = {arm: cohort(root, receipt, arm) for arm in ARMS}
    protocol = dict(
        phase="matched_two_source_poison_development_only", t5_lock_sha256=T5_LOCK_SHA,
        runner_sha256=sha256_file(Path(__file__)), official_test_used=False, development_used_for_selection=False,
        steps=t5.STEPS, effective_batch=t5.RECIPE["effective_batch"], seeds=list(t5.SEEDS),
        fixed_training_stream="medmcqa_matched_poison",
        deconfounding="Positive signed Regime III rows, capped at 512 per injected source and sampled by archived signed weight.",
        arms={arm: info for arm, (_, info) in prepared.items()},
    )
    if lock.is_file():
        if t5.read(lock) != protocol:
            raise ValueError("The locked supplementary comparison differs")
    else:
        save_json(lock, protocol)
    return out, lock, prepared


def run(root, arm):
    out, lock, prepared = atomic_protocol(root)
    fingerprint = sha256_file(lock)
    if arm is None:
        print("V4_DUAL_POISON_PROTOCOL", fingerprint, t5.read(lock)["arms"], flush=True)
        return
    path = out / f"{arm}.json"
    if path.exists() and "mean_protected_macro5" in t5.read(path):
        if t5.read(path)["protocol_sha256"] != fingerprint:
            raise ValueError("The completed supplementary result is bound to another protocol")
        return
    cfg, archive, model, tokenizer = load_base_runtime()
    scorer = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    current = t5.read(path) if path.exists() else {
        "arm": arm, "protocol_sha256": fingerprint, "native": scorer.evaluate(None, include_mmlu_dev158=True), "seeds": {}}
    if current["protocol_sha256"] != fingerprint:
        raise ValueError("The retained supplementary result is bound to another protocol")
    bundle, _ = prepared[arm]
    for seed in t5.SEEDS:
        name = f"{arm}.s{seed}"
        if name in current["seeds"]:
            continue
        weights = out / f"{name}.pt"
        train_file = out / f"{name}.TRAIN.json"
        if weights.exists() and train_file.exists():
            training = t5.read(train_file)
            if sha256_file(weights) != training["sha256"] or training["protocol_sha256"] != fingerprint:
                raise ValueError("The retained supplementary weight differs from its receipt")
            net = load_minimal_controller(weights, device=model.device)
        else:
            net, training = t5.train(
                archive, model, tokenizer, bundle, "T5_Poison_MedOnly_Raw", seed, 0., [],
                out / f"{name}.resume.pt", fingerprint,
            )
            training["sha256"] = save_rank1_controller(weights, net, dict(
                arm=name, rank=1, steps=t5.STEPS, seed=seed, training_lock_sha256=fingerprint))
            training["protocol_sha256"] = fingerprint
            save_json(train_file, training)
        scores = scorer.evaluate(net, include_mmlu_dev158=True)
        unlabeled = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
        current["seeds"][name] = dict(seed=seed, train=training, eval=scores, unlabeled=unlabeled,
                                      protected_macro5=t5.protected(scores))
        save_json(path, current)
    current["mean_protected_macro5"] = sum(r["protected_macro5"] for r in current["seeds"].values()) / len(t5.SEEDS)
    save_json(path, current)
    print("V4_DUAL_POISON_ARM_DONE", arm, current["mean_protected_macro5"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.root, args.arm)

