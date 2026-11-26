"""Development-only capacity comparison after the certified V4 T5 trigger."""
import argparse
from pathlib import Path

import torch

import d45_controls
import v4_t5_locked as t5
from t5_capacity_train import train
from v4_common import (
    FastUnifiedDevEvaluator, UNLABELED_AUDIT, evaluate_unlabeled_manifold,
    load_base_runtime, save_json, sha256_file,
)

FAMILIES = ("gated_rank1", "ortho_rank2")
BACKBONE_ARM = "T5_CFIT_Top6_4096"
CHECKED_FILES = ("T5_CAPACITY_NUCLEAR_WITNESS.pt",
                 "T5_CAPACITY_RANK1_WITNESSES.pt", "T5_CAPACITY_RANK2_WITNESSES.pt")


def initialize(family):
    net = d45_controls.build(family)
    if family == "ortho_rank2":
        with torch.no_grad():
            net.encode.weight.zero_()
    return net


def input_lock(root):
    original = root / "exp_t5_mixing"
    cert_path = original / "CAPACITY_CERTIFICATE.json"
    cert = t5.read(cert_path)
    margin = cert["rank_feasible_lower"]["2"]["objective"] - cert["upper_nuclear"]
    if not (cert["expansion_triggered"] and cert["dual_gap"] <= 1e-5
            and margin > cert["required_rank2_margin"]):
        raise ValueError("The frozen capacity certificate does not permit expansion")
    for name, fingerprint in zip(CHECKED_FILES, (
            cert["nuclear_witness_sha256"],
            cert["rank_feasible_lower"]["1"]["witness_sha256"],
            cert["rank_feasible_lower"]["2"]["witness_sha256"]), strict=True):
        if sha256_file(original / name) != fingerprint:
            raise ValueError(f"The registered capacity witness changed: {name}")
    t5_lock = root / "handoff_codex/T5_TRAINING_LOCK.json"
    t5_data = t5.read(t5_lock)
    if t5_data["runner_sha256"] != sha256_file(Path(t5.__file__)):
        raise ValueError("The matched Rank-1 comparator differs from its recipe")
    out = original / "capacity_expansion"
    out.mkdir(exist_ok=True)
    lock = out / "PROTOCOL.json"
    plan = dict(
        capacity_sha256=sha256_file(cert_path), t5_sha256=sha256_file(t5_lock),
        runner_sha256=sha256_file(Path(__file__)),
        trainer_sha256=sha256_file(Path(__file__).with_name("t5_capacity_train.py")),
        family_module_sha256=sha256_file(Path(d45_controls.__file__)),
        reference=BACKBONE_ARM, families=list(FAMILIES),
        steps=t5.STEPS, seeds=list(t5.SEEDS), official_test_used=False,
        initialization="Matched cold zero write. Rank-2 uses initialized orthogonal writes and zero reads.",
        unchanged="Same source/item streams, loss, budget, relative RMS bound, injection, development cohorts and evaluator.",
    )
    if lock.exists():
        if t5.read(lock) != plan:
            raise ValueError("The locked expansion configuration differs")
    else:
        save_json(lock, plan)
    return out, lock, t5_data


def run(root, family):
    out, lock, recipe = input_lock(root)
    fingerprint = sha256_file(lock)
    bundle, trust = t5.setup_arm(root, recipe, BACKBONE_ARM)
    if trust:
        raise ValueError("The fixed capacity comparator is the pure ranking arm")
    for shape in FAMILIES:
        example = initialize(shape)
        if example.delta(torch.ones((1, d45_controls.WIDTH))).abs().max().item() != 0.:
            raise ValueError("The capacity comparator must start at Native")
    if family is None:
        print("V4_CAPACITY_EXPANSION_LOCK", fingerprint, flush=True)
        return
    path = out / f"{family}.json"
    if path.exists() and "mean_protected_macro5" in t5.read(path):
        if t5.read(path)["protocol_sha256"] != fingerprint:
            raise ValueError("The completed family does not match the frozen protocol")
        return
    cfg, archive, model, tokenizer = load_base_runtime()
    scorer = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    current = t5.read(path) if path.exists() else {
        "family": family, "protocol_sha256": fingerprint, "seeds": {},
        "native": scorer.evaluate(None, include_mmlu_dev158=True)}
    if current["protocol_sha256"] != fingerprint:
        raise ValueError("The retained family does not match the frozen protocol")
    for seed in t5.SEEDS:
        tag = f"{family}.s{seed}"
        if tag in current["seeds"]:
            continue
        weights = out / f"{tag}.pt"
        train_file = out / f"{tag}.TRAIN.json"
        if weights.exists() and train_file.exists():
            detail = t5.read(train_file)
            if sha256_file(weights) != detail["sha256"] or detail["protocol_sha256"] != fingerprint:
                raise ValueError("The retained expansion weights changed")
            net = d45_controls.deployed_controller(weights, model.device)
        else:
            net, detail = train(
                archive, model, tokenizer, bundle, BACKBONE_ARM, seed, 0., [],
                out / f"{tag}.resume.pt", fingerprint, family, initialize,
            )
            partial = weights.with_suffix(".partial")
            torch.save(dict(form=family, state={k: v.detach().cpu() for k, v in net.state_dict().items()},
                            meta=dict(seed=seed, updates=t5.STEPS, protocol=fingerprint)), partial)
            partial.replace(weights)
            detail.update(sha256=sha256_file(weights), protocol_sha256=fingerprint)
            save_json(train_file, detail)
        scores = scorer.evaluate(net, include_mmlu_dev158=True)
        unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
        current["seeds"][tag] = dict(train=detail, eval=scores, unlabeled=unlab,
                                     protected_macro5=t5.protected(scores))
        save_json(path, current)
        print("V4_CAPACITY_FAMILY_SEED", tag, current["seeds"][tag]["protected_macro5"], flush=True)
    current["mean_protected_macro5"] = sum(row["protected_macro5"] for row in current["seeds"].values()) / len(t5.SEEDS)
    save_json(path, current)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.root, args.family)

