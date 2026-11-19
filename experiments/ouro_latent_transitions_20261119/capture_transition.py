"""Capture consecutive native full-width states and blinded boundary interventions."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ouro_latent_organization_20261118"))
from capture_org import capture, positions, projection
from manifest_org import TASKS, digest, dump_line, get_encoding, load_lines, write_json
from collect import load_runtime
from intervene import source_region

MANIFEST = "4a255369c16f6fdf52d4f2569f47097a10044f4379ae533d8fc21587f96f608d"
WORLDS = ("r", "x", "y", "d")
SWAP = dict(zip(WORLDS, ("x", "r", "d", "y")))
ROLES = ("source", "bridge", "mapping", "query", "readout")
SPLITS = {"fit": 300, "discover": 100, "confirm": 300, "transfer": 300}
CAUSAL_COUNTS = {"confirm": 60, "transfer": 60}


def rank(key):
    return hashlib.sha256(key.encode()).digest()


def cohorts(groups, past, task, pilot):
    if pilot:
        return groups[:2], {g["id"]: j for j, g in enumerate(groups[:2])}
    old = {r["id"] for r in load_lines(past / "causal" / task / "rows.jsonl")}
    if task == "relation":
        old |= {r["id"] for r in load_lines(past / "route_retest" / "rows.jsonl")}
    chosen = {}
    for split, size in CAUSAL_COUNTS.items():
        candidates = sorted((g["id"] for g in groups if g["split"] == split and g["id"] not in old),
                            key=lambda gid: rank("transition-intervention-20261119/" + gid))
        assert len(candidates) >= size
        chosen.update({gid: j for j, gid in enumerate(candidates[:size])})
    return groups, chosen


def trace(model, enc, roles, source, depth, edit=None, full=False, detailed=False):
    select = sorted({i for s in roles.values() for i in s} | set(source))
    edits = {(1, 24): edit} if edit is not None else None
    score, head, bank, _, coords = capture(model, enc, select, edits=edits, depth=depth,
                                          full=full, tap_layers={0, 5, 11, 17, 23} if detailed else ())
    index = {p: p if full else coords.index(p) for p in select}
    pooled = torch.stack([
        torch.stack([bank[k, 24][[index[i] for i in v]].float().mean(0) for v in roles.values()])
        for k in range(1, depth + 1)
    ])
    details = None
    if detailed:
        details = {str(k): torch.stack([
            bank[k][[index[i] for i in v]].float().mean(0) for v in roles.values()
        ]).cpu().numpy() for k in bank}
    vectors = bank[1, 24] if full else bank[1, 24][[index[i] for i in source]]
    return pooled, score, head, vectors, details


def scored_example(g, world):
    return {"id": g["id"], "world": world, "split": g["split"], "task": g["task"],
            "family": g["family"], "mode": g["mode"], "gold": g["worlds"][world]["gold"]}


def run(past, output, model_path, task, pilot):
    if digest(past / "groups.jsonl") != MANIFEST:
        raise RuntimeError("Expected the slot-corrected canonical manifest")
    saved = load_lines(past / "natural" / task / "rows.jsonl")
    native = {(r["id"], r["world"]): r for r in saved}
    groups = [g for g in load_lines(past / "groups.jsonl") if g["task"] == task]
    assert Counter(g["split"] for g in groups) == SPLITS
    groups, cohort = cohorts(groups, past, task, pilot)
    out = output / (("pilot-" if pilot else "") + task)
    out.mkdir(parents=True)
    model, tok, runtime = load_runtime(model_path)
    write_json(out / "runtime.json", runtime)
    arrays = np.lib.format.open_memmap(out / "natural_full.npy", mode="w+", dtype=np.float32,
                                      shape=(len(groups) * 4, 2, len(ROLES), 2048))
    counter = np.lib.format.open_memmap(out / "causal_full.npy", mode="w+", dtype=np.float32,
                                       shape=(4 * len(cohort) + 2 * min(8, len(cohort)),
                                              2, 2, len(ROLES), 2048))
    old = np.load(past / "natural" / task / "vectors.npy", mmap_mode="r")
    frame = torch.from_numpy(projection()).cuda()
    started = time.perf_counter()
    max_reference = 0.
    min_reference_cosine = 1.
    max_score_gap = 0.
    checks = Counter()
    i = j = 0
    with (torch.inference_mode(),
          (out / "natural.jsonl").open("w", encoding="utf-8") as nfile,
          (out / "causal.jsonl").open("w", encoding="utf-8") as cfile):
        for g in groups:
            key = g["id"]
            in_cohort = key in cohort
            detail = in_cohort and (pilot or (cohort[key] == 0))
            full_check = pilot or (g is groups[0])
            depth = 4 if detail or full_check else 2
            enc = {w: get_encoding(tok, g, g["worlds"][w]) for w in WORLDS}
            roles = {w: positions(g, e) for w, e in enc.items()}
            src = source_region(g, enc["r"])
            assert all(e["roles"] == enc["r"]["roles"] for e in enc.values())
            assert not set(src) & set(roles["r"]["mapping"] + roles["r"]["readout"])
            base = {}
            for w in WORLDS:
                reference = native[key, w]
                assert reference["ids"] == enc[w]["ids"]
                state, scores, heads, vectors, event_trace = trace(
                    model, enc[w], roles[w], src, depth, full=full_check, detailed=detail)
                base[w] = state, scores, heads, vectors, event_trace
                arrays[i] = state[:2].cpu().numpy()
                if g["split"] == "fit" and i < 40:
                    old_state = old[reference["i"], :2, 24].astype(np.float32)
                    current = (state[:2] @ frame).cpu().numpy()
                    rel = float(np.linalg.norm(old_state-current) / np.linalg.norm(old_state))
                    cosine = float(np.vdot(old_state, current) /
                                   (np.linalg.norm(old_state)*np.linalg.norm(current)))
                    max_reference = max(max_reference, rel)
                    min_reference_cosine = min(min_reference_cosine, cosine)
                    max_score_gap = max(max_score_gap, float(np.max(np.abs(
                        np.array(reference["scores"][:depth]) - np.array(scores)))))
                    # Historical vectors are float16 aggregates from a different GPU.
                    assert np.isfinite(rel) and cosine > .98
                    checks["projected_historical_parity"] += 1
                dump_line(nfile, {**scored_example(g, w), "i": i, "legacy_i": reference["i"],
                                  "scores": scores[:2], "prior_scores_K1_to_K4": reference["scores"]})
                i += 1
            if full_check:
                cur, donor = "r", "x"
                all_pos = list(range(len(enc[cur]["ids"])))
                for w in (cur, donor):
                    independent = trace(model, enc[w], roles[w], src, 1)
                    assert torch.equal(independent[2][0], base[w][2][0])
                    two_loops = trace(model, enc[w], roles[w], src, 2)
                    assert torch.equal(two_loops[2], base[w][2][:2])
                    assert torch.equal(two_loops[0], base[w][0][:2])
                for w in (cur, donor):
                    rerun = trace(model, enc[cur], roles[cur], src, depth, (all_pos, base[w][3]))
                    assert torch.equal(rerun[2], base[w][2])
                    assert torch.equal(rerun[0], base[w][0])
                checks["exact_self_and_full_boundary_swap"] += 1
            if not in_cohort:
                if i % 400 == 0:
                    print(task, i, "natural forwards", round(time.perf_counter() - started, 1), flush=True)
                continue
            for w in WORLDS:
                donor = SWAP[w]
                assert g["worlds"][w]["semantic"]["map_factor"] == g["worlds"][donor]["semantic"]["map_factor"]
                assert g["worlds"][w]["semantic"]["intermediate"] != g["worlds"][donor]["semantic"]["intermediate"]
                original = base[w][3][src] if full_check else base[w][3]
                borrowed = base[donor][3][src] if full_check else base[donor][3]
                dose = (borrowed.float() - original.float()).norm()
                arms = [("semantic", borrowed)]
                if g["split"] in ("confirm", "fit") and cohort[key] < 8 and w in ("r", "x"):
                    seed = int.from_bytes(rank(f"causal-random/{key}/{w}")[:8], "big")
                    z = torch.randn(original.shape, device="cuda",
                                    generator=torch.Generator(device="cuda").manual_seed(seed))
                    arms.append(("random", (original.float() + z * (dose/z.norm())).to(original.dtype)))
                for kind, patch in arms:
                    measured, scores, _, _, details = trace(model, enc[w], roles[w], src, depth,
                                                            (src, patch), detailed=detail)
                    clean = base[w][0]
                    assert torch.equal(measured[0, -1], clean[0, -1])
                    assert scores[0] == base[w][1][0]
                    assert not torch.equal(measured[1], clean[1])
                    delta = patch.float() - original.float()
                    counter[j, 0] = clean[:2].cpu().numpy()
                    counter[j, 1] = measured[:2].cpu().numpy()
                    row = {**scored_example(g, w), "i": j, "donor": donor, "arm": kind,
                           "donor_gold": g["worlds"][donor]["gold"], "position": src,
                           "dose_l2": float(delta.norm()), "original_scores": base[w][1],
                           "edited_scores": scores, "donor_scores": base[donor][1]}
                    if detail and kind == "semantic":
                        assert details.keys() == base[w][4].keys()
                        d = {str(k): np.linalg.norm(details[k]-base[w][4][k], axis=-1).tolist()
                             for k in details}
                        row["role_deltas_by_event"] = d
                    dump_line(cfile, row)
                    j += 1
            if i % 400 == 0:
                print(task, i, "natural forwards", round(time.perf_counter() - started, 1), flush=True)
    arrays.flush()
    counter.flush()
    assert j == len(counter) and i == len(arrays)
    write_json(out / "capture.json", {"natural_worlds": i, "causal_worlds": j, "groups": len(groups),
               "cohort": cohort, "input_sha256": MANIFEST, "capture_sha256": digest(Path(__file__)),
               "checks": dict(checks), "max_historical_projected_relative_L2": max_reference,
               "min_historical_projected_cosine": min_reference_cosine,
               "max_same_candidate_logprob_gap": max_score_gap,
               "seconds": time.perf_counter()-started,
               "natural_sha256": digest(out/"natural_full.npy"), "causal_sha256": digest(out/"causal_full.npy")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("past", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("task", choices=TASKS)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    run(args.past, args.output, args.model, args.task, args.pilot)
