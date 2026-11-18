"""Run deterministic Ouro BF16 trajectory and patch experiments on one CUDA device."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from design import TASKS, build

# The experiment is a sibling of the historical native hook, including on copied runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ouro_latent_state_mechanism_v1"))
from core import REVISION, sha256_file, write_json
from native_capture import capture, candidate_logprob

LOOPS, LAYERS, WIDTH = 4, 24, 2048
# The early core.py constant dropped the initial digit. These are the official
# LFS hash and Git blob identities at the pinned revision.
MODEL_SHA256 = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
MODEL_BLOBS = {
    "config.json": "3c30cf9918b3a881cc2d020f490deb01b75556a8",
    "configuration_ouro.py": "ed783160ba1a64292bb0b060bb0e2e920afbe3b2",
    "modeling_ouro.py": "daf1cc7b84d22cf67dd903528240590a407e2d84",
    "merges.txt": "69503b13f727ba3812b6803e97442a6de05ef5eb",
    "vocab.json": "6a22c055228041f2f7b055686644c157cb5d21fc",
    "tokenizer.json": "5046b313b3e16fa14cd3662ef70c105b08c92092",
    "tokenizer_config.json": "b5a0c01a64da3076bcbb3ea0251f57cf0c7af7d4",
    "special_tokens_map.json": "b4ba320a61e7ea7943bd8d0a11666e53820af952",
}
SITES = tuple((k, l) for k in (1, 2) for l in (5, 11, 17))
SEMANTIC_EVENTS = (5, 11, 17, 23, 24)
CONDITIONS = ("semantic_same_answer", "semantic_different_answer", "map_only",
              "self", "irrelevant", "random")
DIRECTIONS = (
    ("base", "same_answer", "first", "second", "irrelevant"),
    ("same_answer", "base", "second", "first", "irrelevant_same"),
)


def dump_line(file, item):
    file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def load_runtime(model_path):
    # An inherited torchvision installation can be built for a different Torch.
    # Ouro uses only the text model implementation.
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers.utils.import_utils._torchvision_available = False
    if sha256_file(model_path / "model.safetensors") != MODEL_SHA256:
        raise RuntimeError("Checkpoint SHA-256 differs from the pinned repository revision")
    for name, expected in MODEL_BLOBS.items():
        data = (model_path / name).read_bytes()
        blob = f"blob {len(data)}\0".encode() + data
        if hashlib.sha1(blob).hexdigest() != expected:
            raise RuntimeError(f"Unexpected official revision file: {name}")
    torch.manual_seed(20261118)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA BF16 required")
    torch.set_grad_enabled(False)
    torch.cuda.init()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    model.config.use_cache = False
    assert len(model.model.layers) == LAYERS and model.config.hidden_size == WIDTH
    assert model.config._attn_implementation == "sdpa"
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    runtime = {
        "model_revision": REVISION, "model_sha256": MODEL_SHA256, "torch": torch.__version__,
        "transformers": transformers.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "cuda_capability": torch.cuda.get_device_capability(),
        "capacity_bytes": torch.cuda.get_device_properties(0).total_memory,
        "dtype": "bfloat16", "attention": "sdpa", "use_cache": False,
        "files_sha256": {p.name: sha256_file(p) for p in model_path.iterdir()
                        if p.name in MODEL_BLOBS or p.name == "model.safetensors"},
    }
    return model, tokenizer, runtime


def encode(tokenizer, world, options, probe=False):
    prompt = world["probe_prompt" if probe else "prompt"]
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    target = []
    for j in range(len(options)):
        appended = tokenizer.encode(prompt + " " + chr(65 + j), add_special_tokens=False)
        if appended[:-1] != ids:
            raise RuntimeError(f"Expected common context plus one candidate token: {prompt[-100:]!r}")
        target.append(appended[-1])
    boundary = world["anchor"]
    anchor = len(ids) // 2 if boundary is None else max(
        i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < boundary and b <= boundary
    )
    assert anchor < len(ids) - 1 and len(set(target)) == len(target)
    return {"ids": ids, "target": target, "anchor": anchor, "read": len(ids) - 1,
            "anchor_token": tokenizer.convert_ids_to_tokens(ids[anchor]),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def aligned_inputs(tokenizer, group):
    encoded = [
        encode(tokenizer, world, group["options"])
        for world in group["worlds"].values()
    ]
    return len({(e["anchor"], e["anchor_token"], len(e["ids"])) for e in encoded}) == 1


@contextmanager
def runtime_hooks(model, enc, sites, edits=None, trace=False):
    """Apply same-layer replacements and collect targeted native downstream witnesses."""
    core = model.model
    old = core.total_ut_steps
    handles, bank, norm, states, branch = [], {}, [], {}, {}
    expected = [(k, l) for k in range(1, LOOPS + 1) for l in range(LAYERS)]
    events = []

    def layer_hook(layer):
        def hook(module, args, kwargs, out):
            event = (kwargs["current_ut"] + 1, layer)
            if event != expected[len(events)]:
                raise RuntimeError(f"Native loop/layer order mismatch at {event}")
            events.append(event)
            if event in (edits or {}):
                pos, value = edits[event]
                out = out.clone()
                out[0, pos] = value
            if event in sites:
                bank[event] = out[0, enc["anchor"]].detach().clone()
            if trace:
                states[event] = out[0, enc["read"]].detach().clone()
            return out
        return hook

    def norm_hook(module, args, out):
        norm.append(out[0, enc["read"]].detach().clone())

    def contribution(layer, kind):
        def hook(module, args, out):
            event = expected[len(events)]
            assert event[1] == layer
            tensor = out[0] if isinstance(out, tuple) else out
            branch[(*event, kind)] = tensor[0, enc["read"]].detach().clone()
        return hook

    try:
        core.total_ut_steps = LOOPS
        for i, layer in enumerate(core.layers):
            handles.append(layer.register_forward_hook(layer_hook(i), with_kwargs=True))
            if trace and i in (5, 11, 17, 23):
                handles.append(layer.self_attn.register_forward_hook(contribution(i, "attention")))
                handles.append(layer.mlp.register_forward_hook(contribution(i, "mlp")))
        handles.append(core.norm.register_forward_hook(norm_hook))
        yield bank, norm, states, branch
        if events != expected or len(norm) != LOOPS:
            raise RuntimeError("Incomplete native run")
    finally:
        for handle in handles:
            handle.remove()
        core.total_ut_steps = old


def depth_scores(model, norms, targets):
    outputs = torch.cat(
        [model.lm_head(state.view(1, 1, WIDTH)).view(1, -1) for state in norms]
    )
    scores = outputs.float().log_softmax(dim=-1)[:, targets]
    assert torch.isfinite(scores).all()
    return scores.tolist(), outputs


def fast_forward(model, enc, sites=frozenset(), edits=None, trace=False):
    ids = torch.tensor([enc["ids"]], device="cuda")
    with runtime_hooks(model, enc, sites, edits, trace) as (bank, norm, states, branch):
        native = model(ids, use_cache=False, exit_at_step=3, logits_to_keep=1).logits
    scores, full = depth_scores(model, norm, enc["target"])
    if not torch.equal(full[-1], native[0, 0]):
        raise RuntimeError("Captured depth-4 logits differ from native endpoint")
    return scores, bank, states, branch, full


def natural_forward(model, enc, gate=False):
    ids = torch.tensor([enc["ids"]], device="cuda")
    positions = [enc["anchor"], enc["read"]]
    with capture(model, positions, LOOPS) as state:
        end = model(ids, use_cache=False, exit_at_step=3, logits_to_keep=1).logits
    norms = [entry[1].cuda() for entry in state["loop_post_norm"]]
    scores, full = depth_scores(model, norms, enc["target"])
    assert torch.equal(full[-1], end[0, 0])
    if gate:
        raw = model(ids, use_cache=False, exit_at_step=3, logits_to_keep=1).logits
        assert torch.equal(raw, end)
        old = model.model.total_ut_steps
        try:
            for k in range(LOOPS):
                model.model.total_ut_steps = k + 1
                independent = model(ids, use_cache=False, exit_at_step=k, logits_to_keep=1).logits
                assert torch.equal(independent[0, 0], full[k])
                official = [candidate_logprob(model, state["loop_post_norm"][k], [target], [1])
                            for target in enc["target"]]
                assert official == scores[k]
        finally:
            model.model.total_ut_steps = old
        self_value = state["layer_states"]["loop1_layer11"][0].cuda()
        other = fast_forward(model, enc, edits={(1, 11): (enc["anchor"], self_value)})
        assert torch.equal(other[4], full)
    layers = state["layer_states"]
    stack = torch.stack([
        torch.stack([*(layers[f"loop{k}_layer{l:02d}"] for l in range(LAYERS)),
                     state["loop_post_norm"][k-1]])
        for k in range(1, LOOPS + 1)
    ])
    stats = stack.float()
    norms = stats.norm(dim=-1)
    cosine = F.cosine_similarity(stats[:-1], stats[1:], dim=-1)
    relative = (stats[1:] - stats[:-1]).norm(dim=-1) / norms[:-1].clamp_min(1e-8)
    return scores, norms.numpy(), cosine.numpy(), relative.numpy(), stack[:, SEMANTIC_EVENTS]


def predictions(scores):
    # -1 records exact top ties. Ties are retained in all cohort denominators.
    return [int(np.argmax(s)) if s.count(max(s)) == 1 else -1 for s in scores]


def run_natural(model, tokenizer, groups, out, pilot=False):
    results = out / "trajectory_results"
    results.mkdir()
    stats, cosines, relatives, semantic, semantic_ids, row_ids = [], [], [], [], [], []
    pair_ids, pair_cosines, pair_relative = [], [], []
    started = time.perf_counter()
    task_seconds = {}
    audited = set()
    with (results / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for group in groups:
            task = group["task"]
            group_started = time.perf_counter()
            paired_vectors = {}
            for name, world in group["worlds"].items():
                enc = encode(tokenizer, world, group["options"])
                gate = pilot and task not in audited
                scores, norm, cos, rel, vectors = natural_forward(model, enc, gate)
                if gate:
                    audited.add(task)
                stats.append(norm)
                cosines.append(cos)
                relatives.append(rel)
                row_ids.append(f"{group['id']}/{name}")
                if task in TASKS:
                    paired_vectors[name] = vectors.float()
                if task in TASKS and name in ("base", "first"):
                    semantic.append(vectors.view(torch.int16).numpy())
                    semantic_ids.append(f"{group['id']}/{name}")
                entry = {"id": group["id"], "task": task, "split": group["split"], "world": name,
                         **enc, "scores": scores, "predictions": predictions(scores),
                         "gold": world["answer"]}
                if task in TASKS and name in ("base", "first"):
                    probe = encode(tokenizer, world, group["middle_options"], probe=True)
                    probe_results = fast_forward(model, probe)[0]
                    entry.update(probe_scores=probe_results, probe_predictions=predictions(probe_results),
                                 probe_gold=world["middle_answer"], probe_encoding=probe)
                dump_line(file, entry)
            if paired_vectors:
                ref = paired_vectors["base"]
                others = torch.stack([paired_vectors[s] for s in
                                      ("first", "same_answer", "second", "irrelevant")])
                pair_cosines.append(F.cosine_similarity(ref.unsqueeze(0), others, dim=-1).numpy())
                pair_relative.append(((others - ref).norm(dim=-1) /
                                      ref.norm(dim=-1).clamp_min(1e-8)).numpy())
                pair_ids.append(group["id"])
            elapsed = time.perf_counter() - group_started
            task_seconds.setdefault(task, []).append(elapsed)
            if len(row_ids) % 25 < len(group["worlds"]):
                print(f"trajectory {group['id']}: {len(row_ids)} worlds; {time.perf_counter() - started:.1f}s", flush=True)
    np.savez_compressed(results / "geometry.npz", ids=np.array(row_ids), rms=np.array(stats) / np.sqrt(WIDTH),
                        cosine=np.array(cosines), relative_l2=np.array(relatives))
    np.savez_compressed(results / "semantic_bf16.npz", ids=np.array(semantic_ids),
                        events=np.array(SEMANTIC_EVENTS), raw_bits=np.array(semantic))
    np.savez_compressed(results / "paired_geometry.npz", ids=np.array(pair_ids),
                        events=np.array(SEMANTIC_EVENTS),
                        contrasts=np.array(["first", "same_answer", "second", "irrelevant"]),
                        cosine=np.array(pair_cosines), relative_l2=np.array(pair_relative))
    return {"natural_worlds": len(row_ids), "audited_tasks": sorted(audited),
            "per_task_seconds": task_seconds}


def norm_delta(a, b):
    return float(torch.linalg.vector_norm(a.float() - b.float()))


def dose_matched(base, direction, dose):
    size = direction.norm().item()
    if size == 0 or dose == 0:
        return base
    return (base.float() + direction * (dose / size)).to(base.dtype)


def trace_deltas(base, edited):
    sites = list(base)
    if not sites:
        return []
    a = torch.stack([base[event] for event in sites]).float()
    b = torch.stack([edited[event] for event in sites]).float()
    norms = (a - b).norm(dim=1).tolist()
    return [[*e, float(d)] for e, d in zip(sites, norms)]


def margin(scores, goal, anti):
    return [row[goal] - row[anti] for row in scores]


def evaluate_causal_group(model, tokenizer, group, sites, confirm):
    worlds = group["worlds"]
    encs = {name: encode(tokenizer, world, group["options"]) for name, world in worlds.items()}
    bank_sites = set(sites)
    if confirm:
        bank_sites.update((k, min(l + 6, 23)) for k, l in sites)
    natural = {name: fast_forward(model, enc, bank_sites, trace=confirm)
               for name, enc in encs.items()}
    for receive, same, diff, map_only, nuisance in DIRECTIONS:
        renc = encs[receive]
        for k, layer in sites:
            site = k, layer
            if any(encs[name]["anchor"] != renc["anchor"] or
                   encs[name]["ids"][encs[name]["anchor"]] != renc["ids"][renc["anchor"]]
                   for name in (same, diff, map_only, nuisance)):
                yield {"id": group["id"], "task": group["task"], "receive": receive,
                       "site": list(site), "status": "unaligned_anchor"}
                continue
            scores, bank, trace, branches, full = natural[receive]
            base = bank[site]
            real = natural[same][1][site]
            # The chosen anchor precedes the second mapping. This checks actual states
            # from complete prompts, including their possibly different suffix lengths.
            # PyTorch kernels can have shape-dependent BF16 rounding, so use the
            # same-first, different-map patch as an empirical control as well.
            same_prefix = all(
                encs[a]["ids"][:renc["anchor"]+1] == encs[b]["ids"][:renc["anchor"]+1]
                for a, b in ((receive, map_only), (same, diff))
            )
            if not same_prefix:
                raise RuntimeError("Compensation changed the pre-map token prefix")
            goal = worlds[diff]["answer"]
            anti = worlds[receive]["answer"]
            dose = norm_delta(real, base)
            seed = int.from_bytes(hashlib.sha256(f"{group['id']}/{receive}/{site}".encode()).digest()[:8], "big")
            rng = torch.Generator(device="cuda").manual_seed(seed)
            rand_vec = torch.randn(base.shape, device=base.device, generator=rng)
            disturbance = natural[nuisance][1][site].float() - base.float()
            variants = {
                "semantic_same_answer": real,
                "semantic_different_answer": natural[diff][1][site],
                "map_only": natural[map_only][1][site],
                "self": base,
                "irrelevant": dose_matched(base, disturbance, dose),
                "random": dose_matched(base, rand_vec, dose),
            }
            conditions = {}
            for name, target in variants.items():
                patched = fast_forward(model, renc, edits={site: (renc["anchor"], target)}, trace=confirm)
                pc, _, read_states, read_branches, pf = patched
                if name == "self" and not torch.equal(pf, full):
                    raise RuntimeError("Self replacement changed full-vocabulary logits")
                if k > 1 and not torch.equal(pf[:k-1], full[:k-1]):
                    raise RuntimeError("A future-loop intervention changed past logits")
                conditions[name] = {
                    "scores": pc, "pred": predictions(pc), "margin": margin(pc, goal, anti),
                    "dose": norm_delta(target, base),
                }
                if confirm and name in ("semantic_same_answer", "irrelevant", "random", "map_only"):
                    conditions[name]["readout_l2"] = trace_deltas(trace, read_states)
                    conditions[name]["branch_l2"] = trace_deltas(branches, read_branches)
            if confirm:
                block_site = k, min(layer + 6, 23)
                read = renc["read"]
                restore_read = (read, trace[block_site])
                blocked = fast_forward(model, renc, edits={site: (renc["anchor"], real), block_site: restore_read})
                block_only = fast_forward(model, renc, edits={block_site: restore_read})
                if not torch.equal(block_only[4], full):
                    raise RuntimeError("Natural readout restoration changed the native endpoint")
                pc = blocked[0]
                conditions["block_readout"] = {"scores": pc, "margin": margin(pc, goal, anti),
                                              "site": block_site, "pred": predictions(pc)}
                conditions["block_only"] = {"scores": block_only[0],
                                            "margin": margin(block_only[0], goal, anti)}
                probe = encode(tokenizer, worlds[receive], group["middle_options"], probe=True)
                probed = fast_forward(model, probe, {site})
                if probe["ids"][:probe["anchor"]+1] != renc["ids"][:renc["anchor"]+1]:
                    raise RuntimeError("Intermediate query altered the first-stage token prefix")
                own_probe = probed[1][site]
                probe_main = fast_forward(model, probe, edits={site: (probe["anchor"], real)})[0]
                probe_control = fast_forward(
                    model, probe, edits={site: (probe["anchor"], dose_matched(own_probe, rand_vec, dose))}
                )[0]
                p_goal = worlds[same]["middle_answer"]
                p_anti = worlds[receive]["middle_answer"]
                conditions["intermediate_query"] = {
                    "target": p_goal, "anti": p_anti, "natural": margin(probed[0], p_goal, p_anti),
                    "semantic": margin(probe_main, p_goal, p_anti),
                    "random": margin(probe_control, p_goal, p_anti),
                    "natural_scores": probed[0], "semantic_scores": probe_main,
                    "random_scores": probe_control,
                }
            yield {
                "id": group["id"], "task": group["task"], "receive": receive,
                "site": list(site), "status": "ok", "counterfactual": goal, "original": anti,
                "same_answer_donor": same, "same_answer_donor_native_predictions": predictions(natural[same][0]),
                "same_answer_donor_native_scores": natural[same][0],
                "anchor": renc["anchor"], "read": renc["read"], "length": len(renc["ids"]),
                "dose": dose, "dose_to_state": dose / base.float().norm().item(),
                "natural": {"scores": scores, "pred": predictions(scores), "margin": margin(scores, goal, anti)},
                "conditions": conditions,
            }


def run_causal(model, tokenizer, groups, out, freeze=None):
    count, successful = 0, 0
    chosen = None if freeze is None else freeze["selected"]
    with (out / "causal_rows.jsonl").open("w", encoding="utf-8") as file:
        for group in groups:
            sites = SITES if chosen is None else (tuple(chosen[group["task"]]),)
            for row in evaluate_causal_group(model, tokenizer, group, sites, chosen is not None):
                dump_line(file, row)
                count += 1
                successful += row["status"] == "ok"
            file.flush()
            print(f"causal {group['id']}: {successful}/{count} sites-direction rows", flush=True)
    return {"causal_rows": count, "valid_causal_rows": successful}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "pilot", "trajectory", "dev", "confirm"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        groups = build(args.data, lambda group: aligned_inputs(tokenizer, group))
        with args.inputs.open("x", encoding="utf-8") as file:
            for row in groups:
                dump_line(file, row)
        return
    with args.inputs.open(encoding="utf-8") as file:
        groups = [json.loads(line) for line in file]
    args.out.mkdir(parents=True)
    freeze = None
    if args.phase == "confirm":
        freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
        if sha256_file(args.inputs) != freeze["inputs_sha256"]:
            raise RuntimeError("Confirmation input manifest differs from development")
        if sha256_file(Path(__file__)) != freeze["runner_sha256"]:
            raise RuntimeError("Confirmation runner differs from development")
    if args.phase == "trajectory":
        if not (args.freeze and args.freeze.is_file()):
            raise RuntimeError("Freeze the development decision before reading trajectory confirmation rows")
        selection = json.loads(args.freeze.read_text(encoding="utf-8"))
        if sha256_file(args.inputs) != selection["inputs_sha256"]:
            raise RuntimeError("Trajectory input manifest differs from the frozen study")
    model, tokenizer, runtime = load_runtime(args.model)
    write_json(args.out / "runtime.json", runtime)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    with torch.inference_mode():
        if args.phase in ("pilot", "trajectory"):
            # The combined trajectory deliberately excludes untouched confirmation
            # groups until the causal site has been frozen from development alone.
            if args.phase == "pilot":
                selected = [g for g in groups if g["split"] == "pilot"]
            else:
                selected = groups
            results = run_natural(model, tokenizer, selected, args.out, args.phase == "pilot")
        else:
            split = "confirmation" if freeze else "development"
            selected = [g for g in groups if g["split"] == split and g["task"] in TASKS]
            results = run_causal(model, tokenizer, selected, args.out, freeze)
    torch.cuda.synchronize()
    write_json(args.out / "completed.json", {
        **results, "phase": args.phase, "work_seconds": time.perf_counter() - begin,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "phased_groups": len(selected), "input_sha256": sha256_file(args.inputs),
        "runner_sha256": sha256_file(Path(__file__)),
    })


if __name__ == "__main__":
    main()


