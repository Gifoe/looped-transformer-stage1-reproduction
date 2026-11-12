"""Full-dataset native K1..K4 joint-option evaluation and latent-state capture.

Inference never opens labels. All outputs are streaming, resumable and
content-addressed; targets are only letter continuations, not gold labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from core import (DEFAULT_DEMOS, DEFAULT_MODEL_DIR, LOOPS, REVISION, MODEL_SHA256,
                  demo_prefix, make_requests, encode_pair, load_ouro,
                  sha256_file, sha256_text, read_jsonl, write_json, target_log_likelihoods)
from latent_hooks import Patch, LayerRecorder, capture_positions


def pack_state(file_path: Path, array: Optional[np.ndarray], final: np.ndarray,
               positions: list[int], full_head: Optional[np.ndarray]):
    payload = {"normalized_bf16_bits": final, "positions": np.asarray(positions, dtype=np.int32)}
    if array is not None:
        payload["raw_layer_bf16_bits"] = array
    if full_head is not None:
        payload["last_prediction_full_vocab_head_fp32"] = full_head.astype(np.float32)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = file_path.with_name(file_path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(tmp, file_path)


def run_single(model, tok, input_ids: list[int], pred_positions: list[int], demo: str,
               capture_mode: str, state_path: Path, save_head: bool, patch: Optional[Patch]):
    import torch
    selected = capture_positions(capture_mode, input_ids, pred_positions, tok, demo)
    if patch is not None and not selected:
        raise RuntimeError("Patch requires selected positions")
    x = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode(), LayerRecorder(model, selected, capture_mode != "none", patch) as rec:
        outputs, states, gates = model.model(input_ids=x, use_cache=False)
    if len(states) != LOOPS:
        raise RuntimeError("Expected exactly four native normalized recurrent states")
    final = torch.stack([s[0, selected].detach().cpu().contiguous() for s in states], dim=0)
    final_bits = final.view(torch.int16).numpy().view(np.uint16).copy()
    capture_arr = rec.raw_uint16()
    stats = rec.statistics()
    return states, dict(selected_positions=selected, final_bits=final_bits,
                        layer_bits=capture_arr, layer_statistics=stats)


def compute_request_scores(model, tok, requests, demo: str, capture_mode: str,
                           state_dir: Path, item_tag: str, save_head: bool,
                           patch: Optional[Patch]):
    """Deduplicate *exact* tokenized forward requests; no KV cache or mask."""
    pairs = [encode_pair(tok, context, continuation) for context, continuation in requests]
    groups = {}
    for i, (prefix, target) in enumerate(pairs):
        prediction_positions = list(range(len(prefix)-1, len(prefix)+len(target)-1))
        input_ids = prefix + target[:-1]
        key = (tuple(input_ids), tuple(prediction_positions))
        groups.setdefault(key, []).append((i, target))
    scores = np.zeros((4, len(requests)), dtype=np.float64)
    forward_records = []
    for serial, ((ids, positions), members) in enumerate(groups.items()):
        file_name = f"{item_tag}_f{serial:02d}.npz"
        state_path = state_dir / file_name
        states, cap = run_single(model, tok, list(ids), list(positions), demo, capture_mode,
                                 state_path, save_head, patch)
        # Gather each candidate continuation at native head output; no answer generation.
        any_full_head = None
        for i, target in members:
            score, head = target_log_likelihoods(model, states, list(positions), target)
            scores[:, i] = score
            if any_full_head is None and save_head:
                any_full_head = head
        if capture_mode != "none" or save_head:
            pack_state(state_path, cap["layer_bits"], cap["final_bits"],
                       cap["selected_positions"], any_full_head)
        forward_records.append(dict(input_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int32).tobytes()).hexdigest(),
                                    input_tokens=len(ids), prediction_positions=list(positions),
                                    targets=[i for i, _ in members],
                                    positions_saved=len(cap["selected_positions"]),
                                    state_file=str(state_path) if (capture_mode != "none" or save_head) else None,
                                    stats=cap["layer_statistics"]))
        del states
    return scores, forward_records


def run_question(model, tok, row, demo, rotations, capture_mode, state_dir, save_head, patch):
    n = len(row["choices"])
    if rotations not in ("original", "cyclic"):
        raise ValueError(rotations)
    orders = [list(range(n))]
    if rotations == "cyclic":
        orders += [[(j + shift) % n for j in range(n)] for shift in range(1, n)]
    item_hash = sha256_text(row["id"])[:18]
    rotate_records = []
    for rot, order in enumerate(orders):
        prompt, requests = make_requests(row, demo, order)
        score_by_display, forwards = compute_request_scores(
            model, tok, requests, demo, capture_mode, state_dir, f"{item_hash}_r{rot}", save_head, patch)
        original_index = np.empty_like(score_by_display)
        original_index[:, order] = score_by_display
        displayed_preds = np.argmax(score_by_display, axis=1).tolist()
        mapped_preds = [order[index] for index in displayed_preds]
        display_ties = [int(np.sum(score_by_display[k] == np.max(score_by_display[k]))) for k in range(4)]
        rotate_records.append(dict(order=order, prompt_sha256=sha256_text(prompt),
                                   scores_in_original_content_order=original_index.tolist(),
                                   displayed_letter_predictions_by_depth=displayed_preds,
                                   mapped_content_predictions_by_depth=mapped_preds,
                                   display_max_tie_counts_by_depth=display_ties,
                                   forwards=forwards))
    avg = np.mean([np.asarray(x["scores_in_original_content_order"]) for x in rotate_records], axis=0)
    return dict(id=row["id"], split=row["split"], n=n, input_sha256=sha256_text(json.dumps(row,sort_keys=True,ensure_ascii=False)),
                rotations=rotate_records, native_prediction_by_depth=np.argmax(
                    np.asarray(rotate_records[0]["scores_in_original_content_order"]),axis=1).tolist(),
                cyclic_average_score_by_depth=avg.tolist() if rotations=="cyclic" else None,
                cyclic_prediction_by_depth=np.argmax(avg,axis=1).tolist() if rotations=="cyclic" else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Gold-free prepared split JSONL")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--demonstrations", type=Path, default=DEFAULT_DEMOS)
    p.add_argument("--rotations", choices=("original", "cyclic"), default="original")
    p.add_argument("--capture", choices=("none", "answer", "current", "all"), default="current")
    p.add_argument("--save-head", action="store_true", help="Store final answer-position full-vocab logits for K1..4")
    p.add_argument("--reserve-gb", type=float, default=80.0)
    p.add_argument("--patch-layer", type=int, default=None)
    p.add_argument("--patch-source-loop", type=int, default=1)
    p.add_argument("--patch-target-loop", type=int, default=2)
    p.add_argument("--patch-fraction", type=float, default=0.0)
    a = p.parse_args()
    if not a.input.is_file():
        raise FileNotFoundError(a.input)
    a.out.mkdir(parents=True,exist_ok=True)
    patch = None if a.patch_layer is None else Patch(a.patch_layer,a.patch_source_loop,a.patch_target_loop,a.patch_fraction)
    if patch: patch.validate()
    manifest = {"model": "ByteDance/Ouro-1.4B", "revision": REVISION, "model_sha256": MODEL_SHA256,
                "input_file": str(a.input.resolve()), "input_sha256": sha256_file(a.input),
                "demonstrations_sha256": sha256_file(a.demonstrations), "rotations": a.rotations,
                "capture": a.capture, "save_head":a.save_head, "patch": None if patch is None else vars(patch),
                "implementation_sha256": {str(q.name):sha256_file(q) for q in [Path(__file__),Path(__file__).with_name("latent_hooks.py"),Path(__file__).with_name("core.py")]}}
    manifest_path = a.out / "RUN_CONFIG.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf8")) != manifest:
            raise RuntimeError("Existing output belongs to a different run; choose a new directory")
    else: write_json(manifest_path,manifest)
    predictions = a.out / "predictions.jsonl"
    done = set()
    if predictions.exists():
        for rec in read_jsonl(predictions):
            if rec["id"] in done: raise RuntimeError("Duplicate record in output")
            done.add(rec["id"])
    demo = demo_prefix(json.loads(a.demonstrations.read_text(encoding="utf8")))
    tok, model = load_ouro(a.model_dir)
    print(json.dumps({"checkpoint_verified":True,"input":str(a.input),"completed_existing":len(done),"patch":None if patch is None else vars(patch)}),flush=True)
    total = count = 0
    state_dir = a.out / "states"
    with predictions.open("a",encoding="utf8") as dest:
        for row in read_jsonl(a.input):
            total += 1
            if row["id"] in done:
                continue
            free = shutil.disk_usage(a.out).free
            if free < a.reserve_gb * (1024**3):
                raise RuntimeError("Disk space below reserved minimum; change output mount or compact existing states")
            result = run_question(model,tok,row,demo,a.rotations,a.capture,state_dir,a.save_head,patch)
            dest.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n")
            dest.flush();os.fsync(dest.fileno())
            count+=1
            if count%10==0: print(json.dumps({"new_completed":count,"encountered":total,"id":row["id"]}),flush=True)
    print(json.dumps({"status":"complete","new_completed":count,"encountered":total,"output":str(predictions)}),flush=True)

if __name__=="__main__": main()
