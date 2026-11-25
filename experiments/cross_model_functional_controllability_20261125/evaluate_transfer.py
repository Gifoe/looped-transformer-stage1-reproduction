"""Evaluate an unchanged Rank-1 T1 on an official Ouro checkpoint at K4."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.stats import binomtest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from controller_io import controller
from evaluate_benchmarks import candidate_row, mmlu_metrics
from evaluate_core import finish, verified
from fetch_ouro import checksum
import minimal


T1_SHA = "91d4919308bf9060ce0c44e4046e4851e7e59f65b9692c77045535f96534efd2"
TASKS = {
    "arc_challenge": (1172, "acc_norm"), "arc_easy": (2376, "acc_norm"), "sciq": (1000, "acc_norm"),
    "mmlu": (14042, "acc"), "hellaswag": (10042, "acc_norm"), "winogrande": (1267, "acc"),
    "piqa": (1838, "acc_norm"), "mmlu_pro": (12032, "correct"),
}
ARM_NAMES = ("native", "t1")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime(a):
    metadata = read(a.model / "MODEL_LOCK.json")
    pin = read(a.models)[a.name]
    if any(metadata[key] != value for key, value in pin.items()):
        raise ValueError("The local model is not the selected upstream revision")
    for name, entry in metadata["files"].items():
        path = a.model / name
        heavy = name == "model.safetensors"
        expected = pin["weight_sha256"] if heavy else entry["blob_id"]
        if path.stat().st_size != entry["bytes"] or checksum(path, heavy) != expected:
            raise ValueError(f"Unexpected checkpoint file {name}")
    if minimal.sha(a.t1) != T1_SHA:
        raise ValueError("Unexpected historical T1 checkpoint")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The evaluation requires BF16 CUDA support")
    transformers.utils.import_utils._torchvision_available = False
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").requires_grad_(False).eval()
    if model.config.hidden_size != 2048 or len(model.model.layers) != metadata["layers"]:
        raise ValueError("The official architecture differs from the checkpoint receipt")
    model.model.total_ut_steps = 4
    model.config.total_ut_steps = 4
    if hasattr(model.config, "early_exit_threshold"):
        model.config.early_exit_threshold = None
    if hasattr(model, "early_exit_threshold"):
        model.early_exit_threshold = None
    net = controller(a.t1, "cuda")
    token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    ids = torch.tensor([[token, token]], device="cuda")
    _, states, _ = model.model(input_ids=ids, use_cache=False)
    if len(states) != 4:
        raise ValueError("The official core did not execute exactly four loops")
    return model, tokenizer, net, dict(
        model=metadata, depth=4, early_exit=False, dtype="bfloat16", decoding_seed=0,
        tokenizer_tokens=dict(bos=tokenizer.bos_token_id, eos=tokenizer.eos_token_id,
                              pad=tokenizer.pad_token_id, config_bos=model.config.bos_token_id,
                              config_eos=model.config.eos_token_id),
        gpu=torch.cuda.get_device_name(0), capacity_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch=torch.__version__, transformers=transformers.__version__,
    )


def timestamp_cost(total, part):
    for key, val in part.items():
        total[key] = max(total[key], val) if key.startswith("peak_") else total[key] + val


def candidate_task(a, task, arm, plan, model, tok, net, runtime):
    import seven_unified as scoring
    import stage2_score as stage
    file = a.stage2 / "inputs" / f"{task}.jsonl"
    inputs = plan["reference"]["seven_requests_sha256"][task]
    if minimal.sha(file) != inputs or minimal.sha(Path(stage.__file__)) != plan["reference"]["stage2_sha256"]:
        raise ValueError(f"Archival input or scorer has changed: {task}")
    stage.APPLE_BENCHMARKS = dict.fromkeys(stage.APPLE_BENCHMARKS, 0)
    chosen, _ = stage.read_export(SimpleNamespace(input=file, manifest=file.with_suffix(".jsonl.manifest.json"),
                                                   plan=False, benchmarks=(task,)))
    if len(chosen) != TASKS[task][0]:
        raise ValueError(f"Wrong full population for {task}")
    receipt = read(a.stage2 / "scores" / task / "LOCK.json")
    fields = ("max_length", "max_batch", "token_cap", "attention_cap", "head_rows", "shard_size")
    config = {name: receipt[name] for name in fields}
    spec = dict(task=task, depth=4, mode="candidate", arm=arm, model_lock_sha256=plan["model_lock_sha256"],
                checkpoint=plan["arms"][arm], input_sha256=inputs, config=config)
    target = a.out / f"{task}__{arm}.jsonl"
    if value := verified(target, spec, [r["id"] for r in chosen]):
        return value
    shards = a.out / "shards" / target.stem
    shards.mkdir(parents=True, exist_ok=True)
    encoder = stage.HarnessEncoder(tok, config["max_length"])
    complete, total = [], Counter()
    stride = config["shard_size"]
    for start in range(0, len(chosen), stride):
        rows = chosen[start:start + stride]
        ids = [r["id"] for r in rows]
        p = shards / f"{start:05d}.jsonl"
        item = dict(spec, start=start, end=start + len(rows))
        previous = verified(p, item, ids)
        if previous:
            out, timing = previous
        else:
            scores, timing = scoring.shard_score(stage, minimal, model, encoder, net, rows, config, next(model.parameters()).device)
            out = [candidate_row(raw, scoring.UNIFIED) for raw in scores]
            for got, wanted in zip(out, rows, strict=True):
                if (got["id"], got["choices"], got["gold"]) != (wanted["id"], wanted["choices"], wanted["target"]):
                    raise ValueError(f"Choice or ID mismatch at {got['id']}")
            finish(p, out, item, timing, runtime)
        complete.extend(out)
        timestamp_cost(total, timing)
    finish(target, complete, spec, total, runtime)
    return complete, total


def generation_task(a, arm, plan, model, tok, net, runtime):
    import generation_followup_run as decoder
    task = "mmlu_pro"
    source = a.followup / "data/mmlu_pro.parquet"
    if (minimal.sha(source) != plan["reference"]["mmlu_pro_sha256"]
            or minimal.sha(Path(decoder.__file__)) != plan["reference"]["mmlu_pro_code_sha256"]):
        raise ValueError("MMLU-Pro original generation protocol differs")
    items = decoder.prompt_items(source, task, tok)
    if len(items) != TASKS[task][0]:
        raise ValueError("Wrong MMLU-Pro population")
    max_new, batch = decoder.TASKS[task]
    layers = len(model.model.layers)
    capacity_tokens = (14500 * 24) // layers
    slots = 4 * layers
    bounded = decoder.cache_type(model)

    def cache(capacity):
        result = bounded(capacity)
        result.slots = [None] * slots
        result.lengths = [0] * slots
        return result

    batches = list(decoder.batches(items, max_new, batch, capacity_tokens))
    spec = dict(task=task, mode="generated", depth=4, arm=arm, checkpoint=plan["arms"][arm],
                model_lock_sha256=plan["model_lock_sha256"], input_sha256=minimal.sha(source),
                max_new=max_new, batch=batch, capacity_tokens=capacity_tokens, cache_layers=slots)
    target = a.out / f"{task}__{arm}.jsonl"
    if value := verified(target, spec, [r.key for block in batches for r in block]):
        return value
    shards = a.out / "shards" / target.stem
    shards.mkdir(parents=True, exist_ok=True)
    complete, total = [], Counter()
    for i, batch_rows in enumerate(batches):
        p = shards / f"{i:05d}.jsonl"
        item = dict(spec, batch_index=i)
        previous = verified(p, item, [r.key for r in batch_rows])
        if previous:
            out, cost = previous
        else:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            raw, _ = decoder.greedy(task, model, tok, batch_rows, "feedback" if net else "native",
                                    net, 1.0, max_new, cache)
            torch.cuda.synchronize()
            cost = dict(seconds=time.perf_counter()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), calls=max(len(row["tokens"]) for row in raw),
                        generated_tokens=sum(len(row["tokens"]) for row in raw))
            out = mmlu_metrics(raw, batch_rows, decoder)
            finish(p, out, item, cost, runtime)
        complete.extend(out)
        timestamp_cost(total, cost)
    finish(target, complete, spec, total, runtime)
    return complete, total


def pair(native, transfer, key):
    x = np.array([r["metrics"][key] for r in native], dtype=bool)
    y = np.array([r["metrics"][key] for r in transfer], dtype=bool)
    n = len(x)
    rescue = int((~x & y).sum())
    damage = int((x & ~y).sum())
    p = [damage/n, (n-rescue-damage)/n, rescue/n]
    draw = np.random.default_rng(261125137).multinomial(n, p, size=10000)
    ci = np.quantile((draw[:, 2]-draw[:, 0])/n, (.025, .975))
    return dict(n=n, native_count=int(x.sum()), t1_count=int(y.sum()), native=float(x.mean()), t1=float(y.mean()),
                delta=(rescue-damage)/n, rescue=rescue, damage=damage, paired_ci95=ci.tolist(),
                mcnemar=float(binomtest(rescue, rescue+damage).pvalue) if rescue+damage else 1.)


@torch.inference_mode()
def run(a):
    sys.path[:0] = [str(a.stage2 / "code"), str(a.followup), str(a.unified)]
    a.out.mkdir(exist_ok=True, parents=True)
    reference = read(a.reference)
    if reference["depth"] != 4 or reference["expected"] != {k: v[0] for k, v in TASKS.items()}:
        raise ValueError("The candidate/generation archive differs from the eight-benchmark protocol")
    plan = dict(model_lock_sha256=minimal.sha(a.model / "MODEL_LOCK.json"),
                reference_sha256=minimal.sha(a.reference), reference=reference,
                evaluator_sha256=minimal.sha(Path(__file__)), arms=dict(native=None, t1=T1_SHA),
                injection="historical T1 predictor positions before the fourth native loop",
                relative_rms_cap=.45, dose=1., depth=4, no_dynamic_exit=True, no_tuning=True)
    path = a.out / "FREEZE.json"
    if path.exists() and read(path) != plan:
        raise ValueError("The scored cross-model protocol was already locked")
    minimal.dump(path, plan)
    model, tok, control, runtime = load_runtime(a)
    result = a.out / "RESULTS.json"
    outcomes = read(result) if result.exists() else {}
    todo = (a.task,) if a.task != "all" else tuple(TASKS)
    for task in todo:
        readouts = {}
        for arm in ARM_NAMES:
            net = control if arm == "t1" else None
            readouts[arm] = generation_task(a, arm, plan, model, tok, net, runtime) if task == "mmlu_pro" else candidate_task(
                a, task, arm, plan, model, tok, net, runtime)
        native, edited = (readouts[arm][0] for arm in ARM_NAMES)
        if ([(x["id"], x["gold"]) for x in native] != [(x["id"], x["gold"]) for x in edited]
                or len(native) != TASKS[task][0]):
            raise ValueError(f"Mismatch in the official paired {task} rows")
        keys = ("correct", "parseable", "strict_format", "strict") if task == "mmlu_pro" else ("acc", "acc_norm")
        record = dict(primary=TASKS[task][1], metrics={key: pair(native, edited, key) for key in keys},
                      cost={arm: dict(readouts[arm][1]) for arm in ARM_NAMES},
                      predictions={arm: minimal.sha(a.out / f"{task}__{arm}.jsonl") for arm in ARM_NAMES})
        outcomes[task] = record
        minimal.dump(result, outcomes)
        print("PAIRED", a.name, task, json.dumps(record["metrics"][record["primary"]]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "models", "t1", "stage2", "followup", "unified", "reference", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--task", choices=("all", *TASKS), default="all")
    run(p.parse_args())
