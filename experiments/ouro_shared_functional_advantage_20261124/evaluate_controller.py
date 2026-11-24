"""Run the same unrestricted decoder on held-out data for every controller."""

import argparse
import json
from pathlib import Path
import re

import torch

from collect_moments import DATASETS
from evaluate_generation import candidates, data_mode, generate, summarize
import minimal
from task_data import hash_order, load, save


MODES = {"arc_challenge": ("content", "candidate"), "csqa": ("content", "candidate"),
         "squad": ("span",), "gsm8k": ("gsm_process",)}
DECODING = {"content": (24, 48), "candidate": (1, 12), "span": (24, 32), "gsm_process": (192, 24)}
FINAL_MARKER = re.compile(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?")


def jobs(data, phase, depths, limit, transfer_limit):
    for ds in DATASETS:
        filename = f"{ds}_development.jsonl" if phase == "development" else f"test_{ds}.jsonl"
        path = data / filename
        signature = minimal.sha(path)
        for mode in MODES[ds]:
            max_new, batch = DECODING[mode]
            for k in depths:
                n = limit if phase == "development" or k == 4 else transfer_limit
                jobspec = dict(dataset=ds, mode=mode, depth=k, limit=n, batch=batch, max_new=max_new)
                yield dict(jobspec, path=str(path), input_sha256=signature)


def annotate(records):
    for row in records:
        marker = bool(FINAL_MARKER.search(row["prediction"]))
        row["metrics"]["premature_eos"] = row["stop"] == "eos" and not marker
        row["metrics"]["eos_before_16"] = row["stop"] == "eos" and row["answer_tokens"] < 16
        row["metrics"]["process_correct"] = row["metrics"]["strict"] and row["metrics"]["process"]


def rows_for(spec):
    ds, mode = spec["dataset"], spec["mode"]
    path = Path(spec["path"])
    if minimal.sha(path) != spec["input_sha256"]:
        raise RuntimeError(f"Trial input fingerprint differs {path}")
    eligible = load(path)
    ordered = sorted(eligible, key=lambda x: hash_order(x["sample_id"], f"advantage/{ds}/evaluation"))
    selected = ordered[:spec["limit"]]
    if mode == "gsm_process":
        for row in selected:
            prompt = "Solve the problem step by step. End with the final answer after ####."
            row["instruction"] = prompt
            row["prompt"] = f"Question: {row['context']}\n{prompt}\nAnswer:"
    return data_mode(selected, ds, mode)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "previous", "model", "data", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--depths", type=int, nargs="+", default=(2, 3, 4))
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--transfer-limit", type=int, default=96)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--slice", choices=("k4", "transfer"))
    a = p.parse_args()
    checkpoint_sha = minimal.sha(a.checkpoint) if a.checkpoint else "native"
    arm = a.arm
    if a.protocol:
        frozen = json.loads(a.protocol.read_text())
        if frozen["arm_sha256"][arm] != checkpoint_sha or frozen["evaluator_sha256"] != minimal.sha(__file__):
            raise RuntimeError("The confirmation controller or evaluator differs from the sealed version")
        a_spec = [case for case in frozen["cases"] if arm in case["arms"]]
        if a.slice:
            a_spec = [case for case in a_spec if (case["depth"] == 4) == (a.slice == "k4")]
        protocol_sha = minimal.sha(a.protocol)
    else:
        a_spec = list(jobs(a.data, "development", a.depths, a.limit, a.transfer_limit))
        protocol_sha = None
    archive, model, tok, runtime = minimal.runtime(a.archive, a.previous, a.model)
    net = minimal.load_net(a.checkpoint).requires_grad_(False) if a.checkpoint else None
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in a_spec:
        rows = rows_for(spec)
        ds, mode, depth = spec["dataset"], spec["mode"], spec["depth"]
        filename = a.out / f"{ds}__{arm}__k{depth}__{mode}.jsonl"
        receipt = filename.with_suffix(".json")
        plan = dict(spec, arm=arm, n=len(rows), checkpoint_sha256=checkpoint_sha, protocol_sha256=protocol_sha)
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["plan"] != plan or old["rows_sha256"] != minimal.sha(filename):
                raise RuntimeError(f"Existing receipt differs {receipt}")
            continue
        torch.cuda.reset_peak_memory_stats()
        if mode == "candidate":
            vector, cost = candidates(archive, model, tok, net, rows, depth, spec["batch"])
        else:
            vector, cost = generate(archive, model, tok, net, rows, mode, depth, spec["batch"], spec["max_new"])
        if mode == "gsm_process":
            annotate(vector)
        save(filename, vector)
        compact = summarize(vector)
        minimal.dump(receipt, dict(plan=plan, rows_sha256=minimal.sha(filename), metrics=compact, cost=cost, runtime=runtime))
        print(filename.stem, len(vector), {key: round(v["value"], 4) for key, v in compact.items()},
              round(cost["seconds"], 2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
