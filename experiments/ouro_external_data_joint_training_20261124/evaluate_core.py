"""Run the shared checkpoint on frozen core tasks with resumable raw-output receipts."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from controller_io import controller
from evaluate_controller import annotate
from evaluate_generation import candidates, data_mode, generate, summarize
import minimal
from task_data import hash_order, load, save


TASKS = {
    "arc_challenge": ("candidate", "content", "letter"), "csqa": ("candidate", "content", "letter"),
    "obqa": ("candidate", "content", "letter"), "squad": ("span",), "gsm8k": ("gsm_process",),
}
DECODE = {"candidate": (1, 12), "content": (24, 48), "letter": (8, 48), "span": (24, 32), "gsm_process": (192, 24)}
STRIDE = 576
LIMITS = {"dev-small": 120, "dev-large": 360, "test": None, "transfer": 192}


def job_sets(data, phase, depths, requested):
    group = "dev" if phase.startswith("dev-") else "test"
    manifest = json.loads((data / "PANEL.json").read_text())["files"]
    for ds in requested:
        path = data / f"{group}_{ds}.jsonl"
        sha = minimal.sha(path)
        if sha != manifest[path.name]["sha256"]:
            raise ValueError(f"Unexpected panel fingerprint {path}")
        rows = load(path)
        maximum = LIMITS[phase]
        if phase == "transfer":
            rows = sorted(rows, key=lambda r: hash_order(r["sample_id"], f"external/final-depth/{ds}"))
        rows = rows[:maximum] if maximum else rows
        modes = TASKS[ds]
        if group == "dev":
            modes = tuple(mode for mode in modes if mode != "letter")
        for mode in modes:
            batch_rows = data_mode(rows, ds, mode)
            for k in depths:
                max_new, batch = DECODE[mode]
                yield dict(dataset=ds, mode=mode, depth=k, input_sha256=sha, phase=phase, n=len(batch_rows),
                           max_new=max_new, batch=batch, stride=STRIDE), batch_rows


def math_diagnostics(rows):
    annotate(rows)
    for row in rows:
        seq = row["generated_tokens"]
        grams = Counter(zip(seq, seq[1:], seq[2:], seq[3:]))
        row["metrics"].update(token_limit=row["stop"] == "token_limit",
                              repetitive=len(seq) >= 32 and max(grams.values(), default=0) >= 4)


def verified(path, spec, sample_ids):
    receipt = path.with_suffix(".json")
    if not receipt.exists():
        return None
    result = json.loads(receipt.read_text())
    if result["plan"] != spec or minimal.sha(path) != result["rows_sha256"]:
        raise ValueError(f"Changed result receipt {receipt}")
    rows = load(path)
    if [r["id"] for r in rows] != sample_ids or result["metrics"] != summarize(rows):
        raise ValueError(f"Incorrect cohort or aggregate score {receipt}")
    return rows, result["cost"]


def finish(path, rows, spec, cost, hardware):
    save(path, rows)
    minimal.dump(path.with_suffix(".json"), dict(plan=spec, metrics=summarize(rows), cost=cost, runtime=hardware,
                                                rows_sha256=minimal.sha(path)))


@torch.inference_mode()
def run(a):
    if a.phase in ("test", "transfer"):
        if a.freeze is None:
            raise ValueError("Official scoring requires a sealed checkpoint registry and protocol")
        freeze = json.loads(a.freeze.read_text())
        if freeze["core_scorer_sha256"] != minimal.sha(Path(__file__)) or freeze["panel_sha256"] != minimal.sha(a.data / "PANEL.json"):
            raise ValueError("Official frozen core inputs or scorer changed")
        slot = freeze["arms"][a.arm]
        if (minimal.sha(a.checkpoint) if a.checkpoint else "native") != slot["sha256"]:
            raise ValueError("Official checkpoint fingerprint differs from frozen registry")
        if a.phase == "test" and a.depths != [4]:
            raise ValueError("Primary registered full tests use K4")
        if a.phase == "transfer" and set(a.depths) - {2, 3, 5, 6}:
            raise ValueError("Transfer uses registered held-out depths")
    checksum = minimal.sha(a.checkpoint) if a.checkpoint else "native"
    specs = list(job_sets(a.data, a.phase, a.depths, a.tasks))
    archive, model, tok, hardware = minimal.runtime(a.archive, a.previous, a.model)
    net = controller(a.checkpoint, next(model.parameters()).device)
    a.out.mkdir(parents=True, exist_ok=True)
    for spec, rows in specs:
        spec = dict(spec, arm=a.arm, checkpoint_sha256=checksum, freeze_sha256=minimal.sha(a.freeze) if a.freeze else None)
        ds, mode, k = (spec[key] for key in ("dataset", "mode", "depth"))
        peak = a.out / f"{ds}__{a.arm}__k{k}__{mode}.jsonl"
        if verified(peak, spec, [r["sample_id"] for r in rows]):
            continue
        fragments = a.out / "shards" / peak.stem
        fragments.mkdir(parents=True, exist_ok=True)
        all_rows, costs = [], []
        for i in range(0, len(rows), STRIDE):
            batch = rows[i:i+STRIDE]
            piece = fragments / f"{i // STRIDE:04d}.jsonl"
            subplan = dict(spec, start=i, stop=i+len(batch))
            stored = verified(piece, subplan, [r["sample_id"] for r in batch])
            if stored:
                value, cost = stored
            else:
                torch.cuda.reset_peak_memory_stats()
                if mode == "candidate":
                    value, cost = candidates(archive, model, tok, net, batch, k, spec["batch"])
                else:
                    value, cost = generate(archive, model, tok, net, batch, mode, k, spec["batch"], spec["max_new"])
                    if mode == "gsm_process":
                        math_diagnostics(value)
                finish(piece, value, subplan, cost, hardware)
            all_rows.extend(value)
            costs.append(cost)
        cost = {key: (max(c[key] for c in costs) if key == "cuda_peak" else sum(c[key] for c in costs)) for key in costs[0]}
        finish(peak, all_rows, spec, cost, hardware)
        print(peak.stem, len(all_rows), {k:round(v["value"],4) for k,v in summarize(all_rows).items()}, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "data", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--phase", choices=LIMITS, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[2,3,4])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--freeze", type=Path)
    run(parser.parse_args())
