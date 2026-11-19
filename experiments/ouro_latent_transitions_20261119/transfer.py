"""Leave one task entirely out of fitting and configuration selection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from fit_study import (Data, Scale, TASKS, SPLITS, export, fit_system,
                       grouped_rows, load_lines, cosine, mse, projection)


def config_from_source(files, method):
    votes = {}
    for path in files:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        baseline = float(next(r["discover_mse"] for r in rows if r["method"] == "identity_raw"))
        for r in rows:
            if r["method"] == method:
                desc = (r["encoder"] or None, int(r["latent"]) if r["latent"] else None,
                        float(r["penalty"]) if r["penalty"] else None)
                votes.setdefault(desc, []).append(float(r["discover_mse"])/baseline)
    e,d,p = min(votes, key=lambda k: np.mean(votes[k]))
    return dict(method=method, encoder=e, latent=d, penalty=p)


def run(past, runs, result):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data = {t: Data(past, runs, t, 96) for t in TASKS}
    records = []
    choices = {}
    for target in TASKS:
        others = [d for name,d in data.items() if name != target]
        pooled = np.concatenate([d.x[d.data_mask("fit")] for d in others])
        context = np.concatenate([d.context[d.data_mask("fit")] for d in others])
        scale = Scale(pooled)
        train = scale.forward(pooled)
        candidates = ("identity_raw", "ridge_raw", "ridge", "mlp", "dictionary")
        dst = data[target]
        raw = np.load(runs/target/"causal_full.npy", mmap_mode="r") @ projection()
        rows = load_lines(runs/target/"causal.jsonl")
        state = scale.forward(raw)
        true = scale.downstream(state[:, 1, 1]-state[:, 0, 1])
        ctx_rows = dst.context[[dst.ids[r["id"],r["world"]] for r in rows]]
        choices[target] = {}
        for method in candidates:
            config = config_from_source([result/"96"/d.task/"discovery.csv" for d in others], method)
            choices[target][method] = config
            system = fit_system(config, train, scale, context)
            delta = scale.downstream(system(state[:, 1, 0], ctx_rows)-system(state[:, 0, 0], ctx_rows))
            for split in SPLITS:
                eligible = np.array([r["split"] == split and r["arm"] == "semantic" for r in rows])
                natural = dst.data_mask(split)
                natural_rows = np.array(dst.rows, dtype=object)[natural]
                a,b = scale.forward(dst.x[natural]).transpose(1,0,2)
                pred = system(a, dst.context[natural])
                tag = {"source_tasks": "+".join(d.task for d in others),
                       "task": target, "method": method, "split": split}
                records.extend(grouped_rows(natural_rows, {"mse": mse(scale.downstream(pred-b)),
                    "identity_energy": mse(scale.downstream(b-a)), "cosine": np.zeros(len(a))},
                    {**tag, "arm": "natural"}))
                records.extend(grouped_rows(np.array(rows, dtype=object)[eligible],
                    {"mse": mse(delta[eligible]-true[eligible]), "identity_energy": mse(true[eligible]),
                     "cosine": cosine(delta[eligible],true[eligible])}, {**tag,"arm":"semantic"}))
        print("held-out task", target, flush=True)
    export(result/"cross_task_by_group.csv", records)
    (result/"cross_task_frozen.json").write_text(json.dumps(choices,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("past", type=Path)
    p.add_argument("runs", type=Path)
    p.add_argument("results", type=Path)
    a = p.parse_args()
    run(a.past, a.runs, a.results)
