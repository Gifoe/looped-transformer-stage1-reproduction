"""Check where the arithmetic task remains difficult when a value is supplied."""
import argparse
import csv
import json
from pathlib import Path
import re
import time

import torch

from design import choice_prompt
from run import dump_line, encode, fast_forward, load_runtime, predictions, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("model", type=Path)
    args = p.parse_args()
    out = args.study / "arithmetic_diagnostic"
    out.mkdir()
    model, tok, runtime = load_runtime(args.model)
    write_json(out / "runtime.json", runtime)
    groups = [json.loads(line) for line in
              (args.study / "semantic_interventions.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = [g for g in groups if g["task"] == "arithmetic"]
    observations = []
    started = time.perf_counter()
    with torch.inference_mode(), (out / "rows.jsonl").open("w", encoding="utf-8") as file:
        for group in groups:
            worlds, choices = group["worlds"], group["options"]
            for receiver, alternate in (("base", "first"), ("same_answer", "second")):
                w = worlds[receiver]
                middle = int(group["middle_options"][w["middle_answer"]])
                counter = int(group["middle_options"][worlds[alternate]["middle_answer"]])
                factor, bias = map(int, re.search(
                    r"Multiply the result of the first step by (\d+), then add (-?\d+)\.",
                    w["prompt"],
                ).groups())
                gold, cf = w["answer"], worlds[alternate]["answer"]
                assert choices[gold] == str(middle*factor+bias)
                assert choices[cf] == str(counter*factor+bias)
                variants = {
                    "native": (w["prompt"], gold),
                    "supplied_middle": (choice_prompt(
                        f"The input value is {middle}. Multiply it by {factor}, then add {bias}.\nWhat is the final result?",
                        choices), gold),
                    "supplied_counterfactual": (choice_prompt(
                        f"The input value is {counter}. Multiply it by {factor}, then add {bias}.\nWhat is the final result?",
                        choices), cf),
                    "supplied_product": (choice_prompt(
                        f"The product of {middle} and {factor} is {middle*factor}. Add {bias} to this product.\nWhat is the final result?",
                        choices), gold),
                }
                for condition, (prompt, expected) in variants.items():
                    entry = encode(tok, {"prompt": prompt, "anchor": None}, choices)
                    scores = fast_forward(model, entry)[0]
                    row = {"id": group["id"], "receive": receiver, "condition": condition,
                           "prompt": prompt, "scores": scores, "gold": expected,
                           "pred": predictions(scores), "middle": middle, "alternate": counter,
                           "factor": factor, "bias": bias}
                    observations.append(row)
                    dump_line(file, row)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        w = csv.writer(file)
        w.writerow(("condition", "readout_K", "groups", "directions", "correct", "ties"))
        for condition in variants:
            rows = [r for r in observations if r["condition"] == condition]
            for k in range(4):
                w.writerow((condition, k+1, len(groups), len(rows),
                            sum(r["pred"][k]==r["gold"] for r in rows),
                            sum(r["pred"][k]==-1 for r in rows)))
    torch.cuda.synchronize()
    write_json(out / "completed.json", {"groups": len(groups), "rows": len(observations),
               "work_seconds": time.perf_counter()-started,
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()})
    print(f"Arithmetic diagnostic wrote {len(observations)} rows", flush=True)


if __name__ == "__main__":
    main()
