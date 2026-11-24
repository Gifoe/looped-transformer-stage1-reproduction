"""Lock the development winner and score the twelve historical/official K4 tasks."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string

import minimal
import evaluate_benchmarks as historical
import evaluate_core as core
from task_data import load


EXTRA = {"csqa": 1221, "obqa": 500, "squad": 10570, "gsm8k": 1319}
LEGACY = {
    "arc_challenge": 1172, "arc_easy": 2376, "sciq": 1000, "mmlu": 14042,
    "hellaswag": 10042, "winogrande": 1267, "piqa": 1838, "mmlu_pro": 12032,
}
BASE = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
EXTRA_ARMS = {
    "native": "native",
    "t1": "91d4919308bf9060ce0c44e4046e4851e7e59f65b9692c77045535f96534efd2",
    "t2": "117f5f2fe9c5afe363ba8c032f42cd25155f7a88cb465e4e79bfb3bf496c8032",
    "mixed": "ec2e55a64564f608cfeb245da079015db4347a9391072cba7f2949db05185b87",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(a):
    selection = read(a.selection)
    dev = a.selection.with_name("DEV_PROTOCOL.json")
    winner = Path(selection["path"]).resolve()
    winner_hash = minimal.sha(winner)
    if (selection["checkpoint_sha256"] != winner_hash
            or selection["development_protocol_sha256"] != minimal.sha(dev)
            or read(dev)["checkpoint_sha256"][selection["best"]] != winner_hash):
        raise ValueError("Mismatch between the selected checkpoint and sealed development")
    lock = read(a.reference_freeze)
    if (lock["depth"] != 4 or lock["expected"] != LEGACY or lock["model_sha256"] != BASE
            or minimal.sha(a.model / "model.safetensors") != BASE):
        raise ValueError("Historical eight-benchmark protocol or Ouro base do not match")
    for name in LEGACY.keys() - {"mmlu_pro"}:
        wanted = lock["seven_requests_sha256"][name]
        if minimal.sha(a.stage2 / "inputs" / f"{name}.jsonl") != wanted:
            raise ValueError(f"Historical zero-shot request digest differs for {name}")
    check_files = {
        "stage2_sha256": a.stage2 / "code/stage2_score.py",
        "mmlu_pro_sha256": a.followup / "data/mmlu_pro.parquet",
        "mmlu_pro_code_sha256": a.followup / "generation_followup_run.py",
        "mmlu_pro_reference_lock_sha256": a.followup / "results/mmlu_pro/LOCK.json",
    }
    for key, file in check_files.items():
        if key in lock and lock[key] != minimal.sha(file):
            raise ValueError(f"Original scoring code or input was changed: {file}")
    panel = a.panel / "PANEL.json"
    tasks = read(panel)["files"]
    for name, n in EXTRA.items():
        path = a.panel / f"test_{name}.jsonl"
        if tasks[path.name] != {"rows": n, "sha256": minimal.sha(path)}:
            raise ValueError(f"Official panel digest or size differs for {name}")
        ids = [row["sample_id"] for row in load(path)]
        if len(ids) != n or len(set(ids)) != n:
            raise ValueError(f"Official panel IDs do not cover the full set for {name}")

    arms = {"new": {"path": str(winner), "sha256": winner_hash},
            "native": {"path": None, "sha256": "native"}}
    for name in ("t1", "t2", "mixed"):
        weight = getattr(a, name).resolve()
        if minimal.sha(weight) != EXTRA_ARMS[name]:
            raise ValueError(f"Unexpected historical {name} checkpoint")
        arms[name] = {"path": str(weight), "sha256": EXTRA_ARMS[name]}
    plan = {
        "selection_sha256": minimal.sha(a.selection),
        "historical_freeze_sha256": minimal.sha(a.reference_freeze),
        "depth": 4, "expected": LEGACY | EXTRA, "arms": arms,
        "model_sha256": BASE, "panel_sha256": minimal.sha(panel),
        "primary": lock["primary"] | {"csqa": "acc_norm", "obqa": "acc_norm",
                                      "squad": "standard_f1", "gsm8k": "standard_exact"},
        "core_scorer_sha256": minimal.sha(Path(core.__file__)),
        "large_benchmark_scorer_sha256": minimal.sha(Path(historical.__file__)),
        "study_config_sha256": minimal.sha(Path(core.__file__).parent / "configs/study.json"),
        "launcher_sha256": minimal.sha(Path(__file__)),
        **{key: minimal.sha(file) for key, file in check_files.items()},
        "seven_requests_sha256": lock["seven_requests_sha256"],
    }
    path = a.out / "FREEZE.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Official benchmark lock has changed")
    else:
        minimal.dump(path, plan)
    return path, plan


def squad_tokens(value):
    without_punctuation = value.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(?:a|an|the)\b", " ", without_punctuation).split()


def squad_score(prediction, references):
    predicted = squad_tokens(prediction)
    exact, f1 = False, 0.0
    for reference in references:
        gold = squad_tokens(reference)
        exact |= predicted == gold
        common = sum((Counter(predicted) & Counter(gold)).values())
        if common:
            f1 = max(f1, 2 * common / (len(predicted) + len(gold)))
    return {"standard_exact": exact, "standard_f1": f1}


def gsm_score(prediction, references):
    match = re.search(r"####\s*(-?[0-9.,]+)", prediction)
    value = match.group(1).replace(",", "") if match else None
    return {"standard_exact": value == references[0].replace(",", "")}


def standard_generation(a, plan):
    for dataset, mode, scorer in (("squad", "span", squad_score), ("gsm8k", "gsm_process", gsm_score)):
        source = a.panel / f"test_{dataset}.jsonl"
        rows = load(source)
        raw = a.out / f"{dataset}__{a.arm}__k4__{mode}.jsonl"
        stamp = read(raw.with_suffix(".json"))
        spec = dict(arm=a.arm, depth=4, source_sha256=minimal.sha(source),
                    freeze_sha256=minimal.sha(a.freeze), core_rows_sha256=stamp["rows_sha256"],
                    standard_runner_sha256=plan["launcher_sha256"])
        ids = [r["sample_id"] for r in rows]
        target = a.out / f"{dataset}__{a.arm}__k4__standard.jsonl"
        if core.verified(target, spec, ids):
            continue
        output, costs = core.verified(raw, stamp["plan"], ids)
        for row, item in zip(output, rows, strict=True):
            row["metrics"].update(scorer(row["prediction"], item["references"]))
        core.finish(target, output, spec, costs, stamp["runtime"])


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    a.freeze, plan = freeze(a)
    if a.block in ("all", "legacy"):
        from sys import path
        path[:0] = [str(a.unified), str(a.stage2 / "code")]
        import stage2_score
        import seven_unified

        # These registries also support historical few-shot runs. The archive
        # receipt and exact request sequence still have to match at scoring time.
        stage2_score.APPLE_BENCHMARKS = dict.fromkeys(stage2_score.APPLE_BENCHMARKS, 0)
        seven_unified.TASKS = {name: (0, n, metrics)
                               for name, (_, n, metrics) in seven_unified.TASKS.items()}
        a.arm = "new"
        a.minimal = a.minimal_code
        a.out = a.root / "legacy"
        for a.task in ("seven", "mmlu_pro"):
            historical.run(a)
    if a.block in ("all", "extra"):
        a.out = a.root / "extra"
        a.data, a.phase, a.depths, a.tasks = a.panel, "test", [4], list(EXTRA)
        for a.arm in ("new", "native", "t1", "t2", "mixed"):
            weight = plan["arms"][a.arm]["path"]
            a.checkpoint = Path(weight) if weight is not None else None
            core.run(a)
            standard_generation(a, plan)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("selection", "reference-freeze", "archive", "previous", "model",
                "panel", "stage2", "minimal-code", "unified", "followup", "historical",
                "out", "t1", "t2", "mixed"):
        p.add_argument(f"--{key}", required=True, type=Path)
    p.add_argument("--block", choices=("freeze", "legacy", "extra", "all"), default="all")
    a = p.parse_args()
    a.root = a.out.resolve()
    main(a)
