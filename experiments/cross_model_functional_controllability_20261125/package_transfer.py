"""Verify and package the complete paired runs of the three frozen Ouro transfers."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import file_digest
from itertools import combinations
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from evaluate_transfer import ARM_NAMES, TASKS, T1_SHA
import plot_transfer
from report_transfer import mmlu_letter_audit


CODE = ("MODELS.json", "REPRODUCE.md", "REPORT_ZH.md", "evaluate_transfer.py",
        "report_transfer.py", "run_cross_models.sh", "fetch_ouro.py", "package_transfer.py",
        "plot_transfer.py", "publish_transfer.py")
HUGINN_CODE = ("HUGINN_RELEASE.json", "fetch_huginn.py", "huginn_control.py", "evaluate_huginn.py",
               "audit_transfer_mechanism.py", "audit_huginn_mechanism.py", "run_transfer_mechanism.sh",
               "run_huginn.sh", "run_cross_followup.sh", "LOOPCD_PAPER.json", "evaluate_loopcd.py", "run_loopcd.sh")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_map(path, expected):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    table = {row["id"]: row for row in rows}
    if len(table) != expected or len(rows) != expected:
        raise ValueError(f"Wrong unique question population in {path}")
    return table


def audit_populations(root, models):
    """Verify matching item content and compare actual answer changes across versions."""
    findings = {}
    identity_fields = ("gold", "choices", "chars")
    for task, (n, metric) in TASKS.items():
        reference, effects, origins = None, {}, {}
        subject_effects = {}
        for model in models:
            directory = root / "results" / model
            native, t1 = [prediction_map(directory / f"{task}__{arm}.jsonl", n) for arm in ARM_NAMES]
            identity = {key: tuple(row.get(name) for name in identity_fields) for key, row in native.items()}
            if reference is None:
                reference = identity
            transfer_identity = {key: tuple(row.get(name) for name in identity_fields) for key, row in t1.items()}
            if identity != reference or transfer_identity != reference:
                raise ValueError(f"Question identity or candidate text changed at {model}/{task}")
            effects[model] = {key: int(t1[key]["metrics"][metric]) - int(row["metrics"][metric])
                              for key, row in native.items()}
            origins[model] = {key: row["metrics"][metric] for key, row in native.items()}
            if task == "mmlu":
                by_subject = defaultdict(Counter)
                for key, delta in effects[model].items():
                    by_subject[key.split(":", maxsplit=1)[0]][delta] += 1
                subject_effects[model] = {subject: {"n": sum(counts.values()), "rescued": counts[1],
                                                     "harmed": counts[-1]}
                                          for subject, counts in sorted(by_subject.items())}
        patterns = Counter(tuple(effects[model][key] for model in models) for key in reference)
        findings[task] = {"n": n, "metric": metric,
                          "paired_change_counts": [{"signs": list(key), "n": count}
                                                   for key, count in sorted(patterns.items())]}
        comparisons = []
        for left, right in combinations(models, 2):
            record = {"models": [left, right]}
            for name, initially_correct, direction in (("rescues", False, 1), ("damage", True, -1)):
                eligible = (key for key in reference if origins[left][key] == origins[right][key] == initially_correct)
                square = Counter((effects[left][key] == direction, effects[right][key] == direction)
                                 for key in eligible)
                total = sum(square.values())
                record[name] = {"n_initially_matched": total, "both": square[True, True],
                                "left_only": square[True, False], "right_only": square[False, True],
                                "neither": square[False, False]}
            comparisons.append(record)
        findings[task]["overlap_with_same_native_correctness"] = comparisons
        if subject_effects:
            findings[task]["mmlu_subject_counts"] = subject_effects
            findings[task]["mmlu_answer_letter"] = {model: mmlu_letter_audit(root / "results" / model)
                                                   for model in models}
    return {"model_order": list(models), "sign_legend": {"-1": "harmed", "0": "unchanged", "1": "rescued"},
            "questions_matched_on": ["id", *identity_fields], "results": findings}


def loopcd_files(root, models):
    output = {}
    paper = read(root / "LOOPCD_PAPER.json")["published_zero_shot"]
    for model in models:
        folder = root / "results" / "loopcd" / model
        rows = read(folder / "RESULTS.json")
        if rows.keys() != TASKS.keys():
            raise ValueError(f"Incomplete LoopCD coverage for {model}")
        prefix = f"results/loopcd/{model}"
        for item in ("RESULTS.json", "FREEZE.json"):
            output[f"{prefix}/{item}"] = folder / item
        for task, (n, primary) in TASKS.items():
            row = rows[task]
            if row["primary"] != primary:
                raise ValueError(f"Wrong primary metric for LoopCD {model}/{task}")
            if task in paper.get(model, {}):
                if row["source"] != "paper_aggregate_only" or [
                    row["paper_native"], row["fixed"], row["adaptive"]
                ] != paper[model][task]:
                    raise ValueError(f"Wrong published zero-shot aggregate for {model}/{task}")
                continue
            if row["source"] != "current_matched":
                raise ValueError(f"Missing matched LoopCD evaluations for {model}/{task}")
            for arm, item in row["arms"].items():
                preds = folder / f"{task}__{arm}.jsonl"
                receipt = preds.with_suffix(".json")
                if (item["metrics"][primary]["versus_native"]["n"] != n or digest(preds) != item["rows_sha256"]
                        or read(receipt)["rows_sha256"] != item["rows_sha256"]):
                    raise ValueError(f"Unverified LoopCD result at {model}/{task}/{arm}")
                output[f"{prefix}/{preds.name}.gz"] = preds
                output[f"{prefix}/{receipt.name}"] = receipt
    return output


def pack(a):
    out = a.out.resolve()
    root = a.root.resolve()
    if out == root or out.is_relative_to(root / "results"):
        raise ValueError("Stage the release in a separate directory")
    out.mkdir(exist_ok=True, parents=True)
    models = read(root / "MODELS.json")
    if a.only:
        if not set(a.only) <= models.keys():
            raise ValueError("Unknown model in the release request")
        models = {name: models[name] for name in a.only}
    ouro_count = len(models)
    files = {name: root / name for name in (*CODE, *HUGINN_CODE)}
    mechanism = root / "results" / "mechanism"
    complete_huginn = a.with_huginn or a.huginn_benchmarks
    if complete_huginn or a.huginn_partial:
        models["huginn-0125"] = None
    if a.with_huginn:
        files.update(loopcd_files(root, models))
    if a.with_huginn or (a.only is None and (mechanism / "MECHANISM.json").is_file()):
        path = mechanism / "MECHANISM.json"
        measured = read(path)
        if measured["operator_sha256"] != T1_SHA or (measured["train"], measured["validation"]) != (1119, 24):
            raise ValueError("The closed Ouro mechanism summary has different data or a different controller")
        files["results/mechanism/MECHANISM.json"] = path
        for name in models.keys() - {"huginn-0125"}:
            folder = mechanism / name / "finite"
            artifact = folder / "finite.jsonl"
            if read(folder / "FINITE.json")["finite_sha256"] != digest(artifact):
                raise ValueError(f"Changed finite-dose rows for {name}")
            for artifact in ("MECHANISM_LOCK.json", "finite/FINITE.json", "finite/finite.jsonl"):
                files[f"results/mechanism/{name}/{artifact}"] = mechanism / name / artifact
    if digest(a.t1) != T1_SHA:
        raise ValueError("Historical T1 release identity differs")
    files["weights/rank1_t1.pt"] = a.t1
    historical = read(a.historical)
    if historical.get("completed_questions") != sum(n for n, _ in TASKS.values()):
        raise ValueError("Historical Base has a different population")
    reference, reference_loopcd = {}, {}
    for task, (count, metric) in TASKS.items():
        old = historical["results"][task]
        if old["n"] != count or old["metric"] != metric:
            raise ValueError(f"Historical Base protocol differs at {task}")
        reference[task] = old["paired"]["native"]
        reference_loopcd[task] = {arm: old["paired"][arm] for arm in ("fixed", "adaptive")}
    partial_huginn_tasks = []
    for name in models:
        directory = root / "results" / name
        results = read(directory / "RESULTS.json")
        partial = name == "huginn-0125" and a.huginn_partial
        if partial:
            if not results or not results.keys() <= TASKS.keys() or len(results) == len(TASKS):
                raise ValueError("A partial Huginn release requires at least one and fewer than eight closed benchmarks")
            partial_huginn_tasks.extend(task for task in TASKS if task in results)
        elif set(results) != TASKS.keys():
            raise ValueError(f"Missing official benchmarks for {name}")
        lock = read(directory / "FREEZE.json")
        metadata = a.models / name / "MODEL_LOCK.json"
        if name == "huginn-0125":
            fit = directory / "fit"
            controller = fit / "rank1_huginn.pt"
            receipt = read(fit / "FIT.json")
            if (lock["depth"] != 32 or lock["model_sha256"] != digest(metadata)
                    or lock["controller_sha256"] != digest(controller)
                    or lock["controller_sha256"] != receipt["checkpoint_sha256"]
                    or (receipt["steps"], receipt["examples"], receipt["trainable_params"]) != (280, 1119, 10560)):
                raise ValueError("Huginn training, model and evaluation receipts disagree")
            files["weights/rank1_huginn.pt"] = controller
            files[f"results/{name}/fit/FIT.json"] = fit / "FIT.json"
            if not partial:
                for artifact in ("CACHE_PARITY_native.json", "CACHE_PARITY_t1.json"):
                    files[f"results/{name}/{artifact}"] = directory / artifact
            if a.with_huginn or (directory / "mechanism" / "MECHANISM.json").is_file():
                for artifact in ("mechanism/FINITE.json", "mechanism/LOCK.json", "mechanism/MECHANISM.json"):
                    files[f"results/{name}/{artifact}"] = directory / artifact
        elif lock["arms"]["t1"] != T1_SHA or lock["model_lock_sha256"] != digest(metadata):
            raise ValueError(f"Weight or model receipt differs for {name}")
        for artifact in ("RESULTS.json", "FREEZE.json"):
            files[f"results/{name}/{artifact}"] = directory / artifact
        files[f"models/{name}/MODEL_LOCK.json"] = metadata
        for task, (n, metric) in TASKS.items():
            if task not in results and partial:
                continue
            row = results[task]
            if row["primary"] != metric or row["metrics"][metric]["n"] != n:
                raise ValueError(f"Incomplete paired population for {name}/{task}")
            for arm in ARM_NAMES:
                base = directory / f"{task}__{arm}.jsonl"
                receipt = base.with_suffix(".json")
                stamp = read(receipt)
                if digest(base) != row["predictions"][arm] or stamp["rows_sha256"] != row["predictions"][arm]:
                    raise ValueError(f"Changed model prediction rows in {base}")
                files[f"results/{name}/{base.name}.gz"] = base
                files[f"results/{name}/{receipt.name}"] = receipt

    if a.only is None:
        figure_dir = root / "figures"
        plot_transfer.run(SimpleNamespace(results=root / "results", historical=a.historical, out=figure_dir,
                                          with_huginn=complete_huginn))
        for figure in figure_dir.glob("*"):
            if figure.suffix in (".pdf", ".svg"):
                files[f"figures/{figure.name}"] = figure
    records = []
    audit = out / "PAIRED_POPULATION_AUDIT.json"
    matched_models = [name for name in models if not (a.huginn_partial and name == "huginn-0125")]
    audit.write_text(json.dumps(audit_populations(root, matched_models), indent=2) + "\n", encoding="utf-8")
    records.append({"path": audit.name, "bytes": audit.stat().st_size, "sha256": digest(audit)})
    base = out / "HISTORICAL_OURO_1.4B_BASE.json"
    base.write_text(json.dumps({"original_summary_sha256": digest(a.historical), "results": reference,
                                "loopcd_comparisons": reference_loopcd},
                               indent=2) + "\n", encoding="utf-8")
    records.append({"path": base.name, "bytes": base.stat().st_size, "sha256": digest(base)})
    for key, source in sorted(files.items()):
        dst = out / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if key.endswith(".jsonl.gz"):
            with source.open("rb") as original, dst.open("wb") as wrapped:
                with gzip.GzipFile(filename="", mode="wb", fileobj=wrapped, mtime=0) as compressed:
                    shutil.copyfileobj(original, compressed)
        else:
            shutil.copyfile(source, dst)
        records.append({"path": key, "bytes": dst.stat().st_size, "sha256": digest(dst), "original_sha256": digest(source)})
    manifest = {"complete_ouro_models": ouro_count, "complete_huginn": complete_huginn,
                "loopcd_supplement_complete": a.with_huginn,
                "partial_huginn_tasks": partial_huginn_tasks, "official_benchmarks": len(matched_models) * len(TASKS),
                "benchmark_questions_per_arm": sum(n for n, _ in TASKS.values()),
                "measurement_arms": list(ARM_NAMES), "t1_sha256": T1_SHA, "files": records}
    stage_manifest = out / "RELEASE_MANIFEST.partial"
    stage_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    stage_manifest.replace(out / "RELEASE_MANIFEST.json")
    print("PACKAGED", len(records), "verified release files", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "models", "t1", "historical", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--only", nargs="+")
    variants = p.add_mutually_exclusive_group()
    variants.add_argument("--with-huginn", action="store_true")
    variants.add_argument("--huginn-benchmarks", action="store_true")
    variants.add_argument("--huginn-partial", action="store_true")
    args = p.parse_args()
    if (args.with_huginn or args.huginn_benchmarks or args.huginn_partial) and args.only:
        p.error("Publish all three Ouro comparisons with the Huginn release")
    pack(args)
