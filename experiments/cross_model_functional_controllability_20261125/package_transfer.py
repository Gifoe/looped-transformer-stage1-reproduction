"""Verify and package the complete paired runs of the three frozen Ouro transfers."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from evaluate_transfer import ARM_NAMES, TASKS, T1_SHA
import plot_transfer


CODE = ("MODELS.json", "REPRODUCE.md", "REPORT_ZH.md", "evaluate_transfer.py",
        "report_transfer.py", "run_cross_models.sh", "fetch_ouro.py", "package_transfer.py",
        "plot_transfer.py", "publish_transfer.py")


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
        reference, effects = None, {}
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
        if subject_effects:
            findings[task]["mmlu_subject_counts"] = subject_effects
    return {"model_order": list(models), "sign_legend": {"-1": "harmed", "0": "unchanged", "1": "rescued"},
            "questions_matched_on": ["id", *identity_fields], "results": findings}


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
    files = {name: root / name for name in CODE}
    if digest(a.t1) != T1_SHA:
        raise ValueError("Historical T1 release identity differs")
    files["weights/rank1_t1.pt"] = a.t1
    historical = read(a.historical)
    if historical.get("completed_questions") != sum(n for n, _ in TASKS.values()):
        raise ValueError("Historical Base has a different population")
    reference = {}
    for task, (count, metric) in TASKS.items():
        old = historical["results"][task]
        if old["n"] != count or old["metric"] != metric:
            raise ValueError(f"Historical Base protocol differs at {task}")
        reference[task] = old["paired"]["native"]
    for name in models:
        directory = root / "results" / name
        results = read(directory / "RESULTS.json")
        if set(results) != TASKS.keys():
            raise ValueError(f"Missing official benchmarks for {name}")
        lock = read(directory / "FREEZE.json")
        metadata = a.models / name / "MODEL_LOCK.json"
        if lock["arms"]["t1"] != T1_SHA or lock["model_lock_sha256"] != digest(metadata):
            raise ValueError(f"Weight or model receipt differs for {name}")
        for artifact in ("RESULTS.json", "FREEZE.json"):
            files[f"results/{name}/{artifact}"] = directory / artifact
        files[f"models/{name}/MODEL_LOCK.json"] = metadata
        for task, (n, metric) in TASKS.items():
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
        plot_transfer.run(SimpleNamespace(results=root / "results", historical=a.historical, out=figure_dir))
        for figure in figure_dir.glob("*"):
            if figure.suffix in (".pdf", ".svg"):
                files[f"figures/{figure.name}"] = figure
    records = []
    audit = out / "PAIRED_POPULATION_AUDIT.json"
    audit.write_text(json.dumps(audit_populations(root, models), indent=2) + "\n", encoding="utf-8")
    records.append({"path": audit.name, "bytes": audit.stat().st_size, "sha256": digest(audit)})
    base = out / "HISTORICAL_OURO_1.4B_BASE.json"
    base.write_text(json.dumps({"original_summary_sha256": digest(a.historical), "results": reference},
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
    manifest = {"complete_ouro_models": len(models), "official_benchmarks": len(models) * len(TASKS),
                "benchmark_questions_per_arm": sum(n for n, _ in TASKS.values()),
                "measurement_arms": list(ARM_NAMES), "t1_sha256": T1_SHA, "files": records}
    (out / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("PACKAGED", len(records), "verified release files", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "models", "t1", "historical", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--only", nargs="+")
    pack(p.parse_args())
