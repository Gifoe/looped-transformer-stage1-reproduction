"""Verify and package the complete paired runs of the three frozen Ouro transfers."""

import argparse
import gzip
from hashlib import file_digest
import json
from pathlib import Path
import shutil

from evaluate_transfer import ARM_NAMES, TASKS, T1_SHA


CODE = ("MODELS.json", "REPRODUCE.md", "REPORT_ZH.md", "evaluate_transfer.py",
        "report_transfer.py", "run_cross_models.sh", "fetch_ouro.py", "package_transfer.py",
        "plot_transfer.py", "publish_transfer.py")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pack(a):
    out = a.out.resolve()
    root = a.root.resolve()
    if out == root or out.is_relative_to(root / "results"):
        raise ValueError("Stage the release in a separate directory")
    out.mkdir(exist_ok=True, parents=True)
    models = read(root / "MODELS.json")
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

    for figure in (root / "figures").glob("*"):
        if figure.suffix in (".pdf", ".svg"):
            files[f"figures/{figure.name}"] = figure
    records = []
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
    pack(p.parse_args())
