"""Assemble a reproducible release without redistributing the upstream datasets."""

import argparse
from hashlib import file_digest
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import shutil


CLOSED = ("TRAIN.json", "TRAIN_LOCK.json", "FULL_COVERAGE.json", "TANGENT_LOCK.json", "controller.pt")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def copy_artifact(root, path, out, index):
    relative = path.relative_to(root)
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        target = target.with_suffix(".jsonl.gz")
        with path.open("rb") as source, target.open("wb") as store:
            with gzip.GzipFile(filename="", mode="wb", fileobj=store, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
    else:
        shutil.copyfile(path, target)
    index.append({"path": target.relative_to(out).as_posix(), "bytes": target.stat().st_size,
                  "sha256": digest(target), "input_sha256": digest(path)})


def official(root):
    report = json.loads((root / "results/official/RESULTS.json").read_text(encoding="utf-8"))
    selected = root / "results/official/FREEZE.json"
    if report["freeze_sha256"] != digest(selected):
        raise ValueError("Official report is not bound to the same checkpoint freeze")
    if len(report["primary"]) != 12 or len({t["task"] for t in report["tasks"]}) != 12:
        raise ValueError("Do not package incomplete official performance as final")
    files = [selected]
    files.extend((root / "results/official").glob("RESULTS.*"))
    for part in ("extra", "legacy"):
        for data in sorted((root / "results/official" / part).glob("*.jsonl")):
            receipt = data.with_suffix(".json")
            plan = json.loads(receipt.read_text(encoding="utf-8"))
            if plan["rows_sha256"] != digest(data) or plan["plan"]["freeze_sha256"] != report["freeze_sha256"]:
                raise ValueError(f"Official prediction receipt differs for {data}")
            files.extend((data, receipt))
    return files


def run(a):
    root = a.root.resolve()
    out = a.out.resolve()
    if out == root or any(out.is_relative_to(root / name) for name in ("results", "logs", "data", "figures")):
        raise ValueError("Package into a separate release staging path")
    out.mkdir(parents=True, exist_ok=True)
    files = official(root)
    files.extend(path for path in root.iterdir() if path.suffix in (".md", ".py", ".sh") and path.is_file())
    files.append(root / "EARLY_MECHANISM.json")
    files.extend((root / "data/pools_complete").glob("*.json"))
    files.extend((root / "data/reading").glob("READING_INPUTS.json"))
    files.extend((root / "data/pools_complete").glob("*_excluded.jsonl"))
    files.extend((root / "results").glob("*.json"))
    for folder in (root / "results").glob("forecast_*"):
        files.extend(folder.glob("*.json"))
        files.extend(folder.glob("DIRECTIONS.pt"))
        files.extend((folder / "confirmation").glob("*.json*"))
    files.extend((root / "results").glob("filtered_moments/*/*/k4/LOCK.json"))
    files.extend((root / "results").glob("moments/*/*/k*/LOCK.json"))
    for name in CLOSED:
        files.extend((root / "results/fits").glob(f"*/{name}"))
    for part in (root / "results").glob("dev*"):
        if part.is_dir():
            files.extend(part.glob("*.json*"))
    files.extend((root / "results/finite").glob("*/*.json*"))
    files.extend((root / "figures").glob("**/*.pdf"))
    files.extend((root / "figures").glob("**/*.svg"))
    files.extend(root.glob("run_*.log"))
    files.extend((root / "logs").glob("*.log"))
    index = []
    for path in sorted(set(files)):
        copy_artifact(root, path, out, index)
    dense_roots = (root / "results/filtered_moments", root / "results/moments")
    dense = [path for folder in dense_roots for path in sorted(folder.glob("**/fold_*.pt"))]
    matrix_manifest = out / "MATRIX_FILES.json"
    matrix_manifest.write_text(json.dumps(
        [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
         for path in dense], indent=2
    ) + "\n", encoding="utf-8")
    index.append({"path": matrix_manifest.name, "bytes": matrix_manifest.stat().st_size,
                  "sha256": digest(matrix_manifest)})
    settings = {name: version(name) for name in ("torch", "transformers", "pyarrow", "numpy", "scipy", "matplotlib")}
    manifest = {"official_benchmarks": 12, "environment": settings, "files": index}
    (out / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
