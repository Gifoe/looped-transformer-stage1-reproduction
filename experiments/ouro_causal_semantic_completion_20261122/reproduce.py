"""Replay the frozen Semantic Completion protocol, with a shared GPU time limit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


MODEL_SHA = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
PHASES = {"native": "csc_data.py", "bank": "csc_bank.py", "oracle": "csc_oracle.py",
          "learned": "csc_learn.py", "lookup": "csc_lookup.py", "diagnostics": "csc_diagnostics.py",
          "template": "csc_template.py", "formation": "csc_formation.py", "lookup_late": "csc_lookup_late.py"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archived_environment(experiment):
    declaration = next(line for line in (experiment / "env.sh").read_text().splitlines()
                       if line.startswith("export PYTHONPATH="))
    paths = []
    for path in declaration.split("=", 1)[1].split(":"):
        name = Path(path).name.replace("-", "_")
        if name == "ouro_native_state_20261121":
            name = "ouro_native_state_computation_20261121"
        dep = experiment.parent / name
        if not dep.is_dir():
            raise FileNotFoundError(f"Archived dependency missing: {dep}")
        if dep not in paths:
            paths.append(dep)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="0")
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    return env


def ledger_used(ledger):
    if not ledger.exists():
        return 0
    lines = ledger.read_text().splitlines()
    if lines and not lines[-1].startswith("END\t"):
        raise RuntimeError("GPU ledger contains an unfinished stage")
    return sum(int(row.split("\t")[4]) for row in lines if row.startswith("END\t"))


def run_phase(stage, experiment, model, root, env):
    ledger = root / "GPU_LEDGER.tsv"
    left = 18000 - ledger_used(ledger)
    if left <= 0:
        raise RuntimeError("Five-hour GPU budget exhausted")
    start = int(time.time())
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(f"START\t{stage}\t{start}\t{left}\n")
    code = 1
    cmd = [sys.executable, "-u", str(experiment / PHASES[stage]), str(model), str(root)]
    try:
        with (root / f"{stage}.log").open("ab") as stream:
            code = subprocess.run(cmd, env=env, cwd=root, timeout=left, stdout=stream,
                                  stderr=subprocess.STDOUT).returncode
    except subprocess.TimeoutExpired:
        code = 124
    finally:
        end = int(time.time())
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(f"END\t{stage}\t{start}\t{end}\t{end-start}\t{code}\n")
    if code != 0:
        raise RuntimeError(f"{stage} failed ({code}); see {root / (stage + '.log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--verify-artifacts", action="store_true")
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    deck = experiment / "results" / "DECK.json"
    if args.verify_artifacts:
        entries = (experiment / "ARTIFACTS.sha256").read_text().splitlines()
        for entry in entries:
            expected, name = entry.split("  ", 1)
            if sha(experiment / name) != expected:
                raise ValueError(f"Artifact hash mismatch: {name}")
        print(f"Verified {len(entries)} archived artifacts")
        return
    if not args.checkpoint or not args.output:
        parser.error("checkpoint and output are required for replay")
    model, root = args.checkpoint.resolve(), args.output.resolve()
    if root == experiment or experiment in root.parents:
        parser.error("output must be outside the archived experiment")
    if sha(model / "model.safetensors") != MODEL_SHA:
        parser.error("checkpoint SHA-256 differs from the frozen Ouro-1.4B base")
    verified = json.loads((experiment / "results" / "VERIFICATION.json").read_text())
    for name, expected in verified["checkpoint_components"].items():
        if sha(model / name) != expected:
            parser.error(f"checkpoint component SHA-256 differs: {name}")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "DECK.json"
    if destination.exists():
        if sha(destination) != sha(deck):
            parser.error("output contains a different deck")
    else:
        shutil.copyfile(deck, destination)
    env = archived_environment(experiment)
    for stage, final in (("native", "NATIVE_SUMMARY.json"), ("bank", None),
                         ("oracle", "ORACLE_SUMMARY.json")):
        if final is None or not (root / final).exists():
            run_phase(stage, experiment, model, root, env)
    oracle = json.loads((root / "ORACLE_SUMMARY.json").read_text())
    if not (root / "ORACLE_COORDS.pt").exists():
        subprocess.run([sys.executable,str(experiment/"export_oracle.py"),str(root)],env=env,check=True)
    complete = all((root / f"LEARNED_SUMMARY_K{k}.json").exists() for k in (1, 2))
    if oracle["gate"]["full"] and not complete:
        run_phase("learned", experiment, model, root, env)
    if oracle["gate"]["full"]:
        source = experiment / "results" / "LOOKUP_DECK.json"
        destination = root / "LOOKUP_DECK.json"
        if not destination.exists():
            shutil.copyfile(source, destination)
        elif sha(destination) != sha(source):
            parser.error("output contains a different lookup deck")
        if not (root / "LOOKUP_SUMMARY.json").exists():
            run_phase("lookup", experiment, model, root, env)
        if not (root / "OPERATOR_DIAGNOSTIC.json").exists():
            run_phase("diagnostics", experiment, model, root, env)
        early = json.loads((root / "LEARNED_SUMMARY_K1.json").read_text())
        if not early["gate"]["passes"] and not (root / "TEMPLATE_SUMMARY.json").exists():
            run_phase("template", experiment, model, root, env)
        if not (root / "FORMATION_SUMMARY.json").exists():
            run_phase("formation", experiment, model, root, env)
        if not (root / "LOOKUP_LATE_SUMMARY.json").exists():
            run_phase("lookup_late", experiment, model, root, env)


if __name__ == "__main__":
    main()
