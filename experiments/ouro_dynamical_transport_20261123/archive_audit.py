"""Audit archived token moments and matched transport before new interventions."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(file):
    return json.loads(file.read_text(encoding="utf-8"))


def dump(file, payload):
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(file)


def weights(file):
    p = torch.load(file, weights_only=True, map_location="cpu")["state"]
    return p["out.weight"].flatten().double(), p["encode.weight"].flatten().double()


def decompose(moment, arms):
    raw = moment["matrix"].double()
    # `mean_gradient` is a per-question sum, `mean_state` a mean over tokens.
    # Their outer product already equals (T/N) * mean_token(g) mean_token(h)^T.
    mean = torch.outer(moment["mean_gradient"].double(), moment["mean_state"].double())
    cov = raw - mean
    divisor = float(raw.square().sum())
    top_u, top_w = (moment[key][:, 0].double() for key in ("u", "w"))
    norm = float(2*(cov*mean).sum()/divisor)
    probes = {}
    for name, (u, w) in {"archived_top": (top_u, top_w), **arms}.items():
        probes[name] = {k:float(u @ (m @ w)) for k,m in (("raw",raw),("mean",mean),("centered",cov))}
    return dict(n=moment["n"], tokens=moment["token_count"], mean_energy=float(mean.square().sum()/divisor),
                centered_energy=float(cov.square().sum()/divisor), cross_energy=norm, projections=probes,
                top1_archive_energy=float(moment["s"][0].square() / divisor))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--t1", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    results = a.archive/"results"
    arms = {"t1":weights(a.t1)}
    for label, ckpt in {
        "arc_k2":"arc/depth/s261123137/k2/controller.pt",
        "mixed":"mixed/source/s261123137/k234/controller.pt",
        "qasc":"qasc/source/s261123137/k234/controller.pt"}.items():
        arms[label] = weights(results/"HOLDOUT"/ckpt)
    details = {}
    for dataset in ("arc", "csqa", "qasc", "obqa", "mixed"):
        directory = results/"moments"/dataset
        details[dataset] = {}
        for condition, meta in read(directory/"MOMENTS.json")["conditions"].items():
            file = directory/f"{condition}.pt"
            if sha(file) != meta["matrix_sha256"]:
                raise RuntimeError(f"Published moment differs: {file}")
            value = decompose(torch.load(file, weights_only=True, map_location="cpu"), arms)
            value["sha256"] = meta["matrix_sha256"]
            details[dataset][condition] = value
            print(dataset, condition, round(value["mean_energy"], 4), round(value["centered_energy"], 4), flush=True)
    transport = read(results/"D_TRANSPORT/arc_confirm/TRANSPORT_SUMMARY.json")
    empirical = {key:{"n":transport["n"], "all_candidates_matched":value["questions_with_all_candidates_dose_matched"],
                      "late_minus_early":value["contrasts"]["late_minus_early"]["utility"],
                      "matched_late_minus_early":value["contrasts"]["late_matched_minus_early_matched"]["utility"],
                      "match_early":value["conditions"]["early_matched"]["gradient_vs_observed"],
                      "match_late":value["conditions"]["late_matched"]["gradient_vs_observed"]}
                 for key,value in transport["results"].items()}
    old = read(results/"ROUND2_AUDIT/RESULTS.json")
    failed = [row for row in old["per_cell"] if row["log_odds"]<0]
    horizons = read(results/"state_horizon/arc_confirm/STATE_HORIZON.json")
    archived = {k:{arm:{metric:rec[metric] for metric in ("utility","acc_norm")} for arm,rec in v["result"].items()}
                for k,v in horizons["conditions"].items()}
    a.out.mkdir(parents=True, exist_ok=True)
    dump(a.out/"ARCHIVE_AUDIT.json",dict(cells=details, prior_finite_transport=empirical,
                                       prior_state_horizon=archived, prior_negative_transfer=failed,
                                       source_commit="e4d2d219a995b0380b875da547f2db3f40353e8d",
                                       archive_note="Archived states and gradient means are sufficient for centering; "
                                                    "no question-paired g/h vectors were retained here for a permutation."))


if __name__ == "__main__":
    main()
