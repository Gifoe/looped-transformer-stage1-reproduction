"""Freeze the main study before any confirmation forward."""
import argparse
import json
from pathlib import Path

from collect import digest
from intervene import SITES

STUDY_BASE="ce0ee7d98a35fb284b9af41ed6014bfe0388a2ef"

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("study",type=Path)
    a=parser.parse_args()
    s=a.study
    result={"input_sha256":digest(s/"inputs.jsonl"),
        "intervention_sha256":digest(Path(__file__).with_name("intervene.py")),
        "calibration_sha256":digest(s/"calibration"/"rows.jsonl"),
        "base_commit":STUDY_BASE,"sites":SITES,
        "unit":"independent group; reciprocal directions averaged before 30000 percentile bootstrap resamples",
        "main":"K4 natural/irrelevant/random difference on each predeclared task payload, 120 per task",
        "A":"90 fresh owner swap worlds with exact same token ids at both payload spans; distinguish same-token contextual perturbation vs owner tokens; primary K4",
        "B":"90 fresh K1/L5 source patch worlds with native K2/K3 edge masks, K1 boundary erasure, and K2 restoration",
        "C":"90 fresh groups each for rule/operand positions, invariant bridge positions, and whole first-stage prefix",
        "primary_ci":1-.05/3,
        "natural_competence_assess":"same options across all four factorial worlds; joint correctness reported by family"}
    with (s/"frozen_protocol.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
