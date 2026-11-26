"""Confirm the previous waveform closed and the assigned physical GPU has no process."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess

LOCK_SHA="326cb3f9d635a348e8e531828be1c9b7a5af8105729adcd0f8b42c4f45b57287"
POOLS={
    "old0":("quartz","swag","abductive_nli","reclor"),
    "dual0":("logiqa","medqa_usmle","aqua_rat","anli","scitail","mnli","ag_news"),
    "dual1":("obqa","race")
}
CARD={"old0":0,"dual0":0,"dual1":1}
SETS=("Top_RegimeIII","Top_RegimeII","Top_RegimeI","Top_Unsigned_Grad","Random_512")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def ready(root, slot):
    if sha(root/"prospective_unseen/FORECAST_LOCK_V4_PROSPECTIVE.json")!=LOCK_SHA:
        raise ValueError("The unseen-source forecast changed")
    for source in POOLS[slot]:
        try:
            if slot=="dual1":
                p=root/"exp_t3b_reversal"/f"T3B_{source}.json"
                values=read(p)
                arms={f"{source}.{cohort}.s{seed}" for cohort in SETS for seed in (26112711,26112712)}
                if values["source"]!=source or set(values["arms"])!=arms or set(values.get("summary",{}))!=set(SETS):
                    return False
            else:
                p=root/"prospective_unseen/trained"/f"TRAIN_{source}.json"
                values=read(p)
                arms={f"{source}.cold840.s{seed}" for seed in (26112721,26112722)}
                if (values["source"]!=source or values["lock_sha256"]!=LOCK_SHA
                        or set(values["arms"])!=arms or values.get("mean_delta_macro5_pp") is None):
                    return False
        except FileNotFoundError:
            return False
    possible=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    uuids={int(row.split(",")[0]):row.split(",")[1].strip() for row in possible.splitlines()}
    occupied=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True)
    if uuids[CARD[slot]] in occupied.splitlines():
        return False
    recipe=read(root/"handoff_codex/T5_TRAINING_LOCK.json")
    expected=recipe["runner_sha256"]
    if sha(root/"handoff_codex/v4_t5_locked.py")!=expected:
        raise ValueError("The pre-training recipe is not bound to the T5 runner")
    print("V4_WAVE3_GATE_PASSED",slot,sha(root/"handoff_codex/T5_TRAINING_LOCK.json"),flush=True)
    return True


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--slot",choices=tuple(CARD),required=True)
    args=p.parse_args()
    raise SystemExit(0 if ready(args.root,args.slot) else 3)
