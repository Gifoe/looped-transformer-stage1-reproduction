"""Audit all sources under the unchanged V4 pre-training forecast and distinguish historical fitting evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from score_forecast import prospective_metrics
from v4_common import save_json,sha256_file

PIN="326cb3f9d635a348e8e531828be1c9b7a5af8105729adcd0f8b42c4f45b57287"
V3=Path("/root/ouro-v3-controllability-20261126/stage4_d3")
SEEDS=(26112721,26112722)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def result(names, pred, actual):
    truth=np.asarray([actual[name] for name in names])
    model=np.asarray([pred[name] for name in names])
    info=prospective_metrics(model,truth,names,26112735)
    top=info["predicted_ranking"][:3]
    best=info["actual_ranking"][:3]
    info["top3_first_choice_regret_pp"]=max(actual.values())-max(actual[name] for name in top)
    info["top3_mean_regret_pp"]=sum(actual[name] for name in best)/len(best)-sum(actual[name] for name in top)/len(top)
    return info


def audit(root):
    base=root/"prospective_unseen"
    forecast=base/"FORECAST_LOCK_V4_PROSPECTIVE.json"
    if sha256_file(forecast)!=PIN or forecast.with_suffix(".sha256").read_text().split()[0]!=PIN:
        raise ValueError("The new 11-source forecast differs from its registration")
    authority=read(forecast)
    if len(authority["sources"])!=11 or not authority["strict_pre_training_lock"]:
        raise ValueError("The full set of prospectively declared sources is required")
    train=base/"trained"
    ready, outcomes,cloze={}, {}, {}
    for source in authority["sources"]:
        locked=authority["per_source"][source]
        pool=root/"prospective_unseen6/pools"/f"{source}.jsonl"
        if sha256_file(pool)!=locked["pool_sha256"]:
            raise ValueError(f"The trained source pool was modified after registration: {source}")
        path=train/f"TRAIN_{source}.json"
        record=read(path)
        expected={f"{source}.cold840.s{seed}" for seed in SEEDS}
        if record["lock_sha256"]!=PIN or set(record["arms"])!=expected:
            raise ValueError("The source lacks both original seeds")
        weights={}
        for tag,arm in record["arms"].items():
            checkpoint=train/f"{tag}.pt"
            payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
            qual=payload["meta"]
            if qual["lock_sha256"]!=PIN or qual["arm"]!=tag or qual["steps"]!=840 or qual["source"]!=source:
                raise ValueError("A source weight did not use the locked 840-step recipe")
            if sum(v.numel() for v in payload["state"].values())!=4096:
                raise ValueError("The controller does not contain exactly 4096 parameters")
            c=sha256_file(checkpoint)
            if arm["train"].get("sha256",c)!=c:
                raise ValueError("A claimed seed checkpoint has a different fingerprint")
            registered=(arm["eval"]["macro5"]-record["native"]["macro5"])*100
            if abs(registered-arm["delta_macro5_pp"])>1e-10:
                raise ValueError("The source macro was not based on the archived matched native")
            weights[tag]=c
        outcomes[source]=float(np.mean([arm["delta_macro5_pp"] for arm in record["arms"].values()]))
        cloze[source]=float(np.mean([arm["delta_cloze3_pp"] for arm in record["arms"].values()]))
        if abs(outcomes[source]-record["mean_delta_macro5_pp"])>1e-10:
            raise ValueError("Source ordering is inconsistent with the two folded seeds")
        ready[source]=dict(pool_sha256=locked["pool_sha256"],training_receipt_sha256=sha256_file(path),weights=weights)
    names=sorted(outcomes)
    attributes={name:{s:authority["per_source"][s]["predictors"][name] for s in names}
                for name in authority["per_source"][names[0]]["predictors"]}
    prospective={metric:{"macro5":result(names,values,outcomes),"cloze3":result(names,values,cloze)}
                 for metric,values in attributes.items()}
    candidate_diagnostics=root/"cfit_14_features/CFIT_14_DIAGNOSTIC_SPEARMAN.json"
    history=read(candidate_diagnostics)
    earlier=read(V3/"FORECAST_LOCK_V3.json")
    assembled={source:dict(history["per_source"][source],**earlier["per_source"][source]["features"]) for source in history["per_source"]}
    cols=set.intersection(*(set(body) for body in assembled.values()))
    cols={key for key in cols if all(isinstance(row[key],(int,float)) and np.isfinite(row[key]) for row in assembled.values())}
    panels={p:sorted(group) for p,group in earlier["partition_groups"].items()}
    panels["all"]=sorted(assembled)
    retrospective={}
    for group, roster in panels.items():
        retrospect={}
        for cond in ("gt_cold","gt_warm"):
            truth={source:history[cond][source] for source in roster}
            retrospect[cond]={key:result(roster,{source:assembled[source][key] for source in roster},truth) for key in sorted(cols)}
        retrospective[group]=retrospect
    pack=dict(lock_sha256=PIN,n_completed=11,verified_sources=ready,actual_macro5_pp=outcomes,actual_cloze3_pp=cloze,
              predictor_audit=prospective,paired_bootstrap_source_draws=10000,
              retrospective_14=retrospective,retrospective_source_sha256=sha256_file(candidate_diagnostics),
              interpretation="The known 14-source outcomes were already available during V4 development. Only 11 new sources have prospective V4 claims.")
    target=base/"PROSPECTIVE_UNSEEN11_STRICT_AUDIT.json"
    if target.exists() and read(target)!=pack:
        raise ValueError("An independently verified source audit changed")
    if not target.exists():
        save_json(target,pack)
    print("V4_11_SOURCE_AUDIT_COMPLETE",sha256_file(target),flush=True)
    for key,value in prospective.items():
        print(key,"rho",value["macro5"]["spearman"],"top3_hits",value["macro5"]["top3_matches"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    audit(p.parse_args().root)
