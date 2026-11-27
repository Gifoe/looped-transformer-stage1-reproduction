"""Freeze V4 development winners, then run and pair the historical K4 protocols."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

import aggregate_v3 as pairing
import official_v3 as official
from d2_worker import digest, save_json
from v4_common import E0_CONFIG_PATH, load_controller_weights

ALIASES=("Best_Rank1_4096","Best_NonARC_4096","Best_ZeroLabel_4096")
BASELINES=official.BASELINE_ARMS
LEGACY=("arc_challenge","arc_easy","sciq","mmlu","hellaswag","winogrande","piqa","mmlu_pro")
EXTRA=("csqa","obqa","squad","gsm8k")
official.ALIASES=ALIASES
pairing.ALIASES=ALIASES
CFIT=("T5_CFIT_BLUE_All12","T5_CFIT_Top6_4096","T5_CFIT_Top6_Trust_4096",
      "T5_CFIT_Top6_PlusMMLUCloze","T5_CFIT_NonARC_4096")
ALL=CFIT+("T5_Uniform_All12","T5_Poison_Raw","T5_Poison_CFIT_Rescued","T5_Poison_MedOnly_Raw")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_pinned(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.is_file():
        partial=target.with_suffix(".partial")
        shutil.copyfile(source,partial)
        partial.replace(target)
    pin=digest(target)
    if pin!=digest(source):
        raise ValueError(f"Copied inputs differ: {target}")
    return pin


def selection(work,root):
    seal=root/"SELECTION_V4.json"
    signature=seal.with_suffix(".sha256")
    if seal.is_file():
        official.checked_selection(seal)
        return
    root.mkdir(parents=True,exist_ok=True)
    lock=work/"handoff_codex/T5_TRAINING_LOCK.json"
    computed=digest(lock)
    partial=work/"prospective_unseen/PROSPECTIVE_UNSEEN11_STRICT_AUDIT.json"
    prior=read(partial)
    if prior["n_completed"]!=11 or prior["lock_sha256"]!="326cb3f9d635a348e8e531828be1c9b7a5af8105729adcd0f8b42c4f45b57287":
        raise ValueError("The eleven-source locked prospect has not closed")
    reversal=list(work.glob("exp_t3b_reversal/T3B_*.json"))
    if {p.stem.removeprefix("T3B_") for p in reversal}!={"medmcqa","qasc","obqa","race"} or any(
        len(read(p)["arms"])!=10 or len(read(p).get("summary",{}))!=5 for p in reversal):
        raise ValueError("The four predeclared reversal sources must have all ten arms")
    out=work/"exp_t5_mixing"
    capacity=out/"CAPACITY_CERTIFICATE.json"
    if read(capacity)["expansion_triggered"]:
        expansion=out/"CAPACITY_EXPANSION_DEVELOPMENT.json"
        if not expansion.is_file():
            raise ValueError("Capacity gate triggered; complete the two declared expansion ablations before official scoring")
    candidates={}
    for name in ALL:
        path=out/f"T5_{name}.json"
        record=read(path)
        if record["lock_sha256"]!=computed or len(record["seeds"])!=2:
            raise ValueError(f"Missing equal-budget controls: {path}")
        best=max(record["seeds"],key=lambda arm:(record["seeds"][arm]["protected_macro5"],arm))
        seed=record["seeds"][best]
        weights=out/f"{best}.pt"
        if digest(weights)!=seed["train"]["sha256"] or seed["train"]["n_params"]!=4096:
            raise ValueError("The evaluated controller is not the locked Rank-1 model")
        candidates[name]=(record["mean_protected_macro5"],best,path)
    winner=max(CFIT,key=lambda arm:(candidates[arm][0],arm))
    if winner=="T5_CFIT_NonARC_4096":
        winner=max((name for name in CFIT if name!=winner),key=lambda name:(candidates[name][0],name))
    self_summary=work/"exp_t2a_self_distill/T2A_SELF_DISTILL.json"
    self_report=read(self_summary)
    self_anchor=self_report["frozen_self_anchor"]["arm"]
    anchor=torch.load(work/"V4_SELF_ANCHOR.pt",map_location="cpu",weights_only=False)
    if anchor["human_labels_used"] or anchor["source_arm"]!=self_anchor:
        raise ValueError("The selected T2 controller must be the pre-existing zero-label anchor")
    self_checkpoint=self_summary.parent/f"{self_anchor}.pt"
    u,w=load_controller_weights(self_checkpoint)
    torch.testing.assert_close(u,anchor["u_self"].double().flatten(),rtol=0,atol=0)
    torch.testing.assert_close(w,anchor["w_self"].double().flatten(),rtol=0,atol=0)
    frozen={}
    for alias,name in zip(ALIASES[:2],(winner,"T5_CFIT_NonARC_4096"),strict=True):
        metric,tag,dev=candidates[name]
        frozen[alias]=(out/f"{tag}.pt",dev,out/f"{tag}.TRAIN.json",tag,metric)
    frozen[ALIASES[2]]=(self_checkpoint,self_summary,self_summary,self_anchor,None)
    final={}
    for alias,(ck,dev,train,tag,metric) in frozen.items():
        pieces={"controller":ck,"development":dev,"training":train}
        record=dict(selected_arm=tag,n_params=4096,protected_development_macro=metric)
        for kind,source in pieces.items():
            destination=root/"selection_inputs"/alias/(kind+source.suffix)
            record[kind]=str(destination)
            record[kind+"_sha256"]=copy_pinned(source,destination)
        record.update(historical=record["development"],historical_sha256=record["development_sha256"])
        final[alias]=record
    value=dict(status="FROZEN_BEFORE_OFFICIAL_EVALUATION",development_only=True,
               frozen_utc=datetime.now(timezone.utc).isoformat(),candidates=final,gate_sha256=digest(capacity),
               env_sha256=read(E0_CONFIG_PATH)["approved_env_sha256"],t5_recipe_sha256=computed,
               prospective_audit_sha256=digest(partial),
               first_alias_rule="highest seed-mean protected macro among general CFIT Rank-1 arms, separate strict NonARC",
               t2_rule="use previously frozen unlabeled-validation anchor; no official or dev re-selection",
               heldout_mmlu="158 independent development questions for selection across all T5 arms")
    save_json(seal,value)
    signature.write_text(f"{digest(seal)}  {seal.name}\n",encoding="utf-8")
    official.checked_selection(seal)
    print("V4_OFFICIAL_SELECTION_FROZEN",digest(seal),{k:v["selected_arm"] for k,v in final.items()},flush=True)


def run_one(root,alias,phase,task,arm):
    return official.main(SimpleNamespace(
        selection=root/"SELECTION_V4.json",config=E0_CONFIG_PATH,out=root/"official",
        alias=alias,phase=phase,task=task,arm=arm))


def audit(root):
    from aggregate_full import legacy_rows
    from run_full import LEGACY as N_LEGACY, EXTRA as N_EXTRA
    import numpy as np

    seal=root/"SELECTION_V4.json"
    auth=official.checked_selection(seal)
    launch=SimpleNamespace(selection=seal,config=E0_CONFIG_PATH,out=root/"official")
    out=launch.out
    paired,table={},{}
    for alias in auth["candidates"]:
        launch.alias=alias
        runtime=official.arguments(launch,auth)
        runtime.release=runtime.panel.parents[1]
        runtime.minimal=runtime.minimal_code
        receipt=read(runtime.out/"FREEZE.json")
        methods={}
        for task,n in (N_LEGACY|N_EXTRA).items():
            mode="generated" if task=="mmlu_pro" else ("standard" if task in ("gsm8k","squad") else "candidate")
            modes=(mode,"content","letter") if task in ("csqa","obqa") else (mode,)
            for detail in modes:
                rows,clusters=(legacy_rows(runtime,task,n) if task in N_LEGACY else
                              pairing.extra(runtime,__import__("aggregate_full"),out/ALIASES[0],task,n,detail))
                report=pairing.summarize(task,rows,clusters)
                methods[f"{task}/{detail}"]=report
                if mode==detail:
                    metric=receipt["primary"][task]
                    scores={arm:report["paired"][arm][metric]["baseline"] for arm in BASELINES}
                    current=table.setdefault(task,dict(n=n,metric=metric,scores=scores))
                    if any(scores[arm]!=current["scores"][arm] for arm in BASELINES):
                        raise ValueError("Historical scores changed between checkpoints")
                    current["scores"][alias]=report["paired"]["native"][metric]["new"]
        paired[alias]=methods
    model_names=(*BASELINES,*ALIASES)
    macro={section:{arm:float(np.mean([table[task]["scores"][arm] for task in names]))
                     for arm in model_names}
           for section,names in (("legacy_eight",LEGACY),("official_twelve",LEGACY+EXTRA))}
    save_json(out/"FINAL_8PLUS4_TABLE.json",dict(table=table,macro=macro,
                    historical_count=sum(N_LEGACY.values()),total_count=sum((N_LEGACY|N_EXTRA).values()),
                    selection_sha256=digest(seal)))
    save_json(out/"PAIRED_SIGNIFICANCE.json",dict(models=paired,selection_sha256=digest(seal),paired_draws=10000))
    heading=["Benchmark","N","Metric",*model_names]
    rows=["# V4 固定 K4 全量官方评测","",
          "| "+" | ".join(heading)+" |","| "+" | ".join(["---"]*len(heading))+" |"]
    for name,record in table.items():
        sizes=[name,str(record["n"]),record["metric"],*(f"{record['scores'][a]*100:.2f}%" for a in model_names)]
        rows.append("| "+" | ".join(sizes)+" |")
    for name,values in macro.items():
        rows.append("| "+" | ".join([name,"—","macro",*(f"{values[a]*100:.2f}%" for a in model_names)])+" |")
    target=out/"FINAL_8PLUS4_TABLE.md"
    temporary=target.with_suffix(".partial")
    temporary.write_text("\n".join(rows)+"\n",encoding="utf-8")
    temporary.replace(target)
    print("V4_FULL_OFFICIAL_CLOSED",sum(N_LEGACY.values()),sum((N_LEGACY|N_EXTRA).values()),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--work",type=Path)
    p.add_argument("--phase",choices=("selection","freeze","legacy","extra","aggregate"),required=True)
    p.add_argument("--alias",choices=ALIASES)
    p.add_argument("--task")
    p.add_argument("--arm",default="new")
    a=p.parse_args()
    if a.phase=="selection":
        selection(a.work,a.root)
    elif a.phase=="aggregate":
        audit(a.root)
    else:
        run_one(a.root,a.alias,a.phase,a.task,a.arm)
