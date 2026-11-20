"""Select fixed inherited prompts and run staged refinement on one GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from solver import Operators, PREVIOUS, integrate_a, integrate_b, timer, weight_hash
sys.path.insert(0,str(PREVIOUS.parent/"ouro_cross_task_20261118"))
from run import load_runtime


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save(path, data):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,allow_nan=False,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


def cohort_key(w):
    return hashlib.sha256(("574fa66/continuum/"+w["item"]).encode()).hexdigest()


def cohorts():
    source=PREVIOUS/"results"
    frozen=json.loads((source/"FREEZE.json").read_text())
    ids=set(frozen["arc_high_resolution"])
    data=read(source/"data_manifest.jsonl")
    out={}
    for family in ("tableau","twohop"):
        rows=sorted((w for w in data if w["family"]==family and w["cohort"]=="new_structure"),key=cohort_key)
        by_group={}
        for w in rows:
            by_group.setdefault(w["group"],w)
        out[family]=list(by_group.values())[:6]
    arc=[w for w in data if w["item"] in ids and len(w["content"])==4]
    for status in ("00","01","10","11"):
        matches=sorted([w for w in arc if "".join(str(int(i==w["y"])) for i in w["native_scan"])==status],key=cohort_key)
        assert len(matches)>=2
        if status=="00":
            out["arc"]=[]
        out["arc"].extend(matches[:2])
    a=out["arc"]
    out["arc"]=a[::2]+a[1::2]
    return {name:{"primary":values[:4],"holdout":values[4:],
                  "precision":values[:1]} for name,values in out.items()}


def piloting(model, items, path):
    rows=[]
    for family,group in items.items():
        op=Operators(model,group["primary"])
        receipt=op.parity_receipt()
        log=[]
        with timer(op,"pilot/"+family+"/A16",log):
            a=integrate_a(op,16,op.k1)
        with timer(op,"pilot/"+family+"/B16",log):
            b,_=integrate_b(op,16,op.k1)
        assert torch.isfinite(a[32]).all() and torch.isfinite(b[192]).all()
        rows.append(dict(family=family,**receipt,gpu=log))
        del op,a,b
    save(path/"NATIVE_PARITY.json",dict(status="PASS",rows=rows))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("stage",choices=("pilot","primary","precision","holdout","diagnostics","reference","stability"))
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    groups=cohorts()
    model,_,runtime=load_runtime(a.model)
    if a.stage=="pilot":
        save(a.out/"MANIFEST.json",groups)
        save(a.out/"RUNTIME.json",runtime)
        piloting(model,groups,a.out)
        return
    assert json.loads((a.out/"NATIVE_PARITY.json").read_text())["status"]=="PASS"
    from measurements import sequence
    sequence(model,groups,a.out,a.stage)


if __name__=="__main__":
    main()
