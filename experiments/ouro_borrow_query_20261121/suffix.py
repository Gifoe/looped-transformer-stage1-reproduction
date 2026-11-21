"""Run on-policy continuation after a verified first-pivot operation."""
import argparse
from pathlib import Path

import torch

from context import Edit, pre
from interchange import independent_like, load_cases, terminated
from native import Forward, load, output, save, setup, sha
from rule import MAX_NEW, text_results
from state import rotate_about

MODES=("native_suffix","next_readout","all_readout","all_sham","full")


def generate(f,tok,cases,prior,k,mode,layer):
    tokens=[c["prefix"]+[taken[k]["tokens"][len(c["prefix"])]]
            for c,taken in zip(cases,prior,strict=True)]
    active=[i for i,p in enumerate(tokens) if not terminated(tok,p) and len(p)<MAX_NEW]
    step=0
    while active:
        current=[dict(cases[i],prefix=tokens[i]) for i in active]
        seq,at,tr,h1,h2=pre(f,current)
        if mode=="native_suffix" or (mode=="next_readout" and step>0):
            state=(h1,h2)[k]
        else:
            got=tr.data[1-k,layer]
            if mode=="all_sham":
                got=rotate_about(tr.data[k,layer],got,independent_like(got,current,layer,89))
            edit=Edit(at,layer,got,"full" if mode=="full" else "readout")
            state=f.run(seq,1,start=h1 if k else None,edit=edit)[0]
            if mode=="full":
                assert torch.equal(state,(h1,h2)[1-k])
        nxt=output(f.model,state,at).argmax(-1).tolist()
        remain=[]
        for i,v in zip(active,nxt,strict=True):
            tokens[i].append(v)
            if not terminated(tok,tokens[i]) and len(tokens[i])<MAX_NEW:
                remain.append(i)
        active=remain
        step+=1
    return text_results(tok,tokens)


def summarize(rows):
    part={}
    for mode in MODES:
        answers=[r["outputs"][mode] for r in rows]
        failure=[r for r in rows if r["outputs"]["native_suffix"][0]["number"]!=r["target"]]
        part[mode]=dict(K1_correct=sum(z[0]["number"]==r["target"] for r,z in zip(rows,answers,strict=True)),
                        K2_foil=sum(z[1]["number"]==r["foil"] for r,z in zip(rows,answers,strict=True)),
                        joint=sum([x["number"] for x in z]==[r["target"],r["foil"]]
                                  for r,z in zip(rows,answers,strict=True)),
                        K1_correct_after_failed_pivot=sum(r["outputs"][mode][0]["number"]==r["target"] for r in failure),
                        failed_pivot=len(failure))
    return part


@torch.inference_mode()
def run(model,root):
    a=load(root/"CONTEXT_SUMMARY.json")["repair"]
    assert a["n"]-a["arms"]["full"]["K1_correct"]>=8
    site=load(root/"CONTEXT_SELECTION.json")["pick"]["readout"]
    mod,tok=setup(model)
    f=Forward(mod)
    _,cases=load_cases(root,tok,"confirmation")
    reference={r["id"]:r["pred"]["full"] for r in load(root/"CONTEXT_RAW.json")["rows"] if r["kind"]=="repair"}
    path=root/"SUFFIX_RAW.json"
    rows=load(path)["rows"] if path.exists() else []
    assert [r["id"] for r in rows]==[c["id"] for c in cases[:len(rows)]]
    for i in range(len(rows),len(cases),8):
        block=cases[i:i+8]
        prior=[reference[c["id"]] for c in block]
        assert all(all(c["reach"]) for c in block)
        seq={mode:list(zip(*(generate(f,tok,block,prior,k,mode,site) for k in (0,1)),strict=True))
             for mode in MODES}
        for j,c in enumerate(block):
            result={a:x[j] for a,x in seq.items()}
            assert list(result["native_suffix"])==prior[j]
            assert list(result["full"])==c["native"][::-1]
            rows.append(dict(id=c["id"],target=c["target"],foil=c["foil"],outputs=result))
        save(path,dict(rows=rows,source_sha256=sha(root/"CONTEXT_RAW.json"),layer=site))
        print("on-policy suffix",len(rows),len(cases),flush=True)
    s=summarize(rows)
    save(root/"SUFFIX_SUMMARY.json",s)
    print(s,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

