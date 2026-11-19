"""Evaluate the frozen factorial models on old and untouched generated groups."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from capture_composition import ARMS, MAIN, WORLDS, digest, projection, write_json
from model_composition import (Data, ROLE_BLOCK, apply, collection, fit, lift)

NATURAL=("confirm","transfer","fresh_same","fresh_structure")
ALL_MODELS=("zero",)
ARM_NAMES=("S","M","SM","I")


def export(path,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        out=csv.DictWriter(f,list(rows[0]),lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def margins(a,base,alt,alt2=None):
    arange=np.arange(len(a))
    reference=a[arange,alt] if alt2 is None else (a[arange,alt]+a[arange,alt2])/2
    return a[arange,base]-reference


def world_meta(rows,what):
    x=np.array([[r[what][w] for w in WORLDS] for r in rows])
    return x.reshape(-1) if what=="mid" else x


class Probe:
    def __init__(self,task,key,fit_rows,fit_features):
        self.key=key
        self.numeric=task=="arithmetic"
        options=[r[key] for r in fit_rows for _ in WORLDS]
        targets=world_meta(fit_rows,"mid" if key=="intermediate_options" else "gold").reshape(-1)
        truths=[o[j] for o,j in zip(options,targets)]
        self.vocab={v:i for i,v in enumerate(sorted(set(truths)))}
        if self.numeric:
            y=np.array(truths,dtype=np.float32).reshape(-1,1)
        else:
            y=np.eye(len(self.vocab),dtype=np.float32)[[self.vocab[v] for v in truths]]
        self.params=fit(fit_features,y,.1,centered=True)

    def scores(self,states,rows):
        y=apply(self.params,states)
        options=[r[self.key] for r in rows]
        out=np.empty((len(rows),max(map(len,options))),np.float32)
        out.fill(-10000)
        for i,a in enumerate(options):
            if self.numeric:
                out[i,:len(a)]=-np.abs(y[i,0]-np.array(a,np.float32))
            else:
                out[i,:len(a)]=[y[i,self.vocab[w]] if w in self.vocab else -10000 for w in a]
        return out


class CandidateHead:
    def __init__(self,checkpoint,rows):
        self.ids=sorted({v for r in rows for v in r["target_ids"]})
        self.index={v:i for i,v in enumerate(self.ids)}
        with safe_open(checkpoint/"model.safetensors",framework="pt") as f:
            w=f.get_slice("lm_head.weight")
            self.weight=torch.cat([w[v:v+1] for v in self.ids]).to(torch.bfloat16).T.contiguous()

    def score(self,h,rows):
        logits=(torch.from_numpy(np.ascontiguousarray(h)).to(torch.bfloat16)@self.weight).float().numpy()
        out=np.full((len(h),max(len(r["target_ids"]) for r in rows)),-10000,np.float32)
        for i,r in enumerate(rows):
            choices=[self.index[t] for t in r["target_ids"]]
            out[i,:len(choices)]=logits[i,choices]
        return out


def observed(rows,arm,loop):
    result=np.full((len(rows),max(len(r["target_ids"]) for r in rows)),-10000,np.float32)
    for i,r in enumerate(rows):
        score=r["scores"][arm][loop]
        result[i,:len(score)]=score
    return result


def norm(a):
    return np.sqrt(np.sum(a*a,axis=-1))


def geometric(pred,true,scale):
    delta=(pred-true)/scale
    z=(pred/scale)[:,ROLE_BLOCK].reshape(len(pred),-1)
    y=(true/scale)[:,ROLE_BLOCK].reshape(len(pred),-1)
    n0,n1=norm(z),norm(y)
    den=n0*n1
    return dict(response_mse=np.mean((z-y)**2,axis=1),
                cosine=np.divide(np.sum(z*y,axis=1),den,out=np.zeros_like(den),where=den>1e-12),
                norm_ratio=n0/np.maximum(n1,1e-8),
                **{f"mse_{name}":np.mean(delta[:,j]**2,axis=1) for j,name in
                   enumerate(("source","bridge","mapping","query","readout"))})


def diagnostic_fit(train,checkpoint,all_meta,scale):
    ids=train.select("fit")
    rows=[train.meta[i] for i in ids]
    feat,states=collection(train,ids,("physical",))
    task=rows[0]["task"]
    mids=Probe(task,"intermediate_options",rows,states["y"][:,2]/scale[2])
    composition=Probe(task,"options",rows,states["y"][:,4]/scale[4])
    k1_diagnostics={r:Probe(task,"intermediate_options",rows,states[r]) for r in ("mean","token")}
    head=CandidateHead(checkpoint,all_meta)
    g=world_meta(rows,"gold")
    def parity(score):
        return margins(score,g[:,0],g[:,1],g[:,2])
    r4=parity(observed(rows,"r",3))
    k4=np.stack([parity(observed(rows,a,3))-r4 for a in MAIN],axis=1)
    q=projection()[:,:32]
    x=((feat["physical","y"]/scale)@q).reshape(3*len(rows),-1)
    later=fit(x,k4.reshape(-1,1),.1)
    return mids,composition,head,k1_diagnostics,later


def four_effect(score):
    return np.concatenate((score,(score[:,2]-score[:,0]-score[:,1])[:,None]),axis=1)


def prediction(model,cfg,f,which):
    rep=cfg["rep"]
    X=lift(f[which,rep],cfg["kind"])
    return apply(model,X).reshape(-1,3,5,2048)


def evaluate_natural(data,split,models,scale,diag):
    mids,final,head,diag_k1,_=diag
    ids=data.select(split)
    rows=[data.meta[i] for i in ids]
    f,states=collection(data,ids,())
    out=[]
    n=len(rows)
    expanded=[r for r in rows for _ in WORLDS]
    y=states["y"]
    incoming=states["full"].reshape(-1,5,2048)
    true_choice=np.array([np.argmax(r["scores"][w][1]) for r in rows for w in WORLDS])
    for name in ("identity","natural-full","natural-mean","natural-token"):
        z=(incoming if name=="identity" else apply(models[name],states[name.split("-")[-1]])
           .reshape(-1,5,2048)*scale)
        errors=np.mean(((z-y)/scale)[:,ROLE_BLOCK]**2,axis=(1,2))
        choices=np.argmax(head.score(z[:,4],expanded),axis=1)
        mid_fit=np.argmax(mids.scores(z[:,2]/scale[2],expanded),axis=1)==world_meta(rows,"mid")
        aux=(np.argmax(diag_k1[name.split("-")[-1]].scores(states[name.split("-")[-1]],expanded),
                       axis=1)==world_meta(rows,"mid")) if name in ("natural-mean","natural-token") else np.zeros(4*n)
        for i,r in enumerate(rows):
            a=slice(4*i,4*i+4)
            out.append(dict(task=r["task"],split=split,id=r["id"],difficulty=r["difficulty"],method=name,
                  state_mse=float(errors[a].mean()),update_mse=float(errors[a].mean()),
                  native_K2_top1_agreement=float((choices[a]==true_choice[a]).mean()),
                  diagnostic_K2_mid_accuracy=float(mid_fit[a].mean()),
                  diagnostic_K1_mid_accuracy=float(aux[a].mean()) if name in ("natural-mean","natural-token") else ""))
    return out


def evaluate_response(data,split,models,configs,scale,diag):
    mids,final,head,_,later=diag
    indices=np.array([i for i in data.select(split) if "SM" in data.meta[i]["scores"]])
    rows=[data.meta[i] for i in indices]
    f,_=collection(data,indices,("physical",))
    true=four_effect(f["physical","y"].reshape(-1,3,5,2048))
    base=data.raw[indices,ARMS.index("r")]
    gold=world_meta(rows,"gold")
    mid=world_meta(rows,"mid").reshape(-1,4)
    n=len(rows)
    repeat=[r for r in rows for _ in MAIN]
    def diag_state(model,z,region):
        v=model.scores(z.reshape(3*n,5,2048)[:,region]/scale[region],repeat)
        return v.reshape(n,3,v.shape[-1])
    def numeric_state(model,z,region):
        return apply(model.params,z.reshape(3*n,5,2048)[:,region]/scale[region]).reshape(n,3)
    def head_state(z):
        v=head.score(z.reshape(3*n,5,2048)[:,4],repeat)
        return v.reshape(n,3,v.shape[-1])
    def parity(v):
        return np.stack([margins(v[:,j],gold[:,0],gold[:,1],gold[:,2]) for j in range(3)],axis=1)
    def mid_delta(v):
        return np.stack([margins(v[:,j],mid[:,1],mid[:,0]) for j in range(3)],axis=1)
    def mid_semantics(z):
        if mids.numeric:
            direction=np.array([np.sign(int(r["intermediate_options"][m[1]])-
                                        int(r["intermediate_options"][m[0]])) for r,m in zip(rows,mid)])
            return numeric_state(mids,z,2)*direction[:,None]
        return mid_delta(diag_state(mids,z,2))
    def final_semantics(z):
        if final.numeric:
            return numeric_state(final,z,4)
        return parity(diag_state(final,z,4))
    anchor=np.broadcast_to(base[:,None],(n,3,5,2048))
    mid_base=mid_semantics(anchor)
    sem_base=final_semantics(anchor)
    head_base=parity(head_state(anchor))
    true_score={loop:four_effect(np.stack([
        margins(observed(rows,a,loop),gold[:,0],gold[:,1],gold[:,2])
        -margins(observed(rows,"r",loop),gold[:,0],gold[:,1],gold[:,2]) for a in MAIN],axis=1))
                for loop in (1,2,3)}
    oracle=anchor+true[:,:3]
    oracle_mid=four_effect(mid_semantics(oracle)-mid_base)
    oracle_comp=four_effect(final_semantics(oracle)-sem_base)
    original_top=np.argmax(observed(rows,"r",3),axis=1)
    replay=four_effect(parity(head_state(oracle))-head_base)
    assert np.max(np.abs(replay-true_score[1]))<.02
    q=projection()[:,:32]
    results=[]
    for method in (*ALL_MODELS,*(name for name in configs if not name.startswith("natural-"))):
        estimate=np.zeros_like(true) if method=="zero" else four_effect(
            prediction(models[method],configs[method],f,"physical")*scale)
        candidate=anchor+estimate[:,:3]
        m=four_effect(mid_semantics(candidate)-mid_base)
        c=four_effect(final_semantics(candidate)-sem_base)
        k2=four_effect(parity(head_state(candidate))-head_base)
        late_x=((estimate[:,:3]/scale)@q).reshape(3*n,-1)
        k4=four_effect(apply(later,late_x).reshape(n,3))
        dor_gold=(gold[:,1],gold[:,2],gold[:,3],gold[:,3])
        for j,arm in enumerate(ARM_NAMES):
            geo=geometric(estimate[:,j],true[:,j],scale)
            choice=np.argmax(observed(rows,arm if arm!="I" else "SM",3),axis=1)
            for i,r in enumerate(rows):
                results.append({"task":r["task"],"split":split,"difficulty":r["difficulty"],
                    "id":r["id"],"method":method,"arm":arm,
                    **{key:float(v[i]) for key,v in geo.items()},
                    "mid_error":float(abs(m[i,j]-oracle_mid[i,j])),
                    "comp_error":float(abs(c[i,j]-oracle_comp[i,j])),
                    "k2_error":float(abs(k2[i,j]-true_score[1][i,j])),
                    "k4_error":float(abs(k4[i,j]-true_score[3][i,j])),
                    "mid_pred":float(m[i,j]),"mid_true":float(oracle_mid[i,j]),
                    "comp_pred":float(c[i,j]),"comp_true":float(oracle_comp[i,j]),
                    "k2_pred":float(k2[i,j]),"k2_true":float(true_score[1][i,j]),
                    "k3_true":float(true_score[2][i,j]),"k4_true":float(true_score[3][i,j]),
                    "k4_pred":float(k4[i,j]),"k4_target":int(dor_gold[j][i]),
                    "k4_clean_choice":int(original_top[i]),"k4_choice":int(choice[i])})
    return results


def run(dev,fresh,frozen,checkpoint,out):
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    old,new=Data(dev),Data(fresh)
    f=json.loads((frozen/"freeze.json").read_text())
    assert f["fit_groups"]==300 and f["discovery_groups"]==100
    for name,h in f["implementation"].items():
        assert digest(Path(__file__).parent/name)==h,name
    scale=np.load(frozen/"role_scaler.npy")
    assert digest(frozen/"role_scaler.npy")==f["scaler"]
    models={}
    for name,h in f["weights"].items():
        p=frozen/f"{name}.npz"
        assert digest(p)==h
        with np.load(p) as array:
            models[name]=dict(array)
    diagnostics=diagnostic_fit(old,checkpoint,old.meta,scale)
    natural,causal=[],[]
    for split in NATURAL:
        data=new if split.startswith("fresh") else old
        natural+=evaluate_natural(data,split,models,scale,diagnostics)
        causal+=evaluate_response(data,split,models,f["configurations"],scale,diagnostics)
        print(new.meta[0]["task"],split,len(data.select(split)),flush=True)
    export(out/"natural.csv",natural)
    export(out/"causal.csv",causal)
    write_json(out/"complete.json",{"task":old.meta[0]["task"],"natural_records":len(natural),
               "causal_records":len(causal),"locked_models":digest(frozen/"freeze.json"),
               "independent_manifest":new.complete["source_manifest_sha256"]})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("dev","fresh","frozen","checkpoint","out"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    run(a.dev,a.fresh,a.frozen,a.checkpoint,a.out)
