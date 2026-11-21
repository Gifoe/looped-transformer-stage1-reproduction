"""Bidirectional mediator intervention with prompt-only native K2 suffixes."""
import argparse
from pathlib import Path

import torch

from native import Forward, Recorder, load, save, sha, setup, timer
from route_ops import lane_plan, norms

UP_ROLL=137
PRODUCT_ROLL=317


class Writer:
    def __init__(self,n,strength,permuted=False):
        self.n,self.strength,self.permuted=n,strength,permuted
        self.norm=0.

    def resid(self,k,l,x):
        return x

    def mlp(self,key,z,rec):
        return z

    def attn(self,key,a,v):
        return a,v

    def write(self,key,name,w):
        if key!=(0,0) or name!="a_write":
            return w
        out=w.clone()
        part=w[:,1:self.n]
        self.norm=float((self.strength*part).norm())
        d=part.roll(37,-1) if self.permuted else part
        out[:,1:self.n]-=self.strength*d
        return out


def capture(f,ids,n,dose=0.,permuted=False):
    trace=Recorder(keys={(1,0)})
    w=Writer(n,dose,permuted) if dose else None
    y=f.run(ids,2,record=trace,edit=w)
    return y,trace.data[1,0],w


def moment(a,b):
    return dict(norm=float(a.norm()),ratio=float(a.norm()/b.norm().clamp_min(1e-10)),
                same_lane_max=float((a-a[:1]).abs().max()))


def patch_down(f,ids,h1,n,product):
    def replace(_,args):
        z=args[0].clone()
        z[:,:n]=product
        return (z,)
    handle=f.layers[0].mlp.down_proj.register_forward_pre_hook(replace)
    try:
        return f.run(ids,1,start=h1)[0]
    finally:
        handle.remove()


def geometry(u,g):
    d=u["clean"]-u["drop"]
    cross=(g["clean"]-g["drop"])*d
    clean=g["clean"]*d
    drop=g["drop"]*d
    assert float((clean-drop-cross).norm()/clean.norm().clamp_min(1e-10))<2e-5
    def cosine(a,b):
        return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-10))
    return dict(up=moment(d,u["clean"]),gate=moment(g["clean"]-g["drop"],g["clean"]),
                cross=moment(cross,clean),cosine_realized=cosine(clean,drop),
                product_norm=[float(clean.norm()),float(drop.norm())])


def evaluate(f,tok,question):
    assert "gold" not in question
    t=timer()
    ids,lens,read=lane_plan(tok,question["prompt"],question["choices"])
    n=lens[0]
    assert all(x==n for x in lens)
    h={}
    m={}
    for name,d,p in (("clean",0.,False),("drop",1.,False),("half",.5,False),("permute_writer",1.,True)):
        h[name],m[name],_=capture(f,ids,n,d,p)
    up={k:v["u"][:,:n] for k,v in m.items() if k in ("clean","drop")}
    act=f.layers[0].mlp.act_fn
    gate={k:act(v["g_pre"][:,:n]) for k,v in m.items() if k in up}
    g=geometry(up,gate)
    base={name:[read(f.model,z) for z in states] for name,states in h.items()}
    arms={}
    norm=f.layers[0].post_attention_layernorm
    proj=f.layers[0].mlp.up_proj
    restored={}
    H={k:v["input"][:,:n] for k,v in m.items() if k in up}
    A={k:v["a_write"][:,:n] for k,v in m.items() if k in up}
    for k in up:
        recompute=proj(norm(H[k]+A[k]))
        assert float((recompute-up[k]).norm()/up[k].norm().clamp_min(1e-10))<2e-5
    restored["inherited"]=proj(norm(H["clean"]+A["drop"]))
    restored["fresh_attention"]=proj(norm(H["drop"]+A["clean"]))
    restored["removed_up_norm_rel"]={
        k:float((m["clean"][k][:,:n]-m["drop"][k][:,:n]).norm()/m["clean"][k][:,:n].norm().clamp_min(1e-10))
        for k in ("input","a_write","mlp_input","u")
    }
    support={}
    for receiver,donor in (("drop","clean"),("clean","drop")):
        z=up[receiver]*gate[receiver]
        delta=up[donor]-up[receiver]
        product=delta*gate[receiver]
        controls={
            "up":up[donor]*gate[receiver],
            "up_norm_control":(up[receiver]+delta.roll(UP_ROLL,-1))*gate[receiver],
            "product_norm_control":z+product.roll(PRODUCT_ROLL,-1),
            "gate":up[receiver]*gate[donor],
            "gate_and_up":up[donor]*gate[donor],
        }
        if receiver=="drop":
            for name in ("inherited","fresh_attention"):
                controls[name]=restored[name]*gate[receiver]
        noop=patch_down(f,ids,h[receiver][0],n,z)
        parity=float((noop-h[receiver][1]).abs().max())
        assert parity<5e-5,(question["id"],receiver,parity)
        moments=dict(noop_error=parity,actual_raw_up=float(delta.norm()),
                     matched_raw_up=float(delta.roll(UP_ROLL,-1).norm()),
                     actual_product=float(product.norm()),
                     matched_product=float(product.roll(PRODUCT_ROLL,-1).norm()))
        results={}
        for name,v in controls.items():
            o=patch_down(f,ids,h[receiver][0],n,v)
            results[name]=dict(ll=read(f.model,o),z_dose=float((v-z).norm()),
                               state_norm_ratio=norms(o,h["clean"][1],lens))
        moments["raw_up_control_output_dose"]=results["up_norm_control"]["z_dose"]
        moments["native_candidate_lane_alignment_error"]=float((up[receiver]-up[receiver][:1]).abs().max())
        support[receiver]=moments
        arms[receiver]=results
    return dict(id=question["id"],native=base,arms=arms,geometry=g,support=support,
                carrier=restored["removed_up_norm_rel"],seconds=timer()-t)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("dataset")
    a=p.parse_args()
    rows=load(a.manifest)["cohorts"][a.dataset]["rows"]
    model,tok=setup(a.model)
    f=Forward(model)
    done=load(a.out)["rows"] if a.out.exists() else []
    assert [x["id"] for x in done]==[x["id"] for x in rows[:len(done)]]
    for q in rows[len(done):]:
        done.append(evaluate(f,tok,{k:v for k,v in q.items() if k!="gold"}))
        if len(done)%4==0 or len(done)==len(rows):
            save(a.out,dict(expected=len(rows),manifest_sha256=sha(a.manifest),rows=done,
                            peak_bytes=torch.cuda.max_memory_allocated()))
            print(a.dataset,len(done),len(rows),flush=True)


if __name__=="__main__":
    main()

