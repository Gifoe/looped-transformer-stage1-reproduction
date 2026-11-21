"""Native discovery uses prompts and IDs only, never correctness labels."""
import argparse
import platform
from pathlib import Path

import numpy as np
import torch
import transformers

from native import Forward, Recorder, REV, MODEL_HASH, SEED, dump_witness, flatnorm, identity_trial, load, output, save, setup, sha, tensors, timer


def magnitude(x):
    return float(x.float().square().mean().sqrt())


def split(f,rec,left,right,l,n):
    r,s=rec[left,l],rec[right,l]
    a,b=r["a"][...,:n,:n],s["a"][...,:n,:n]
    v,w=r["v"][...,:n,:],s["v"][...,:n,:]
    dv=w-v
    da=b-a
    att=[da@v,a@dv,da@dv]
    real=s["av"][:,:n].transpose(1,2)-r["av"][:,:n].transpose(1,2)
    proj=f.layers[l].self_attn.o_proj
    p=lambda x:proj(x.transpose(1,2).reshape(1,n,-1))
    assert proj.bias is None
    applied=[p(t) for t in att]
    end=s["att_projected"][:,:n]-r["att_projected"][:,:n]
    att_res=magnitude(sum(att)-real)/max(magnitude(real),1e-8)
    p_res=magnitude(sum(applied)-end)/max(magnitude(end),1e-8)
    if max(att_res,p_res)>1e-4:
        raise ArithmeticError((l,left,att_res,p_res))
    g=f.layers[l].mlp.act_fn(r["g_pre"][:,:n])
    gg=f.layers[l].mlp.act_fn(s["g_pre"][:,:n])
    u,uu=r["u"][:,:n],s["u"][:,:n]
    dg,du=gg-g,uu-u
    mlp=[dg*u,g*du,dg*du]
    d=f.layers[l].mlp.down_proj
    dy=s["z"][:,:n]-r["z"][:,:n]
    down=s["down"][:,:n]-r["down"][:,:n]
    projected=[d(z) for z in mlp]
    zres=magnitude(sum(mlp)-dy)/max(magnitude(dy),1e-8)
    dres=magnitude(sum(projected)-down)/max(magnitude(down),1e-8)
    if max(zres,dres)>1e-4:
        raise ArithmeticError((l,left,zres,dres))
    # Post-norm changes report observed magnitudes. The one-corner counterfactual
    # accounting is stored separately from linear projection shares.
    def norm_panel(norm,old,terms,real):
        baseline=norm(old)
        changes=[norm(old+x)-baseline for x in terms]
        return dict(single_rms=list(map(magnitude,changes)),actual_rms=magnitude(real),
          remainder_rms=magnitude(real-sum(changes)))
    valid=lambda z:z[:,:n]
    entropy=float(-(a*a.clamp_min(1e-40).log()).sum(-1).mean())
    h=lambda x:x.float().square().mean(dim=(0,2,3)).sqrt().tolist()
    return dict(layer=l+1,transition=f"K{left+1}toK{right+1}",attention=dict(
        heads=list(map(h,att)),head_actual=h(real),pre=list(map(magnitude,att)),projected=list(map(magnitude,applied)),
        actual=magnitude(real),projection_actual=magnitude(end),residual=[att_res,p_res],entropy=entropy,
        post_norm=norm_panel(f.layers[l].input_layernorm_2,valid(r["att_projected"]),applied,valid(s["a_write"]-r["a_write"]))),
        mlp=dict(pre=list(map(magnitude,mlp)),projected=list(map(magnitude,projected)),
         actual=magnitude(dy),projection_actual=magnitude(down),residual=[zres,dres],
         sign_change=float(((g>0)!=(gg>0)).float().mean()),post_norm=norm_panel(f.layers[l].post_attention_layernorm_2,
         valid(r["down"]),projected,valid(s["m_write"]-r["m_write"]))),
        state_rms=magnitude(valid(r["input"])),delta_state_rms=magnitude(valid(s["input"]-r["input"])),
        update_rms=magnitude(valid(r["output"]-r["input"])))


def collect(f,tok,rows,out,name,witness):
    result=[]
    dest=out/f"ATLAS_{name}.json"
    if dest.exists():
        return
    for i,row in enumerate(rows):
        assert "gold" not in row
        ids,pos=tensors(tok,[row["prompt"]])
        t=timer()
        r=Recorder()
        states=f.run(ids,4,record=r)
        logits=[output(f.model,h,pos) for h in states]
        measures=[split(f,r.data,k,k+1,l,ids.shape[-1]) for k in range(3) for l in range(24)]
        result.append(dict(id=row["id"],family=row["family"],tokens=ids.shape[-1],measures=measures,
             top5=[torch.topk(z,5).indices[0].tolist() for z in logits],
             state_rms=[magnitude(h) for h in states],gpu_seconds=timer()-t))
        if i==0 and witness:
            dump_witness(out,ids,states,r)
            np.savez_compressed(out/"TRACE_VOCAB_LOGITS.npz",**{f"K{k+1}":x.cpu().numpy() for k,x in enumerate(logits)})
        if (i+1)%8==0:
            print(name,i+1,len(rows),flush=True)
        del r,states
    save(dest,result)


def freeze(root,blind):
    phase=load(root/"ATLAS_dev.json")
    scores={}
    head_scores={}
    for k in ("attention","mlp"):
        mat=np.asarray([[r[k]["post_norm"]["single_rms"][0]/max(r[k]["post_norm"]["actual_rms"],1e-9)
                         for r in q["measures"][:24]] for q in phase])
        scores[k]=np.median(mat,axis=0).tolist()
    att_layer=int(np.argmax(scores["attention"]))
    mlp_layer=int(np.argmax(scores["mlp"]))
    for l in (att_layer,(att_layer+12)%24):
        head_scores[str(l+1)]=np.median([np.divide(q["measures"][l]["attention"]["heads"][0],
                     np.maximum(q["measures"][l]["attention"]["head_actual"],1e-9)) for q in phase],axis=0).tolist()
    hs=sorted(np.argsort(head_scores[str(att_layer+1)])[-4:].astype(int).tolist())
    save(root/"CANDIDATES.json",dict(source="median natural K1→K2 first-factor change / total after post-norm over all valid tokens",
         candidate_attention_layer=att_layer+1,candidate_mlp_layer=mlp_layer+1,heads_zero_based=hs,
         control_heads_zero_based=sorted(set(range(16))-set(hs))[:4],
         support="all valid queries/tokens; no preselected semantic ROI",selection_scores=scores,
         head_ranking_scores=head_scores,discovery_manifest_sha256=sha(blind),atlas_sha256=sha(root/"ATLAS_dev.json"),
         attention_control_layer=(att_layer+12)%24+1,mlp_control_layer=(mlp_layer+12)%24+1))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("blind",type=Path)
    p.add_argument("stage",choices=("technical","pilot","dev","confirm_arc","confirm_external","controlled","freeze"))
    opt=p.parse_args()
    opt.out.mkdir(exist_ok=True,parents=True)
    if opt.stage=="freeze":
        freeze(opt.out,opt.blind)
        return
    file=opt.model/"model.safetensors"
    if opt.stage=="technical":
        assert sha(file)==MODEL_HASH
    inputs=load(opt.blind)
    model,tok=setup(opt.model)
    f=Forward(model)
    if opt.stage=="technical":
        rows=inputs["pilot"][:3]
        checks=[]
        for dtype in (torch.float32,torch.bfloat16):
            model.to(dtype)
            for row in rows:
                ids,pos=tensors(tok,[row["prompt"]])
                checks.append(dict(dtype=str(dtype),id=row["id"],**identity_trial(f,ids,pos)))
        save(opt.out/"RUNTIME.json",dict(model_revision=REV,model_sha256=MODEL_HASH,backend="eager/no-cache/eval",
                   torch=torch.__version__,transformers=transformers.__version__,python=platform.python_version(),
                   cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(), seed=SEED,
                   peak_bytes=torch.cuda.max_memory_allocated(),parity=checks))
        return
    collect(f,tok,inputs[opt.stage],opt.out,opt.stage,opt.stage=="pilot")


if __name__=="__main__":
    main()

