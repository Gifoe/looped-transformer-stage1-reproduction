"""Predeclared source, context, and cross-task interventions."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import torch

from collect import get_encoding, load_runtime, load_lines, dump_line, write_json, digest, predictions
from run import fast_forward, dose_matched, depth_scores

SITES={"relation":(2,5),"logic":(2,11),"arithmetic":(2,5)}
DIRECTIONS=(("r","d","x","n","y"),("d","r","y","m","x"))
TAPS=(0,5,8,11,17,23)


def rng(key):
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big")
    return torch.Generator(device="cuda").manual_seed(seed)


def l2(v):
    return float(v.float().norm())


def contrasts(sc,target,base):
    return [x[target]-x[base] for x in sc]


def region(e,role):
    if role=="prefix":
        return list(range(e["anchor"]+1))
    if role=="read":
        return [e["read"]]
    if role=="query_options":
        return sorted(set(e["roles"]["query"]+e["roles"]["option"]+[e["read"]]))
    return e["roles"][role]


def source_region(g,e):
    if g["task"]!="relation":
        return region(e,"signal")
    return sorted(region(e,"payload_0")+region(e,f'payload_{g["meta"]["other"]}'))


def encode_bank(model, e, positions, site):
    return fast_forward(model,{**e,"anchor":positions},{site})


def scores_patch(model,e,site,pos,vector):
    return fast_forward(model,e,edits={site:(pos,vector)})


def source_experiment(model, tok, g, job):
    site=SITES[g["task"]]
    worlds={n:get_encoding(tok,g,w) for n,w in g["worlds"].items()}
    for rx,don,alt,noise,map_only in DIRECTIONS:
        e=worlds[rx]
        signal=source_region(g,e)
        roles={"payload":signal}
        if job=="A":
            assert g["mode"]=="context"
            assert all(e["ids"][i]==worlds[don]["ids"][i] for i in signal)
            roles.update(binding=region(e,"bind"),prefix=region(e,"prefix"))
        if job=="C":
            roles.update(bridge=region(e,"bridge"),prefix=region(e,"prefix"))
        noise_pos=region(e,"noise")
        tracked=sorted(set(i for v in roles.values() for i in v).union(noise_pos))
        index={p:i for i,p in enumerate(tracked)}
        bank={n:encode_bank(model,worlds[n],tracked,site) for n in (rx,don,noise,map_only)}
        r,s=g["worlds"][rx],g["worlds"][don]
        target=g["worlds"][alt]["gold"]
        assert s["gold"]==r["gold"]!=target
        key=f"{g['id']}/{rx}"
        output={}
        noise_indices=[index[i] for i in noise_pos]
        noise_base=bank[rx][1][site][noise_indices]
        noise_vec=bank[noise][1][site][noise_indices].float()-noise_base.float()
        assert l2(noise_vec)>0
        for name,pos in roles.items():
            selection=[index[i] for i in pos]
            base=bank[rx][1][site][selection]
            give=bank[don][1][site][selection]
            map_state=bank[map_only][1][site][selection]
            assert torch.equal(map_state,base)
            dose=l2(give.float()-base.float())
            z=torch.randn(base.shape,device="cuda",generator=rng(f"{key}/{name}"))
            values={"semantic":(pos,base,give),"self":(pos,base,base),"map_only":(pos,base,map_state),
                    "irrelevant":(noise_pos,noise_base,dose_matched(noise_base,noise_vec,dose)),
                    "random":(pos,base,dose_matched(base,z,dose))}
            scores={}
            for arm,(indices,original,value) in values.items():
                y=scores_patch(model,e,site,indices,value)
                if arm in ("self","map_only"):
                    assert torch.equal(y[4],bank[rx][4])
                assert torch.equal(y[4][0],bank[rx][4][0])
                scores[arm]={"scores":y[0],"pred":predictions(y[0]),"dose":l2(value.float()-original.float())}
            if job in ("A","C") or name=="payload":
                pe=get_encoding(tok,g,r,True)
                assert pe["ids"][:max(pos)+1]==e["ids"][:max(pos)+1]
                pr=encode_bank(model,pe,pos,site)
                probe_base=pr[1][site]
                mid={}
                for arm,value in {"native":probe_base,"semantic":give,
                                  "random":dose_matched(probe_base,z,l2(give.float()-probe_base.float()))}.items():
                    pred=scores_patch(model,pe,site,pos,value)[0]
                    mid[arm]={"scores":pred,"pred":predictions(pred)}
            else:
                mid=None
            output[name]={"positions":pos,"arms":scores,"intermediate":mid,
                          "same_token_ids":all(e["ids"][i]==worlds[don]["ids"][i] for i in pos)}
        # Trace effect at the natural receiving token before analyzing final scores.
        primary=output["payload"]
        selected=primary["positions"]
        donor_patch=bank[don][1][site][[index[i] for i in selected]]
        trace_roles={"source":selected,"mapping":region(e,"map"),
                     "query_options":region(e,"query_options"),"readout":region(e,"read")}
        baseline=instrument(model,e,trace_roles)
        edited=instrument(model,e,trace_roles,edits={site:(selected,donor_patch)})
        yield {"id":g["id"],"task":g["task"],"family":g["family"],"job":job,"receive":rx,"donor":don,
               "site":site,"original":r["gold"],"counterfactual":target,"mid_original":r["mid"],
               "mid_counterfactual":s["mid"],"arms":output,"native":bank[rx][0],"donor_native":bank[don][0],
               "input_cf_native":fast_forward(model,worlds[alt])[0],
               "trace":change(baseline,edited,trace_roles)}


@contextmanager
def taps(model,e,roles,edits,route,mask_loops):
    core=model.model
    prior=core.total_ut_steps
    core.total_ut_steps=4
    handles=[]
    states={}
    branches={}
    norms=[]
    events=[]
    index=[p for name, positions in roles.items() for p in positions]
    bank={}
    def layer_hook(layer):
        def hook(module,args,kw,out):
            event=(1+kw["current_ut"],layer)
            assert event==(len(events)//24+1,len(events)%24)
            if event in edits:
                indices,values=edits[event]
                out=out.clone()
                out[0,indices]=values
            events.append(event)
            if layer in TAPS:
                states[event]=out[0,index].clone()
            return out
        return hook
    def before_attention(module,args,kw):
        k=kw["current_ut"]+1
        if k not in mask_loops:
            return
        h=kw["hidden_states"]
        n=h.shape[-2]
        mask=kw["attention_mask"]
        if mask is None:
            assert not module.sliding_window
            valid=torch.ones((n,n),device=h.device,dtype=torch.bool).tril()
            mask=torch.zeros((1,1,n,n),device=h.device,dtype=h.dtype)
            mask.masked_fill_(~valid,torch.finfo(h.dtype).min)
        else:
            assert mask.shape[-2:]==(n,n)
            mask=mask.clone()
        for dest,src in route.get(k,()):
            dst=torch.tensor(dest,device=h.device)[:,None]
            origin=torch.tensor(src,device=h.device)[None,:]
            minimum=False if mask.dtype==torch.bool else torch.finfo(mask.dtype).min
            mask[...,dst,origin]=minimum
        kw={**kw,"attention_mask":mask}
        return args,kw
    def contribution(layer,name):
        def hook(module,args,out):
            k=len(events)//24+1
            v=out[0] if isinstance(out,tuple) else out
            branches[k,layer,name]=v[0,index].clone()
        return hook
    def norm_hook(module,args,out):
        k=len(norms)+1
        event=(k,"norm")
        if event in edits:
            positions,values=edits[event]
            out=out.clone()
            out[0,positions]=values
        norms.append(out[0,e["read"]].clone())
        states[event]=out[0,index].clone()
        bank[k]=out[0].clone()
        return out
    try:
        for l,layer in enumerate(core.layers):
            handles.append(layer.register_forward_hook(layer_hook(l),with_kwargs=True))
            if mask_loops:
                handles.append(layer.self_attn.register_forward_pre_hook(before_attention,with_kwargs=True))
            if l in TAPS:
                handles.append(layer.self_attn.register_forward_hook(contribution(l,"attention")))
                handles.append(layer.mlp.register_forward_hook(contribution(l,"mlp")))
        handles.append(core.norm.register_forward_hook(norm_hook))
        yield states,branches,norms,bank
        assert len(events)==96 and len(norms)==4
    finally:
        for hook in handles:
            hook.remove()
        core.total_ut_steps=prior


def instrument(model,e,roles,edits=None,route=None,mask_loops=()):
    device=model.lm_head.weight.device
    inputs=torch.tensor([e["ids"]],device=device)
    with taps(model,e,roles,edits or {},route or {},mask_loops) as (states,branches,norms,bank):
        native=model(inputs,use_cache=False,exit_at_step=3,logits_to_keep=1).logits
    sc,full=depth_scores(model,norms,e["target"])
    assert torch.equal(full[-1],native[0,0])
    return {"scores":sc,"states":states,"branches":branches,"norm":bank,"head":full}


def change(a,b,roles):
    result={}
    for field in ("states","branches"):
        events=list(a[field])
        delta=torch.stack([b[field][k].float()-a[field][k].float() for k in events])
        pos=0
        for name,positions in roles.items():
            width=len(positions)
            l2s=delta[:,pos:pos+width].square().sum(dim=2).mean(dim=1).sqrt().tolist()
            result[f"{field}/{name}"]={":".join(map(str,key)):v for key,v in zip(events,l2s)}
            pos+=width
    return result


def loop_experiment(model,tok,g):
    worlds={n:get_encoding(tok,g,w) for n,w in g["worlds"].items()}
    for rx,don,alt,noise,_ in DIRECTIONS:
        e=worlds[rx]
        src=source_region(g,e)
        src_site=(1,5)
        recv=encode_bank(model,e,src,src_site)
        give=encode_bank(model,worlds[don],src,src_site)
        nz=region(e,"noise")
        nzbase=encode_bank(model,e,nz,src_site)[1][src_site]
        nzalt=encode_bank(model,worlds[noise],nz,src_site)[1][src_site]
        assert l2(nzalt.float()-nzbase.float())>0
        base=recv[1][src_site]
        semantic=give[1][src_site]
        dose=l2(semantic.float()-base.float())
        no=dose_matched(nzbase,nzalt.float()-nzbase.float(),dose)
        rnd=torch.randn(base.shape,device="cuda",generator=rng(f"B/{g['id']}/{rx}"))
        rnd=dose_matched(base,rnd,dose)
        roles={"source":src,"mapping":region(e,"map"),"query_options":region(e,"query_options")}
        roles["readout"]=region(e,"read")
        mapping=roles["mapping"]
        query=roles["query_options"]
        latent={"none":{}, "semantic":{src_site:(src,semantic)}}
        out={}
        traces={}
        native=instrument(model,e,roles)
        implicit=instrument(model,e,roles,latent["semantic"])
        assert torch.equal(native["head"],recv[4])
        for arm,pos,ref,v in (("self",src,base,base),("irrelevant",nz,nzbase,no),("random",src,base,rnd)):
            y=instrument(model,e,roles,{src_site:(pos,v)})
            if arm=="self":
                assert torch.equal(y["head"],native["head"])
            out[arm]={"scores":y["scores"],"pred":predictions(y["scores"]),"dose":l2(v.float()-ref.float())}
        masks=(2,3)
        routes={
            "explicit_open":{},
            "k2_map":{2:[(mapping,src)]},
            "k2_direct":{2:[(query,src)]},
            "k2_both":{2:[(mapping,src),(query,src)]},
            "k3_both":{3:[(mapping,src),(query,src)]},
            "k2k3_both":{2:[(mapping,src),(query,src)],3:[(mapping,src),(query,src)]},
        }
        for arm,route in routes.items():
            pair={v:instrument(model,e,roles,p,route,masks) for v,p in latent.items()}
            out[arm]={k:{"scores":v["scores"],"pred":predictions(v["scores"])} for k,v in pair.items()}
            if arm=="explicit_open":
                opened=pair
                assert torch.equal(pair["none"]["head"][0],native["head"][0])
            if arm=="k2_both":
                traces["k2_both"]=change(pair["none"],pair["semantic"],roles)
        for arm,positions,k in (("k2_both_restore_map",mapping,2),("k2_both_restore_query",query,2)):
            pair={}
            for v,p in latent.items():
                restoration={(k,"norm"):(positions,opened[v]["norm"][k][positions])}
                pair[v]=instrument(model,e,roles,{**p,**restoration},routes["k2_both"],masks)
            out[arm]={k:{"scores":v["scores"],"pred":predictions(v["scores"])} for k,v in pair.items()}
        for arm,positions in (("erase_k1_source",src),("erase_k1_map",mapping),("erase_k1_query",query),
                              ("erase_k1_suffix",sorted(set(mapping+query)))):
            clean={(1,"norm"):(positions,native["norm"][1][positions])}
            y=instrument(model,e,roles,{**latent["semantic"],**clean})
            out[arm]={"scores":y["scores"],"pred":predictions(y["scores"])}
        clean=instrument(model,e,roles,{**latent["semantic"],(1,"norm"):(list(range(len(e["ids"]))),native["norm"][1])})
        assert torch.equal(clean["head"],native["head"])
        traces["implicit"]=change(native,implicit,roles)
        traces["explicit_open"]=change(opened["none"],opened["semantic"],roles)
        r,d=g["worlds"][rx],g["worlds"][don]
        target=g["worlds"][alt]["gold"]
        pe=get_encoding(tok,g,r,True)
        assert pe["ids"][:max(src)+1]==e["ids"][:max(src)+1]
        mid={"native":fast_forward(model,pe)[0]}
        for arm,v in (("semantic",semantic),("random",rnd)):
            mid[arm]=fast_forward(model,pe,edits={src_site:(src,v)})[0]
        yield {"id":g["id"],"task":g["task"],"job":"B","family":g["family"],"receive":rx,"donor":don,
               "site":src_site,"original":r["gold"],"counterfactual":target,"mid_original":r["mid"],
               "mid_counterfactual":d["mid"],"mid":mid,"native":native["scores"],"injected":implicit["scores"],
               "donor_native":give[0],"input_cf_native":fast_forward(model,worlds[alt])[0],
               "dose":dose,"edges":{str(k):sum(len(dst)*len(src) for dst,src in r)
                                    for k,r in routes["k2k3_both"].items()},
               "arms":out,"trace":traces}


def run(job,task,model_dir,study):
    path=study/"inputs.jsonl"
    frozen=json.loads((study/"frozen_protocol.json").read_text())
    assert frozen["input_sha256"]==digest(path)
    assert frozen["intervention_sha256"]==digest(Path(__file__))
    split="confirm" if job=="main" else job
    candidates=[g for g in load_lines(path) if g["split"]==split and g["task"]==task]
    folder=study/f"{job}-{task}"
    folder.mkdir()
    model,tok,runtime=load_runtime(model_dir)
    write_json(folder/"runtime.json",runtime)
    count=0
    start=time.perf_counter()
    with torch.inference_mode(), (folder/"rows.jsonl").open("w",encoding="utf-8") as file:
        for g in candidates:
            rows=loop_experiment(model,tok,g) if job=="B" else source_experiment(model,tok,g,job)
            for row in rows:
                dump_line(file,row)
                count+=1
            if count%20==0:
                file.flush()
                print(job,task,count,round(time.perf_counter()-start,1),flush=True)
    torch.cuda.synchronize()
    write_json(folder/"complete.json",{"job":job,"task":task,"groups":len(candidates),"rows":count,
              "gpu_phase_seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("job",choices=("main","A","B","C"))
    p.add_argument("task",choices=("relation","logic","arithmetic"))
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--study",type=Path,required=True)
    a=p.parse_args()
    run(a.job,a.task,a.model,a.study)


if __name__=="__main__":
    main()
