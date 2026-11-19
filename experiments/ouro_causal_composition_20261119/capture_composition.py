"""Real two-factor interventions at the native Ouro K1 normalization boundary."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent / "ouro_latent_organization_20261118"))
from capture_org import capture, positions, projection
from manifest_org import TASKS, digest, dump_line, get_encoding, load_lines, write_json
from collect import load_runtime

SOURCE_MANIFEST = "4a255369c16f6fdf52d4f2569f47097a10044f4379ae533d8fc21587f96f608d"
WORLDS = ("r", "x", "y", "d")
MAIN = ("S", "M", "SM")
CONTROLS = ("randS", "randM", "randSM", "noiseS", "noiseSM", "wrongM", "wrongSM")
ARMS = WORLDS+MAIN+CONTROLS
ROLE_NAMES = ("source", "bridge", "mapping", "query", "readout")
RNG_SEED = 2026111919


def key_hash(s):
    import hashlib
    return hashlib.sha256(s.encode()).digest()


def protocol_masks(g, e):
    q = e["roles"]
    if g["task"] == "relation":
        src = sorted(set(q["payload_0"] + q[f'payload_{g["meta"]["other"]}'] + q["bind"]))
    else:
        src = q["signal"]
    return src, q["map"], q["noise"]


def gather(model, e, grid, site=None, detail=False, atlas=True, frame=None):
    edits = {(1,24):site} if site is not None else None
    scores, head, bank, _, _ = capture(model,e,[],edits=edits,full=True,
                                        tap_layers=None if atlas else ())
    k1 = bank[1,24]
    states = torch.stack(list(bank.values())).float()
    f = torch.zeros((len(grid), states.shape[1]), device=states.device)
    for j, role in enumerate(grid):
        f[j,role] = 1/len(role)
    means = f @ states
    if atlas:
        visual = (means @ frame[:,:32]).reshape(4,25,5,32).cpu().numpy()
    else:
        visual = None
    k2 = means[49 if atlas else 1].cpu().numpy()
    if not detail:
        head = None
    return dict(k1=k1, k2=k2, score=scores, head=head, visual=visual)


def combine(a,b):
    return a[0]+b[0],torch.cat((a[1],b[1]))


def dose(x,y):
    return float((x.float()-y.float()).norm())


def scaled(x,d,n):
    return (x.float()+d*n/d.norm().clamp_min(1e-12)).to(x.dtype)


def geometry(records,specification,offset=0):
    d = [records[w]["visual"] for w in specification]
    signal, mapping, dual = (v-d[0] for v in d[1:])
    interaction = dual-signal-mapping
    return np.stack([np.sqrt(np.mean(v*v,axis=-1)) for v in (signal,mapping,dual,interaction)],axis=-1)


def matching(g,enc):
    base=enc["r"]
    assert all(e["roles"]==base["roles"] and e["target"]==base["target"] and len(e["ids"])==len(base["ids"])
               for e in enc.values())
    src,mp,noise=protocol_masks(g,base)
    assert not set(src)&set(mp) and not set(src+mp)&set(noise)
    for a,b,where in (("r","x",src),("r","y",mp),("r","n",noise),
                      ("y","d",src),("x","d",mp)):
        changed={i for i,(v,w) in enumerate(zip(enc[a]["ids"],enc[b]["ids"])) if v!=w}
        assert changed and changed<=set(where),(g["id"],a,b,changed-set(where))
    for a,b in (("r","y"),("x","d")):
        assert g["worlds"][a]["mid"]==g["worlds"][b]["mid"]
    assert g["worlds"]["r"]["mid"]!=g["worlds"]["x"]["mid"]
    return src,mp,noise


def run_group(model,tok,g,physical,controls,technical,frame):
    enc={w:get_encoding(tok,g,g["worlds"][w]) for w in (*WORLDS,"n")}
    src,mp,noise=matching(g,enc)
    base=enc["r"]
    roles=positions(g,base)
    grid=list(roles.values())
    result={w:gather(model,enc[w],grid,detail=technical,frame=frame) for w in WORLDS}
    clean=result["r"]["k1"]
    if technical:
        full_site=(list(range(len(base["ids"]))),result["x"]["k1"])
        parity=gather(model,base,grid,full_site,detail=True,frame=frame)
        assert torch.equal(parity["head"],result["x"]["head"])
        self_site=(src+mp,clean[src+mp])
        self_trace=gather(model,base,grid,self_site,detail=True,frame=frame)
        assert torch.equal(self_trace["head"],result["r"]["head"])
    moves={"S":(src,result["x"]["k1"][src]),"M":(mp,result["y"]["k1"][mp])}
    moves["SM"]=combine(moves["S"],moves["M"])
    if controls:
        seed=int.from_bytes(key_hash("controls/"+g["id"])[:8],"big")
        rng=torch.Generator(device="cuda").manual_seed(seed)
        for k in ("S","M"):
            idx, val=moves[k]
            x=clean[idx]
            delta=torch.randn(x.shape,generator=rng,device=x.device)
            moves["rand"+k]=(idx,scaled(x,delta,dose(x,val)))
        moves["randSM"]=combine(moves["randS"],moves["randM"])
        witness=gather(model,enc["n"],grid,atlas=False,frame=frame)
        x=clean[noise]
        moves["noiseS"]=(noise,scaled(x,witness["k1"][noise].float()-x.float(),
                                    dose(clean[src],moves["S"][1])))
        moves["noiseSM"]=combine(moves["noiseS"],moves["M"])
        m_alt=result["d"]["k1"][mp]
        moves["wrongM"]=(mp,scaled(clean[mp],m_alt.float()-clean[mp].float(),
                                   dose(clean[mp],moves["M"][1])))
        moves["wrongSM"]=combine(moves["S"],moves["wrongM"])
    dosages={}
    if physical:
        for name,edit in moves.items():
            assert len(edit[0])==len(set(edit[0]))
            result[name]=gather(model,base,grid,edit,frame=frame)
            assert result[name]["score"][0]==result["r"]["score"][0]
            dosages[name]=dose(clean[edit[0]],result[name]["k1"][edit[0]])
        if controls:
            for real,control in (("S","randS"),("M","randM"),("S","noiseS"),("M","wrongM")):
                assert abs(dosages[real]-dosages[control])/max(dosages[real],1.e-6)<.04
        # The full K1 boundary edit is exactly additive after assignment, before original K2.
        for w,site in (("S",src),("M",mp),("SM",src+mp)):
            mask=sorted(set(range(len(clean)))-set(site))
            assert torch.equal(result[w]["k1"][mask],clean[mask])
        assert torch.equal(result["SM"]["k1"][src],result["S"]["k1"][src])
        assert torch.equal(result["SM"]["k1"][mp],result["M"]["k1"][mp])
    atlases={"natural":geometry(result,WORLDS)}
    for a,k in ((("r","S","M","SM"),"patch"),(("r","randS","randM","randSM"),"random"),
                (("r","noiseS","M","noiseSM"),"irrelevant"),(("r","S","wrongM","wrongSM"),"mismatch")):
        if all(w in result for w in a):
            atlases[k]=geometry(result,a)
    if physical:
        assert np.max(atlases["patch"][0,24,:,-1]) < .001
    meta=dict(id=g["id"],split=g["split"],task=g["task"],family=g["family"],mode=g["mode"],
              difficulty=g.get("difficulty","legacy"),tokens=len(base["ids"]),
              roles=roles,source_patch=src,map_patch=mp,noise_patch=noise,arm_dose=dosages,
              ids={w:enc[w]["ids"] for w in WORLDS},target_ids=base["target"],
              options=g["options"],intermediate_options=g["intermediate_options"],
              scores={w:tr["score"] for w,tr in result.items()},
              gold={w:g["worlds"][w]["gold"] for w in WORLDS},mid={w:g["worlds"][w]["mid"] for w in WORLDS})
    return meta,result,atlases


def run(inputs,out,checkpoint,task,phase):
    g=[r for r in load_lines(inputs) if r["task"]==task and "meta" in r]
    if phase=="development":
        assert digest(inputs)==SOURCE_MANIFEST
        assert Counter(r["split"] for r in g)=={"fit":300,"discover":100,"confirm":300,"transfer":300}
    if phase=="independent":
        assert Counter(r["split"] for r in g)=={"fresh_same":80,"fresh_structure":80}
    if phase=="pilot":
        g=g[:4]
    out.mkdir(parents=True)
    model,tok,runtime=load_runtime(checkpoint)
    frame=torch.from_numpy(projection()[:,:64].copy()).cuda()
    retrospective={}
    for split in ("confirm","transfer"):
        candidates=[r["id"] for r in g if r["split"]==split]
        retrospective.update({gid:j for j,gid in enumerate(sorted(candidates,
                                   key=lambda s:key_hash("old-composition/"+s))[:60])})
    stats=Counter()
    t0=time.perf_counter()
    raw=np.lib.format.open_memmap(out/"role_k2.npy",mode="w+",dtype=np.float32,
                                  shape=(len(g),len(ARMS),5,2048))
    archive=np.lib.format.open_memmap(out/"atlas.npy",mode="w+",dtype=np.float32,
                                     shape=(len(g),5,4,25,5,4))
    archive.fill(np.nan)
    raw.fill(np.nan)
    index=0
    with (torch.inference_mode(),(out/"groups.jsonl").open("w",encoding="utf-8") as log,
          (out/"token_k1.bf16").open("wb") as binary,(out/"token_k1.f16").open("wb") as projected):
        for i,group in enumerate(g):
            split=group["split"]
            physical=split not in ("confirm","transfer") or group["id"] in retrospective or phase=="pilot"
            control=phase=="pilot" or (physical and int.from_bytes(key_hash("test-controls/"+group["id"])[:8],
                                                                   "big")%4 == 0
                                       and split in ("discover","fresh_same","fresh_structure"))
            info,tr,atlas=run_group(model,tok,group,physical,control,technical=i<4 and phase=="pilot",
                                    frame=frame)
            info["i"]=i
            info["token_offset"]=index
            for w in WORLDS:
                a=tr[w]["k1"]
                binary.write(a.contiguous().view(torch.int16).cpu().numpy().tobytes())
                projected.write((a.float()@frame).half().cpu().numpy().tobytes())
                index+=len(a)
            for k in tr:
                raw[i,ARMS.index(k)]=tr[k]["k2"]
            for j,k in enumerate(("natural","patch","random","irrelevant","mismatch")):
                if k in atlas:
                    archive[i,j]=atlas[k]
            stats[f"{split}/groups"]+=1
            stats[f"{split}/physical"]+=int(physical)
            stats[f"{split}/controls"]+=int(control)
            dump_line(log,info)
            if (i+1)%40==0:
                log.flush()
                print(task,phase,i+1,round(time.perf_counter()-t0,2),flush=True)
    raw.flush()
    archive.flush()
    torch.cuda.synchronize()
    write_json(out/"completion.json",{"time_seconds":time.perf_counter()-t0,"task":task,"phase":phase,
               "source_manifest_sha256":digest(inputs),"capture_sha256":digest(Path(__file__)),
               "counts":dict(stats),"token_total":index, "runtime":runtime,
               "arrays":{name:{"size":(out/name).stat().st_size,"sha256":digest(out/name)}
                         for name in ("role_k2.npy","atlas.npy","token_k1.bf16","token_k1.f16","groups.jsonl")}})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("inputs","out","checkpoint"):
        p.add_argument(k,type=Path)
    p.add_argument("task",choices=TASKS)
    p.add_argument("phase",choices=("pilot","development","independent"))
    a=p.parse_args()
    run(a.inputs,a.out,a.checkpoint,a.task,a.phase)
