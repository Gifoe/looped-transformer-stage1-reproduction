"""Trace a K1 normalized MLP write into K2 propagation and generation."""
import argparse
from pathlib import Path

import torch

from block import Intervene, Trace, decomposition, geom, matched, mixed
from interchange import Patch, complete, fit_margin
from native import Forward, load, output, pad, save, setup, sha, timer
from origin_stats import ARMS, summarize

LAYER = 14


class K2Trace(Trace):
    def put(self, key, name, value):
        if key[1] == LAYER:
            super().put(key, name, value)


def decompose(f, ids, pos, start=None, edit=None):
    trace = K2Trace(f)
    with trace.phases():
        end = f.run(ids, 1, start=start, edit=edit, record=trace)[0]
    y, comp, info = decomposition(f, pos, trace, 0, LAYER)
    return end, y, comp, info, trace


def dose(v, ref):
    return v*(ref/v.norm(dim=-1).clamp_min(1e-12)).unsqueeze(-1)


def check_early_prefix(f, cases):
    valid = [True]*len(cases)
    for j in range(max(len(s["prefix"]) for s in cases)):
        active = [(i,p+s["prefix"][:j],s["prefix"][j]) for i,s in enumerate(cases)
                  if len(s["prefix"])>j for p in s["quad"]]
        ids, at = pad([p for _,p,_ in active])
        for end in f.run(ids,2):
            got = output(f.model,end,at).argmax(-1).tolist()
            for (i,_,wanted),seen in zip(active,got,strict=True):
                valid[i] = valid[i] and wanted==seen
    return valid


def evaluate(f, tok, cases):
    ids, pos = pad([p+c["prefix"] for c in cases for p in c["quad"]])
    i = torch.arange(len(pos),device=pos.device)
    j = i[3::4]
    at = pos[j]
    rec_ids = ids[j]
    wanted = torch.tensor([c["labels"] for c in cases],device=pos.device)
    native1, first_site, first, _, t1 = decompose(f,ids,pos)
    native2, ref, clean, natural_info, t2 = decompose(f,ids,pos,start=native1)
    native = f.run(ids,2)
    assert all(torch.equal(a,b) for a,b in zip(native,(native1,native2),strict=True))
    a1 = t1.saved[(0,LAYER),"m_write"][i,pos]
    m = first["full"][3]
    assert all(c["kind"]=="repair" and d["kind"]=="safe" for c,d in zip(cases[::2],cases[1::2],strict=True))
    # The paired safe recipient receives exactly the matched borrow source vector.
    source = m[::2].repeat_interleave(2,dim=0)
    geometry = geom(first["writes"][1],clean["writes"][1],source,cases,LAYER,901)
    n = len(source)
    full = source.new_zeros(n,4,source.size(-1))
    full[:,3] = -source
    contrast = source.new_tensor([-1,1,1,-1])[None,:,None]/4
    changes = dict(joint=full,quarter=full/4,symmetric=source[:,None]*contrast,
                   source_geometry=geometry[:,None]*contrast)
    for name in ("joint","symmetric"):
        torch.testing.assert_close(mixed(a1+changes[name].flatten(0,1)),mixed(a1)-source,
                                   rtol=3e-4,atol=3e-4)
    restore = {}
    summaries = {}
    states = {"clean":native2}
    backgrounds = {"clean":(native1,ref,clean)}
    for name, change in changes.items():
        k1 = f.run(ids,1,edit=Patch(pos,writes={(LAYER,"m_write"):a1+change.flatten(0,1)}))[0]
        z, site, pert, info, tr = decompose(f,ids,pos,start=k1)
        states[name] = z
        backgrounds[name] = (k1,site,pert)
        delta = [u-v for u,v in zip(clean["full"],pert["full"],strict=True)]
        if name in ("joint","quarter"):
            other = i[i%4!=3]
            torch.testing.assert_close(k1[other],native1[other],rtol=0.,atol=0.)
            old = t2.saved[(0,LAYER),"input"]
            new = tr.saved[(0,LAYER),"input"]
            torch.testing.assert_close(old[other],new[other],rtol=0.,atol=0.)
            for field in (0,2,3):
                torch.testing.assert_close(delta[field],torch.zeros_like(delta[field]),rtol=0.,atol=0.)
            torch.testing.assert_close(delta[1],ref-site,rtol=2e-4,atol=3e-4)
        if name in ("joint","symmetric"):
            restore[name] = {"p":delta[1]}
            if name=="symmetric":
                restore[name].update(m=delta[3],both=delta[1]+delta[3])
        summaries[name] = dict(P_shift=delta[1].norm(dim=-1).tolist(),
                               A_shift=delta[2].norm(dim=-1).tolist(),
                               M_shift=delta[3].norm(dim=-1).tolist(),
                               readout_shift=(ref-site).norm(dim=-1).tolist(),
                               K2_reference_outside=info["outside"])

    def outcome(bg, vector):
        k1, base, _ = backgrounds[bg]
        with Intervene(f,LAYER,at,base+vector):
            h = f.run(rec_ids,1,start=k1[j])[0]
        return output(f.model,h,at), fit_margin(f,h,at,wanted)

    logits = {key:(output(f.model,z[j],at),fit_margin(f,z[j],at,wanted)) for key,z in states.items()}
    for k, z in enumerate((native1,native2)):
        number = output(f.model,z[j],at).argmax(-1).tolist()
        assert number==[c["native"][k]["tokens"][len(c["prefix"])] for c in cases]
    for bg, variants in restore.items():
        for label,v in variants.items():
            anchor = backgrounds[bg][1]
            logits[f"{bg}_restore_{label}"] = outcome(bg,v)
            control = geom(anchor,ref,v,cases,LAYER,913+len(label))
            logits[f"{bg}_restore_{label}_geometry"] = outcome(bg,control)
    for bg in ("clean","symmetric"):
        _, base, cmp = backgrounds[bg]
        p, w = cmp["full"][1], cmp["full"][3]
        for arm,v in (("p",p),("m",w),("both",p+w)):
            logits[f"{bg}_remove_{arm}"] = outcome(bg,-v)
        lower = torch.minimum(p.norm(dim=-1),w.norm(dim=-1))
        for arm,v in (("p",dose(p,lower)),("m",dose(w,lower))):
            logits[f"{bg}_dose_{arm}"] = outcome(bg,-v)
            logits[f"{bg}_dose_{arm}_geometry"] = outcome(bg,-geom(first_site,base,v,cases,LAYER,933))
        verify,_ = outcome(bg,torch.zeros_like(base))
        assert verify.argmax(-1).tolist()==logits[bg][0].argmax(-1).tolist()
    assert set(logits)==set(ARMS)
    chosen = {a:z.argmax(-1).tolist() for a,(z,_) in logits.items()}
    margins = {a:t.tolist() for a,(_,t) in logits.items()}
    new = list(dict.fromkeys((i,v) for vs in chosen.values() for i,v in enumerate(vs)))
    finished = complete(f,tok,[cases[i] for i,_ in new],[v for _,v in new],1)
    lookup = dict(zip(new,finished,strict=True))
    assert [lookup[i,v] for i,v in enumerate(chosen["clean"])]==[c["native"][1] for c in cases]
    rec = []
    for i,c in enumerate(cases):
        rec.append(dict(pred={a:lookup[i,v[i]] for a,v in chosen.items()},
                        first_token={a:v[i] for a,v in chosen.items()},
                        margin={a:v[i] for a,v in margins.items()},
                        source_norm=source[i].norm().item(),
                        native_P=natural_info["propagated"][i],native_A=natural_info["a"][i],
                        native_M=natural_info["m"][i],
                        shifts={a:{key:v[i] for key,v in payload.items()} for a,payload in summaries.items()}))
    return rec


@torch.inference_mode()
def run(model, root):
    started = timer() if torch.cuda.is_initialized() else None
    m,tok = setup(model)
    f = Forward(m)
    if started is None:
        started = timer()
    pairs = matched(f,tok,root,"confirmation")
    src = root/"NATIVE.json"
    path = root/"ORIGIN_RAW.json"
    previous = load(path) if path.exists() else {"rows":[],"elapsed":0.}
    rows = previous["rows"]
    if rows:
        assert previous["native_sha256"]==sha(src)
    expect = [(r["g"]["id"],c["kind"]) for r,s,_,_ in pairs for c in (r["pair"][0],s)]
    assert [(r["id"],r["kind"]) for r in rows]==expect[:len(rows)]
    for i in range(len(rows)//2,len(pairs),3):
        group = pairs[i:i+3]
        flags = [r["aligned"] and all(s["reach"]) for r,s,_,_ in group]
        cases = [c for (r,s,_,_),ok in zip(group,flags,strict=True) if ok for c in (r["pair"][0],s)]
        support = iter(check_early_prefix(f,cases) if cases else [])
        flags = [all((next(support),next(support))) if ok else False for ok in flags]
        cases = [c for (r,s,_,_),ok in zip(group,flags,strict=True) if ok for c in (r["pair"][0],s)]
        measured = iter(evaluate(f,tok,cases) if cases else [])
        for (r,s,margins,distance),ok in zip(group,flags,strict=True):
            for case in (r["pair"][0],s):
                data = next(measured) if ok else dict(pred={a:case["native"][1] for a in ARMS})
                rows.append(dict(id=r["g"]["id"],kind=case["kind"],source_id=case["id"],
                                 target=case["target"],foil=case["foil"],aligned=ok,match=distance,
                                 native=case["native"],**data))
        save(path,dict(rows=rows,native_sha256=sha(src),
                       elapsed=previous["elapsed"]+timer()-started,
                       peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1)))
        print("K1 origin and K2 factorial",len(rows)//2,len(pairs),flush=True)
    data = summarize(rows)
    save(root/"ORIGIN_SUMMARY.json",data)
    print("causal source gates",data["gates"],flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a = p.parse_args()
    run(a.model,a.root)
