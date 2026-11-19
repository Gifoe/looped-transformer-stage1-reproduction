"""Execute native Ouro semantic perturbations and record all-layer causal responses."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from mechanism_design import FORKS, create, dump, encode, load, seed, tableau

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ouro_cross_task_20261118"))
from run import load_runtime

LAYERS = 24
LOOPS = 4
TAPS = ("input", "q", "k", "v", "attention", "attention_update", "mlp", "mlp_update", "residual", "loop_norm")
ROLE_NAMES = ("noise", "source", "bridge", "mapping", "query", "candidate", "label", "readout")
ALIGNED_ROLES = ("source", "bridge", "mapping", "query", "readout")
CONTRASTS = ("source", "mapping", "same_answer", "interaction")
TABLEAU_FORKS = ("open", "content", "letter")
INTERCHANGE_LANES = 20


def add_line(f, x):
    f.write(dump(x) + "\n")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def predict(scores):
    scores = np.asarray(scores)
    return np.where((scores==scores.max(axis=-1,keepdims=True)).sum(axis=-1)==1,
                    scores.argmax(axis=-1), -1).tolist()


def norm_at(model, vec):
    return model.model.norm.forward(vec)


def single_mask(length, batch, device, dtype, max_lookback):
    pos = torch.arange(length, device=device)
    dist = pos[:, None] - pos[None, :]
    ok = dist >= 0
    if max_lookback:
        ok &= dist < max_lookback
    mn = torch.finfo(dtype).min
    return torch.zeros((batch, 1, length, length), device=device, dtype=dtype).masked_fill_(~ok, mn)


@contextmanager
def hooks(model, encs, data, depth, trace=False, full=False, prefix=False, raw=None, edits=None, cuts=None, record_lens=True, intervention_pairs=None):
    core = model.model
    handles, norms = [], []
    event = [0, -1]
    visits = []
    rows = torch.arange(len(encs), device="cuda")
    read = torch.tensor([e["read"] for e in encs], device="cuda")
    contents = torch.tensor([e["content"] for e in encs],device="cuda")
    letters = torch.tensor([e["letter"] for e in encs],device="cuda")
    if trace:
        roles = encs[0]["roles"]
        length = len(encs[0]["ids"])
        weights = torch.zeros((len(ROLE_NAMES), length), device="cuda")
        for r, role in enumerate(ROLE_NAMES):
            pos = roles.get(role, [])
            if pos:
                weights[r, pos] = 1 / len(pos)
        target_content = model.lm_head.weight[contents]
        target_label = model.lm_head.weight[letters]
    if intervention_pairs:
        ac, bc = (torch.tensor(v,device="cuda") for v in zip(*intervention_pairs))
    if prefix:
        end = encs[0]["prefix"]
        assert all(e["ids"][:end] == encs[0]["ids"][:end] and e["prefix"] == end for e in encs)
    old = core.total_ut_steps

    def take(name, out):
        if isinstance(out, tuple):
            out = out[0]
        if prefix and name in ("input", "residual", "loop_norm"):
            equal = (out[:, :end] == out[0:1, :end]).all(dim=(-1, -2))
            data["parity"].append(equal)
        if raw is not None and name in TAPS:
            arr = out[:2, :encs[0]["read"]+1].view(torch.int16).cpu().numpy()
            raw.write(np.ascontiguousarray(arr).tobytes())
            data["raw_layout"].append([*event, name, list(arr.shape)])
        if intervention_pairs:
            rms = (out[ac].float()-out[bc].float()).square().mean(-1).sqrt()
            data["tokens"].setdefault(name, []).append(rms.T)
        if trace:
            h = out.float()
            # Full-width, at every token. Axial responses average squared distances
            # over all six grid edges. Signed answer evidence is recorded separately.
            h = h.view(3, 3, *h.shape[1:])
            s = (h[1:]-h[:1]).square().mean(dim=(0,1,3)).sqrt()
            m = (h[:,1:]-h[:,:1]).square().mean(dim=(0,1,3)).sqrt()
            same = (h[1,2]-h[0,0]).square().mean(-1).sqrt()
            i = (h[1:,1:]-h[1:,:1]-h[:1,1:]+h[:1,:1]).square().mean(dim=(0,1,3)).sqrt()
            vec = torch.stack((s,m,same,i),dim=-1)
            data["trace"].setdefault(name, []).append(weights@vec)
            if full:
                data["tokens"].setdefault(name, []).append(vec)

    def pre(index):
        def f(module, args, kwargs):
            event[:] = [kwargs["current_ut"]+1, index]
            if tuple(event) != (len(visits)//LAYERS+1,len(visits)%LAYERS):
                raise RuntimeError(f"unexpected native order {event}")
            visits.append(tuple(event))
            if trace or prefix or raw is not None or intervention_pairs:
                take("input", args[0])
        return f

    def post(index):
        def f(module, args, kwargs, out):
            if index != event[1]:
                raise RuntimeError("physical layer mismatch")
            if trace or prefix or raw is not None or intervention_pairs:
                take("residual", out)
            if trace and record_lens:
                h = norm_at(model,out[rows,read])
                c = torch.einsum("bd,bod->bo",h,target_content).float()
                l = torch.einsum("bd,bod->bo",h,target_label).float()
                data["lens"].append(torch.stack((c,l),dim=1))
        return f

    def tap(name):
        def f(module, args, out):
            take(name, out)
        return f

    def norm(module, args, out):
        event[1] = 24
        if edits and event[0] in edits:
            out = out.clone()
            for r, positions, values in edits[event[0]]:
                out[r, positions] = values
        norms.append(out[rows, read])
        data["boundary"].append(out)
        if trace or prefix or raw is not None or intervention_pairs:
            take("loop_norm", out)
        return out

    mask_bank = {}
    def edit_attention(module,args,kwargs):
        if event[0] not in (2,3):
            return
        h = kwargs["hidden_states"]
        mask = kwargs["attention_mask"]
        if mask is None:
            mw = module.sliding_window
            if mw not in mask_bank:
                mask_bank[mw] = single_mask(h.shape[1], h.shape[0], h.device, h.dtype, mw)
            mask = mask_bank[mw].clone()
        else:
            mask = mask.expand(len(encs),-1,-1,-1).clone()
        for batch_row, target, source in cuts.get(event[0], []):
            y = torch.tensor(target, device=h.device)[:,None]
            x = torch.tensor(source, device=h.device)[None,:]
            mask[batch_row,:,y,x] = False if mask.dtype == torch.bool else torch.finfo(h.dtype).min
        return args,{**kwargs,"attention_mask":mask}

    try:
        core.total_ut_steps = depth
        for l, layer in enumerate(core.layers):
            handles.append(layer.register_forward_pre_hook(pre(l),with_kwargs=True))
            handles.append(layer.register_forward_hook(post(l),with_kwargs=True))
            if cuts is not None:
                handles.append(layer.self_attn.register_forward_pre_hook(edit_attention, with_kwargs=True))
            if trace or raw is not None or intervention_pairs:
                kwargs = {"q":layer.self_attn.q_proj,"k":layer.self_attn.k_proj,"v":layer.self_attn.v_proj,
                          "attention":layer.self_attn,"attention_update":layer.input_layernorm_2,
                          "mlp":layer.mlp,"mlp_update":layer.post_attention_layernorm_2}
                for name,module in kwargs.items():
                    handles.append(module.register_forward_hook(tap(name)))
        handles.append(core.norm.register_forward_hook(norm))
        yield norms,contents,letters
        if len(visits)!=depth*LAYERS or len(norms)!=depth:
            raise RuntimeError("incomplete native forward")
    finally:
        for h in handles:
            h.remove()
        core.total_ut_steps = old


def forward(model, encs, *, depth=4, trace=False, full=False, prefix=False, raw=None, edits=None, cuts=None, intervention_pairs=None):
    length = max(len(e["ids"]) for e in encs)
    ids = torch.tensor([e["ids"]+[0]*(length-len(e["ids"])) for e in encs],device="cuda")
    data = {"boundary":[],"trace":{},"tokens":{},"lens":[],"parity":[],"raw_layout":[]}
    with hooks(model,encs,data,depth,trace,full,prefix,raw,edits,cuts,intervention_pairs=intervention_pairs) as (norms,contents,letters):
        model.model(input_ids=ids,use_cache=False)
    # Unmodified full-vocabulary native head, FP32 log-softmax with canonical candidate IDs.
    logits = torch.stack([model.lm_head(h) for h in norms])
    lp = logits.float().log_softmax(-1)
    scores = torch.stack((lp.gather(-1,contents.expand(depth,-1,-1)),
                          lp.gather(-1,letters.expand(depth,-1,-1))),dim=2).permute(1,0,2,3)
    data["scores"] = scores.cpu().numpy()
    data["top_token"] = logits.argmax(-1).T.cpu().numpy().tolist()
    data["logits"] = logits
    for key in ("trace","tokens"):
        data[key] = {k: torch.stack(v).half().cpu().numpy() for k,v in data[key].items()}
    if trace:
        data["lens"] = torch.stack(data["lens"]).reshape(depth,LAYERS,len(encs),2,5).cpu().numpy()
    if prefix:
        parity = torch.stack(data["parity"])
        data["parity"] = {"all_tensor_sites":len(parity),"before_divergence_tokens":encs[0]["prefix"],
                          "exact_tensor_sites_by_fork":parity.sum(0).cpu().numpy().tolist()}
        if (parity.sum(0) != len(parity)).any():
            raise RuntimeError(f"prefix invariant failed {data['parity']}")
    return data


def replace_at(tensor, positions, control, magnitude=None):
    base = tensor[positions]
    change = control.float()-base.float()
    if magnitude is not None:
        change = change * (magnitude / change.norm().clamp_min(1e-12))
    return (base.float()+change).to(base.dtype)


def score_record(g,e,values,top,kind,arm=None,loop=None):
    mode = int(e["fork"]=="letter")
    return {"id":g["id"],"split":g["split"],"fork":e["fork"],"kind":kind,"arm":arm,"loop":loop,
            "s":e["s"],"m":e["m"],"y":e["y"],"pred":predict(values[:,mode,:]),
            "pred_content":predict(values[:,0,:]),"pred_letter":predict(values[:,1,:]),
            "scores":values.tolist(),"top_token":top,"content_tokens":e["content"],"letter_tokens":e["letter"]}


def checks(model, base):
    ids = torch.tensor([base["ids"]], device="cuda")
    observed = forward(model,[base])
    with torch.inference_mode():
        for k in range(4):
            old = model.model.total_ut_steps
            model.model.total_ut_steps = k+1
            try:
                native = model(ids,use_cache=False,exit_at_step=k,logits_to_keep=1).logits
            finally:
                model.model.total_ut_steps = old
            if not torch.equal(native[0,0],observed["logits"][k,0]):
                raise RuntimeError(f"native wrapper endpoint mismatch at K{k+1}")
    return {"independent_native_wrapper_bitwise":True,"depths":4}


def interchange(model,g,worlds,natural,loop):
    base = worlds[0]
    s_pos, m_pos = [base["roles"][key] for key in ("source","mapping")]
    norm = natural["boundary"][loop-1]
    encs, edits, tags = [], [], []
    def arm(label,e,source=None,mapping=None):
        idx = len(encs)
        encs.append(e)
        tags.append(label)
        for pos, val in ((s_pos,source),(m_pos,mapping)):
            if val is not None:
                edits.append((idx,pos,val))
    generator = torch.Generator(device="cuda").manual_seed(seed("noise",g["id"],base["fork"],loop))
    def random_delta(pos,target):
        ref = norm[0,pos]
        delta = torch.randn(ref.shape,device=ref.device,generator=generator)
        return replace_at(norm[0],pos,ref.float()+delta,(target.float()-ref.float()).norm())
    for e in worlds:
        s,m=e["s"],e["m"]
        if s==0 and m==0:
            arm("self",e,norm[0,s_pos],norm[0,m_pos])
        else:
            arm("tableau",e,norm[s*3,s_pos],norm[m,m_pos])
    for target_idx in (4,8):
        e=worlds[target_idx]
        s,m=e["s"],e["m"]
        value_s = norm[s*3,s_pos]
        value_m = norm[m,m_pos]
        dose = (value_m.float()-norm[0,m_pos].float()).norm()
        same_y = [i for i,w in enumerate(worlds) if w["y"]==0 and w["s"]!=0]
        other = same_y[0]
        arm("random",e,random_delta(s_pos,value_s),random_delta(m_pos,value_m))
        arm("same_y_mapping_control",e,value_s,replace_at(norm[0],m_pos,norm[other,m_pos],dose))
        arm("oracle_joint_mapping",e,norm[target_idx,s_pos],norm[target_idx,m_pos])
        arm("whole_boundary",e)
        edits.append((len(encs)-1,list(range(len(base["ids"]))),norm[target_idx,:len(base["ids"])]))
    original_count = len(encs)
    clean = [base]*original_count
    pairs = [(0, len(clean), 0)]
    clean.append(base)
    for i,t in enumerate(tags):
        if t=="whole_boundary":
            pairs.append((i,len(clean),encs[i]["s"]*3+encs[i]["m"]))
            clean.append(encs[i])
    if len(clean)!=INTERCHANGE_LANES:
        raise RuntimeError("Mismatched donor and receiver batch widths")
    out = forward(model,clean,edits={loop:edits})
    for edited,check,target_idx in pairs:
        x,y=out["logits"][loop-1:,edited],out["logits"][loop-1:,check]
        if not torch.equal(x,y):
            raise RuntimeError(f"interchange parity failed for {tags[edited]}: {float((x.float()-y.float()).abs().max())}")
        if not np.array_equal(out["scores"][edited,loop-1:],natural["scores"][target_idx,loop-1:]):
            raise RuntimeError("edited endpoint disagrees with native donor scores")
    if not torch.equal(out["logits"][:loop,0],out["logits"][:loop,1]):
        raise RuntimeError("interchange modified earlier scores")
    for i,(tag,e) in enumerate(zip(tags,encs)):
        yield score_record(g,e,out["scores"][i],out["top_token"][i],"interchange",tag,loop)
    return


def route(model,g,base,sample_atlas=None):
    src, noise = [base["roles"][key] for key in ("source","noise")]
    maps = base["roles"]["mapping"]
    query = sorted(set(base["roles"]["query"]+base["roles"]["readout"]))
    unrel = [i for i in range(min(maps)) if i not in maps]
    assert len(unrel)>=len(maps)
    unrel=unrel[-len(maps):]
    tags = ("sham","source_cut","noise_cut","rescue","source_cut_recut","rescue_recut",
            "source_cut_recut_control","rescue_recut_control","recut_only","recut_control_only")
    sham = forward(model,[base]*len(tags),cuts={})
    rescue = sham["boundary"][1][0,maps]
    cuts={2:[(i,maps,src) for i in (1,3,4,5,6,7)]+[(2,maps,noise)],
          3:[(i,query,maps) for i in (4,5,8)]+[(i,query,unrel) for i in (6,7,9)]}
    intervention_pairs=[(1,0),(2,0),(3,1),(5,4),(7,6)] if sample_atlas else None
    out = forward(model,[base]*len(tags),edits={2:[(i,maps,rescue) for i in (3,5,7)]},cuts=cuts,
                  intervention_pairs=intervention_pairs)
    if sample_atlas:
        np.savez_compressed(sample_atlas,contrasts=["cut-sham","noise-sham","rescue-cut",
                           "kept_rescue_when_recut","kept_rescue_under_control_recut"],
                           **{f"token_{k}":v for k,v in out["tokens"].items()})
    if not torch.equal(out["logits"][0],out["logits"][0,:1].expand(len(tags),-1)):
        raise RuntimeError("route mask leaked backwards")
    if not torch.equal(out["logits"][:2,1],out["logits"][:2,3]):
        raise RuntimeError("rescue at K2 boundary changed K2 readout")
    return [score_record(g,base,v,t,"route",arm,2) for arm,v,t in zip(tags,out["scores"],out["top_token"])]


def run_tableaux(model,tok,manifest,out,*,pilot=False):
    groups=list(load(manifest))
    if pilot:
        groups=[next(g for g in groups if g["split"]==split) for split in ("development","new_entities","new_structure")]
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    parity=[]
    raw_path=Path(os.environ.get("OURO_RAW_ATLAS","/dev/shm/ouro_hidden_raw_bf16.bin"))
    with (out/"natural.jsonl").open("w",encoding="utf-8") as nat, (out/"causal.jsonl").open("w",encoding="utf-8") as c, (out/"forks.jsonl").open("w",encoding="utf-8") as f:
        for i,g in enumerate(groups):
            boxes=tableau(tok,g,TABLEAU_FORKS)
            if i==0:
                (out/"native_gate.json").write_text(dump(checks(model,boxes["open"][0])),encoding="utf-8")
            ifaces=[encode(tok,g,0,0,fork) for fork in FORKS]
            fork_data=forward(model,ifaces,prefix=True)
            parity.append({"id":g["id"],"forks":list(FORKS),**fork_data["parity"]})
            for j,e in enumerate(ifaces):
                add_line(f,{**score_record(g,e,fork_data["scores"][j],fork_data["top_token"][j],"fork"),
                            "ids":e["ids"],"roles":e["roles"],"prompt":e["text"],
                            "options":e["options"],"binding":e["by_content"]})
            for fork,worlds in boxes.items():
                sample = (g["id"].endswith(("-000","-001","-002")))
                raw_capture = (not pilot and i==0 and fork=="content")
                if raw_capture:
                    with raw_path.open("xb") as raw:
                        data=forward(model,worlds,trace=True,full=sample,raw=raw)
                    (out/"raw_layout.json").write_text(dump({"path":str(raw_path),"sha256":sha(raw_path),
                        "bytes":raw_path.stat().st_size,"format":"contiguous little-endian BF16 bit patterns",
                        "worlds":[[0,0],[0,1]],"taps":data["raw_layout"],"group":g["id"],"fork":fork}),encoding="utf-8")
                else:
                    data=forward(model,worlds,trace=True,full=sample)
                np.savez_compressed(out/f"{g['id']}_{fork}.npz",role_names=ROLE_NAMES,contrasts=CONTRASTS,
                                   **{f"role_{k}":v for k,v in data["trace"].items()},
                                   **{f"token_{k}":v for k,v in data["tokens"].items()},lens=data["lens"].astype("float16"))
                for j,e in enumerate(worlds):
                    add_line(nat,{**score_record(g,e,data["scores"][j],data["top_token"][j],"natural"),
                                  "ids":e["ids"],"roles":e["roles"],"prompt":e["text"] if sample else None})
                if fork!="letter":
                    patch_bank=forward(model,worlds+[worlds[0]]*(INTERCHANGE_LANES-len(worlds)))
                    for k in (1,2,3):
                        for row in interchange(model,g,worlds,patch_bank,k):
                            add_line(c,row)
                    atlas=out/f"{g['id']}_{fork}_causal.npz" if sample else None
                    for row in route(model,g,worlds[0],atlas):
                        add_line(c,row)
            for stream in (nat,c,f):
                stream.flush()
            if (i+1)%5==0 or i==0:
                print(f"tableaux {i+1}/{len(groups)} elapsed {time.perf_counter()-start:.1f}s",flush=True)
    (out/"prefix_parity.json").write_text(dump(parity),encoding="utf-8")
    (out/"tableaux_complete.json").write_text(dump({"groups":len(groups),"manifest_sha256":sha(manifest),
         "python_sha256":sha(Path(__file__)),"seconds":time.perf_counter()-start,
         "peak_allocated":torch.cuda.max_memory_allocated()}),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("prepare","pilot","tableaux"))
    p.add_argument("checkpoint",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    if args.command=="prepare":
        tok=AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=True)
        rows=list(create(tok))
        with args.manifest.open("x",encoding="utf-8") as file:
            for row in rows:
                add_line(file,row)
        print(f"manifest={len(rows)} SHA256={sha(args.manifest)}",flush=True)
        return
    model,tok,env=load_runtime(args.checkpoint)
    print(dump(env),flush=True)
    with torch.inference_mode():
        run_tableaux(model,tok,args.manifest,args.out,pilot=(args.command=="pilot"))
    (args.out/"runtime.json").write_text(dump(env),encoding="utf-8")


if __name__=="__main__":
    main()
