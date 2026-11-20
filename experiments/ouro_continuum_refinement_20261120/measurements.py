"""Measure full-state, directional and token-level changes on fixed prompts."""
import json
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from solver import Operators, integrate_a, integrate_b, timer, weight_hash

LEVELS=(16,32,64,128,256,512)


def dump(p,data):
    t=p.with_suffix(".tmp")
    t.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n",encoding="utf-8")
    t.replace(p)


def per_token(t,op):
    return [row[:length] for row,length in zip(t.cpu().tolist(),op.lengths.tolist(),strict=True)]


def norm(x,op):
    return (x.float().square().sum(-1)*op.valid).sum(-1).sqrt()


def dist(x,y,op,tokens=False):
    dx=x.float()-y.float()
    g=(norm(dx,op)/norm(y,op).clamp_min(1e-10)).tolist()
    if not tokens:
        return g
    tok=dx.norm(dim=-1)/y.float().norm(dim=-1).clamp_min(1e-10)
    return dict(full=g,tokens=per_token(tok,op))


def cos(x,y,op):
    d=min(x.ndim,y.ndim)
    assert d==3
    return (((x.float()*y.float()).sum(-1)*op.valid).sum(-1) /
            (norm(x,op)*norm(y,op)).clamp_min(1e-12)).clamp(-1,1).tolist()


def path_stats(states,op,scale):
    axis=sorted(states)
    xs=[states[i] for i in axis]
    steps=[b.float()-a.float() for a,b in zip(xs,xs[1:])]
    lengths=[norm(v,op).tolist() for v in steps]
    time=[i/scale for i in axis]
    distance=[dist(x,xs[0],op) for x in xs]
    turn=[]
    bend=[]
    for i,(a,b) in enumerate(zip(steps,steps[1:])):
        overlap=cos(a,b,op)
        turn.append(overlap)
        bend.append([2*math.sqrt(max(0,(1-c)/2))/max((x+y)/2,1e-10)
                     for c,x,y in zip(overlap,lengths[i],lengths[i+1],strict=True)])
    displacement=norm(xs[-1]-xs[0],op).tolist()
    return dict(t=time,to_initial=distance,arc_lengths=lengths,
                speed=[[v/(j-i) for v in ds] for i,j,ds in zip(time,time[1:],lengths)],
                token_speed=[per_token(v.norm(dim=-1)/(j-i),op) for i,j,v in zip(time,time[1:],steps)],
                turn_cos=turn,curvature=bend,
                net_over_path=[d/max(s,1e-10) for d,s in zip(displacement,map(sum,zip(*lengths)),strict=True)])


def paths_compare(a,b,op,scale,token=True):
    ticks=sorted(set(a)&set(b))
    return dict(t=[k/scale for k in ticks],state=[dist(a[k],b[k],op,token) for k in ticks])


def endpoint(x,y,op,anchor):
    z1,z2=op.scores(x),op.scores(y)
    return dict(state=dist(x,y,op,True),direction_cos=cos(x-anchor,y-anchor,op),
                logits_first=z1.tolist(),logits_second=z2.tolist(),
                max_candidate_logit_delta=(z1-z2).abs().amax(-1).tolist(),
                changed_argmax=(z1.argmax(-1)!=z2.argmax(-1)).tolist())


def profile(op, start, mode, log, levels, rk_levels, b_grid):
    calculate=integrate_a if mode=="A" else integrate_b
    result={}
    extras={} if mode=="A" else {"local_grid":b_grid}
    for solver,levels_to_run in (("euler",levels),("rk4",rk_levels)):
        for m in levels_to_run:
            with timer(op,f"{mode}/{solver}/{m}",log):
                result[(solver,m)]=calculate(op,m,start,rk4=solver=="rk4",**extras)
    return result


def analyze_profile(results,op,mode,anchor,b_grid=4):
    pure={k:(v if mode=="A" else v[0]) for k,v in results.items()}
    terminal={k:(v[max(v)] if mode=="A" else v[1]) for k,v in results.items()}
    scale=32 if mode=="A" else 48*b_grid
    comparisons=[]
    for key in pure:
        method,m=key
        linked=[(method,2*m)]
        if method=="rk4":
            linked.extend([("euler",m),("euler",4*m)])
        for right in linked:
            if right in pure:
                comparisons.append(dict(left=key,right=right,
                    path=paths_compare(pure[key],pure[right],op,scale),
                    endpoint=endpoint(terminal[key],terminal[right],op,anchor)))
    fine=max(m for s,m in pure if s=="euler")
    baseline={}
    norm_size={}
    for key,value in pure.items():
        name=f"{key[0]}{key[1]}"
        baseline[name]=endpoint(terminal[key],op.k2,op,anchor)
        if mode=="B":
            norm_size[name]=dist(terminal[key],value[scale],op,True)
    geometry={f"{s}{m}":path_stats(v,op,scale) for (s,m),v in pure.items() if s=="euler"}
    return dict(comparisons=comparisons,native_endpoint=baseline,terminal_norm=norm_size,
                geometry=geometry,reference_m=fine)


def local_defects(op, ref, log, grid):
    rows=[]
    with timer(op,"B/matched_start_local",log):
        for branch in range(48):
            k=branch*grid
            start=ref[k]
            native=start+op.branch(branch,start).float()
            r=ref[k+grid]
            row=dict(branch=branch+1,native_vs_fine=dist(native,r,op,True),
                     native_displacement=dist(native,start,op),fine_displacement=dist(r,start,op),
                     matched_paths={})
            for m in (128,256):
                v,_=integrate_b(op,m,start,start_branch=branch,end_branch=branch+1,local_grid=grid)
                row["matched_paths"][m]=[dist(v[k+j],ref[k+j],op,True) for j in range(1,grid+1)]
            rows.append(row)
    return rows


def normal_benchmark(model, group, full):
    op=Operators(model,group)
    log=[]
    levels=LEVELS if full else (64,128,256)
    b_grid=4 if full else 1
    a=profile(op,op.k1,"A",log,levels,(16,32,64) if full else (),b_grid)
    b=profile(op,op.k1,"B",log,levels,(32,64,128) if full else (),b_grid)
    a_ref=a["euler",levels[-1]]
    b_ref,b_end=b["euler",levels[-1]]
    a_end=a_ref[32]
    b_pre=b_ref[48*b_grid]
    ac_b={i:b_ref[i*48*b_grid//32] for i in a_ref if i*48*b_grid %32==0}
    native_path={i:h for i,h in enumerate(op.actual)}
    native_pre=op.actual[-1]
    data={"items":[w["item"] for w in group],"counts_before_profiles":{k:v-sum(t[k] for t in log)
          for k,v in op.counts.items()},"timings":log,
          "A":analyze_profile(a,op,"A",op.k1,b_grid),"B":analyze_profile(b,op,"B",op.k1,b_grid),
          "cross":paths_compare(a_ref,ac_b,op,32),
          "B_vs_native_path":paths_compare({i:b_ref[i*b_grid] for i in range(49)},native_path,op,48),
          "native_geometry":path_stats(native_path,op,48),
          "endpoints":{"A_vs_B_post":endpoint(a_end,b_end,op,op.k1),
            "A_vs_B_pre":endpoint(a_end,b_pre,op,op.k1),"B_pre_vs_native_pre":endpoint(b_pre,native_pre,op,op.k1),
            "B_post_vs_native_K2":endpoint(b_end,op.k2,op,op.k1),
            "A_vs_native_K2":endpoint(a_end,op.k2,op,op.k1),
            "native_norm":dist(op.k2,native_pre,op,True),"B_norm":dist(b_end,b_pre,op,True),
            "K1_vs_native_K2":dist(op.k1,op.k2,op,True)}}
    if full:
        data["B_local"]=local_defects(op,b_ref,log,b_grid)
        with timer(op,"A/omit_norm/euler/128",log):
            no_norm=integrate_a(op,128,op.k1,terminal_norm=False)
        data["A_norm_intervention"]=dict(pre=endpoint(no_norm[32],a["euler",128][32],op,op.k1),
                         post=endpoint(op.norm(no_norm[32]),a["euler",128][32],op,op.k1))
    return data


def sequence(model, groups, out, stage):
    if stage=="precision":
        precision_benchmark(model,groups,out)
    elif stage=="diagnostics":
        dynamics_benchmark(model,groups,out)
    elif stage=="reference":
        reference_benchmark(model,groups,out)
    elif stage=="stability":
        stability_confirmation(model,groups,out)
    else:
        for family,c in groups.items():
            path=out/f"{stage}_{family}.json"
            if not path.exists():
                data=normal_benchmark(model,c[stage],stage=="primary")
                dump(path,data)


def dtype_probe(op):
    recorded={}
    layer=op.layers[0]
    handles=[]
    for name,module in (("attention_projection",layer.self_attn.q_proj),("mlp_projection",layer.mlp.gate_proj),
                        ("sandwich_norm",layer.input_layernorm_2),("terminal_norm",op.core.norm)):
        def hear(_,args,out,tag=name):
            recorded[tag]={"input":str(args[0].dtype),"output":str(out.dtype)}
        handles.append(module.register_forward_hook(hear))
    try:
        op.field(op.k1)
    finally:
        for handle in handles:
            handle.remove()
    assert len(recorded)==4 and all(
        v["input"]==v["output"]==str(op.dtype) for v in recorded.values())
    return recorded


def field_probe(op, reference, branch_states):
    outputs={}
    for k in (0,16,32):
        x=reference[k]
        with sdpa_kernel(SDPBackend.MATH):
            y=op.field(x)
            q=op.field(x.bfloat16().float()) if op.dtype==torch.float32 else y
        outputs[f"A/{k/32}"]=(op.field(x),y,q)
    for index in (0,1,10,11,24,25,46,47):
        x=branch_states[index]
        with sdpa_kernel(SDPBackend.MATH):
            y=op.branch(index,x)
            q=op.branch(index,x.bfloat16().float()) if op.dtype==torch.float32 else y
        outputs[f"B/{index+1}"]=(op.branch(index,x),y,q)
    return outputs


def precision_benchmark(model, groups, out):
    bf_hash=weight_hash(model,torch.bfloat16)
    comparison={}
    for family,c in groups.items():
        op=Operators(model,c["precision"])
        stages=[]
        a=profile(op,op.k1,"A",stages,(16,32,64,128,256),(16,32,64),1)
        b=profile(op,op.k1,"B",stages,(16,32,64,128,256),(32,64),1)
        with timer(op,"A/euler/256/BF16_state",stages):
            ar=integrate_a(op,256,op.k1,round_state=True)
        with timer(op,"B/euler/256/BF16_state",stages):
            br,br_end=integrate_b(op,256,op.k1,local_grid=1,round_state=True)
        with timer(op,"BF16/default_vs_math_field_probes",stages):
            probes=field_probe(op,a["euler",256],b["euler",256][0])
            activations=dtype_probe(op)
        comparison[family]=dict(anchor=op.k1, native=op.k2, a=a["euler",256], b=b["euler",256],
            a_rk=a["rk4",64], b_rk=b["rk4",64], probes=probes,timings=stages,
            dtype=activations, reports={"A":analyze_profile(a,op,"A",op.k1,1),
                "B":analyze_profile(b,op,"B",op.k1,1),
                "rounded_A":endpoint(ar[32],a["euler",256][32],op,op.k1),
                "rounded_B":endpoint(br_end,b["euler",256][1],op,op.k1),
                "rounded_B_pre":dist(br[48],b["euler",256][0][48],op,True),
                "rounded_A_displacement":dist(ar[32],op.k1,op),
                "rounded_B_displacement":dist(br[48],op.k1,op)})
        dump(out/f"precision_bf16_{family}.json",dict(timings=stages, dtype=activations,
                                                    **comparison[family]["reports"]))
        del op,a,b,ar,br,br_end
    assert weight_hash(model,torch.bfloat16)==bf_hash
    model.float()
    assert {p.dtype for p in model.parameters()}=={torch.float32}
    assert weight_hash(model,torch.bfloat16)==bf_hash
    fp_hash=weight_hash(model,torch.float32)
    for family,c in groups.items():
        op=Operators(model,c["precision"])
        log=[]
        bf=comparison[family]
        anchor=bf["anchor"].float()
        native_parity=op.parity_receipt()
        a=profile(op,anchor,"A",log,(16,32,64,128,256),(16,32,64),1)
        b=profile(op,anchor,"B",log,(16,32,64,128,256),(32,64,128),1)
        with timer(op,"FP32/default_vs_math_field_probes",log):
            probes=field_probe(op,bf["a"],bf["b"][0])
            dtype=dtype_probe(op)
        terms={}
        for name,(default,math_out,round_in) in probes.items():
            bd,bm,_=bf["probes"][name]
            terms[name]={"state_cast_FP32":dist(round_in,math_out,op),
                "bf16_vs_fp32_math_same_rounded_input":dist(bm,round_in,op),
                "bf16_default_vs_math":dist(bd,bm,op),
                "fp32_default_vs_math":dist(default,math_out,op)}
        a_end=a["euler",256][32]
        b_pre=b["euler",256][0][48]
        b_post=b["euler",256][1]
        native_common=op.field(anchor)
        data=dict(item=c["precision"][0]["item"],A=analyze_profile(a,op,"A",anchor,1),
            B=analyze_profile(b,op,"B",anchor,1),dtype=dtype,native_parity=native_parity,
            arithmetic_a=paths_compare(bf["a"],a["euler",256],op,32),
            arithmetic_b=paths_compare(bf["b"][0],b["euler",256][0],op,48),
            terms=terms,timings=log,endpoints={
                "BF16_vs_FP32_K1":dist(bf["anchor"],op.k1,op,True),
                "FP32_K2_own_vs_common_anchor":dist(op.k2,native_common,op,True),
                "BF16_vs_FP32_native_K2":dist(bf["native"],op.k2,op,True),
                "BF16_vs_FP32_A256":endpoint(bf["a"][32],a_end,op,anchor),
                "BF16_vs_FP32_B256_pre":endpoint(bf["b"][0][48],b_pre,op,anchor),
                "BF16_vs_FP32_B256_post":endpoint(bf["b"][1],b_post,op,anchor),
                "B_norm_only_BF16_vs_FP32_same_input":dist(bf["b"][1],op.norm(bf["b"][0][48]),op,True),
                "FP32_A_vs_B_pre":endpoint(a_end,b_pre,op,anchor),
                "FP32_A_vs_B_post":endpoint(a_end,b_post,op,anchor),
                "FP32_A_vs_native_common":endpoint(a_end,native_common,op,anchor),
                "FP32_B_vs_native_common":endpoint(b_post,native_common,op,anchor)})
        dump(out/f"precision_fp32_{family}.json",data)
        del op,a,b,probes,terms,data
    actual=weight_hash(model,torch.float32)
    assert actual==fp_hash and weight_hash(model,torch.bfloat16)==bf_hash
    dump(out/"WEIGHT_PRECISION_AUDIT.json",dict(status="PASS",bf16_hash=bf_hash,fp32_hash=fp_hash,
         fp32_hash_after=actual,all_parameters_fp32=True,original_values_preserved=True,
         allow_tf32=False,matmul_precision=torch.get_float32_matmul_precision(),autocast=False))


def perturb(op,h,amplitude,seed):
    generator=torch.Generator(device=h.device).manual_seed(seed)
    d=torch.randn(h.shape,generator=generator,device=h.device)*op.valid.unsqueeze(-1)
    gain=amplitude*norm(h,op)/norm(d,op).clamp_min(1e-10)
    return h.float()+gain[:,None,None]*d


def commutator(op, branch_states, log):
    result=[]
    with timer(op,"FP32/adjacent_branch_commutator",log):
        for i in range(24):
            x=branch_states[2*i]
            a=op.branch(2*i,x)
            m=op.branch(2*i+1,x)
            mag=(norm(a,op)+norm(m,op)).clamp_min(1e-10)
            levels=[]
            for dt in (.02,.01,.005):
                difference=(op.branch(2*i+1,x+dt*a)-m)-(op.branch(2*i,x+dt*m)-a)
                levels.append(dict(scale=dt,coefficient=(norm(difference,op)/(dt*mag)).tolist(),
                                   relative_two_map_order_gap=(dt*norm(difference,op)/norm(x,op)).tolist()))
            result.append(dict(layer=i,shared_state_direction_cos=cos(a,m,op),levels=levels))
    return result


def dynamics_benchmark(model, groups, out):
    for family,c in groups.items():
        op=Operators(model,c["precision"])
        log=[]
        with timer(op,"BF16/A64/extend_to_T4",log):
            reference=integrate_a(op,64,op.k1,horizon=4,grid=8)
        with timer(op,"BF16/B64/perturb_reference",log):
            branch_ref,_=integrate_b(op,64,op.k1,local_grid=1)
        trials={}
        for seed,amplitude in ((574,1e-3),(575,1e-3),(574,1e-2)):
            anchor=perturb(op,op.k1,amplitude,seed)
            dose=norm(anchor-op.k1,op)
            with timer(op,f"BF16/paired_perturb/{seed}/{amplitude}",log):
                aa=integrate_a(op,64,anchor,horizon=4,grid=8)
                bb,_=integrate_b(op,64,anchor,local_grid=1)
            trials[f"{seed}/{amplitude}"]={
                "A_expansion":[(norm(aa[k]-h,op)/dose).tolist() for k,h in reference.items()],
                "B_expansion":[(norm(bb[k]-h,op)/dose).tolist() for k,h in branch_ref.items()]}
        with timer(op,"BF16/A_field_speed_along_T4",log):
            residuals=[dist(op.field(h),h,op) for h in reference.values()]
        op3=Operators(model,c["precision"],depth=3)
        anchor3=op3.parity_receipt()
        with timer(op3,"BF16/K2_to_K3/A128_B128",log):
            aa3=integrate_a(op3,128,op3.k1)
            bb3,be3=integrate_b(op3,128,op3.k1,local_grid=1)
        dump(out/f"dynamics_{family}.json",dict(horizon_geometry=path_stats(reference,op,8),
              relative_field_speed=residuals,trials=trials,timings=log,K3={"parity":anchor3,
              "A":endpoint(aa3[32],op3.k2,op3,op3.k1),"B":endpoint(be3,op3.k2,op3,op3.k1),
              "cross":endpoint(aa3[32],be3,op3,op3.k1),"B_norm":dist(be3,bb3[48],op3,True)}))
        del op,op3,reference,branch_ref,aa,bb,aa3,bb3,be3
    model.float()
    for family,c in groups.items():
        op=Operators(model,c["precision"])
        log=[]
        with timer(op,"FP32/euler/B128_for_commutator",log):
            states,_=integrate_b(op,128,op.k1,local_grid=1)
        result=commutator(op,states,log)
        dump(out/f"order_{family}.json",dict(timings=log,levels=result))


def reference_benchmark(model, groups, out):
    anchors={}
    for family,c in groups.items():
        op=Operators(model,c["precision"])
        anchors[family]=op.k1.float()
    del op
    model.float()
    for family,c in groups.items():
        path=out/f"reference_{family}.json"
        if path.exists():
            continue
        op=Operators(model,c["precision"])
        h=anchors[family]
        log=[]
        a=profile(op,h,"A",log,(256,512),(64,128),1)
        b=profile(op,h,"B",log,(256,512),(64,128),1)
        a_path=a["rk4",128]
        b_path,b_post=b["rk4",128]
        b_pre=b_path[48]
        a_end=a_path[32]
        with timer(op,"FP32/F_common_anchor",log):
            native=op.field(h)
        a_e=a["euler",512][32]
        b_e=b["euler",512][1]
        dump(path,dict(item=c["precision"][0]["item"],timings=log,
             A=analyze_profile(a,op,"A",h,1),B=analyze_profile(b,op,"B",h,1),
             endpoints={"A_RK128_vs_B_RK128_pre":endpoint(a_end,b_pre,op,h),
                        "A_RK128_vs_B_RK128_post":endpoint(a_end,b_post,op,h),
                        "A_RK128_vs_native_common":endpoint(a_end,native,op,h),
                        "B_RK128_vs_native_common":endpoint(b_post,native,op,h),
                        "A_E512_vs_RK128":endpoint(a_e,a_end,op,h),
                        "B_E512_vs_RK128":endpoint(b_e,b_post,op,h),
                        "B_RK128_Norm":dist(b_post,b_pre,op,True)},
             shared_progress_path=paths_compare(a_path,{i:b_path[int(i*48/32)] for i in a_path
                                                       if i*48%32==0},op,32)))


def stability_confirmation(model, groups, out):
    anchors={}
    for family,c in groups.items():
        op=Operators(model,c["precision"])
        anchors[family]=op.k1.float()
    del op
    model.float()
    for family,c in groups.items():
        filename=out/f"stability_fp32_{family}.json"
        if filename.exists():
            continue
        op=Operators(model,c["precision"])
        log=[]
        h=anchors[family]
        with timer(op,"FP32/A64_T4_B64_baselines",log):
            aa=integrate_a(op,64,h,horizon=4,grid=8)
            bb,_=integrate_b(op,64,h,local_grid=1)
            rel_fields=[dist(op.field(x),x,op) for x in aa.values()]
        trials={}
        for seed,amplitude in ((574,1e-3),(575,1e-3),(574,1e-2)):
            start=perturb(op,h,amplitude,seed)
            dose=norm(start-h,op)
            with timer(op,f"FP32/paired_perturb/{seed}/{amplitude}",log):
                a=integrate_a(op,64,start,horizon=4,grid=8)
                b,_=integrate_b(op,64,start,local_grid=1)
            trials[f"{seed}/{amplitude}"]={"A_expansion":[(norm(a[k]-x,op)/dose).tolist() for k,x in aa.items()],
                      "B_expansion":[(norm(b[k]-x,op)/dose).tolist() for k,x in bb.items()]}
        dump(filename,dict(item=c["precision"][0]["item"],time_A=[k/8 for k in aa],
             progress_B=[k/48 for k in bb],relative_field_speed=rel_fields,trials=trials,timings=log))
