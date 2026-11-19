"""Aggregate the prespecified causal contrasts and render figures 2–6."""
import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

from mechanism_design import FORKS, dump, load, seed

K = np.arange(1,5)
PALETTE = ("#2D5986","#B05532","#458264","#765787","#707B86")
SPLITS = ("development","new_entities","new_structure")
PROTOCOLS = ("open","content","letter")
ROLES = ("noise","source","bridge","mapping","query","candidate","label","readout")


def mean_ci(arr, key):
    values=np.asarray(arr,dtype=float)
    idx=np.random.default_rng(seed("boot",key)).integers(len(values),size=(3000,len(values)))
    ci=np.quantile(values[idx].mean(axis=1),[.025,.975],axis=0)
    return {"n":len(values),"mean":values.mean(axis=0).tolist(),"lo":ci[0].tolist(),"hi":ci[1].tolist()}


def scores(row,channel=None):
    channel=int(row["fork"]=="letter") if channel is None else channel
    return np.asarray(row["scores"])[:,channel]


def margin(row,channel=None,target=None,alts=None):
    sg=scores(row,channel)
    y=row["y"] if target is None else target
    wrong=[i for i in range(sg.shape[-1]) if i!=y] if alts is None else alts
    return sg[:,y]-sg[:,wrong].mean(axis=-1)


def hit(row,channel=None):
    if channel is None:
        key="pred"
    else:
        key=("pred_content","pred_letter")[channel]
    if key in row:
        return np.asarray(row[key])==row["y"]
    sg=scores(row,channel)
    return (sg.argmax(axis=-1)==row["y"]) & ((sg==sg.max(axis=-1,keepdims=True)).sum(axis=-1)==1)


def inventory(rows,fields):
    result=defaultdict(list)
    for r in rows:
        result[tuple(r[f] for f in fields)].append(r)
    return result


def behavior(rows):
    tab = {}
    for (split,fork),part in inventory(rows,("split","fork")).items():
        groups = inventory(part,("id",))
        per_group=[]
        for g in groups.values():
            outcomes=np.stack([hit(r) for r in g])
            top=np.asarray([r["top_token"] for r in g])
            targets=np.array([r["letter_tokens" if fork=="letter" else "content_tokens"][r["y"]] for r in g])
            prev,resp=outcomes[:,:-1],outcomes[:,1:]
            per_group.append(np.r_[outcomes.mean(0),(top==targets[:,None]).mean(0), (~prev&resp).mean(0),(prev&~resp).mean(0)])
        tab[f"{split}/{fork}"]={"answers":len(part),
            "accuracy":mean_ci(np.array(per_group)[:,:4],f"{split}-{fork}-acc"),
            "greedy_accuracy":mean_ci(np.array(per_group)[:,4:8],f"{split}-{fork}-greedy"),
            "wrong_to_right":np.array(per_group)[:,8:11].mean(0).tolist(),
            "right_to_wrong":np.array(per_group)[:,11:14].mean(0).tolist(),
            "native_margin":mean_ci([np.mean([margin(r) for r in g],axis=0) for g in groups.values()],f"{split}-{fork}-margin")}
    return tab


def routes(rows):
    by_group=inventory(rows,("split","fork","id"))
    values=defaultdict(list)
    for (split,protocol,_),part in by_group.items():
        d={r["arm"]:r for r in part}
        margins={k:margin(r) for k,r in d.items()}
        hits={k:hit(r) for k,r in d.items()}
        effects={
          "source_path_specific_damage":margins["noise_cut"]-margins["source_cut"],
          "restore_same_receiver":margins["rescue"]-margins["source_cut"],
          "readout_recut_reduces_restoration":
              (margins["rescue_recut_control"]-margins["source_cut_recut_control"])-
              (margins["rescue_recut"]-margins["source_cut_recut"]),
          "readout_recut_without_prior_cut":margins["recut_control_only"]-margins["recut_only"],
          "cut_accuracy_loss":hits["sham"].astype(float)-hits["source_cut"],
          "restored_accuracy":hits["rescue"].astype(float)-hits["source_cut"],
          "recut_rescue_accuracy":
              hits["rescue_recut_control"].astype(float)-hits["source_cut_recut_control"]-
              (hits["rescue_recut"].astype(float)-hits["source_cut_recut"]),
          "native_K2wrong_K4right":(~hits["sham"][1] & hits["sham"][3]),
          "natural_corrected_become_wrong_after_cut":(~hits["sham"][1] & hits["sham"][3] & ~hits["source_cut"][3]),
          "natural_corrected_retain_K4_after_rescue":(~hits["sham"][1] & hits["sham"][3] & hits["rescue"][3]),
        }
        for contrast,value in effects.items():
            values[split,protocol,contrast].append(value)
    return {"/".join(key):mean_ci(v,key) for key,v in values.items()}


def interchanges(rows,original):
    values=defaultdict(list)
    nats={(r["id"],r["fork"],r["s"],r["m"]):r for r in original}
    for (split,fork,group,loop),part in inventory(rows,("split","fork","id","loop")).items():
        arms={(r["s"],r["m"],r["arm"]):r for r in part}
        base=arms[0,0,"self"]
        m=defaultdict(list)
        for s,t in ((1,1),(2,2)):
            joint=arms[s,t,"tableau"]
            joint_targets=[s,t]
            # For either nonzero diagonal, the correct composed target is
            # absent from both corresponding single-factor target slots.
            competitors=sorted(set([0,*joint_targets]))
            assert joint["y"] not in competitors
            score=lambda r:margin(r,target=joint["y"],alts=competitors)
            for name in ("tableau","random","same_y_mapping_control","oracle_joint_mapping","whole_boundary"):
                row=arms[s,t,name]
                m[name+"_vs_base"].append(score(row)-score(base))
                m[name+"_hit"].append(hit(row))
            m["tableau_vs_source_only"].append(score(joint)-score(arms[s,0,"tableau"]))
            m["tableau_vs_rule_only"].append(score(joint)-score(arms[0,t,"tableau"]))
            m["tableau_vs_random"].append(score(joint)-score(arms[s,t,"random"]))
            m["tableau_vs_same_y_mapping_control"].append(score(joint)-score(arms[s,t,"same_y_mapping_control"]))
            m["tableau_vs_oracle"].append(score(joint)-score(arms[s,t,"oracle_joint_mapping"]))
            m["tableau_vs_whole"].append(score(joint)-score(arms[s,t,"whole_boundary"]))
            m["natural_hit"].append(hit(nats[group,fork,s,t]))
        for contrast,v in m.items():
            values[split,fork,loop,contrast].append(np.mean(v,axis=0))
    return {"/".join(map(str,key)):mean_ci(v,key) for key,v in values.items()}



def paired_protocol_routes(rows):
    pair={}
    for (split,protocol,group),part in inventory(rows,("split","fork","id")).items():
        d={r["arm"]:margin(r) for r in part}
        if "recut_control_only" not in d:
            continue
        pair.setdefault((split,group),{})[protocol]=np.stack((
            d["noise_cut"]-d["source_cut"],
            d["rescue"]-d["source_cut"],
            (d["rescue_recut_control"]-d["source_cut_recut_control"])-
            (d["rescue_recut"]-d["source_cut_recut"])))
    out=defaultdict(list)
    for (split,_),p in pair.items():
        if set(p)=={"open","content"}:
            out[split].append(p["content"]-p["open"])
    return {k:mean_ci(v,("paired-protocol-routes",k)) for k,v in out.items()}


def same_horizon_interchanges(rows):
    data={}
    for (split,fork,g,loop),part in inventory(rows,("split","fork","id","loop")).items():
        arm={(r["s"],r["m"],r["arm"]):r for r in part}
        target=[(1,1),(2,2)]
        response=[]
        for s,m in target:
            r=arm[s,m,"tableau"]
            y=r["y"]
            score=lambda t:margin(t,target=y,alts=[0,s])
            response.append(np.stack((
                score(r)-score(arm[s,m,"random"]),
                score(r)-score(arm[s,m,"same_y_mapping_control"]),
                score(r)-score(arm[s,0,"tableau"]),
                score(r)-score(arm[0,m,"tableau"]),
                hit(r))))
        data.setdefault((split,fork,g),{})[loop]=np.mean(response,axis=0)
    output=defaultdict(list)
    for (split,fork,_),d in data.items():
        if set(d)!={1,2,3}:
            continue
        for name,v in (
            ("one_more_loop_after_K1",d[1][:,1]),("one_more_loop_after_K2",d[2][:,2]),
            ("one_more_loop_after_K3",d[3][:,3]),
            ("K2_minus_K1_with_one_more_loop",d[2][:,2]-d[1][:,1]),
            ("K2_minus_K3_with_one_more_loop",d[2][:,2]-d[3][:,3]),
            ("K2_minus_K1_with_two_more_loops",d[2][:,3]-d[1][:,2])):
            output[split,fork,name].append(v)
    return {"/".join(k):mean_ci(v,k) for k,v in output.items()}


def fork_analysis(forks):
    outputs = defaultdict(list)
    for (split,_),part in inventory(forks,("split","id")).items():
        rows = {r["fork"]:r for r in part}
        old_label=rows["content"]["binding"][str(rows["content"]["y"])]
        new_label=rows["relabel"]["binding"][str(rows["relabel"]["y"])]
        for name,r in rows.items():
            ref=rows["open"] if name=="content" else rows["content"]
            h=hit(r)
            d={"correct":h,"greedy":np.asarray(r["top_token"])==r["letter_tokens" if name=="letter" else "content_tokens"][r["y"]],
               "content_margin":margin(r,0),"letter_margin":margin(r,1),
               "content_choice_flip":np.asarray(r["pred_content"])!=ref["pred_content"],
               "content_margin_delta":margin(r,0)-margin(ref,0)}
            if name=="relabel" and old_label!=new_label:
                same_label_target=next(int(k) for k,v in r["binding"].items() if v==old_label)
                ss=scores(r,1)
                d["correct_label_vs_old_label"]=ss[:,r["y"]]-ss[:,same_label_target]
                d["new_label_selected"]=np.asarray(r["pred_letter"])==r["y"]
                d["old_label_selected"]=np.asarray(r["pred_letter"])==same_label_target
            for k,v in d.items():
                outputs[split,name,k].append(v)
    return {"/".join(key):mean_ci(v,key) for key,v in outputs.items()}


def signature(w):
    return w-w.mean(axis=-1,keepdims=True)-(w-w.mean(axis=-1,keepdims=True)).mean(axis=-2,keepdims=True)


def cos(x,y):
    norm=np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)
    return np.einsum("...d,...d->...",x,y)/np.maximum(norm,1e-12)


def forecast_auc(y,x):
    pos=x[y]
    neg=x[~y]
    if len(pos)==0 or len(neg)==0:
        return None
    d=pos[:,None]-neg
    return float(((d>0).astype(float)+.5*(d==0)).mean())


def phases(atlas_dir, original, groups):
    event_names = ("L05","L11","L17","L23")
    layers=[5,11,17,23]
    scores_by_group=inventory(original,("id","fork"))
    fields={p:{} for p in PROTOCOLS}
    for g in groups:
        gid=g["id"]
        for p in PROTOCOLS:
            with np.load(atlas_dir/f"{gid}_{p}.npz") as z:
                lens=z["lens"][:,layers,:,int(p=="letter"),:].astype(float)
            fields[p][gid]=lens.reshape(16,9,5)
    dev=[g["id"] for g in groups if g["split"]=="development"]
    results={}
    for p in ("content","letter"):
        def matrix(gid):
            a=signature(fields["open"][gid]).reshape(16,-1)
            b=signature(fields[p][gid]).reshape(16,-1)
            return cos(a[:,None,:],b[None,:,:])
        learn=np.mean([matrix(gid) for gid in dev],0)
        # Select the dev stage using one common set of physical checkpoints.
        matches=learn.argmax(0)
        legal=[int(np.argmax(learn[:4*(j//4+1),j])) for j in range(16)]
        value={"learn_matrix":learn.tolist(),"selected_open_events":matches.tolist(),
               "causal_time_selected_open_events":legal,"within_loop_events":list(event_names),"held_out":{}}
        for split in SPLITS[1:]:
            gs=[g["id"] for g in groups if g["split"]==split]
            mats=np.stack([matrix(gid) for gid in gs])
            js=np.arange(16)
            d=mats[:,matches,js]-mats[:,js,js]
            staged={"correlation":mats.mean(0).tolist(),"matched_minus_same_event":mean_ci(d,(split,p,"phase")),
                    "mean_match_gain":mean_ci(d.mean(1),(split,p,"phase-average")),"forecast":[]}
            for loop in range(3):
                j=4*loop+3
                xs,targets,cur=[],[],[]
                for gid in gs:
                    data=sorted(scores_by_group[gid,p],key=lambda r:(r["s"],r["m"]))
                    w=fields["open"][gid][legal[j]]
                    y=np.array([r["y"] for r in data])
                    ms=np.array([margin(r)[loop] for r in data])
                    tm=(w[np.arange(9),y]*5-w.sum(1))/4
                    cur.append(ms)
                    xs.append(tm)
                    targets.append(np.stack([hit(r) for r in data]))
                xx,cc,bb=np.concatenate(xs),np.concatenate(cur),np.concatenate(targets)
                out={"from_loop":loop+1,"to_loop":loop+2}
                for name,ix in (("correction",~bb[:,loop]),("retention",bb[:,loop])):
                    out[name]={"n":int(ix.sum()),"positives":int(bb[ix,loop+1].sum()),
                               "aligned_open_margin_auc":forecast_auc(bb[ix,loop+1],xx[ix]),
                               "current_native_margin_auc":forecast_auc(bb[ix,loop+1],cc[ix])}
                staged["forecast"].append(out)
            value["held_out"][split]=staged
        results[p]=value
    return results


def hotpot(rows):
    values=defaultdict(list)
    for (_,fork),part in inventory(rows,("id","fork")).items():
        d={r["arm"]:r for r in part}
        m={tag:margin(r) for tag,r in d.items()}
        out={"native_accuracy":hit(d["natural"]),
             "greedy_accuracy":np.asarray(d["natural"]["top_token"])==d["natural"]["gold_label_token" if fork=="letter" else "gold_content_token"],
             "source_replacement_damage":m["noise_replace"]-m["source_replace"],
             "mapping_replacement_damage":m["noise_replace"]-m["mapping_replace"],
             "route_cut_damage":m["noise_cut"]-m["source_cut"],
             "route_restore":m["rescue"]-m["source_cut"],
             "route_cut_accuracy":hit(d["noise_cut"]).astype(float)-hit(d["source_cut"]),
             "route_restore_accuracy":hit(d["rescue"]).astype(float)-hit(d["source_cut"])}
        for k,v in out.items():
            values[fork,k].append(v)
    return {"/".join(k):mean_ci(v,k) for k,v in values.items()}


def figures(summary,atlas,out,groups):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
      "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"figure.dpi":140})
    out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(out/(name+".pdf"),bbox_inches="tight")
        fig.savefig(out/(name+".png"),dpi=220,bbox_inches="tight")
        plt.close(fig)
    def line(ax,d,label,col,ls="-"):
        ax.plot(K,d["mean"],color=col,label=label,lw=1.8,ls=ls)
        ax.fill_between(K,d["lo"],d["hi"],color=col,alpha=.14,lw=0)
        ax.set_xticks(K,["K1","K2","K3","K4"])
    def zero(ax):
        ax.axhline(0,color="#6b7280",lw=.65,zorder=0)
    def loop_boundaries(ax):
        for x in (23.5,47.5,71.5):
            ax.axvline(x,color="white",lw=.7,alpha=.85)
        ax.set_xticks([11.5,35.5,59.5,83.5],["K1","K2","K3","K4"])

    examples=[g["id"] for g in groups if g["split"]=="new_structure"]
    fig,axs=plt.subplots(2,3,figsize=(11,5.2),layout="constrained")
    for j,fork in enumerate(PROTOCOLS):
        items=[]
        for gid in examples:
            with np.load(atlas/f"{gid}_{fork}.npz") as z:
                items.append(np.stack([z["role_attention_update"].astype(float),z["role_mlp_update"].astype(float)]))
        vals=np.stack(items).mean(0)
        for i,(title,contrast) in enumerate((("Attention update · source contrast",0),("MLP update · rule contrast",1))):
            ax=axs[i,j]
            display=vals[i,:,:,contrast].T.copy()
            if fork=="open":
                display[5:7]=np.nan
            cmap=plt.get_cmap("magma").with_extremes(bad="#d1d5db")
            img=ax.imshow(display,aspect="auto",cmap=cmap,norm=PowerNorm(.45,vmin=0,vmax=.10))
            loop_boundaries(ax)
            ax.set_title(f"{fork.capitalize()} · {title}",fontsize=8)
            ax.set_yticks(range(len(ROLES)),ROLES if not j else [])
    cb=fig.colorbar(img,ax=axs,shrink=.73,pad=.015,extend="max")
    cb.set_label("Full-width BF16 response RMS (power scale)")
    save(fig,"figure_2_token_role_atlas")
    gid=examples[0]
    fig,axs=plt.subplots(2,2,figsize=(9,5.3),layout="constrained")
    for j,fork in enumerate(("open","content")):
        with np.load(atlas/f"{gid}_{fork}.npz") as a, np.load(atlas/f"{gid}_{fork}_causal.npz") as c:
            for i,(x,label) in enumerate(((a["token_residual"][:,:,0],"Natural source swap"),(c["token_residual"][:,:,0],"K2 route cut − sham"))):
                ax=axs[i,j]
                im=ax.imshow(np.log10(1+x.astype(float).T),aspect="auto",cmap="magma")
                loop_boundaries(ax)
                ax.set_title(f"{fork.capitalize()} · {label}")
                ax.set_ylabel("Token index")
                fig.colorbar(im,ax=ax,shrink=.84,label="log₁₀(1 + RMS)")
    save(fig,"figure_2b_full_token_atlas_example")

    fig,axs=plt.subplots(1,3,figsize=(10.5,3.2),layout="constrained",width_ratios=[1,1,1.25])
    for j,p in enumerate(("content","letter")):
        stage=summary["phase"][p]["held_out"]["new_structure"]
        ax=axs[j]
        im=ax.imshow(stage["correlation"],aspect="equal",cmap="RdBu_r",vmin=-1,vmax=1)
        ax.plot(np.arange(16),summary["phase"][p]["selected_open_events"],".",color="#111111",ms=3,label="Fitted stage")
        ax.set_xticks([1.5,5.5,9.5,13.5],["K1","K2","K3","K4"])
        ax.set_yticks([1.5,5.5,9.5,13.5],["K1","K2","K3","K4"])
        ax.set_xlabel(p.capitalize()+" event")
        ax.set_ylabel("Open event")
    cb=fig.colorbar(im,ax=axs[:2],shrink=.62,pad=.01)
    cb.set_label("Answer-response cosine")
    ax=axs[2]
    for k,split in enumerate(SPLITS[1:]):
        for j,p in enumerate(("content","letter")):
            d=summary["phase"][p]["held_out"][split]["mean_match_gain"]
            y=3-k*2-j
            ax.errorbar(d["mean"],y,xerr=[[d["mean"]-d["lo"]],[d["hi"]-d["mean"]]],fmt="o",color=PALETTE[j],capsize=3)
    ax.axvline(0,color="#6b7280",lw=.65)
    ax.set_yticks(range(4),["New structure, letter","New structure, content","New entities, letter","New entities, content"],fontsize=8)
    ax.set_xlabel("Aligned score minus same-event score")
    save(fig,"figure_3_functional_alignment")

    fig,axs=plt.subplots(1,3,figsize=(10.5,3.15),layout="constrained")
    for j,p in enumerate(("open","content")):
        ax=axs[j]
        for i,(key,label) in enumerate((("source_path_specific_damage","Source path damage"),
                                     ("restore_same_receiver","Same-receiver rescue"),
                                     ("readout_recut_reduces_restoration","Rescue lost on K3 re-cut"))):
            line(ax,summary["routes"][f"new_structure/{p}/{key}"],label,PALETTE[i])
        zero(ax)
        ax.set_title(p.capitalize()+" · new structure")
        ax.set_ylabel("Answer-margin contrast (nats)")
    axs[0].legend(frameon=False,fontsize=7,loc="best")
    ax=axs[2]
    for i,(key,label) in enumerate((("route_cut_damage","Source path damage"),("route_restore","Same-receiver rescue"))):
        for p,ls in (("open","-"),("content","--")):
            line(ax,summary["hotpot"][f"{p}/{key}"],p.capitalize()+" · "+label,PALETTE[i],ls)
    zero(ax)
    ax.set_title("HotpotQA bridge validation")
    ax.set_ylabel("Answer-margin contrast (nats)")
    ax.legend(frameon=False,fontsize=6,loc="best")
    save(fig,"figure_4_delayed_causal_use")

    fig,axs=plt.subplots(1,3,figsize=(10.5,3.1),layout="constrained")
    stats=summary["forks"]
    for i,p in enumerate(("open","content","letter","order","relabel","distractor","evidence")):
        line(axs[0],stats[f"new_structure/{p}/correct"],p.capitalize(),PALETTE[i%5],"-" if i<5 else "--")
    axs[0].set_title("Native target accuracy")
    axs[0].set_ylabel("Fraction correct")
    axs[0].legend(frameon=False,fontsize=7,ncol=2)
    for i,p in enumerate(("order","relabel","distractor","evidence")):
        line(axs[1],stats[f"new_structure/{p}/content_choice_flip"],p.capitalize(),PALETTE[i])
    axs[1].set_title("Change relative to fixed content list")
    axs[1].set_ylabel("Fraction with another content choice")
    axs[1].legend(frameon=False,fontsize=7)
    for i,(k,label) in enumerate((("new_label_selected","Correct new label"),("old_label_selected","Original label"))):
        line(axs[2],stats[f"new_structure/relabel/{k}"],label,PALETTE[i])
    axs[2].set_title("Relabel only · gold label changed")
    axs[2].set_ylabel("Fraction selected")
    axs[2].legend(frameon=False,fontsize=7)
    save(fig,"figure_5_counterfactual_forks")
    if summary.get("fork_atlas"):
        fig,axs=plt.subplots(1,2,figsize=(7.7,3.15),layout="constrained")
        d=summary["fork_atlas"]["new_structure"]
        sites=np.arange(96)
        for j,fork in enumerate(("open","letter","order","relabel","distractor","evidence")):
            counts=d[fork]["readout_rms"]
            axs[0].plot(sites,counts,color=PALETTE[j%5],ls="--" if j==5 else "-",label=fork.capitalize(),lw=1.2)
        axs[0].set_ylim(bottom=0)
        axs[0].set_yticks(np.arange(0,.36,.05))
        axs[0].set_ylabel("Answer-position residual RMS")
        axs[0].legend(frameon=False,fontsize=7,ncol=2)
        axs[0].set_title("Fork minus fixed content list")
        focused=d["relabel"]["letter_preference"]
        axs[1].plot(sites,focused["mean"],color=PALETTE[0],lw=1.3)
        axs[1].fill_between(sites,focused["lo"],focused["hi"],color=PALETTE[0],alpha=.17,lw=0)
        zero(axs[1])
        axs[1].set_ylabel("New label minus old label (logit)")
        axs[1].set_title("Relabel-only fork at Answer position")
        for ax in axs:
            loop_boundaries(ax)
        save(fig,"figure_5b_suffix_and_binding")


    fig,axs=plt.subplots(1,3,figsize=(10.5,3.1),layout="constrained")
    it=summary["interchanges"]
    for j,p in enumerate(("open","content")):
        ax=axs[j]
        for i,(key,label) in enumerate((("tableau_vs_base","Composed target vs clean"),
                                      ("tableau_vs_random","Composed target vs random"),
                                      ("tableau_vs_same_y_mapping_control","Composed target vs same-Y donor"),
                                      ("oracle_joint_mapping_vs_base","Oracle mapping vs clean"))):
            ds=[it[f"new_structure/{p}/{k}/{key}"] for k in (1,2,3)]
            marker=[d["mean"][-1] for d in ds]
            ci=np.array([d["lo"][-1] for d in ds]),np.array([d["hi"][-1] for d in ds])
            x=K[:3]+(i-1.5)*.04
            ax.errorbar(x,marker,yerr=[marker-ci[0],ci[1]-marker],fmt="o-",color=PALETTE[i],lw=1.1,ms=4,capsize=2,label=label)
        zero(ax)
        ax.set_xticks(K[:3],["K1","K2","K3"])
        ax.set_xlabel("Boundary intervened on")
        ax.set_ylabel("K4 composed-target contrast (nats)")
        ax.set_title(p.capitalize()+" · new structure")
    axs[0].legend(frameon=False,fontsize=6)
    for i,(k,label) in enumerate((("tableau_hit","Separate donors"),("oracle_joint_mapping_hit","Oracle mapping"),
                                  ("whole_boundary_hit","Whole donor, 20 lanes"),("natural_hit","Native input, 9 lanes"))):
        ds=[it[f"new_structure/open/{j}/{k}"] for j in (1,2,3)]
        point=np.asarray([d["mean"][-1] for d in ds])
        error=np.asarray([point-[d["lo"][-1] for d in ds],[d["hi"][-1] for d in ds]-point])
        axs[2].errorbar(K[:3]+.025*(i-1.5),point,yerr=error,fmt="o-",color=PALETTE[i],label=label,
                        ms=4,lw=1.1,capsize=2)
    axs[2].set_xticks(K[:3],["K1","K2","K3"])
    axs[2].set_title("Independent composition · open")
    axs[2].set_ylabel("K4 composed-target accuracy")
    axs[2].set_xlabel("Boundary intervened on")
    axs[2].legend(frameon=False,fontsize=7)
    save(fig,"figure_6_competing_mechanisms")
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.1),layout="constrained")
    for j,p in enumerate(("open","content")):
        ax=axs[j]
        for i,label in enumerate(("vs matched random","vs same-Y donor","vs source only","vs rule only")):
            ds=[summary["same_horizon"][f"new_structure/{p}/one_more_loop_after_K{k}"] for k in (1,2,3)]
            center=np.array([d["mean"][i] for d in ds])
            ci=np.array([d["lo"][i] for d in ds]),np.array([d["hi"][i] for d in ds])
            ax.errorbar(K[:3]+(i-1.5)*.044,center,yerr=[center-ci[0],ci[1]-center],fmt="o-",color=PALETTE[i],
                        capsize=2,lw=1.2,ms=4,label=label)
        zero(ax)
        ax.set_xticks(K[:3],["K1 → K2","K2 → K3","K3 → K4"])
        ax.set_ylabel("Composed-target contrast (nats)")
        ax.set_title(p.capitalize()+" · exactly one downstream loop")
    axs[0].legend(frameon=False,fontsize=7)
    save(fig,"figure_6b_matched_remaining_depth")


def write_atlas_index(atlas,groups,out):
    with (out/"ATLAS_INDEX.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.writer(file)
        writer.writerow(["id","split","fork","file","sha256","bytes","full_token_contrast"])
        for g in groups:
            for p in PROTOCOLS:
                for suffix in ("","_causal"):
                    x=atlas/f"{g['id']}_{p}{suffix}.npz"
                    if x.is_file():
                        with x.open("rb") as f:
                            d=hashlib.file_digest(f,"sha256").hexdigest()
                        writer.writerow([g["id"],g["split"],p,x.name,d,x.stat().st_size,
                                         bool(suffix or g["id"].endswith(("-000","-001","-002")))])


def main():
    p=argparse.ArgumentParser()
    for x in ("manifest","atlas","real","results","figures"):
        p.add_argument(x,type=Path)
    p.add_argument("--hard",type=Path)
    p.add_argument("--fork-followup",type=Path)
    args=p.parse_args()
    groups=list(load(args.manifest))
    natural=list(load(args.atlas/"natural.jsonl"))
    causal=list(load(args.atlas/"causal.jsonl"))
    forks=list(load(args.atlas/"forks.jsonl"))
    hot=list(load(args.real/"hotpot_arms.jsonl"))
    import json
    complete=json.loads((args.atlas/"tableaux_complete.json").read_text())
    hotpot_complete=json.loads((args.real/"hotpot_complete.json").read_text())
    parity=json.loads((args.atlas/"prefix_parity.json").read_text())
    assert len(groups)==120 and len(natural)==len(groups)*27 and len(forks)==len(groups)*7
    assert len(parity)==len(groups) and all(all(v==o["all_tensor_sites"] for v in o["exact_tensor_sites_by_fork"]) for o in parity)
    assert len(causal)==len(groups)*2*(17*3+10)
    assert len(hot)==64*3*8
    summary={"counts":{"groups":len(groups),"natural":len(natural),"forks":len(forks),"causal":len(causal),
                      "hotpot":len(hot),"atlas_sites_per_shared_prefix":parity[0]["all_tensor_sites"]},
              "completion":complete,"hotpot_completion":hotpot_complete,
              "behavior":behavior(natural),"routes":routes([r for r in causal if r["kind"]=="route"]),
              "interchanges":interchanges([r for r in causal if r["kind"]=="interchange"],natural),
              "forks":fork_analysis(forks),"phase":phases(args.atlas,natural,groups),"hotpot":hotpot(hot),
              "paired_protocol_routes":paired_protocol_routes([r for r in causal if r["kind"]=="route"]),
              "same_horizon":same_horizon_interchanges([r for r in causal if r["kind"]=="interchange"]),
              "same_horizon_metrics":["margin vs random","margin vs same-Y","margin vs source-only",
                                      "margin vs map-only","unique candidate accuracy"]} 
    if args.hard:
        hard=list(load(args.hard/"hotpot_hard_arms.jsonl"))
        hard_meta=json.loads((args.hard/"complete.json").read_text())
        assert len(hard)==hard_meta["groups"]*3*8
        assert not {r["id"] for r in hard}&{r["id"] for r in hot}
        summary["hard_hotpot"]=hotpot(hard)
        summary["hard_hotpot_completion"]=hard_meta
    if args.fork_followup:
        all_gen=list(load(args.fork_followup/"free_generation.jsonl"))
        summary["free_generation"]={"%s/%s/K%d"%key:{"n":len(v),"correct":sum(r["correct"] for r in v)}
                for key,v in inventory(all_gen,("split","fork","k")).items()}
        z=np.load(args.fork_followup/"fork_atlas.npz")
        summary["fork_atlas"]={"shape":list(z["data"].shape),"taps":z["taps"].tolist(),"forks":z["forks"].tolist(),
                               "metrics":z["metrics"].tolist(),"all_question_rms_exact":bool(np.all(z["data"][...,1]==0))}
        assert summary["fork_atlas"]["all_question_rms_exact"]
        selected = [i for i,g in enumerate(groups) if g["split"]=="new_structure"]
        changed = [i for i in selected if groups[i]["new_labels"][groups[i]["order"].index(0)]!="ABCDE"[groups[i]["order"].index(0)]]
        atlas={}
        for j,fork in enumerate(FORKS):
            atlas[fork]={"readout_rms":z["data"][selected,2,:,j,0].mean(axis=0).tolist()}
            if fork=="relabel":
                atlas[fork]["letter_preference"]=mean_ci(z["data"][changed,2,:,j,3],"heldout-fork-label")
        summary["fork_atlas"]["new_structure"]=atlas
    args.results.mkdir(parents=True,exist_ok=True)
    (args.results/"mechanism_summary.json").write_text(dump(summary)+"\n",encoding="utf-8")
    write_atlas_index(args.atlas,groups,args.results)
    figures(summary,args.atlas,args.figures,groups)


if __name__=="__main__":
    main()
