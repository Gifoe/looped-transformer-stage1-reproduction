"""Aligned cross-world edits and matched suffix execution."""
import torch

from dynamics import NativeBatch, interpolation
from study import dump, policy_seed, write_row

TIMES = (.25, .5, .75)
ARMS = ("native", "semantic", "same_y_different_b", "same_b_different_s",
        "random", "unrelated_token", "linear_clean", "linear_semantic")


def put_dose(base, direction, dose):
    d = direction.float()
    return (base.float() + d * dose / d.flatten().norm().clamp_min(1e-12)).to(base.dtype)


def interventions(model, worlds, run):
    wants = ((0, 0, 0), (1, 0, 0), (1, 0, 2), (1, 2, 0))
    roles = ("s", "m", "r")
    rec, donor, same_y, same_b = [next(x for x in worlds if tuple(x[k] for k in roles)==t)
                                for t in wants]
    assert donor["b"] == same_y["b"] != rec["b"] == same_b["b"]
    assert rec["y"] == same_y["y"] == same_b["y"] != donor["y"]
    encs = [rec, donor, same_y, same_b, rec, rec, rec, rec]
    context = NativeBatch(model,encs,trace=True)
    linear = interpolation(context.k1,context.k2,TIMES)
    gen = torch.Generator(device="cuda").manual_seed(policy_seed(rec["group"], "causal"))
    read = rec["roles"]["readout"][-1]
    noise = rec["roles"]["noise"][-1]
    for method, states, suffix in (
        ("native",context.native,lambda t,h:context.branch_path(1,start=h,from_branch=int(48*t))),
        ("A16",context.euler(16),lambda t,h:context.euler(16,start=h,from_index=int(16*t))),
        ("B16",context.branch_path(16),lambda t,h:context.branch_path(16,start=h,from_branch=int(48*t))),
    ):
        for t in TIMES:
            h = states[t]
            edits = h[:1].repeat(len(ARMS),1,1)
            for i in (1,2,3):
                edits[i,read] = h[i,read]
            delta = h[1,read].float() - h[0,read].float()
            dose = delta.norm()
            random = torch.randn(delta.shape,device=delta.device,generator=gen)
            edits[4,read] = put_dose(h[0,read],random,dose)
            edits[5,noise] = put_dose(h[0,noise],delta,dose)
            edits[6:] = linear[t][0].to(edits.dtype)
            edits[7,read] = linear[t][1,read].to(edits.dtype)
            end = suffix(t,edits)[1.]
            # Span intervention and factual control use identical mask and layout.
            # For these SDPA inference calls, identical factual lane inputs must reproduce.
            assert torch.equal(end[0], states[1.][0]), (rec["group"], method, t)
            scores = context.head(end).cpu().tolist()
            for i, name in enumerate(ARMS):
                write_row(run,dict(group=rec["group"],cohort=rec["cohort"],method=method,t=t,
                    arm=name,gold=rec["y"],target=donor["y"],scores=scores[i],role="answer_colon",
                    donor_worlds=[x["item"] for x in encs[:4]], edit_norm=float(
                        (edits[i,noise if name=="unrelated_token" else read].float()
                         - (linear[t][0,read].float() if i>=6
                            else h[0,noise if name=="unrelated_token" else read].float())).norm())))
    return context.counts


def causality(model, groups, out):
    records = []
    with (out/"causal.jsonl").open("w",encoding="utf-8") as f:
        for worlds in groups:
            if worlds[0]["family"] != "twohop":
                continue
            counts = interventions(model,worlds,f)
            records.append(dict(group=worlds[0]["group"], counts=counts))
            f.flush()
            if len(records)%4==0:
                print("causal groups",len(records),flush=True)
    dump(out/"causal_execution.json",records)
