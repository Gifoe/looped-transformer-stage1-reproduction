"""Descriptive order-pair comparisons against equal-view ordinary CE."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent / "iteration1"
DEPTHS=(1,2,3,4,6)


def load(cohort,arm):
    return json.loads((HERE/cohort/f"predictions.{arm}.json").read_bytes())


def tied_pair(a,b):
    both=Counter((int(x),int(y)) for x,y in zip(a,b))
    win,loss=both[(0,1)],both[(1,0)]
    return {"n":sum(both.values()),"baseline":both[(1,0)]+both[(1,1)],
            "binding":both[(0,1)]+both[(1,1)],"win":win,"loss":loss,
            "paired_one_sided_p":float(binomtest(win,win+loss,alternative="greater").pvalue) if win+loss else 1.0}


def condition(rows,method,budget):
    q=f"native_k{budget}"
    return [r[method][q]["exact"] for r in rows]


def jsd(rows,k):
    q=f"native_k{k}"
    result=[]
    for r in rows:
        a,b=(np.array(r[v][q]["logits_by_content"]) for v in ("original","permuted"))
        p1,p2=a-logsumexp(a),b-logsumexp(b)
        m=np.logaddexp(p1,p2)-np.log(2.)
        result.append(.5*((np.exp(p1)*(p1-m)).sum()+(np.exp(p2)*(p2-m)).sum()))
    return np.array(result)


def jsd_difference(base,test):
    delta=base-test
    draw=np.random.default_rng(101).integers(0,len(delta),size=(9999,len(delta)))
    ci=np.quantile(delta[draw].mean(1),[.025,.975]).tolist()
    return {"baseline_mean":float(base.mean()),"binding_mean":float(test.mean()),
            "baseline_minus_binding":float(delta.mean()),"paired95_nats":ci}


def main():
    result={}
    for cohort in ("dev","diagnostic","transfer"):
        x=load(cohort,"binding")
        refs={a:load(cohort,a) for a in ("base","ce","distill","dropout")}
        report={}
        for k in DEPTHS:
            ga=[r["gold"] for r in x]
            binding_orig,binding_perm=[condition(x,z,k) for z in ("original","permuted")]
            ca=[a is not None and a==b for a,b in zip(binding_orig,binding_perm)]
            cp=[a==b for a,b in zip(binding_perm,ga)]
            candidate_jsd=jsd(x,k)
            for arm,y in refs.items():
                assert [r["id"] for r in x]==[r["id"] for r in y]
                baseline_orig,baseline_perm=[condition(y,z,k) for z in ("original","permuted")]
                sa=[a is not None and a==b for a,b in zip(baseline_orig,baseline_perm)]
                sp=[a==b for a,b in zip(baseline_perm,ga)]
                common=[i for i,g in enumerate(ga) if binding_orig[i]==baseline_orig[i]==g]
                report.setdefault(arm,{})[k]={"mean_remapped_candidate_distribution_jsd":jsd_difference(jsd(y,k),candidate_jsd),
                      "same_content_all":tied_pair(sa,ca),
                      "permuted_correct_all":tied_pair(sp,cp),
                      "permuted_correct_when_both_originally_correct_descriptive_only":
                          tied_pair([sp[i] for i in common],[cp[i] for i in common])}
        result[cohort]=report
    (HERE/"BINDING_SPECIFICITY.json").write_text(json.dumps(result,indent=2)+"\n")
    print({str(k):tuple(result["transfer"]["ce"][k]["same_content_all"][m]
                        for m in ("baseline","binding","paired_one_sided_p")) for k in DEPTHS})


if __name__=="__main__":
    main()

