"""Secondary descriptive margin effects; do not replace the failed repair criterion."""
import csv
import json
import statistics
from linux_runtime import ROOT,save


def describe(v):
    return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),positive=sum(x>0 for x in v),
        zero=sum(x==0 for x in v),negative=sum(x<0 for x in v),min=min(v),max=max(v))


def main():
    with (ROOT/'fresh/feedback_validation/validation_cases.csv').open(encoding='utf8') as f:rows=list(csv.DictReader(f))
    result={}
    for subset in ['all16','transition_survives_dot64_13']:
        rr=rows if subset=='all16' else [r for r in rows if r['transition_survives_dot64']=='True']
        result[subset]={}
        for kind in ['reverse_cut','clean_kv_normmatched']:
            result[subset][kind]=dict(native=describe([float(r[kind+'_margin'])-float(r['on_margin']) for r in rr]),
                dot64=describe([float(r[kind+'_dot64_margin'])-float(r['dot64_on_margin']) for r in rr]))
    save(ROOT/'secondary_margin_effects.json',dict(scope='Post-primary descriptive margin analysis of the fixed16 target set; no new significance claim.',
        results=result,interpretation='Positive margins shifts can occur without repairing an answer. A failed correction gate is not absence of a causal margin effect. Matched direction/random controls and upstream precision robustness remain necessary to interpret semantic specificity.'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
