"""Required descriptive L18 nonadditivity; separate from original positive families."""
import datetime,hashlib,json,pathlib,statistics
from fractions import Fraction
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];O=R/'publication/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_latent_mechanism_completion'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text())
assert read(R/'B2_QUEUE_STATE.json')['status']=='COMPLETE'
assert read(R/'posthoc_review/QUEUE_STATE.json')['status']=='COMPLETE'
assert (O/'NATURAL_MEDIATION_RESULTS.md').exists()
freeze=read(R/'reporting/INTERACTION_SUPPLEMENT_FREEZE.json');assert freeze['script_sha256']==sha(__file__)
result={};sources={};rng=np.random.default_rng(20261114);table=[]
for stage in ('train64','train1030','validation299'):
 p=R/'b2_analysis'/stage/'rows.jsonl';sources[str(p.relative_to(R))]=sha(p)
 assert read(R/'posthoc_review'/stage/'INDEPENDENT_REVIEW.json')['status']=='PASS'
 rows=[json.loads(x) for x in p.read_text().splitlines()];result[stage]={}
 for group in ('all','rescue','damage','stable_correct','stable_wrong'):
  for nt in (False,True):
   key=group+('_no_tie' if nt else '');rr=[r for r in rows if (group=='all' or r['natural_group']==group) and (not nt or r['natural_no_tie'])];cells={}
   for metric in ('meanwrong','fixedwrong','dynamic'):
    vals=[];exact=[]
    for r in rr:
     c=r['conditions'];value=c['N']['gaps'][metric]-c['TX']['gaps'][metric]-c['TA']['gaps'][metric]+c['T']['gaps'][metric]
     fraction=Fraction(c['N']['fraction_gaps'][metric])-Fraction(c['TX']['fraction_gaps'][metric])-Fraction(c['TA']['fraction_gaps'][metric])+Fraction(c['T']['fraction_gaps'][metric]);vals.append(value);exact.append((r['id'],fraction))
    arr=np.asarray(vals,dtype=float)
    if len(arr):
     boots=arr[rng.integers(0,len(arr),size=(5000,len(arr)))].mean(axis=1);ci=np.quantile(boots,[.025,.975]).tolist();mean=float(arr.mean());exactmean=str(sum((v for _,v in exact),Fraction(0))/len(exact))
    else:ci=None;mean=None;exactmean=None
    cells[metric]=dict(n=len(rr),mean=mean,ordinary95=ci,exact_fraction_mean=exactmean,positive_ids=[i for i,v in exact if v>0],negative_ids=[i for i,v in exact if v<0],zero_ids=[i for i,v in exact if v==0])
   result[stage][key]=cells
   if stage!='train64' and not nt:
    c=cells['meanwrong'];table.append('|'+ '|'.join([stage,group,str(c['n']),f"{c['mean']:.6f}",str([round(x,6) for x in c['ordinary95']]),str(len(c['positive_ids'])),str(len(c['negative_ids'])),str(len(c['zero_ids']))])+'|')
receipt=dict(status='COMPLETE_DESCRIPTIVE_SUPPLEMENT',definition='I=G(N)-G(TX)-G(TA)+G(T); negative means subadditivity of individual restorations in this measured margin, not duplicated semantic information.',bootstrap_draws=5000,seed=20261114,scope='Descriptive unadjusted intervals; not a new confirmatory family, pass criterion, per-case confidence claim or explanatory fraction. Added after Train1030 review and before Validation299 effects review; no intervention selection changed.',source_sha256=sources,script_sha256=sha(__file__),reporting_freeze_sha256=sha(R/'reporting/INTERACTION_SUPPLEMENT_FREEZE.json'),results=result)
p=O/'results/B2/INTERACTION_DESCRIPTIVE.json';p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(receipt,indent=2))
text='\n## Descriptive controlled L18 interaction\n\nI=G(N)-G(TX)-G(TA)+G(T), using per-question paired scores. Negative I means individual X/A restoration gains sum to more than the joint gain on this receiver and metric; it does not identify a unique semantic redundancy or first writer. Residual restoration is a controlled bypass using x_N while the actual T attention increment remains a_T, followed by the real shared MLP/suffix.\n\n|Cohort|Natural group|n|Mean I|Ordinary95%CI|Exact positive|Exact negative|Exact zero|\n|---|---|---|---|---|---|---|---|\n'+'\n'.join(table)+'\n\nThese descriptive unadjusted intervals are separate from the original6/12/6 positive families. All raw metric variants, exact signs, IDs and native-no-tie sensitivities are in results/B2/INTERACTION_DESCRIPTIVE.json. No new significance gate or explanatory percentage is introduced.\n'
with (O/'NATURAL_MEDIATION_RESULTS.md').open('a') as h:h.write(text)
print('INTERACTION_SUPPLEMENT_COMPLETE')
