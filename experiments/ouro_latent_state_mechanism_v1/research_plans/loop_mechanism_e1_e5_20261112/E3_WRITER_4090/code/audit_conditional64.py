import json,pathlib,statistics,hashlib
from fractions import Fraction
p=pathlib.Path(r'D:\ouro-e4-e5-20261112\branch_reference\conditional_stage_a_snapshot');s=json.loads((p/'summary.json').read_text());rows=[json.loads(x) for x in (p/'rows.jsonl').read_text().splitlines()];assert len(rows)==len({r['id'] for r in rows})==64;maxerr=0;fracbad=0
for r in rows:
 g=r['gold'];fw=r['fixedwrong'] if 'fixedwrong' in r else None;C=r['conditions']; gap=lambda v: v[g]-sum(x for i,x in enumerate(v) if i!=g)/(len(v)-1)
 for name,c in C.items():
  scores=c['scores'];top=max(scores);ties=[i for i,v in enumerate(scores) if v==top];assert ties==c['ties'] and ties[0]==c['prediction'] and c['correct']==(g==ties[0]);maxerr=max(maxerr,abs(gap(scores)-c['gaps']['allwrong']))
 for fac in ['E','L']:
  names={'rest':(f'P_plus_{fac}_main','P'),'loss':('N',f'N_minus_{fac}_main'),'rest_norm':(f'P_plus_{fac}_norm','P'),'loss_norm':('N',f'N_minus_{fac}_norm'),'rest_unrelated':(f'P_plus_{fac}_unrelated','P'),'loss_unrelated':('N',f'N_minus_{fac}_unrelated')}
  for metric in ('allwrong','fixedwrong','dynamic'):
   d={k:C[a]['gaps'][metric]-C[b]['gaps'][metric] for k,(a,b) in names.items()};f={k:Fraction(C[a]['raw_fraction_gaps'][metric])-Fraction(C[b]['raw_fraction_gaps'][metric]) for k,(a,b) in names.items()};d['rest_specific']=d['rest']-d['rest_unrelated'];d['loss_specific']=d['loss']-d['loss_unrelated'];f['rest_specific']=f['rest']-f['rest_unrelated'];f['loss_specific']=f['loss']-f['loss_unrelated']
   for k in d:maxerr=max(maxerr,abs(d[k]-r['contrasts'][fac]['production'][k][metric]));fracbad+=f[k]!=Fraction(r['contrasts'][fac]['raw_fraction'][k][metric])
assert maxerr<1e-12 and fracbad==0
claims=['rest','loss','rest_norm','loss_norm','rest_specific','loss_specific'];result={}
for fac in ['E','L']:
 group=[r for r in rows if r['actual_group_4090']=='rescue'];nt=[r for r in group if r['natural_no_tie']];result[fac]={}
 for k in claims:
  v=[r['contrasts'][fac]['production'][k]['allwrong'] for r in group];n=[r['contrasts'][fac]['production'][k]['allwrong'] for r in nt];m=statistics.mean(v);mn=statistics.mean(n);sv=s['groups']['rescue']['factors'][fac][k];st=s['groups']['rescue_no_tie']['factors'][fac][k];assert abs(m-sv['mean'])<1e-12 and abs(mn-st['mean'])<1e-12;passed=m>0 and mn>=0;assert passed==s['qualification'][fac]['checks'][k];result[fac][k]={'mean':m,'no_tie_mean':mn,'passed':passed}
 assert all(v['passed'] for v in result[fac].values())==s['qualification'][fac]['qualifies']
out={'status':'STAGE_A64_INDEPENDENT_SAVED_SCORE_CHECK_PASSED','count':64,'rescue':len(group),'no_tie_rescue':len(nt),'max_abs_error':maxerr,'fraction_difference_mismatch':fracbad,'qualification':result,'scope':'Recomputed scores, prediction/ties and Fraction contrasts from saved analysis rows; Fraction source target logits not separately reread; not an independent model forward.'};o=p/'INDEPENDENT_SAVED_SCORE_AUDIT.json';o.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2));print('receiptSHA',hashlib.sha256(o.read_bytes()).hexdigest())
