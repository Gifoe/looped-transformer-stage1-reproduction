"""DRAFT descriptive P/Q/R report; refuses partial split or stale source."""
import json,hashlib,argparse,gzip,os,statistics
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python forbidden')
from pqr_runner import ROOT,OLD,bindings,sha,canonical,write_once
from pqr_metrics import evaluate,describe,NAMES
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--binding',required=True);p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.run).resolve();run.relative_to(ROOT.resolve());out=Path(a.out).resolve();out.relative_to(ROOT.resolve());assert not out.exists();bp=Path(a.binding).resolve();bh=sha(bp);frozen=json.loads(bp.read_text());assert frozen['binding']==bindings();allrows={};sources={}
for split,n in [('train',1119),('validation',299),('test',1172)]:
 cv=run/('coverage_'+split+'.json');c=json.loads(cv.read_text());assert c['status']=='PQR_COMPLETE_CUDA_REPLAY_ZERO' and c['binding_sha256']==bh and c['expected_questions']==c['complete_questions']==n and c['replayed_condition_logits']==4*n and len({q['id'] for q in c['questions']})==n;old=OLD/('coverage_'+split+'.json');oc=json.loads(old.read_text());assert {q['id'] for q in c['questions']}=={q['id'] for q in oc['questions']};items=[]
 for q in c['questions']:
  ident=q['id'];key=hashlib.sha256(ident.encode()).hexdigest();mp=run/split/(key+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==q['metadata_sha256'] and sha(tp)==q['logits_sha256'];j=json.loads(mp.read_text());payload=j['payload'];assert j['payload_sha256']==hashlib.sha256(canonical(payload)).hexdigest() and payload['id']==ident and payload['split']==split and payload['binding_sha256']==bh
  oldm=OLD/split/(key+'.json');assert sha(oldm)==payload['old_metadata_sha256'];oldj=json.loads(oldm.read_text());v=payload['result'];corners=v['corners'];assert set(corners)==set(NAMES);g=v['gold'];wrong=v['fixed_wrong'];scores={k:corners[k]['scores'] for k in NAMES};assert all(len(z)==len(v['target_ids']) and z[g]-z[wrong]==corners[k]['gold_vs_fixed_wrong_gap'] for k,z in scores.items())
  oldresult=oldj['payload']['result'];k1good=oldresult['baseline_K1']['prediction']==g;k2good=oldresult['baseline_K2']['prediction']==g;group={(False,True):'rescue',(True,False):'damage',(True,True):'stable_correct',(False,False):'stable_wrong'}[(k1good,k2good)];no_tie=not oldresult['baseline_K1']['top_tie'] and not oldresult['baseline_K2']['top_tie'];metrics=evaluate({k:corners[k]['gold_vs_fixed_wrong_gap'] for k in NAMES});per_wrong={str(i):evaluate({k:scores[k][g]-scores[k][i] for k in NAMES}) for i in range(len(scores['111'])) if i!=g}
  items.append({'id':ident,'demo':oldj['payload']['identity']['is_demonstration'],'natural_group':group,'natural_no_top_tie':no_tie,'gold':g,'fixed_wrong':wrong,'corners':corners,'metrics':metrics,'candidate_metrics':per_wrong,'metadata_sha256':q['metadata_sha256'],'logits_sha256':q['logits_sha256']})
 allrows[split]=items;sources[split]=sha(cv)
assert sha(bp)==bh and bindings()==frozen['binding']
def summarize(items):
 names=items[0]['metrics'].keys() if items else [];return {'n':len(items),'metrics':{k:describe([r['metrics'][k] for r in items]) for k in names}}
summary={'status':'PQR_FULL_COVERAGE_DESCRIPTIVE_PENDING_REVIEW','binding_sha256':bh,'coverage_sha256':sources,'interpretation':'position-level interventions under each R background; not information-source or attention/MLP localization','splits':{s:{'all':summarize(v),'non_demo':summarize([r for r in v if not r['demo']]),'natural_no_top_tie':summarize([r for r in v if r['natural_no_top_tie']]),'natural_groups':{g:summarize([r for r in v if r['natural_group']==g]) for g in ('rescue','damage','stable_correct','stable_wrong')}} for s,v in allrows.items()}}
raw=out.with_suffix('.jsonl.gz');temp=raw.with_suffix('.gz.partial');assert not raw.exists() and not temp.exists()
with temp.open('xb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
  for split,items in allrows.items():
   for r in items:z.write(canonical({'split':split,**r})+b'\n')
os.link(temp,raw);temp.unlink();summary['rows_sha256']=sha(raw);write_once(out,summary);print('PQR_ANALYSIS_COMPLETE',sha(out))
