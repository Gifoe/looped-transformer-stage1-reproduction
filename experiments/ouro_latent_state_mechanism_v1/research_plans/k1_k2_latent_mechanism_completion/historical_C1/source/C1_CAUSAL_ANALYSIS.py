"""Frozen CPU-only C1 outcome analysis; Train/Val isolated labels, no Test."""
import argparse
import collections
import datetime
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from C1_AUDIT_TENSOR_IO import archive_receipt

SEED=20261114
BOOT=10000
ENDPOINTS=['N','P','P_plus_L','U','UX','UA','UXA','U_upstream_norm','U_upstream_unrelated',
           'U_X_norm','U_X_unrelated','U_A_norm','U_A_unrelated',
           'N_X_norm_rollback','N_X_unrelated_rollback','N_A_norm_rollback','N_A_unrelated_rollback']
CONTRASTS={
 'total_upstream_loss':{'N':1,'U':-1},
 'X_restore':{'UX':1,'U':-1}, 'X_loss':{'N':1,'UA':-1},
 'A_restore':{'UA':1,'U':-1}, 'A_loss':{'N':1,'UX':-1},
 'interaction':{'N':1,'UX':-1,'UA':-1,'U':1},
 'X_restore_vs_norm':{'UX':1,'U_X_norm':-1},
 'X_restore_vs_unrelated':{'UX':1,'U_X_unrelated':-1},
 'A_restore_vs_norm':{'UA':1,'U_A_norm':-1},
 'A_restore_vs_unrelated':{'UA':1,'U_A_unrelated':-1},
 'X_loss_vs_norm':{'N_X_norm_rollback':1,'UA':-1},
 'X_loss_vs_unrelated':{'N_X_unrelated_rollback':1,'UA':-1},
 'A_loss_vs_norm':{'N_A_norm_rollback':1,'UX':-1},
 'A_loss_vs_unrelated':{'N_A_unrelated_rollback':1,'UX':-1},
 'upstream_loss_vs_norm':{'U_upstream_norm':1,'U':-1},
 'upstream_loss_vs_unrelated':{'U_upstream_unrelated':1,'U':-1},
 'P_forward_restoration':{'P_plus_L':1,'P':-1}}
ROUTE_TESTS=[f'{b}_{direction}{suffix}' for b in ['X','A'] for direction in ['restore','loss']
             for suffix in ['','_vs_norm','_vs_unrelated']]
METRICS=['meanwrong','fixed_K1_wrong','dynamic_wrong','correct']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
 tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def lines(p):
 with Path(p).open(encoding='utf-8-sig') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def score_values(logits,gold,fixed):
 v=[Fraction(float(x)) for x in logits];top=max(v);ties=[i for i,x in enumerate(v) if x==top]
 return {'meanwrong':v[gold]-sum(x for i,x in enumerate(v) if i!=gold)/(len(v)-1),
         'fixed_K1_wrong':v[gold]-v[fixed],
         'dynamic_wrong':v[gold]-max(x for i,x in enumerate(v) if i!=gold),
         'correct':Fraction(int(ties[0]==gold))},ties
def interval(values):
 # Whole-question bootstrap. Shared resamples across the columns of one table.
 x=np.asarray(values,dtype=np.float64);assert x.ndim==2 and np.isfinite(x).all()
 n,k=x.shape
 if n==0:return None
 rng=np.random.default_rng(SEED);means=np.empty((BOOT,k))
 for start in range(0,BOOT,128):
  count=min(128,BOOT-start);weights=rng.multinomial(n,np.full(n,1/n),size=count)
  means[start:start+count]=weights@x/n
 return {'mean':x.mean(0),'lo95':np.quantile(means,.025,axis=0),'hi95':np.quantile(means,.975,axis=0),
         'lo_one_sided_family12':np.quantile(means,.05/12,axis=0)}
def labels(path):
 result={}
 for r in lines(path):
  # Match existing isolated E6/H2 row vocabulary without loading any Test file.
  gold=r.get('gold',r.get('gold_index'))
  if gold is None:raise ValueError('Missing gold index in isolated label row')
  assert r['id'] not in result;result[r['id']]=int(gold)
 return result
def question(row,gold,root):
 split=row['split'];ident=row['id'];stem=hashlib.sha256(ident.encode()).hexdigest()
 rp=root/'references'/split/(stem+'.json');r=read(rp)
 archive_receipt(rp.with_suffix('.safetensors'),r)
 source_rows={'K1':r['conditions']['K1']}
 provenance={'reference_json_sha256':sha(rp),'reference_tensor_sha256':r['tensor_sha256'],'treatments':{}}
 for name in ENDPOINTS:
  if name in ['N','P','U']:source_rows[name]=r['conditions'][name]
  else:
   p=root/'treatments'/split/stem/(name+'.json');c=read(p);source_rows[name]=c
   provenance['treatments'][name]={'metadata_sha256':sha(p)}
   if 'candidate_logits' in c:
    archive_receipt(p.with_suffix('.safetensors'),c)
    provenance['treatments'][name]['tensor_sha256']=c['tensor_sha256']
 k1=source_rows['K1']['candidate_logits'];assert 0<=gold<len(k1)
 fixed=max((i for i in range(len(k1)) if i!=gold),key=lambda i:k1[i])
 endpoint={};fractions={};production_metrics={};ties={}
 for name,c in source_rows.items():
  if 'candidate_logits' not in c:
   endpoint[name]={'status':c.get('status','UNDEFINED')};continue
  v=c['candidate_logits'];assert len(v)==len(k1)
  exact,tie=score_values(v,gold,fixed);pred=tie[0]
  assert c['prediction']==pred and c['top_ties']==tie,(ident,name,'prediction/ties')
  fs=c['scores'];offset_deviation=max(abs((fs[i]-fs[0])-(v[i]-v[0])) for i in range(len(v)))
  production_metrics[name]={'meanwrong':fs[gold]-sum(x for i,x in enumerate(fs) if i!=gold)/(len(fs)-1),
    'fixed_K1_wrong':fs[gold]-fs[fixed],
    'dynamic_wrong':fs[gold]-max(x for i,x in enumerate(fs) if i!=gold),
    'correct':float(pred==gold)}
  endpoint[name]={'status':'OK','prediction':pred,'top_ties':tie,
                  'candidate_logits':v,'scores':fs,**production_metrics[name],'correct':pred==gold,
                  'score_offset_deviation_vs_BF16_logits':offset_deviation,
                  'exact_BF16_logit_gaps':{k:str(x) for k,x in exact.items()}}
  fractions[name]=exact;ties[name]=tie
 k1correct=endpoint['K1']['correct'];ncorrect=endpoint['N']['correct']
 group=('stable_correct' if k1correct else 'rescue') if ncorrect else ('damage' if k1correct else 'stable_wrong')
 effects={};exact_effects={}
 for name,coef in CONTRASTS.items():
  if not set(coef)<=set(fractions):
   effects[name]=None;exact_effects[name]=None;continue
  v={metric:sum(Fraction(coef[e])*fractions[e][metric] for e in coef) for metric in METRICS}
  effects[name]={metric:sum(coef[e]*production_metrics[e][metric] for e in coef) for metric in METRICS}
  exact_effects[name]={metric:str(value) for metric,value in v.items()}
 no_natural_tie=len(ties['K1'])==len(ties['N'])==1
 no_endpoint_tie=len(ties)==18 and all(len(v)==1 for v in ties.values())
 decisions={}
 for b,restore,reverse in [('X','UX','UA'),('A','UA','UX')]:
  rc=[f'U_{b}_norm',f'U_{b}_unrelated'];lc=[f'N_{b}_norm_rollback',f'N_{b}_unrelated_rollback']
  complete=all(c in ties for c in rc+lc)
  lost=ncorrect and not endpoint['U']['correct']
  recovered=lost and endpoint[restore]['correct']
  reciprocal_loss=ncorrect and not endpoint[reverse]['correct']
  relevant=['K1','N','U',restore,reverse]+rc+lc
  strict=complete and all(len(ties[c])==1 for c in relevant)
  restore_specific=complete and all(not endpoint[c]['correct'] for c in rc)
  loss_specific=complete and all(endpoint[c]['correct'] for c in lc)
  decisions[b]={'N_correct_U_wrong':lost,'recovered_from_U':recovered,
     'reciprocal_loss_in_N':reciprocal_loss,
     'strict_controlled_recovery':bool(recovered and strict and restore_specific),
     'strict_controlled_bidirectional':bool(recovered and reciprocal_loss and strict and restore_specific and loss_specific),
     'controls_complete':complete}
 assert source_rows['UXA']['Rlogits_sha256']==source_rows['N']['Rlogits_sha256'],(ident,'joint structural endpoint')
 return {'id':ident,'split':split,'gold_offline_only':gold,'fixed_K1_wrong':fixed,'natural_group':group,
         'no_natural_tie':no_natural_tie,'all_endpoint_no_tie':no_endpoint_tie,
         'endpoints':endpoint,'contrasts':effects,'exact_fraction_contrasts':exact_effects,
         'branch_decisions':decisions,'provenance':provenance}

def summarise(qs,label):
 tables=[];decision_tables=[]
 for group in ['all','rescue','damage','stable_correct','stable_wrong']:
  groupqs=[q for q in qs if group=='all' or q['natural_group']==group]
  for sensitivity in ['all','no_natural_tie','all_endpoint_no_tie']:
   subset=[q for q in groupqs if sensitivity=='all' or q[sensitivity]]
   masks=collections.defaultdict(list)
   for name in CONTRASTS:
    mask=tuple(i for i,q in enumerate(subset) if q['contrasts'][name] is not None)
    masks[mask].append(name)
   cached={}
   for mask,names in masks.items():
    matrix=[[subset[i]['contrasts'][name][m] for name in names for m in METRICS] for i in mask]
    stats=interval(matrix) if mask else None
    for offset,name in enumerate(names):
     cached[name]=(list(mask),None if stats is None else {k:v[offset*len(METRICS):(offset+1)*len(METRICS)] for k,v in stats.items()})
   for name in CONTRASTS:
    mask,stats=cached[name];defined=[subset[i] for i in mask]
    for j,metric in enumerate(METRICS):
     vals=[q['contrasts'][name][metric] for q in defined]
     exactvals=[Fraction(q['exact_fraction_contrasts'][name][metric]) for q in defined]
     tables.append({'cohort':label,'group':group,'sensitivity':sensitivity,'contrast':name,'metric':metric,
       'n_population':len(subset),'n_defined':len(defined),'n_undefined':len(subset)-len(defined),
       'mean':float(stats['mean'][j]) if stats else None,
       'paired95':[float(stats['lo95'][j]),float(stats['hi95'][j])] if stats else None,
       'one_sided95_lower_Bonferroni12':float(stats['lo_one_sided_family12'][j]) if stats else None,
       'positive':sum(v>0 for v in vals),'negative':sum(v<0 for v in vals),'zero':sum(v==0 for v in vals),
       'exact_BF16_logit_fraction_mean':str(sum(exactvals)/len(exactvals)) if exactvals else None,
       'exact_positive':sum(v>0 for v in exactvals),'exact_negative':sum(v<0 for v in exactvals),
       'exact_zero':sum(v==0 for v in exactvals)})
   for branch in ['X','A']:
    decision_tables.append({'cohort':label,'group':group,'sensitivity':sensitivity,'branch':branch,
      'n':len(subset),'counts':{k:sum(q['branch_decisions'][branch][k] for q in subset)
                   for k in ['N_correct_U_wrong','recovered_from_U','reciprocal_loss_in_N',
                             'strict_controlled_recovery','strict_controlled_bidirectional','controls_complete']},
      'strict_bidirectional_ids':[q['id'] for q in subset if q['branch_decisions'][branch]['strict_controlled_bidirectional']]})
 return tables,decision_tables

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--phase',choices=['freeze','train','validation'],required=True)
 p.add_argument('--train-labels',type=Path,default=Path('/root/autodl-tmp/ouro-e6a/next_step_train_diagnosis/TRAIN_ONLY_INPUTS.jsonl'))
 p.add_argument('--val-labels',type=Path,default=Path('/root/autodl-tmp/ouro-e7/H2_VALIDATION_ONLY_INPUTS.jsonl'))
 a=p.parse_args();out=a.root/'analysis';freeze=out/'C1_ANALYSIS_SOURCE_FREEZE_v2.json'
 source={'analysis':sha(__file__),'order':sha(a.root/'ASTRA_ORDER_C1_UPSTREAM_TO_L18.md'),
         'independent_all_endpoint_audit':sha(Path(__file__).with_name('C1_ALL_ENDPOINT_CPU_AUDIT.py')),
         'independent_tensor_reader':sha(Path(__file__).with_name('C1_AUDIT_TENSOR_IO.py')),
         'synthetic_QA_source':sha(Path(__file__).with_name('C1_ANALYSIS_SYNTHETIC_QA.py')),
         'train_labels_isolated':sha(a.train_labels),'validation_labels_isolated':sha(a.val_labels),
         'Train_goldfree_inputs':sha(a.root/'inputs/train.jsonl'),'Val_goldfree_inputs':sha(a.root/'inputs/validation.jsonl')}
 if a.phase=='freeze':
  assert not freeze.exists();write(freeze,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_hashes':source,'bootstrap':{'n':BOOT,'seed':SEED,'sampling':'Whole questions, multinomial counts equivalent to replacement bootstrap; contrasts sharing a defined cohort share draws across all outcome metrics. Reset seed per fixed group/sensitivity/availability cohort.',
    'ordinary_percentiles':[.025,.975],'route_family_single_sided_lower_percentile':.05/12},
    'primary':'Natural-rescue meanwrong; strong route requires all raw/norm/unrelated-adjusted bidirectional tests in remainingTrain1030 and Val299.',
    'undefined':'Retain endpoint and coverage; defined-only means reported. Incomplete primary population does not receive strong full-cohort route support.',
    'selection':'No re-fit, Test, parameter, subset, dose, site or metric selection.',
    'version':'v2: peer source review replaces an unjustified universal 2e-6 score-offset assertion with recorded deviation; independent every-endpoint FP64 normalizer audit with a priori FP32 envelope remains mandatory.',
    'chronology':'v1 source/freeze retained. v2 prepared after the four technical reference inputs were executed, before their scores or any C1 scientific outcome/label comparison were inspected. No scientific operator, metric, estimator, cohort or gate changes.'});print('C1_ANALYSIS_FROZEN',sha(freeze));return
 assert read(freeze)['source_hashes']==source
 t=time.time();split=a.phase;rowfile=a.root/'inputs'/(split+'.jsonl');inputs=list(lines(rowfile))
 assert len(inputs)==(1094 if split=='train' else 299)
 gs=labels(a.train_labels if split=='train' else a.val_labels);assert set(gs)=={r['id'] for r in inputs}
 # Caller must supply complete producer coverage receipts before this analysis.
 gate=a.root/('TRAIN_TREATMENTS_COMPLETE.json' if split=='train' else 'VALIDATION_TREATMENTS_COMPLETE.json')
 assert gate.exists(),'Producer complete-coverage gate missing'
 auditpath=out/(split.upper()+'_ALL_ENDPOINT_CPU_AUDIT.json');audit=read(auditpath)
 assert audit['status']=='PASS' and audit['n_questions']==len(inputs)
 assert audit['source_sha256']==source['independent_all_endpoint_audit'] and audit['analysis_source_sha256']==source['analysis']
 qs=[question(r,gs[r['id']],a.root) for r in inputs]
 discovery=read('/root/autodl-tmp/ouro-e1-e3-offload-20261113/inputs/E1_DISCOVERY64_FROZEN.json')
 ids64={r['id'] for group in discovery['groups'].values() for r in group};assert len(ids64)==64
 cohorts={'train64':[q for q in qs if q['id'] in ids64],
          'train1030':[q for q in qs if q['id'] not in ids64],'train1094':qs} if split=='train' else {'validation299':qs}
 allstats=[];decisions=[]
 for label,cohort in cohorts.items():
  stats,d=summarise(cohort,label);allstats.extend(stats);decisions.extend(d)
 with gzip.open(out/(split.upper()+'_QUESTION_CAUSAL_EVIDENCE.jsonl.gz'),'wt',encoding='utf-8') as f:
  for q in qs:f.write(json.dumps(q,allow_nan=False)+'\n')
 write(out/(split.upper()+'_CONTRAST_TABLES.json'),allstats);write(out/(split.upper()+'_DECISION_TABLES.json'),decisions)
 write(out/(split.upper()+'_COMPLETE.json'),{'status':'PASS','n':len(qs),'source_freeze_sha256':sha(freeze),
     'producer_gate_sha256':sha(gate),'all_endpoint_audit_sha256':sha(auditpath),'CPU_wall_seconds':time.time()-t,'GPU_forwards':0})
 if split=='validation':
  train=read(out/'TRAIN_CONTRAST_TABLES.json');verdict=[]
  for b in ['X','A']:
   failures=[]
   for cohort,table in [('train1030',train),('validation299',allstats)]:
    for name in [x for x in ROUTE_TESTS if x.startswith(b+'_')]:
     s=next(x for x in table if x['cohort']==cohort and x['group']=='rescue' and x['sensitivity']=='all' and x['metric']=='meanwrong' and x['contrast']==name)
     nt=next(x for x in table if x['cohort']==cohort and x['group']=='rescue' and x['sensitivity']=='no_natural_tie' and x['metric']=='meanwrong' and x['contrast']==name)
     if s['n_undefined'] or not s['n_defined'] or s['one_sided95_lower_Bonferroni12']<=0 or nt['mean'] is None or nt['mean']<0:
      failures.append({'cohort':cohort,'contrast':name,'primary':s,'natural_no_tie':nt})
   verdict.append({'branch':b,'strong_specific_bidirectional_route':not failures,'failures':failures})
  write(out/'C1_ROUTE_GATE_VERDICT.json',{'branches':verdict,'scope':'Conditional specified upstream-to-readout route; no complete-boundary mechanism or explained percentage.'})
 print('C1_ANALYSIS_COMPLETE',split,len(qs),flush=True)

if __name__=='__main__':main()
