"""CPU-only public scalar reproduction. No model, probe fit or raw tensor claim."""
import argparse,gzip,hashlib,io,json,pathlib,statistics
from fractions import Fraction
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output');args=ap.parse_args();P=pathlib.Path(args.root).resolve()
SHARDS=json.loads((P/'SHARDED_ARTIFACTS.json').read_text())
def binary(p):
 if p.exists():return p.open('rb')
 item=SHARDS[str(p.relative_to(P))];data=b''.join((P/part['path']).read_bytes() for part in item['parts'])
 assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
 return io.BytesIO(data)
def read(p):
 with binary(p) as raw:
  if p.suffix=='.gz':
   with gzip.GzipFile(fileobj=raw) as h:return json.load(h)
  return json.loads(raw.read().decode())
def lines(p):
 with binary(p) as raw:
  with gzip.GzipFile(fileobj=raw) as z:
   with io.TextIOWrapper(z,encoding='utf-8') as h:
    for line in h:yield json.loads(line)
def gaps(v,g,w):return dict(meanwrong=v[g]-sum(x for i,x in enumerate(v) if i!=g)/(len(v)-1),fixedwrong=v[g]-v[w],dynamic=v[g]-max(x for i,x in enumerate(v) if i!=g))
def close(x,y):assert np.isclose(x,y,rtol=1e-12,atol=1e-13),(x,y)
def definitions():
 a={'loss':{'N':1,'T':-1},'restore':{'U_plus':1,'U':-1},'norm_loss':{'N':1,'upstream_norm':-1},'norm_restore':{'U_plus_norm_preserve':1,'U':-1},'specific_loss':{'upstream_unrelated':1,'T':-1},'specific_restore':{'U_plus':1,'U_plus_unrelated':-1}}
 m={'loss':{'N':1,'M':-1},'restore':{'U_M_plus':1,'U':-1},'norm_loss':{'N':1,'M_norm_preserve':-1},'norm_restore':{'U_M_plus_norm_preserve':1,'U':-1},'specific_loss':{'M_unrelated':1,'M':-1},'specific_restore':{'U_M_plus':1,'U_M_plus_unrelated':-1}}
 b={}
 for branch,restore,remove in [('X','TX','TA'),('A','TA','TX')]:
  b[branch+'_restore']={restore:1,'T':-1};b[branch+'_loss']={'N':1,remove:-1}
  for kind in ('norm','unrelated'):
   b[branch+'_restore_vs_'+kind]={restore:1,f'T_{branch}_{kind}':-1};b[branch+'_loss_vs_'+kind]={f'N_{branch}_{kind}':1,remove:-1}
 return {'attention':a,'L18_branch':b,'MLP_secondary':m}
manifest=read(P/'PACKAGE_MANIFEST.json')
for item in manifest['files']:
 path=(P/item['path']).resolve();assert path.is_relative_to(P)
 assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],item['path']
reports=[]
for stage in ('train64','train1030','validation299'):
 d=P/'results/B2'/stage;rr=list(lines(d/'compact_scalar_evidence.jsonl.gz'));summary=read(d/'summary.json.gz');defs=definitions();scalars=0;intervals=0
 for r in rr:
  c=r['conditions'];g=r['gold'];k1=c['K1'];n=c['N'];group='rescue' if not k1['correct'] and n['correct'] else 'damage' if k1['correct'] and not n['correct'] else 'stable_correct' if k1['correct'] else 'stable_wrong'
  assert group==r['natural_group'] and r['natural_no_tie']==(len(k1['ties'])==len(n['ties'])==1)
  w=max((i for i in range(len(k1['scores'])) if i!=g),key=lambda i:k1['scores'][i])
  for name,z in c.items():
   if z['status']!='OK':continue
   raw=z['candidate_logits'];ties=[i for i,x in enumerate(raw) if x==max(raw)]
   assert ties==z['ties'] and z['prediction']==ties[0] and z['correct']==(ties[0]==g)
   for key,val in gaps(z['scores'],g,w).items():close(val,z['gaps'][key]);scalars+=1
   assert {k:str(v) for k,v in gaps([Fraction(x) for x in raw],g,w).items()}==z['fraction_gaps']
  for family,cells in defs.items():
   for name,coef in cells.items():
    stored=r['families'][family][name]
    if any(c[k]['status']!='OK' for k in coef):assert stored is None;continue
    for metric in ('meanwrong','fixedwrong','dynamic'):
     value=sum(v*c[k]['gaps'][metric] for k,v in coef.items());close(value,stored['production'][metric]);scalars+=1
     exact=sum((v*Fraction(c[k]['fraction_gaps'][metric]) for k,v in coef.items()),Fraction(0));assert exact==Fraction(stored['fraction'][metric])
 rng=np.random.default_rng(20261114)
 for group in ('all','rescue','damage','stable_correct','stable_wrong'):
  for nt in (False,True):
   key=group+('_no_tie' if nt else '');pop=[r for r in rr if (group=='all' or r['natural_group']==group) and (not nt or r['natural_no_tie'])];assert len(pop)==summary[key]['n']
   for family,cells in defs.items():
    passes=[]
    for name in cells:
     valid=[r for r in pop if r['families'][family][name] is not None];values=np.asarray([r['families'][family][name]['production']['meanwrong'] for r in valid]);cell=summary[key]['families'][family]['contrasts'][name]
     assert cell['n_defined']==len(valid) and cell['n_total']==len(pop) and cell['full_defined_coverage']==(len(valid)==len(pop))
     if len(values):
      boots=values[rng.integers(0,len(values),size=(5000,len(values)))].mean(axis=1);lower=float(np.quantile(boots,.05/len(cells)))
      close(float(values.mean()),cell['mean']);close(lower,cell['lower'])
      for a,b in zip(np.quantile(boots,[.025,.975]),cell['ci95']):close(a,b)
      nv=[r['families'][family][name]['production']['meanwrong'] for r in valid if r['natural_no_tie']]
      passed=bool(len(valid)==len(pop) and values.mean()>0 and lower>0 and nv and statistics.mean(nv)>=0);intervals+=1
     else:passed=False
     assert passed==cell['positive_pass'];passes.append(passed)
     for label,sign in [('positive_ids',1),('strict_counterexample_ids',-1),('zero_ids',0)]:
      ids=[r['id'] for r in valid if (Fraction(r['families'][family][name]['fraction']['meanwrong'])>0 if sign==1 else Fraction(r['families'][family][name]['fraction']['meanwrong'])<0 if sign==-1 else Fraction(r['families'][family][name]['fraction']['meanwrong'])==0)]
      assert ids==cell[label]
    assert all(passes)==summary[key]['families'][family]['all_positive_pass']
   for name,cell in summary[key]['conditions'].items():
    valid=[r for r in pop if r['conditions'][name]['status']=='OK'];assert len(valid)==cell['n_defined']
    for label,pred in [('correct_ids',lambda r:r['conditions'][name]['correct']),('tie_ids',lambda r:len(r['conditions'][name]['ties'])>1),('prediction_changed_from_N_ids',lambda r:r['conditions'][name]['prediction']!=r['conditions']['N']['prediction']),('rescued_from_N_ids',lambda r:not r['conditions']['N']['correct'] and r['conditions'][name]['correct']),('damaged_from_N_ids',lambda r:r['conditions']['N']['correct'] and not r['conditions'][name]['correct'])]:assert [r['id'] for r in valid if pred(r)]==cell[label]
 # Fixed E2 scalar accuracy tables, not a repeated frozen-head feature computation.
 e=P/'results/AB1'/stage;counts={}
 for row in lines(e/'E2_ROWS.jsonl.gz'):
  key=row['condition']+'|'+row['head'];v=counts.setdefault(key,[0,0,0]);v[0]+=1;v[1]+=row['probe_correct'];v[2]+=row['native_correct']
 ed=read(e/'E2_DESCRIPTIVE.json.gz')['summary']['all']
 for key,(total,correct,native) in counts.items():assert ed[key]['n']==total;close(correct/total,ed[key]['accuracy']);close(native/total,ed[key]['native_accuracy'])
 reports.append(dict(stage=stage,n=len(rr),scalar_checks=scalars,bootstrap_interval_checks=intervals,frozen_E2_accuracy_cells=len(counts)))
# Original AB1 corner/control signs and its fixed primary bootstrap order.
ab_intervals=0
for stage in ('train64','train1030','validation299'):
 ab=list(lines(P/'results/AB1'/stage/'rows.jsonl.gz'));bs={r['id']:r for r in lines(P/'results/B2'/stage/'compact_scalar_evidence.jsonl.gz')};summary=read(P/'results/AB1'/stage/'summary.json.gz');primary=summary['PRIMARY_PREDECLARED']
 edges={'Q_R_E':('H10','H00'),'Q_R_B':('H11','H01'),'R_Q_E':('H01','H00'),'R_Q_B':('H11','H10')}
 for r in ab:
  assert (r['natural_group'],r['natural_no_tie'],r['gold'])==(bs[r['id']]['natural_group'],bs[r['id']]['natural_no_tie'],bs[r['id']]['gold'])
  k1=bs[r['id']]['conditions']['K1']['scores'];g=r['gold'];fw=max((i for i in range(len(k1)) if i!=g),key=lambda i:k1[i]);assert fw==r['fixedwrong']
  for name,c in r['conditions'].items():
   if c['status']!='OK':continue
   actual=gaps(c['scores'],g,fw);raw=gaps([Fraction(x) for x in c['candidate_logits']],g,fw)
   for metric,key in [('meanwrong','allwrong'),('fixedwrong','fixedwrong'),('dynamic','dynamic')]:close(actual[metric],c['production_gaps'][key]);assert raw[metric]==Fraction(c['fraction_gaps'][key])
   ties=[i for i,x in enumerate(c['candidate_logits']) if x==max(c['candidate_logits'])];assert ties==c['ties'] and ties[0]==c['prediction']
  c=r['conditions']
  for edge,(left,right) in edges.items():
   cell=r['contrasts'][edge]['allwrong'];effect=c[left]['production_gaps']['allwrong']-c[right]['production_gaps']['allwrong'];close(effect,cell['effect'])
   base='H00' if edge.endswith('_E') else 'H11';sign=1 if base=='H00' else -1
   for kind in ('norm','random'):
    if c[edge+'_'+kind]['status']=='OK':close(sign*(c[edge+'_'+kind]['production_gaps']['allwrong']-c[base]['production_gaps']['allwrong']),cell[kind])
   if cell['random'] is not None:close(effect-cell['random'],cell['specificity'])
  close(c['U_Q9_N']['production_gaps']['allwrong']-c['H01']['production_gaps']['allwrong'],r['contrasts']['Q9']['allwrong']['restoration'])
  close(c['H11']['production_gaps']['allwrong']-c['N_Q9_U']['production_gaps']['allwrong'],r['contrasts']['Q9']['allwrong']['transfer_loss'])
 rescue=[r for r in ab if r['natural_group']=='rescue'];arng=np.random.default_rng(20261114)
 def check_interval(values,cell,alpha):
  global ab_intervals
  arr=np.asarray(values);draws=arr[arng.integers(0,len(arr),size=(5000,len(arr)))].mean(axis=1);close(arr.mean(),cell['mean']);close(np.quantile(draws,alpha),cell['lower_one_sided_bonferroni'])
  for x,y in zip(np.quantile(draws,[.025,.975]),cell['ci95']):close(x,y)
  ab_intervals+=1
 for edge in edges:
  for kind in ('effect','norm','specificity'):
   valid=[r for r in rescue if r['contrasts'][edge]['allwrong'][kind] is not None];cell=primary['edges'][edge][kind];check_interval([r['contrasts'][edge]['allwrong'][kind] for r in valid],cell,.05/12)
   nv=[r['contrasts'][edge]['allwrong'][kind] for r in valid if r['natural_no_tie']];assert bool(valid and nv and cell['mean']>0 and cell['lower_one_sided_bonferroni']>0 and statistics.mean(nv)>=0)==cell['pass']
 for kind in ('restoration','transfer_loss'):
  cell=primary['q9'][kind];check_interval([r['contrasts']['Q9']['allwrong'][kind] for r in rescue],cell,.05/2)
 assert primary['Q_route_pass']==all(primary['edges'][e][k]['pass'] for e in ('Q_R_E','Q_R_B') for k in ('effect','norm','specificity'))
 assert primary['R_route_pass']==all(primary['edges'][e][k]['pass'] for e in ('R_Q_E','R_Q_B') for k in ('effect','norm','specificity'))
 assert primary['Q9_path_pass']==(primary['Q_route_pass'] and all(primary['q9'][k]['pass'] for k in ('restoration','transfer_loss')))

# Descriptive factorial interaction has its own ordinary, unadjusted bootstrap.
ip=P/'results/B2/INTERACTION_DESCRIPTIVE.json'
interaction=read(ip);irng=np.random.default_rng(20261114);interaction_intervals=0
assert interaction['status']=='COMPLETE_DESCRIPTIVE_SUPPLEMENT'
for stage in ('train64','train1030','validation299'):
 population=list(lines(P/'results/B2'/stage/'compact_scalar_evidence.jsonl.gz'))
 for group in ('all','rescue','damage','stable_correct','stable_wrong'):
  for nt in (False,True):
   key=group+('_no_tie' if nt else '');selected=[r for r in population if (group=='all' or r['natural_group']==group) and (not nt or r['natural_no_tie'])]
   for metric in ('meanwrong','fixedwrong','dynamic'):
    values=[];exact=[]
    for r in selected:
     c=r['conditions'];values.append(c['N']['gaps'][metric]-c['TX']['gaps'][metric]-c['TA']['gaps'][metric]+c['T']['gaps'][metric])
     exact.append((r['id'],Fraction(c['N']['fraction_gaps'][metric])-Fraction(c['TX']['fraction_gaps'][metric])-Fraction(c['TA']['fraction_gaps'][metric])+Fraction(c['T']['fraction_gaps'][metric])))
    cell=interaction['results'][stage][key][metric];assert cell['n']==len(selected)
    if values:
     arr=np.asarray(values);boots=arr[irng.integers(0,len(arr),size=(5000,len(arr)))].mean(axis=1);close(arr.mean(),cell['mean'])
     for x,y in zip(np.quantile(boots,[.025,.975]),cell['ordinary95']):close(x,y)
     assert Fraction(cell['exact_fraction_mean'])==sum((v for _,v in exact),Fraction(0))/len(exact)
     interaction_intervals+=1
    for name,test in [('positive_ids',lambda v:v>0),('negative_ids',lambda v:v<0),('zero_ids',lambda v:v==0)]:assert cell[name]==[ident for ident,v in exact if test(v)]

result=dict(status='PASS',stages=reports,AB1_primary_intervals=ab_intervals,descriptive_interaction_intervals=interaction_intervals,scope='Public compact-scalar/manifest/table reproduction only. No model inference, raw-tensor audit, probe refit or recomputation of frozen coefficients on unpublished raw features.')
if args.output:pathlib.Path(args.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result))
