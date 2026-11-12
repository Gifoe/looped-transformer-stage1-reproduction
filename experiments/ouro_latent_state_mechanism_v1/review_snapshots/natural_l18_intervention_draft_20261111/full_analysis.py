"""Full-cohort L18 local-swap metrics after independent coverage only."""
import os,json,hashlib,gzip
from pathlib import Path
import numpy as np
from source_input import ROOT,sha
RUN=ROOT/'research_plans/natural_l18_full_run_20261111';COV=ROOT/'research_plans/NATURAL_L18_INDEPENDENT_COVERAGE_20261111.json';GEO=ROOT/'research_plans/natural_r_geometry_20261111';OUT=ROOT/'research_plans/natural_l18_full_analysis_20261111'
def require(x,msg):
 if not x:raise ValueError(msg)
def desc(values):
 a=np.asarray(values,dtype='float64')
 if not len(a):return {'n':0}
 q=np.quantile(a,[.05,.25,.5,.75,.95]);return {'n':len(a),'mean':float(a.mean()),'median':float(q[2]),'q05':float(q[0]),'q25':float(q[1]),'q75':float(q[3]),'q95':float(q[4]),'positive':int((a>0).sum()),'negative':int((a<0).sum()),'zero':int((a==0).sum())}
def main():
 c=json.loads(COV.read_text());require(c['status']=='INDEPENDENT_L18_CUDA_AND_SOURCE_COVERAGE_PASSED' and c['count']==2590,'coverage');entries={(q['split'],q['id']):q for q in c['questions']};require(len(entries)==2590,'duplicate');source=json.loads((GEO/'source.json').read_text());rows=[]
 for split in ('train','validation','test'):
  p=GEO/(split+'_rows.jsonl');require(sha(p)==source[split]['rows_sha256'],'geometry SHA')
  with p.open() as f:
   for line in f:
    natural=json.loads(line);key=(split,natural['id']);require(key in entries,'missing ID');q=entries[key];name=split+'_'+hashlib.sha256(natural['id'].encode()).hexdigest();mp=RUN/(name+'.json');require(sha(mp)==q['metadata_sha256'],'metadata SHA');m=json.loads(mp.read_text());require(m['id']==natural['id'] and m['gold']==natural['gold'] and m['targets']==natural['targets'] and m['fixed_wrong']==natural['fixed_wrong'] and m['scores']['B1']==natural['scores'][0] and m['scores']['B2']==natural['scores'][1],'natural identity')
    scores=m['scores'];gold=m['gold'];wrong=m['fixed_wrong'];others=[j for j in range(len(m['targets'])) if j!=gold]
    def metrics(b,t):
     if t not in scores:return None
     per=[(scores[b][gold]-scores[b][j])-(scores[t][gold]-scores[t][j]) for j in others]
     return {'all_wrong':float(np.mean(per)),'fixed_wrong':(scores[b][gold]-scores[b][wrong])-(scores[t][gold]-scores[t][wrong]),'per_wrong':per}
    # L2=B2-T21; G1=T12-B1.
    L2=metrics('B2','T21');G1=metrics('T12','B1');L2norm=metrics('B2','T21_norm');G1norm=metrics('T12_norm','B1') if 'T12_norm' in scores else None
    rows.append({'id':natural['id'],'split':split,'group':natural['group'],'demo':natural['demo'],'natural_no_top_tie':natural['natural_no_top_tie'],'gold':gold,'fixed_wrong':wrong,'targets':m['targets'],'scores':scores,'predictions':m['predictions'],'top_ties':m['top_ties'],'controls':m['controls'],'L2':L2,'G1':G1,'L2_norm':L2norm,'G1_norm':G1norm})
 require(len(rows)==2590 and {(r['split'],r['id']) for r in rows}==set(entries),'full IDs');OUT.mkdir(exist_ok=False)
 rowsfile=OUT/'rows.jsonl.gz'
 with gzip.open(rowsfile,'xt') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
 summary={}
 for split in ('train','validation','test'):
  base=[r for r in rows if r['split']==split];groups={'all':base,'natural_no_tie':[r for r in base if r['natural_no_top_tie']]}
  if split=='train':groups.update({'non_demo':[r for r in base if not r['demo']],'demo':[r for r in base if r['demo']]})
  for g in ('rescue','damage','stable_correct','stable_wrong'):
   groups[g]=[r for r in base if r['group']==g];groups['natural_no_tie/'+g]=[r for r in base if r['group']==g and r['natural_no_top_tie']]
  if split=='train':
   for g in ('rescue','damage','stable_correct','stable_wrong'):
    groups['non_demo/'+g]=[r for r in base if not r['demo'] and r['group']==g];groups['non_demo/natural_no_tie/'+g]=[r for r in base if not r['demo'] and r['group']==g and r['natural_no_top_tie']]
   groups['non_demo/natural_no_tie']=[r for r in base if not r['demo'] and r['natural_no_top_tie']]
  result={}
  for group,rs in groups.items():
   values={}
   for name in ('L2','G1','L2_norm','G1_norm'):
    for ref in ('all_wrong','fixed_wrong'):values[name+'_'+ref]=desc([r[name][ref] for r in rs if r[name] is not None])
    values[name+'_undefined']=sum(r[name] is None for r in rs)
   for before,after in [('B2','T21'),('B1','T12'),('B2','T21_norm'),('B1','T12_norm')]:
    eligible=[r for r in rs if after in r['predictions']];rescue=sum(r['predictions'][before]!=r['gold'] and r['predictions'][after]==r['gold'] for r in eligible);damage=sum(r['predictions'][before]==r['gold'] and r['predictions'][after]!=r['gold'] for r in eligible)
    values[before+'_to_'+after]={'n':len(eligible),'changed_prediction':sum(r['predictions'][before]!=r['predictions'][after] for r in eligible),'rescue':rescue,'damage':damage,'net':rescue-damage}
   for name in ('L2','G1'):
    for ref in ('all_wrong','fixed_wrong'):
     pairs=sorted((r[name][ref],r['id']) for r in rs);values[name+'_'+ref+'_extremes']={'min':pairs[:5],'max':pairs[-5:]}
   result[group]=values
  summary[split]=result
 output={'status':'L18_FULL_COHORT_INDEPENDENT_COVERAGE_DESCRIPTIVE','coverage_sha256':sha(COV),'rows_sha256':sha(rowsfile),'script_sha256':sha(Path(__file__)),'definitions':{'L2':'gap(B2)-gap(T21)','G1':'gap(T12)-gap(B1)','all_wrong':'within-question equally weighted non-gold gaps','interpretation':'local output substitution; not an estimate of mediation share or unique semantic source'},'splits':summary}
 with (OUT/'summary.json').open('x') as f:json.dump(output,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print('L18_FULL_ANALYSIS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
