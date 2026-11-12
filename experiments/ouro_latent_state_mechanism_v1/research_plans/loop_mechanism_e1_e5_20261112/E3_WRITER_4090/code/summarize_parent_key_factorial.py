import json,pathlib,statistics,hashlib
from fractions import Fraction
src=pathlib.Path(r'D:\ouro-e4-e5-20261112\branch_reference\writer4090_holdout_snapshot');dst=pathlib.Path(r'D:\ouro-e4-e5-20261112\publication_staging\e3_writer_increment_20261113\results\parent_edge_factorial');dst.mkdir(parents=True,exist_ok=False);raw=src/'rows.jsonl';source=src/'SOURCE.json';orig=[json.loads(z) for z in raw.read_text(encoding='utf8').splitlines()];assert len(orig)==1329 and len({(r['split'],r['id']) for r in orig})==1329
rows=[]
for r in orig:
 c=r['conditions'];gold=r['gold'];wrong=r['fixedwrong'];
 def prod(name,kind):
  v=c[name]['scores'];return {'allwrong':v[gold]-sum(x for i,x in enumerate(v) if i!=gold)/(len(v)-1),'fixedwrong':v[gold]-v[wrong],'dynamic':v[gold]-max(x for i,x in enumerate(v) if i!=gold)}[kind]
 def frac(name,kind):return Fraction(c[name]['fraction_'+kind])
 out={'split':r['split'],'id':r['id'],'group':r['group'],'no_tie':r['no_tie'],'gold':gold,'predictions':{n:c[n]['prediction'] for n in ['P','PK','P_plus_a_main','P_plus_a_main_plus_key']},'contrasts':{}}
 for metric in ['allwrong','fixedwrong','dynamic']:
  d={};f={}
  for nm,a,b in [('key_at_P','PK','P'),('key_at_P_plus_A','P_plus_a_main_plus_key','P_plus_a_main')]:d[nm]=prod(a,metric)-prod(b,metric);f[nm]=frac(a,metric)-frac(b,metric)
  d['interaction']=d['key_at_P_plus_A']-d['key_at_P'];f['interaction']=f['key_at_P_plus_A']-f['key_at_P'];out['contrasts'][metric]={'production':d,'raw_fraction':{k:str(v) for k,v in f.items()}}
 rows.append(out)
summary={}
for split in ['train','validation']:
 summary[split]={}
 for group in ['all','rescue','damage','stable_correct','stable_wrong']:
  for nt in [False,True]:
   key=group+('_no_tie' if nt else '');subset=[r for r in rows if r['split']==split and (group=='all' or r['group']==group) and (not nt or r['no_tie'])];sm={'n':len(subset),'metrics':{}}
   for metric in ['allwrong','fixedwrong','dynamic']:
    sm['metrics'][metric]={}
    for k in ['key_at_P','key_at_P_plus_A','interaction']:
     vals=[r['contrasts'][metric]['production'][k] for r in subset];fr=[Fraction(r['contrasts'][metric]['raw_fraction'][k]) for r in subset];sm['metrics'][metric][k]={'mean':statistics.mean(vals) if vals else None,'median':statistics.median(vals) if vals else None,'production_sign_counts':{'positive':sum(x>0 for x in vals),'negative':sum(x<0 for x in vals),'zero':sum(x==0 for x in vals)},'fraction_exact_sign_counts':{'positive':sum(x>0 for x in fr),'negative':sum(x<0 for x in fr),'zero':sum(x==0 for x in fr)},'fraction_exact_mean':str(sum(fr,Fraction())/len(fr)) if fr else None,'positive_ids':sorted(r['id'] for r in subset if r['contrasts'][metric]['production'][k]>0),'negative_ids':sorted(r['id'] for r in subset if r['contrasts'][metric]['production'][k]<0),'zero_ids':sorted(r['id'] for r in subset if r['contrasts'][metric]['production'][k]==0)}
   summary[split][key]=sm
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();(dst/'rows.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf8');(dst/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8');receipt={'status':'POSTHOC_PARENT_KEY_EDGE_SUMMARY','input_rows_sha256':sha(raw),'input_source_sha256':sha(source),'output_rows_sha256':sha(dst/'rows.jsonl'),'output_summary_sha256':sha(dst/'summary.json'),'scope':'Posthoc aggregation of previously run P, PK, P+A and P+A+key conditions; not pre-registered criterion, not independent model replay, not mediation of the newer L-only intervention.'};(dst/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print('receipt',sha(dst/'RECEIPT.json'))
for split in ['train','validation']:
 s=summary[split]['rescue'];print(split,'rescue_n',s['n'],{k:round(v['mean'],6) for k,v in s['metrics']['allwrong'].items()})
