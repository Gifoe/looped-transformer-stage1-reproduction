import csv,gzip,hashlib,json,math
from pathlib import Path
P=Path(__file__).resolve().parent;root=P.parent
corners=['G111','G112','G121','G122','G211','G212','G221','G222']
effects=['K_Q1V1','K_Q1V2','K_Q2V1','K_Q2V2','V_Q1K1','V_Q1K2','V_Q2K1','V_Q2K2','Q_K1V1','Q_K1V2','Q_K2V1','Q_K2V2','I_Q1','I_Q2','J_QKV']
corner_rows=[];effect_rows=[];sources=[]
for split,n in [('train',1119),('validation',299),('test',1172)]:
 base=root/(split+'_analysis');sp=base/'summary.json';rp=base/'rows.jsonl.gz';summary=json.loads(sp.read_text(encoding='utf8'));raw=rp.read_bytes();assert hashlib.sha256(raw).hexdigest()==summary['rows_sha256']
 rows=[json.loads(x) for x in gzip.decompress(raw).splitlines()];assert len(rows)==n and len({r['id'] for r in rows})==n
 for r in rows:
  assert all(set(r[k])==set(corners) for k in ('scores','predictions','top_ties','score_gap','fixed_gap','dynamic_gap'))
  assert len(r['targets'])>=2 and 0<=r['gold']<len(r['targets'])
  assert 0<=r['fixed_wrong']<len(r['targets']) and r['fixed_wrong']!=r['gold']
  assert all(len(r[k])==64 and all(c in '0123456789abcdef' for c in r[k]) for k in ('metadata_sha256','tensor_sha256'))
  assert set(r['target_logits'])==set(corners)
  for c in corners:
   s=r['scores'][c];assert len(s)==len(r['targets']) and all(math.isfinite(v) for v in s)
   assert len(r['target_logits'][c])==len(s) and all(math.isfinite(v) for v in r['target_logits'][c])
   maximum=max(s);assert r['predictions'][c]==s.index(maximum) and r['top_ties'][c]==(s.count(maximum)>1)
   gap=sum(s[r['gold']]-s[j] for j in range(len(s)) if j!=r['gold'])/(len(s)-1);assert math.isclose(gap,r['score_gap'][c],rel_tol=1e-9,abs_tol=1e-9)
   assert math.isclose(s[r['gold']]-s[r['fixed_wrong']],r['fixed_gap'][c],rel_tol=1e-9,abs_tol=1e-9)
   assert math.isclose(s[r['gold']]-max(s[j] for j in range(len(s)) if j!=r['gold']),r['dynamic_gap'][c],rel_tol=1e-9,abs_tol=1e-9)
 selected=[r for r in rows if not r['demo']] if split=='train' else rows;group=summary['groups']['non_demo' if split=='train' else 'all'];assert len(selected)==group['N']
 for c in corners:
  count=sum(r['predictions'][c]==r['gold'] for r in selected);assert count==group['correct'][c];corner_rows.append([split,'non_demo' if split=='train' else 'all',len(selected),c,count])
 for e in effects:
  mean=sum(r['gap_differences'][e] for r in selected)/len(selected);assert math.isclose(mean,group['gap_differences'][e]['mean'],rel_tol=1e-10,abs_tol=1e-10);effect_rows.append([split,'non_demo' if split=='train' else 'all',len(selected),e,mean])
 sources.extend([(str(sp.relative_to(root)),hashlib.sha256(sp.read_bytes()).hexdigest()),(str(rp.relative_to(root)),hashlib.sha256(raw).hexdigest())]);print(split,'validated',len(rows),'selected',len(selected))
for name,header,rows in [('corners.csv',['split','population','n','corner','correct'],corner_rows),('effects.csv',['split','population','n','contrast','mean_score_gap_change'],effect_rows),('source_sha256.csv',['source','sha256'],sources)]:
 with (P/name).open('w',newline='',encoding='utf8') as f:w=csv.writer(f,lineterminator='\n');w.writerow(header);w.writerows(rows)
