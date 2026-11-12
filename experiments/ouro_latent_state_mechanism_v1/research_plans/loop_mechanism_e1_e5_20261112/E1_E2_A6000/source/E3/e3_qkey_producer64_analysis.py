"""Read only fully audited frozen-64 producer edge effects; no partial analysis."""
import json,hashlib,datetime,os,statistics,gzip
from fractions import Fraction
from pathlib import Path
from safetensors.torch import load_file
B=Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112');RUN=B/'E3/qkey_producer_train64';SEL=B/'orders/E1_DISCOVERY64_FROZEN.json';A=RUN/'independent_audit.json';RC=RUN/'receipt.json';PRIOR=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_eight_corner_20261112/train_analysis/rows.jsonl.gz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(xs):
 x=sorted(xs);n=len(x)
 def quant(q):
  if not n:return None
  pos=q*(n-1);a=int(pos);return x[a]+(x[min(a+1,n-1)]-x[a])*(pos-a)
 return {'n':n,'mean':sum(x)/n if n else None,'median':quant(.5),'q05':quant(.05),'q25':quant(.25),'q75':quant(.75),'q95':quant(.95),'positive':sum(v>0 for v in x),'negative':sum(v<0 for v in x),'zero':sum(v==0 for v in x)}
def main():
 hashes={str(p):sha(p) for p in (A,RC,SEL,PRIOR,Path(__file__))};a=json.loads(A.read_text());r=json.loads(RC.read_text());sel=json.loads(SEL.read_text());groups={v['id']:g for g in ('rescue','damage','stable_correct','stable_wrong') for v in sel['groups'][g]};need=set(groups);assert len(need)==64 and a['status']=='E3_QKEY64_INDEPENDENT_BARE_FORWARD_AND_CUDA_SCORE_AUDIT_PASSED' and a['collector_receipt_sha256']==hashes[str(RC)] and a['selection_sha256']==hashes[str(SEL)] and len(a['questions'])==64 and {x['id'] for x in a['questions']}==need
 assert hashes[str(PRIOR)]=='a8ac7510ec4bbad1a5b8e745a8d41d523ac984f4e2e3bdbc8c0d11b06b5a635e'
 with gzip.open(PRIOR,'rt') as f:natural={v['id']:v for line in f if (v:=json.loads(line))['id'] in need}
 assert set(natural)==need and all(not natural[i]['demo'] and natural[i]['group']==groups[i] for i in need)
 ledger={x['id']:x for x in a['questions']};rows=[]
 for ident in sorted(need):
  key=hashlib.sha256(ident.encode()).hexdigest();mp=RUN/(key+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==ledger[ident]['metadata_sha256'] and sha(tp)==ledger[ident]['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));gold=m['gold'];wrong=[j for j in range(len(m['targets'])) if j!=gold];fixed=m['fixed_wrong'];row={'id':ident,'group':groups[ident],'natural_no_top_tie':natural[ident]['natural_no_top_tie'],'cuts':{}}
  for cut in (0,8,16,17):
   cs=m['conditions'][str(cut)]; rr={}
   for n in ('A0K0','A0K1','A1K1','A1K0'):
    v=cs[n]['scores'];z=t[f'cut{cut}_{n}_logits'];raw=[Fraction.from_float(float(z[0,j])) for j in m['targets']];maxv=max(v);rr[n]={'scores':v,'gold':gold,'pred':v.index(maxv),'ties':[i for i,x in enumerate(v) if x==maxv],'allwrong':sum(v[gold]-v[j] for j in wrong)/len(wrong),'fixed':v[gold]-v[fixed],'dynamic':v[gold]-max(v[j] for j in wrong),'raw_allwrong':sum((raw[gold]-raw[j] for j in wrong),Fraction())/len(wrong),'raw_fixed':raw[gold]-raw[fixed],'raw_dynamic':raw[gold]-max(raw[j] for j in wrong)}
   def edge(start,end):
    a,b=rr[start],rr[end];v={'from':start,'to':end,'allwrong':a['allwrong']-b['allwrong'],'fixed':a['fixed']-b['fixed'],'dynamic':a['dynamic']-b['dynamic'],'raw_allwrong_sign':(a['raw_allwrong']>b['raw_allwrong'])-(a['raw_allwrong']<b['raw_allwrong']),'raw_fixed_sign':(a['raw_fixed']>b['raw_fixed'])-(a['raw_fixed']<b['raw_fixed']),'raw_dynamic_sign':(a['raw_dynamic']>b['raw_dynamic'])-(a['raw_dynamic']<b['raw_dynamic']),'baseline_correct':a['pred']==gold,'treated_correct':b['pred']==gold,'from_pred':a['pred'],'to_pred':b['pred'],'from_ties':a['ties'],'to_ties':b['ties']};return v
   row['cuts'][str(cut)]={'clean_key_transfer_loss':edge('A0K0','A0K1'),'producer_background_restore_loss':edge('A1K0','A1K1'),'background_change':edge('A1K1','A0K0'),'corners':{k:{n:v for n,v in c.items() if not n.startswith('raw_')} for k,c in rr.items()}}
  rows.append(row)
 assert all(sha(Path(k))==v for k,v in hashes.items())
 result={'status':'E3_QKEY64_DESCRIPTIVE_ANALYSIS_COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes':hashes,'scope':'Fixed train64 stratified exploratory, 16/group; native K2 backgrounds and R-query key edge only; index17 technical endpoint. background_change means A1K1 minus A0K0. No natural path or population inference. All earlier ARC outcomes exposed.','groups':{}}
 for g in ('all64','all64_natural_no_top_tie','rescue','damage','stable_correct','stable_wrong','rescue_natural_no_top_tie','damage_natural_no_top_tie','stable_correct_natural_no_top_tie','stable_wrong_natural_no_top_tie'):
  base=g.replace('_natural_no_top_tie','');sub=[r for r in rows if (base=='all64' or r['group']==base) and (not g.endswith('_natural_no_top_tie') or r['natural_no_top_tie'])];result['groups'][g]={}
  for cut in ('0','8','16','17'):
   result['groups'][g][cut]={}
   for edge in ('clean_key_transfer_loss','producer_background_restore_loss','background_change'):
    es=[r['cuts'][cut][edge] for r in sub];result['groups'][g][cut][edge]={metric:summary([v[metric] for v in es]) for metric in ('allwrong','fixed','dynamic','raw_allwrong_sign','raw_fixed_sign','raw_dynamic_sign')};result['groups'][g][cut][edge]['decisions']={'correct_to_wrong':sum(v['baseline_correct'] and not v['treated_correct'] for v in es),'wrong_to_correct':sum(not v['baseline_correct'] and v['treated_correct'] for v in es),'changed':sum(v['from_pred']!=v['to_pred'] for v in es)}
 out=RUN/'analysis';out.mkdir(exist_ok=False)
 for name,data in [('rows.json',rows),('summary.json',result)]:
  with (out/name).open('x') as f:json.dump(data,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print('QKEY64_ANALYSIS_COMPLETE',sha(out/'summary.json'),flush=True)
if __name__=='__main__':main()
