import json,hashlib,os
from pathlib import Path
import torch
from safetensors.torch import load_file
B=Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112');D=B/'E3/qkey_producer_train64';A=D/'independent_audit.json';SEL=B/'orders/E1_DISCOVERY64_FROZEN.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(A.read_text());s=json.loads(SEL.read_text());groups={v['id']:g for g in s['groups'] for v in s['groups'][g]};assert a['status']=='E3_QKEY64_INDEPENDENT_BARE_FORWARD_AND_CUDA_SCORE_AUDIT_PASSED' and len(groups)==64 and {x['id'] for x in a['questions']}==set(groups);rows=[]
for q in a['questions']:
 i=q['id'];key=hashlib.sha256(i.encode()).hexdigest();p=D/(key+'.safetensors');assert sha(p)==q['tensor_sha256'];t=load_file(str(p));r={'id':i,'group':groups[i],'cuts':{}}
 for c in (0,8,16,17):
  d={}
  for n in ('receiver_x_R','a_R','sdpa_R'):
   z=t[f'cut{c}_A1K1_{n}'].double()-t[f'cut{c}_A0K0_{n}'].double();d[n+'_delta_l2']=float(torch.linalg.vector_norm(z))
  r['cuts'][str(c)]=d
 rows.append(r)
out={'status':'AUDITED_QKEY_TARGET_STATE_DIFFERENCES','audit_sha256':sha(A),'selection_sha256':sha(SEL),'script_sha256':sha(Path(__file__)),'scope':'Q-region producer perturbation; full A1K1 versus clean A0K0, Euclidean norms in each stage separately. Different cuts are not matched dose.','rows':rows,'groups':{}}
for g in ['rescue','damage','stable_correct','stable_wrong']:
 out['groups'][g]={}
 for c in ('0','8','16','17'):
  x=[r['cuts'][c] for r in rows if r['group']==g];out['groups'][g][c]={k:sum(v[k] for v in x)/len(x) for k in x[0]}
p=D/'analysis/target_state_norms.json'
with p.open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('TARGET_STATE_DONE',sha(p))
