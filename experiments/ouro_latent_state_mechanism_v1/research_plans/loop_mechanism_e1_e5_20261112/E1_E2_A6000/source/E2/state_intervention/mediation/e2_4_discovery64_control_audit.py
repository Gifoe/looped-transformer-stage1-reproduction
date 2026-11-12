"""Independent CPU check of E2.4 corner anchors and full-delta control math."""
import json,pathlib,hashlib,torch
from safetensors.torch import load_file
from safetensors import safe_open
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';P=E/'state_intervention/mediation/discovery64';PR=E/'state_intervention/discovery64';ST=E/'E2_fixed_probe_states.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
sel=json.loads((B/'orders/E1_DISCOVERY64_FROZEN.json').read_text());expected={('train',q['id']) for g in sel['groups'].values() for q in g};pending=json.loads((P/'PENDING.json').read_text());assert len(expected)==len(pending['cases'])==64 and {(x['split'],x['id']) for x in pending['cases']}==expected;prev=json.loads((PR/'PENDING_INDEPENDENT_AUDIT.json').read_text());links={(r['split'],r['id']):r for r in prev['cases']};checked=[]
with safe_open(str(ST),framework='pt',device='cpu') as st:
 for q in pending['cases']:
  key=hashlib.sha256((q['split']+'\0'+q['id']).encode()).hexdigest();mp=P/(key+'.json');tp=P/(key+'.safetensors');assert sha(mp)==q['metadata_sha256'] and sha(tp)==q['tensor_sha256'];m=json.loads(mp.read_text());o=load_file(str(tp));pm=PR/(key+'.json');pt=PR/(key+'.safetensors');link=links[(q['split'],q['id'])];assert sha(pm)==m['source_metadata_sha256']==link['metadata_sha256'] and sha(pt)==m['source_tensor_sha256']==link['tensor_sha256'];old=load_file(str(pt));x2=old['identity_injected_R'].double();x1=old['rollback_full_injected_R'].double();a2=old['identity_a_R'];ap=old['rollback_full_a_R']
  for name,x,a,generated,anchor in (('C00',x2,a2,a2,'identity'),('C11',x1,ap,ap,'rollback_full'),('C10',x1,a2,ap,None),('C01',x2,ap,a2,None)):
   assert bits(o[name+'_x_injected_R'],x.to(torch.bfloat16)) and bits(o[name+'_a_injected_R'],a) and bits(o[name+'_a_receiver_R'],generated)
   assert bits(o[name+'_y_R'],(o[name+'_x_injected_R']+a).to(torch.bfloat16))
   if anchor:assert bits(o[name+'_logits'],old[anchor+'_logits'])
  d=x2-x1;donor=m['full_control_donor']['id'];dk='train_'+hashlib.sha256(donor.encode()).hexdigest();other=st.get_tensor(dk+'_u1_l18_input').double()-st.get_tensor(dk+'_u0_l18_input').double();x1n=x1.norm();x2n=x2.norm();on=other.norm();dn=d.norm();values={'full_norm':None if x1n==0 and x2n!=0 else (x2 if x1n==0 else x1*(x2n/x1n)),'full_unrelated':None if on==0 and dn!=0 else (x2 if on==0 else x2-other*(dn/on))}
  for name,value in values.items():
   if value is None:assert m['conditions'][name]['status']=='UNDEFINED';continue
   patch=value.to(torch.bfloat16);assert m['conditions'][name]['status']=='OK' and bits(patch,o[name+'_injected_R']) and torch.allclose(o[name+'_cast_residual_fp64'],patch.double()-value,atol=1e-10,rtol=0)
  checked.append(q['id'])
receipt={'status':'E2_4_DISCOVERY64_INDEPENDENT_CONTROL_MATH_PASSED','n':len(checked),'ids':checked,'pending_sha256':sha(P/'PENDING.json'),'script_sha256':sha(pathlib.Path(__file__))};out=P/'CONTROL_MATH_AUDIT.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');print('E2_4_CONTROL_MATH_PASS',sha(out))
