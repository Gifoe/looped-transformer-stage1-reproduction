"""CPU full-stage integrity and mathematical control audit. No model forward."""
import argparse,hashlib,json,pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'storage'))
from ab1_prefix_codec import load

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def th(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def audit_case(m,t):
 E,B=t['source_E'],t['source_B'];R,P=m['R'],m['prefix_len'];ident=m['id'];split=m['split'];assert E.shape==B.shape and E.shape[1]==R+1 and 0<P<R
 H={}
 for nm,q,r in [('H00',E,E),('H10',B,E),('H01',E,B),('H11',B,B)]:
  x=B.clone();x[:,P:R]=q[:,P:R];x[:,R]=r[:,R];H[nm]=x
 assert set(m['conditions'])==set(['H00','H10','H01','H11','U_Q9_N','N_Q9_U']+[e+'_'+k for e in ('Q_R_E','Q_R_B','R_Q_E','R_Q_B') for k in ('norm','random')])
 for n,h in H.items():assert th(h)==m['conditions'][n]['boundary_input_sha256']
 keys=['actual_L1_kwargs','actual_L9_kwargs','actual_L18_sdpa_kwargs','actual_L18_sdpa_tail','actual_L18_sdpa_shapes_strides_dtypes']
 for info in m['conditions'].values():
  if info['status']=='OK':assert all(info[k]==m['conditions']['H11'][k] for k in keys)
 for name,donor in [('U_Q9_N','native_Q9_full'),('N_Q9_U','U_Q9_full')]:
  i=m['conditions'][name];assert i['receiver_Q9_R_sha256']==i['applied_Q9_R_sha256'] and i['receiver_Q9_prefix_sha256']==i['applied_Q9_prefix_sha256'] and i['donor_Q9_Q_sha256']==th(t[donor][:,P:R,:])
 assert equal(t['H11_logits'],t['native_K2_logits'])
 for tensor,key in [('native_K1_logits','K1_scores'),('native_K2_logits','N_scores')]:
  z=t[tensor].reshape(-1);v=z[m['targets']];ss=m['identity'][key];ties=torch.nonzero(v==v.max()).reshape(-1).tolist();lse=float(z.double().logsumexp(0));bound=32*torch.finfo(torch.float32).eps*(abs(lse)+float(z.abs().max())+1)
  assert [float(x) for x in v]==ss['candidate_logits'] and ties==ss['ties'] and ties[0]==ss['prediction'] and abs(lse-ss['logsumexp'])<=bound
  assert all(abs(float(x)-lse-y)<=bound for x,y in zip(v,ss['scores']))
 checked=[]
 for name,info in m['conditions'].items():
  if info['status']!='OK':assert info['status'].startswith('UNDEFINED_');continue
  z=t[name+'_logits'].reshape(-1);assert z.dtype==torch.bfloat16 and z.numel()==49152 and torch.isfinite(z).all();assert th(z)==info['full_logits_sha256']
  s=info['scores'];v=z[m['targets']];ties=torch.nonzero(v==v.max()).reshape(-1).tolist();assert [float(x) for x in v]==s['candidate_logits'] and ties==s['ties'] and ties[0]==s['prediction']
  # Independent CPU normalization: GPU float32 reduction may differ slightly.
  lse=float(torch.logsumexp(z.double(),0));eps=torch.finfo(torch.float32).eps;bound=32*eps*(abs(lse)+float(z.abs().max())+1)
  assert abs(lse-s['logsumexp'])<=bound
  assert all(abs((float(x)-lse)-y)<=bound for x,y in zip(v,s['scores']))
  checked.append({'condition':name,'normalization_error_bound':bound,'max_score_error':max(abs(float(x)-lse-y) for x,y in zip(v,s['scores']))})
 for edge,(src,dst,region) in {'Q_R_E':('H00','H10','Q'),'Q_R_B':('H11','H01','Q'),'R_Q_E':('H00','H01','R'),'R_Q_B':('H11','H10','R')}.items():
  for kind in ('norm','random'):
   name=edge+'_'+kind;info=m['conditions'][name]
   if info['status']!='OK':continue
   positions=list(range(P,R)) if region=='Q' else [R];receipt=info['control_receipt'];assert receipt['status']=='OK' and len(receipt['rows'])==len(positions);rec=[]
   for tok,row in zip(positions,receipt['rows']):
    assert row['position']==tok;a=H[src][0,tok].double();b=H[dst][0,tok].double();dose=(b-a).norm();na=a.norm();nb=b.norm()
    if kind=='norm':assert row['seed'] is None and (nb!=0 or na==0);exact=b if nb==0 else b*(na/nb)
    else:
     seed=int.from_bytes(hashlib.sha256(f'AB1-v1|{split}|{ident}|{edge}|{tok}'.encode()).digest()[:8],'big')&((1<<63)-1);assert row['seed']==seed
     gen=torch.Generator(device='cpu');gen.manual_seed(seed);direction=torch.randn(a.shape,generator=gen,dtype=torch.float64);exact=a.clone() if dose==0 else a+direction*(dose/direction.norm())
    cast=exact.to(E.dtype);assert th(cast.double()-exact)==row['cast_residual']['sha256'];assert abs(float(dose)-row['nominal_displacement'])<1e-12 and abs(float((cast.double()-a).norm())-row['actual_displacement'])<1e-12;rec.append(cast)
   assert equal(torch.stack(rec).unsqueeze(0),t[name+'_injected_region'])
   boundary=H[src].clone();boundary[:,positions,:]=t[name+'_injected_region'];assert th(boundary)==info['boundary_input_sha256']
 return {'id':ident,'status':'PASS','scores':checked,'scope':'codec integrity; full-vocabulary CPU normalization envelope; exact candidate ties; eight control mathematical reconstructions; no independent model forward'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',required=True,choices=['train64','train1030','validation299']);a=ap.parse_args();d=pathlib.Path(a.root)/'data'/a.stage;p=json.loads((d/'PENDING.json').read_text());assert p['status']=='COMPLETE';f=json.loads((pathlib.Path(a.root)/'FULL_SOURCE_FREEZE.json').read_text());assert sha(pathlib.Path(a.root)/'FULL_SOURCE_FREEZE.json')==p['source_freeze_sha256'];assert all(sha(pathlib.Path(a.root)/k)==v for k,v in f['sources'].items());expected={'train64':64,'train1030':1030,'validation299':299}[a.stage];assert len(p['cases'])==p['n']==expected and len({x['id'] for x in p['cases']})==expected;records=[];ip=pathlib.Path(a.root)/'inputs'/(p['split']+'_model_inputs.jsonl');inputs={q['id']:q for line in ip.read_text().splitlines() if (q:=json.loads(line))}
 for c in p['cases']:
  mp=d/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=json.loads(mp.read_text());assert m['id']==c['id'] and m['split']==p['split'];assert m['source_identity']==p['source_identity'];q=dict(inputs[c['id']],prefix_len=json.loads((pathlib.Path(a.root)/'inputs/CPU_PREPARATION_MANIFEST.json').read_text())['prefix_length']);assert all(m[k]==q[k] for k in ('id','split','R','prefix_len','targets','input_ids_sha256'));records.append(audit_case(m,load(m,d)))
 out={'status':'PASS','stage':a.stage,'pending_sha256':sha(d/'PENDING.json'),'script_sha256':sha(__file__),'records':records};(d/'INDEPENDENT_COVERAGE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS',expected)
if __name__=='__main__':main()
