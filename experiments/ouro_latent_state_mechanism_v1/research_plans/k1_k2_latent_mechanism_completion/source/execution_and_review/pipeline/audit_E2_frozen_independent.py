"""Independent CPU arithmetic audit of original frozen heads; no fitting/model forward."""
import argparse,pathlib,sys,json,hashlib,numpy as np
GLOBAL=(330,340,389,414,422)
PARAM_SHA='204a3788301c7aa826a198b9e40ce495ef81284b6cd41a4758dfd942f8650890'
KEYS=tuple(f'{site}_{feature}_trainK{loop}_{control}' for site in ('l18_input','post_norm') for feature in ('raw','unit_l2') for loop in (1,2) for control in ('real','permuted'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def oracle(x,key,targets,arrays):
 x=np.asarray(x,dtype=np.float64).copy();assert x.shape==(1,2048) and np.isfinite(x).all();assert len(set(targets))==len(targets)>=2 and set(targets)<=set(GLOBAL)
 if '_unit_l2_' in key:
  n=np.linalg.norm(x,axis=1,keepdims=True);x=np.divide(x,n,out=np.zeros_like(x),where=n!=0)
 w,mu,std=(arrays[key+k] for k in ('_w','_mu','_std'));assert w.shape==(2049,5) and mu.shape==std.shape==(2048,) and (std>0).all()
 features=(x-mu)/std;aug=np.concatenate([features,np.ones((1,1),dtype=np.float64)],axis=1);all_scores=aug@w;cols=[GLOBAL.index(v) for v in targets];scores=all_scores[:,cols];z=scores.astype(np.longdouble);exp=np.exp(z-z.max(axis=1,keepdims=True));probs=exp/exp.sum(axis=1,keepdims=True);ties=np.flatnonzero(scores[0]==scores.max()).tolist();eps=np.finfo(np.float64).eps
 # A-priori forward-error allowance for 2049-term dot and double softmax.
 gamma=2050*eps/(1-2050*eps);dotbound=gamma*(np.abs(aug)@np.abs(w))[:,cols];pbound=64*eps*(len(targets)+1)*(1+float(np.abs(scores).max()))
 return scores,probs,ties,dotbound,pbound

def run(root,stage):
 sys.path.insert(0,str(root/'code'));sys.path.insert(0,str(root/'storage'))
 from ab1_prefix_codec import load
 d=root/'data'/stage;o=root/'analysis_results'/stage;pp=d/'PENDING.json';ap=d/'INDEPENDENT_COVERAGE_AUDIT.json';npz=root/'inputs/selected_models.npz';assert sha(npz)==PARAM_SHA;p=json.loads(pp.read_text());audit=json.loads(ap.read_text());assert p['status']=='COMPLETE' and audit['status']=='PASS' and audit['pending_sha256']==sha(pp)
 paths=[pp,ap,npz,pathlib.Path(__file__)];before={str(x.relative_to(root)):sha(x) for x in paths};arrays=dict(np.load(npz,allow_pickle=False));assert set(arrays)=={k+s for k in KEYS for s in ('_w','_mu','_std')};records=[]
 for c in p['cases']:
  mp=d/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=json.loads(mp.read_text());t=load(m,d);f=m['frozen_e2'];assert f['secondary_all16'] and f['status']=='FROZEN_E2_DESCRIPTIVE_APPLICATION';count=0;ms=0.;mp_err=0.
  expected={'K1'}|{k for k,v in m['conditions'].items() if v['status']=='OK'};assert set(f['results'])==expected
  for condition,heads in f['results'].items():
   assert set(heads)==set(KEYS)
   for key,stored in heads.items():
    site='l18_input' if key.startswith('l18_input_') else 'post_norm'
    if condition=='K1':state=t['source_K1_L18_x_R'] if site=='l18_input' else t['source_B'][:,m['R'],:]
    else:state=t[condition+('_L18_x_R' if site=='l18_input' else '_finalnorm_R')]
    assert str(state.dtype)=='torch.bfloat16' and list(state.shape)==[1,2048];assert stored['parameter_sha256']==PARAM_SHA and stored['offered_token_ids']==m['targets']
    scores,probs,ties,sbound,pbound=oracle(state.float().numpy().astype(np.float64),key,m['targets'],arrays);ss=np.asarray(stored['scores']);sp=np.asarray(stored['probabilities']);assert ss.shape==sp.shape==scores.shape
    se=np.abs(scores-ss);pe=np.abs(probs-sp);assert (se<=sbound).all() and (pe<=pbound).all();assert stored['top_ties_offered_indices']==[ties] and stored['prediction_offered_index']==[ties[0]];count+=1;ms=max(ms,float(se.max()));mp_err=max(mp_err,float(pe.max()))
  records.append(dict(id=c['id'],head_applications=count,max_score_error=ms,max_probability_error=mp_err,classes=list(GLOBAL),offered_token_ids=m['targets'],site_and_BF16_checks='PASS'))
 assert all(sha(root/k)==v for k,v in before.items());result=dict(status='PASS',stage=stage,source_sha256=before,cases=records,n=len(records),parameter_sha256=PARAM_SHA,scope='Independent mathematical implementation using shared archived parameters and NumPy BLAS; no independent training or model forward. Probability oracle longdouble offered max-shift normalization; a-priori float64 error bounds.');o.mkdir(parents=True,exist_ok=True);out=o/'E2_FROZEN_INDEPENDENT_AUDIT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',required=True,choices=['train64','train1030','validation299']);a=ap.parse_args();print(run(pathlib.Path(a.root),a.stage)['status'])
