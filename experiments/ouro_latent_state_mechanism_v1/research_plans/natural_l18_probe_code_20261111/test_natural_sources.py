import sys,tempfile,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.home()/'.codex/natural_branch_full_review_v4_20261111'))
import torch
from safetensors.torch import save_file
import natural_sources as n
def write(p,x):p.write_text(json.dumps(x,sort_keys=True))
def fail(f):
 try:f()
 except ValueError:return
 raise AssertionError('expected rejection')
def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);n.PLAN=root;n.RUN=root/'run';n.RUN.mkdir();n.COMBINED=root/'combined.json';n.GPU=root/'gpu.json';ident='q1';split='train';base=split+'_'+hashlib.sha256(ident.encode()).hexdigest();mp=n.RUN/(base+'.json');tp=mp.with_suffix('.safetensors');v=torch.arange(2048).reshape(1,2048).bfloat16();save_file({f'u{u}_layer17_{k}':v.clone() for u in (0,1) for k in ('x','a')},str(tp));tsha=n.sha(tp)
  meta={'id':ident,'split':split,'R_absolute':3,'targets':[4,5],'gold':0,'prior_scores':[[1.,2.],[3.,4.]],'natural_metadata_sha256':'a'*64,'natural_tensor_sha256':'b'*64,'probe_tensor_sha256':tsha};write(mp,meta)
  q={'id':ident,'split':split,'natural_metadata_sha256':'a'*64,'natural_tensor_sha256':'b'*64,'branch_metadata_sha256':n.sha(mp),'branch_tensor_sha256':tsha};cp=root/'natural_branch_train_exploration_20261111';cp.mkdir();cp=cp/'train_coverage_preliminary.json';write(cp,{'questions':[q]});write(n.GPU,{'questions':[q]});write(n.COMBINED,{'status':'INDEPENDENT_CPU_STATE_AND_GPU_LOGIT_REPLAY_PASSED','counts':{'train':1119,'validation':299,'test':1172},'splits':{'train':{'cpu_coverage_sha256':n.sha(cp)}},'gpu_replay_sha256':n.sha(n.GPU)});n.COMBINED_SHA=n.sha(n.COMBINED)
  n.coverage=lambda s:[{'id':ident}];n.load_natural=lambda s,q:{'metadata_sha256':'a'*64,'tensor_sha256':'b'*64,'R':3,'targets':[4,5],'gold':0,'scores':[[1.,2.],[3.,4.]]}
  got=n.load('train',ident);assert torch.equal(got['states'][0]['a'],v)
  fail(lambda:n.load('train','wrong_question'))
  n.load_natural=lambda s,q:{'metadata_sha256':'a'*64,'tensor_sha256':'b'*64,'R':4,'targets':[4,5],'gold':0,'scores':[[1.,2.],[3.,4.]]};fail(lambda:n.load('train',ident))
  with tp.open('ab') as f:f.write(b'tamper')
  fail(lambda:n.load('train',ident))
 print('CPU_SOURCE_ID_INPUT_HASH_REJECTION_PASS')
if __name__=='__main__':main()
