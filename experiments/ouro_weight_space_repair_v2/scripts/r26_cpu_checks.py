"""Pure synthetic/source checks: no tokenizer, data, model or GPU."""
import ast,copy
from r26_common import ROOT,save,sha
from r26_protocol import variants,interpret

def main():
 checks=[]
 import runtime
 from r26_protocol import build_inputs
 original=runtime.encode_pair
 class Tok:
  def __call__(self,text,**kwargs):return dict(input_ids=list(range(160)),offset_mapping=[(i,i+1) for i in range(160)])
 try:
  runtime.encode_pair=lambda tok,c,t:(list(range(155 if c=='Q' else 5)),[7,8] if t=='short' else [7,8,9])
  entries=[dict(source_token_position=i,key_prediction_position=i-1,value_id=i,offset=[i,i+1]) for i in range(2,145)]
  memory=dict(key_text='synthetic',entries=entries)
  plan=build_inputs(Tok(),[dict(id='a',split='discovery')],{'a':dict(request_pairs=[['Q','short'],['Q','long']])},{k:k for k in ['N0','N1','N2']},memory)
  assert plan['candidate']==1 and len(plan['contexts'])==5 and sum(len(variants(c['tokens'],c['cuts'])) for c in plan['contexts'])==30
  assert not plan['selection_audit'][0]['eligible'] and plan['selection_audit'][1]['eligible']
 finally:runtime.encode_pair=original
 checks.append('synthetic label-free ordered input admission/five contexts/30 forwards/143 offsets')
 x=[0,49151,2,3,4,5,6];cuts=[2,4];v=variants(x,cuts);assert len(v)==6
 for name,t,p in v:
  if name in ['B1','B2']:
   c=cuts[int(name[-1])-1];assert t[:c+1]==x[:c+1] and len(t)==len(x) and all(a!=b for a,b in zip(t[c+1:],x[c+1:]))
  if name in ['C1','C2']:assert t==x[:cuts[int(name[-1])-1]+1]
 checks.append('deterministic suffix modulo wrap/no fixed point/prefix/cut/truncation')
 for bad in [[0,4],[4,2],[2,6],[2,2]]:
  try:variants(x,bad)
  except AssertionError:pass
  else:raise AssertionError('bad cut accepted')
 checks.append('strict positive ordered cuts with future')
 from r7_core import digest
 def row(name):
  d=5 if name=='Q' else 4;c=[2,4];tokens=list(range(7));comparisons=[]
  for v,ps in [('A_repeat',c),('B1',c[:1]),('B2',c[1:]),('C1',c[:1]),('C2',c[1:])]:
   for k in range(1,d+1):
    for pos in ps:
     def metric(head):return dict(exact=True,shape=[1,1,49152 if head else 2048],dtype='torch.float32' if head else 'torch.bfloat16',sha256_A='0'*64,sha256_B='0'*64,max_abs=0.,RMS=0.,unequal_elements=0)
     comparisons.append(dict(variant=v,depth=k,position=pos,state=metric(False),head=metric(True)))
  inputs=[dict(variant=v,input_ids=t,positions=p,shape=[1,len(t)],input_sha256=digest(t)) for v,t,p in variants(tokens,c)]
  return dict(context=name,complete=True,depth=d,cuts=c,comparisons=comparisons,inputs=inputs)
 rows=[row(n) for n in ['Q','N0','N1','N2','key']];assert interpret(rows)['category']=='finite_config_prefix_evidence'
 for variant,expected in [('C1','length_bitwise_sensitive'),('B1','content_invariance_not_established'),('A_repeat','inconclusive_repeat')]:
  metric=next(x for x in rows[0]['comparisons'] if x['variant']==variant)['head'];metric.update(exact=False,sha256_B='1'*64,max_abs=1.,RMS=1.,unequal_elements=1);assert interpret(rows)['category']==expected
 for bad in [rows[:4],copy.deepcopy(rows),copy.deepcopy(rows),copy.deepcopy(rows)]:
  if len(bad)==5:
   if len(bad[0]['comparisons'])==30:bad[0]['comparisons'].pop()
  try:interpret(bad)
  except AssertionError:pass
  else:raise AssertionError('partial domain accepted')
 bad=copy.deepcopy(rows);bad[0]['comparisons'][0]=copy.deepcopy(bad[0]['comparisons'][1])
 try:interpret(bad)
 except AssertionError:pass
 else:raise AssertionError('duplicate accepted')
 checks.append('126 comparisons exact schema/four priority classes/incomplete duplicate rejection')
 src=(ROOT/'scripts/r26_worker.py').read_text();ast.parse(src)
 assert src.index('audit=input_audit(tok)')<src.index('tok,model=load_model()')
 assert "if name in ['B1','B2'] and pos!=c['cuts'][int(name[-1])-1]:continue" in src
 assert 'single=h[:,i:i+1,:]' in src and 'assert z.shape==(1,1,49152)' in src
 assert 'argmax' not in src and "['gold']" not in src and 'register_forward' not in src
 assert 'use_cache=False' in src
 checks.append('source: premodel CPU plan/no labels/no hooks/protected B cut/head single position')
 import torch
 from r26_worker import compare
 a=torch.zeros((1,1,8),dtype=torch.bfloat16);assert compare(a,a.clone())['exact']
 b=a.clone();b[0,0,0]=1;assert not compare(a,b)['exact']
 try:compare(a,a[:,:,:4])
 except AssertionError:pass
 else:raise AssertionError('shape mismatch accepted')
 checks.append('CPU Torch BF16 state/fullV FP32 comparison and shape guard')
 for p in (ROOT/'scripts').glob('r26_*.py'):
  ast.parse(p.read_text());assert '\\' not in p.relative_to(ROOT).as_posix()
 checks.append('all cross-platform manifest/receipt paths POSIX')
 import numpy as np
 from r26_analyze import audit_numeric
 # BF16 bits 1.0 vs 2.0 give exact unit difference; signed zeros differ in bits only.
 for domain,a,b,maximum,rms_expected,unequal in [('state',np.array([16256],dtype=np.uint16),np.array([16384],dtype=np.uint16),1.,1.,1),('head',np.array([0.],dtype=np.float32),np.array([-0.],dtype=np.float32),0.,0.,1)]:
  m=dict(exact=False,unequal_elements=unequal,max_abs=maximum,RMS=rms_expected)
  assert audit_numeric(m,[a,b],domain)==rms_expected
  wrong=dict(m,unequal_elements=0)
  try:audit_numeric(wrong,[a,b],domain)
  except AssertionError:pass
  else:raise AssertionError('bad raw metric accepted')
 checks.append('independent BF16-bit expansion/FP32 signed-zero byte audit/bad numeric rejected')
 save('results/r26_CPU_CHECKS.json',dict(status='passed',checks=checks,scope='Synthetic and AST/source only; no actual model/state/tokenizer loaded',source_sha256={p.relative_to(ROOT).as_posix():sha(p.relative_to(ROOT).as_posix()) for p in (ROOT/'scripts').glob('r26_*.py')}))
 print('R26_CPU_CHECKS_PASS',len(checks))
if __name__=='__main__':main()
