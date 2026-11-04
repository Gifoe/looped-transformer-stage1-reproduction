"""CPU synthetic hook lifecycle and fixed task protocol checks, no real model."""
import ast,json,types
import torch,numpy as np
from r27_exec_common import ROOT,save,sha
from r27_exec_hooks import SourceReplay
from r27_exec_protocol import PATCHES,label_metrics

def toy():
 return types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(k_proj=torch.nn.Identity(),v_proj=torch.nn.Identity())) for _ in range(24)]))
class Guard:
 def __init__(self):self.n=0
 def invalidate(self):self.n+=1

def run(controller,x,check=None):
 outputs=[]
 for loop in range(4):
  for j,(kind,layer,m) in enumerate(controller.modules):
   y=m(x);outputs.append((loop,kind,layer,y))
   if check:check(loop,kind,layer,y)
 controller.verify();return outputs

def main():
 checks=[];positions=[1,3,5,7];x=torch.arange(40).reshape(1,10,4).to(torch.bfloat16);original=x.clone();model=toy();guard=Guard()
 with SourceReplay(model) as c:
  c.bind(guard);c.configure('capture',positions,10);outputs=run(c,x);assert all(y is x for _,_,_,y in outputs)
  donor={k:dict(v) for k,v in c.captured.items()};assert all(torch.equal(z,x[:,positions]) for group in donor.values() for z in group.values());checks.append('capture returnNone/tensor identity/loop4 only/all48x4 counts')
  for kind in ['K','V']:
   for permutation in [(0,1,2,3),(1,2,3,0)]:
    c.configure('patch',positions,10,kind,donor[kind],permutation)
    def verify(loop,k,layer,y):
     if loop!=3 or k!=kind:assert y is x
     else:
      assert y is not x;assert torch.equal(y[:,positions],x[:,[positions[i] for i in permutation]])
      other=[i for i in range(10) if i not in positions];assert torch.equal(y[:,other],x[:,other])
    run(c,x,verify);assert torch.equal(x,original)
  checks.append('K/V identity and cyclic source replay/first3loop untouched/no in-place/no other rows change')
  c.configure('capture',positions,10);c.modules[0][2](x)
  try:c.configure('capture',positions,10)
  except AssertionError:pass
  else:raise AssertionError('unverified overwrite')
  try:c.verify()
  except AssertionError:pass
  else:raise AssertionError('partial counts accepted')
 assert c.cleanup['original_hooks_restored'] and all(not m._forward_hooks for _,_,m in c.modules)
 checks.append('unverified reset rejected/partial callcount rejected/finally cleanup')
 model=toy();existing=model.model.layers[0].self_attn.k_proj.register_forward_hook(lambda *args:None)
 try:
  with SourceReplay(model) as c:
   c.bind(Guard());c.configure('capture',positions,10);raise ValueError('synthetic forward exception')
 except ValueError:pass
 assert c.cleanup['original_hooks_restored'] and len(model.model.layers[0].self_attn.k_proj._forward_hooks)==1;existing.remove()
 checks.append('exception preserves original error and original pre-existing hook')
 model=toy();bad=model.model.layers[5].self_attn.k_proj
 def fail_registration(*args,**kwargs):raise RuntimeError('synthetic registration failure')
 bad.register_forward_hook=fail_registration;c=SourceReplay(model)
 try:c.__enter__()
 except RuntimeError:pass
 else:raise AssertionError('registration failure not raised')
 assert c.cleanup['original_hooks_restored'] and not c.handles and all(not m._forward_hooks for _,_,m in c.modules)
 model=toy()
 with SourceReplay(model) as c:
  c.bind(Guard());c.configure('capture',positions,10)
  try:c.modules[1][2](x)
  except AssertionError:pass
  else:raise AssertionError('reordered projection accepted')
 checks.append('partial registration cleanup/projection event order failure')
 model=toy()
 with SourceReplay(model) as c:
  c.bind(Guard());c.configure('capture',positions,10);run(c,x);first={k:dict(v) for k,v in c.captured.items()};y=x.clone();y[:,-1,:]+=1
  c.configure('capture',positions,10);run(c,y)
  assert all(torch.equal(first[k][l],c.captured[k][l]) for k in ['K','V'] for l in range(24))
  c.configure('patch',positions,10,'V',first['V']);assert c.donor is first['V'];run(c,y)
 checks.append('query-independent source capture and same first-query donor object shared')
 assert len(PATCHES)==6 and [p[0] for p in PATCHES]==['V_identity','V_clean','V_shuffle','K_clean','K_shuffle','K_identity']
 z=np.zeros(49152);ids={k:i for i,k in enumerate('ABCD')};z[2]=1;m=label_metrics(z,ids,'C');assert m['expected_strictly_highest'] and 0<m['label_probability_mass']<1
 z[:4]=1;assert not label_metrics(z,ids,'C')['expected_strictly_highest']
 try:label_metrics(np.full(49152,np.nan),ids,'A')
 except AssertionError:pass
 else:raise AssertionError('nonfinite accepted')
 checks.append('fixed six replay order/strict label-space tie/no nonfinite/logsoftmax mass')
 from r27_exec_protocol import ordered_instances,execution_schema,functional_summary
 import copy
 manifest=json.loads((ROOT/'configs/r27_CPU_STRUCTURE_MANIFEST.json').read_text());rows=ordered_instances(manifest)
 def ref(shape,key):return dict(key=key,dtype='torch.bfloat16',shape=shape,raw_sha256=key)
 def clock(row,mode,kind=None,shuffle=False):return dict(mode=mode,kind=kind,calls_per_module=[4]*48,source_positions=row['source_positions'],events=[[a,b,c] for a in range(4) for b in range(24) for c in ['K','V']],other_rows_exact_verified=mode=='patch',permutation=[1,2,3,0] if shuffle else [0,1,2,3])
 bases=[]
 for row in rows:
  r={k:row[k] for k in ['table','query_row','version','input_ids_sha256','input_shape','source_positions','expected_label']};r['label_ids']={x['label']:x['token_id'] for x in row['sources']};r['query_states']=[ref([1,1,2048],str(i)*64) for i in range(4)];r['fullV']=dict(ref([1,1,49152],'h'*64),dtype='torch.float32');r['clock']=clock(row,'capture');r['donors']={k:[ref([1,4,512],f'{row["table"]}{row["version"]}{k}{l}'.ljust(64,'0')) for l in range(24)] for k in ['K','V']}
  if row['version']=='B':r['first3_full_states']=[ref([1,row['input_shape'][1],2048],str(i+4)*64) for i in range(3)]
  r['metrics']=dict(expected_strictly_highest=True,label_logits={k:0. for k in 'ABCD'})
  bases.append(r)
 raw=dict(baselines=bases,no_hook=copy.deepcopy(bases[0]),no_op_exact=True,invalid_reasons=[],source_pairs=[],patches=[],cleanup=dict(handles_removed=True,original_hooks_restored=True,all_model_hooks_empty=True,errors=[]),forward_records=[dict(number=1,phase='no_hook')])
 for i,r in enumerate(bases):raw['forward_records'].append(dict(number=i+2,phase='A',**{k:r[k] for k in ['table','query_row','version','clock']}))
 for table in [1,2]:
  for version in ['A','B']:
   a,b=[r for r in bases if r['table']==table and r['version']==version]
   for kind in ['K','V']:
    for layer in range(24):raw['source_pairs'].append(dict(table=table,version=version,kind=kind,layer=layer,A=a['donors'][kind][layer],B=b['donors'][kind][layer],exact=True))
 assert execution_schema(raw,rows,False,512)==9
 for i,row in enumerate([r for r in rows if r['version']=='B']):
  base=bases[2*i+1];firstquery=next(r['query_row'] for r in rows if r['table']==row['table'])
  for name,kind,version,shuffle in PATCHES:
   r=copy.deepcopy(base);donor=next(b for b in bases if b['table']==row['table'] and b['query_row']==firstquery and b['version']==version);r.update(patch=name,clock=clock(row,'patch',kind,shuffle),donor_table=row['table'],donor_query_row=firstquery,donor_version=version,donor_kind=kind,donor_vectors=donor['donors'][kind],first3_full_exact=True,baseline_first3_full_states=base['first3_full_states'])
   if name.endswith('identity'):r['identity_exact']=True
   raw['patches'].append(r);raw['forward_records'].append(dict(number=len(raw['forward_records'])+1,phase='B',**{k:r[k] for k in ['table','query_row','patch','clock']}))
 assert execution_schema(raw,rows,True,512)==33
 for field in ['source_pairs','patches','forward_records']:
  bad=copy.deepcopy(raw);bad[field].pop()
  try:execution_schema(bad,rows,True,512)
  except AssertionError:pass
  else:raise AssertionError('incomplete phase schema accepted')
 # Primary V is not rescued by an independently passing K path.
 for i in range(4):
  a,b=bases[2*i:2*i+2];ya,yb=a['expected_label'],b['expected_label'];a['metrics']['label_logits'][ya]=1.;b['metrics']['label_logits'][yb]=1.
  for r in raw['patches'][i*6:(i+1)*6]:r['metrics']['label_logits']={k:0. for k in 'ABCD'};r['metrics']['label_logits'][ya]=1. if r['patch']=='K_clean' else -1.
 summary=functional_summary(bases,raw['patches']);assert not summary['paths']['V']['fixed_screen_pass'] and summary['paths']['K']['fixed_screen_pass']
 checks.append('9/33 exhaustive schema/192 source pairs/partial rejection/V cannot substitute K')
 from r27_exec_common import Vectors
 import io
 v=Vectors();b=torch.tensor([[[0.,-0.,1.]]],dtype=torch.bfloat16);a=v.keep(b);assert a==v.keep(b.clone())
 decoded=(v.arrays[a['key']].astype(np.uint32)<<16).view(np.float32);assert decoded[0,0,2]==1 and np.signbit(decoded[0,0,1]) and not np.signbit(decoded[0,0,0])
 stream=io.BytesIO();np.savez(stream,**v.arrays);stream.seek(0)
 with np.load(stream,allow_pickle=False) as z:assert z[a['key']].tobytes()==v.arrays[a['key']].tobytes()
 checks.append('BF16 raw bits/signed zero/content dedup/NPZ lossless decode')
 model=toy()
 with SourceReplay(model) as c:
  g=Guard();c.bind(g);c.configure('capture',positions,10);assert g.n==1;run(c,x);d={k:dict(q) for k,q in c.captured.items()};c.configure('patch',positions,10,'V',d['V']);assert g.n==2
  for loop in range(4):
   for _,_,m in c.modules:m(x)
  c.flags[-1]=torch.tensor(False)
  try:c.verify()
  except AssertionError:pass
  else:raise AssertionError('false flag accepted')
 assert g.n==3
 checks.append('false GPU-style boolean flags rejected/invalidate configure+cleanup clock')
 from r27_exec_protocol import check_resources
 sup=dict(status='complete',exit_code=0,overrun=False,gpu_worker_seconds=20.)
 book=dict(used_seconds=20.,inherited_total_seconds=18874.135196831077,total_used_seconds=18894.135196831077);old=[18874.135196831077]+[0.]*14;assert check_resources(sup,book,old)==20.
 for changes in [dict(timed_out=True),dict(overrun=True),dict(exit_code=1),dict(gpu_worker_seconds=181.),dict(gpu_worker_seconds=0.)]:
  try:check_resources(dict(sup,**changes),book,old)
  except AssertionError:pass
  else:raise AssertionError('bad supervisor accepted')
 for badbook in [dict(book,used_seconds=19.),dict(book,total_used_seconds=1.)]:
  try:check_resources(sup,badbook,old)
  except AssertionError:pass
  else:raise AssertionError('bad ledger accepted')
 bad=copy.deepcopy(raw);bad['invalid_reasons']=['synthetic premise failure']
 try:execution_schema(bad,rows,True,512)
 except AssertionError:pass
 else:raise AssertionError('invalid premise accepted')
 bases[0]['metrics']['expected_strictly_highest']=False;assert not functional_summary(bases,[])['phaseA_pass'];bases[0]['metrics']['expected_strictly_highest']=True
 try:functional_summary(bases,raw['patches'][:-1])
 except AssertionError:pass
 else:raise AssertionError('partial PhaseB summary accepted')
 checks.append('timeout/overrun/actual debit/ledger guards/invalid vs completePhaseA/partialPhaseB rejection')
 from r27_exec_prepare import SOURCES
 for p in SOURCES:
  ast.parse((ROOT/p).read_text());assert '\\' not in p
 worker=(ROOT/'scripts/r27_exec_worker.py').read_text();assert worker.index('input_audit=audit(tok)')<worker.index('tok,model=load_model()')
 assert 'full=True' in worker and 'vectors.keep(s) for s in states[:3]' in worker
 checks.append('portable explicit source set/pre-model tokenizer audit/full first3 state scope')
 save('results/r27_EXEC_CPU_CHECKS.json',dict(status='passed_CPU_synthetic_only',checks=checks,real_model_loaded=False,GPU_worker_seconds=0,torch_version=torch.__version__,source_sha256={p:sha(p) for p in SOURCES}))
 print('R27_EXEC_CPU_CHECKS_PASS',len(checks))
if __name__=='__main__':main()
