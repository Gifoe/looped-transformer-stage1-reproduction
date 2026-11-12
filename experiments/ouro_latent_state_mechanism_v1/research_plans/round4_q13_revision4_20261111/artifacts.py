"""Local staging: immutable 12-logit + 4-branch artifacts and replay."""
import hashlib,json,os,uuid,torch
from pathlib import Path
from safetensors.torch import save_file,load_file
from q13_core import Capture,matched,tensor_sha,bits_equal

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def score(z,targets,device):
 if z.dtype!=torch.bfloat16 or z.ndim!=2 or z.shape[0]!=1 or not torch.isfinite(z).all():raise ValueError('logits')
 if not targets or any(type(t)!=int or t<0 or t>=z.shape[1] for t in targets):raise ValueError('targets')
 z=z.to(device);return [float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=device)).sum()) for t in targets]
def save(out,identity,results,targets,prefix,readout,expected_vocab,expected_hidden,prior_logits,source_guard):
 out=Path(out);out.mkdir(exist_ok=False);source_guard();tensors={};scores={};captures={}
 if set(prior_logits)!={'211','221','212','222'}:raise ValueError('missing previous corners')
 for r in ('1','2'):
  pair=results[r];a=pair['capture1'];b=pair['capture2']
  if not isinstance(a,Capture) or not isinstance(b,Capture) or (a.trajectory,b.trajectory)!=('21'+r,'22'+r) or a.input_ids_sha256!=identity['input_ids_tensor_sha256'] or b.input_ids_sha256!=identity['input_ids_tensor_sha256']:raise ValueError('capture trajectory/input')
  matched(a,b,prefix,readout)
  for i,c in (('1',a),('2',b)):
   if c.branch_sha256!=tensor_sha(c.branch) or c.input_sha256!=tensor_sha(c.layer_input):raise ValueError('capture drift')
   captures['a'+i+'r'+r]={'trajectory':c.trajectory,'input_ids_sha256':c.input_ids_sha256,'layer_input_sha256':c.input_sha256,'full_branch_sha256':c.branch_sha256,'r_slice_sha256':tensor_sha(c.branch[:,readout,:]),'site':'ut1/layer12/input_layernorm_2'}
  for key in ('B1','B2','I1','I2','U1','U2'):
   z=pair[key]
   if tuple(z.shape)!=(1,expected_vocab):raise ValueError('vocabulary')
   tensors[key+'r'+r]=z.detach().cpu().contiguous().clone();scores[key+'r'+r]=score(z,targets,z.device)
  if not bits_equal(pair['B1'],prior_logits['21'+r]) or not bits_equal(pair['B2'],prior_logits['22'+r]) or not bits_equal(pair['B1'],pair['I1']) or not bits_equal(pair['B2'],pair['I2']):raise ValueError('baseline/identity logits')
  if bits_equal(a.branch[:,readout,:],b.branch[:,readout,:]) and (not bits_equal(pair['B1'],pair['U1']) or not bits_equal(pair['B2'],pair['U2'])):raise ValueError('identical branch patch')
  for i in ('1','2'):
   a=pair['capture'+i].branch[:,readout,:]
   if a.dtype!=torch.bfloat16 or tuple(a.shape)!=(1,len(readout),expected_hidden) or not torch.isfinite(a).all():raise ValueError('branch')
   tensors['a'+i+'r'+r]=a.detach().cpu().contiguous().clone()
 temp=out/('values.'+uuid.uuid4().hex+'.partial');final=out/'values.safetensors'
 try:
  save_file(tensors,str(temp));check=load_file(str(temp))
  if set(check)!=set(tensors) or any(not torch.equal(check[k],tensors[k]) for k in tensors):raise ValueError('tensor readback')
  with temp.open('r+b') as f:os.fsync(f.fileno())
  os.link(temp,final)
 finally:
  if temp.exists():temp.unlink()
 decisions={}
 for key,values in scores.items():
  i=max(range(len(values)),key=lambda x:(values[x],-x));decisions[key]={'prediction':i,'top_tie':sum(v==values[i] for v in values)>1}
 payload={'decisions':decisions,'identity':identity,'targets':targets,'readout':readout,'prefix':prefix,'captures':captures,'vocab_size':expected_vocab,'hidden_size':expected_hidden,'scores':scores,'tensor_sha256':sha(final)}
 envelope={'payload':payload,'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest()};source_guard()
 m=out/'metadata.json';tmp=out/('metadata.'+uuid.uuid4().hex+'.partial')
 try:
  with tmp.open('xb') as f:f.write(canonical(envelope));f.flush();os.fsync(f.fileno())
  os.link(tmp,m)
 finally:
  if tmp.exists():tmp.unlink()
 return envelope

def replay(out,identity,targets,prefix,readout,expected_vocab,expected_hidden,prior_logits,device,source_guard):
 out=Path(out);source_guard();m=json.loads((out/'metadata.json').read_text());v=m['payload']
 if m['payload_sha256']!=hashlib.sha256(canonical(v)).hexdigest() or v['identity']!=identity or v['targets']!=targets or v['readout']!=readout or v['prefix']!=prefix or v['vocab_size']!=expected_vocab or v['hidden_size']!=expected_hidden or v['tensor_sha256']!=sha(out/'values.safetensors'):raise ValueError('provenance')
 t=load_file(str(out/'values.safetensors'));logits={k+'r'+r for k in ('B1','B2','I1','I2','U1','U2') for r in ('1','2')};branches={'a'+i+'r'+r for i in ('1','2') for r in ('1','2')}
 if set(t)!=logits|branches or set(v['scores'])!=logits:raise ValueError('tensor keys')
 for k in logits:
  if tuple(t[k].shape)!=(1,expected_vocab) or score(t[k],targets,device)!=v['scores'][k]:raise ValueError('score replay')
  values=v['scores'][k];i=max(range(len(values)),key=lambda j:(values[j],-j))
  if v['decisions'][k]!={'prediction':i,'top_tie':sum(x==values[i] for x in values)>1}:raise ValueError('decision')
 if set(prior_logits)!={'211','221','212','222'} or set(v['captures'])!=branches:raise ValueError('prior/capture keys')
 for r in ('1','2'):
  for i in ('1','2'):
   k='a'+i+'r'+r;c=v['captures'][k]
   if c['trajectory']!=('21'+r if i=='1' else '22'+r) or c['input_ids_sha256']!=identity['input_ids_tensor_sha256'] or c['site']!='ut1/layer12/input_layernorm_2' or c['r_slice_sha256']!=tensor_sha(t[k]):raise ValueError('capture receipt')
  if not bits_equal(t['B1r'+r],prior_logits['21'+r].cpu()) or not bits_equal(t['B2r'+r],prior_logits['22'+r].cpu()) or not bits_equal(t['B1r'+r],t['I1r'+r]) or not bits_equal(t['B2r'+r],t['I2r'+r]):raise ValueError('baseline identity')
  if bits_equal(t['a1r'+r],t['a2r'+r]) and (not bits_equal(t['B1r'+r],t['U1r'+r]) or not bits_equal(t['B2r'+r],t['U2r'+r])):raise ValueError('same branch')
 for k in branches:
  if t[k].dtype!=torch.bfloat16 or tuple(t[k].shape)!=(1,len(readout),expected_hidden) or not torch.isfinite(t[k]).all():raise ValueError('branch shape')
 source_guard();return {'metadata_sha256':sha(out/'metadata.json'),'tensor_sha256':sha(out/'values.safetensors'),'device':str(device),'status':'ZERO_SCORE_DIFFERENCE'}
