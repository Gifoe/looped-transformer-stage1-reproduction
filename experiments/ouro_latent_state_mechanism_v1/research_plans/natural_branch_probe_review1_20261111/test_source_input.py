import tempfile,json,hashlib,torch
from pathlib import Path
from safetensors.torch import save_file
import source_input as s
with tempfile.TemporaryDirectory() as td:
 root=Path(td);s.ROOT=root;s.ORIG=root/'orig';s.RES=root/'res';s.SEL=root/'sel.json';s.RES.mkdir();(s.ORIG/'shards/test').mkdir(parents=True)
 ids=[3,4,5];ih=hashlib.sha256(json.dumps(ids).encode()).hexdigest();sel={'selection':{'A':{'id':'x','split':'test','input_sha256':ih,'input_length':3}}};s.SEL.write_text(json.dumps(sel));key=hashlib.sha256(b'x').hexdigest();mp=s.ORIG/'shards/test'/(key+'.json');tp=mp.with_suffix('.safetensors')
 rec=[{'candidate':i,'input_ids_sha256':ih,'encoding':{'input_ids':ids,'predecessor_positions':[2],'target_ids':[7+i]},'captured_positions':[0,2],'scores':[1.,2.]} for i in range(2)];mp.write_text(json.dumps({'id':'x','split':'test','gold_index':1,'records':rec}))
 tensors={}
 for u in (1,2):
  for i in range(24):tensors[f'candidate0_loop{u}_layer{i:02d}']=torch.zeros(2,2048,dtype=torch.bfloat16)
  tensors[f'candidate0_loop{u}_post_norm']=torch.zeros(2,2048,dtype=torch.bfloat16);tensors[f'candidate0_loop{u}_lm_head_logits']=torch.zeros(1,49152,dtype=torch.bfloat16)
 save_file(tensors,str(tp));cv=s.RES/'coverage_test.json';cv.write_text(json.dumps({'questions':[{'id':'x','status':'complete','errors':[],'metadata_sha256':s.sha(mp),'tensor_sha256':s.sha(tp)}]}));s.EXPECTED.update(selection=s.sha(s.SEL),coverage=s.sha(cv))
 v=s.load_one('A');assert v['R']==2 and v['saved_row']==1 and v['gold']==1 and v['targets']==[7,8] and len(v['states'])==48
 mp.write_text('{}')
 try:s.load_one('A')
 except ValueError:pass
 else:raise AssertionError('metadata drift')
print('SOURCE_INPUT_CPU_FIXTURE_PASS; not real model')
