import json,tempfile,torch
from pathlib import Path
from safetensors.torch import save_file
from test_technical import obs,states,norms,logits,scores,targets,R,m as model
from decompose import decompose
model.model.norm.weight=model.model.norm.w
import sys,types
if sys.platform=='win32':sys.modules['fcntl']=types.SimpleNamespace()
from full_runner_draft import check_existing
from source_input import sha
with tempfile.TemporaryDirectory() as td:
 out=Path(td);key='train_'+'a'*64;t={f'u{u}_layer{i:02d}_{k}':v.clone() for (u,i),d in obs['states'].items() for k,v in d.items()};t.update({f'loop{u+1}_logits':z.clone() for u,z in enumerate(logits)});t.update({f'loop{u+1}_post_norm':z.clone() for u,z in enumerate(norms)});tp=out/(key+'.safetensors');save_file(t,str(tp))
 e={'id':'q','split':'train','gold':0,'metadata_sha256':'m','tensor_sha256':'s','scores':scores,'targets':targets,'R':R,'saved_row':2,'states':states,'norms':norms,'logits':logits};start={'model':'abc'}
 W=model.lm_head.weight[targets].detach().numpy();gamma=model.model.norm.weight.detach().numpy();lens=decompose(obs['states'],W,gamma,1e-5,list(range(len(targets))),0)
 m={'id':'q','split':'train','gold':0,'natural_metadata_sha256':'m','natural_tensor_sha256':'s','source_hashes':start,'probe_tensor_sha256':sha(tp),'prior_scores':scores,'targets':targets,'R_absolute':R,'R_saved_row':2,'attention':obs['attention'],'lens':lens};mp=out/(key+'.json');mp.write_text(json.dumps(m))
 assert check_existing(out,key,e,start,'cpu',model,1e-5)
 m['R_saved_row']=1;mp.write_text(json.dumps(m))
 try:check_existing(out,key,e,start,'cpu',model,1e-5)
 except ValueError:pass
 else:raise AssertionError('wrong row')
 mp.unlink()
 try:check_existing(out,key,e,start,'cpu',model,1e-5)
 except RuntimeError:pass
 else:raise AssertionError('orphan')
print('FULL_RESUME_CPU_MOCK_PASS; no GPU or actual recovery')
