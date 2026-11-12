import tempfile,hashlib,json,sys,types,torch
from pathlib import Path
from safetensors.torch import save_file
from production_inputs import read_question,sha
with tempfile.TemporaryDirectory() as td:
 root=Path(td);run=root/'orchestration_runs/pqr-round3-20261111';old=root/'orchestration_runs/round2-full-20261111';orig=root/'orchestration_runs/run-20261107T053919Z-6d01138e';split='train';ident='item';key=hashlib.sha256(ident.encode()).hexdigest();regions={'P':[0],'Q':[1],'R':[2]}
 for d in (run/split,old/split,orig/'shards'/split):d.mkdir(parents=True)
 nm=orig/'shards'/split/(key+'.json');nm.write_text(json.dumps({'records':[{'candidate':0,'scores':[1.,1.],'encoding':{'input_ids':[1,2,3],'target_ids':[4]}}]}));om=old/split/(key+'.json');ot=om.with_suffix('.safetensors');prior={'scores':[1.]};om.write_text(json.dumps({'payload':{'result':{'conditions':{'K2C_K1R':prior,'K2C_K2R':prior}}}}));z=torch.zeros(1,8,dtype=torch.bfloat16);save_file({'core_K2C_K1R':z.clone(),'core_K2C_K2R':z.clone()},str(ot))
 pm=run/split/(key+'.json');pt=pm.with_suffix('.safetensors');save_file({'corner_211':z.clone(),'corner_212':z.clone()},str(pt));p={'id':ident,'split':split,'binding_sha256':'b','old_metadata_sha256':sha(om),'result':{'regions':regions,'corners':{k:prior for k in ('211','212','221','222')},'gold':0,'fixed_wrong':1}};pm.write_text(json.dumps({'payload':p}))
 cv=run/'coverage_train.json';oc=old/'coverage_train.json';nc=orig/'coverage_train.json';cv.write_text(json.dumps({'status':'PQR_COMPLETE_CUDA_REPLAY_ZERO','split':split,'expected_questions':1,'complete_questions':1,'replayed_condition_logits':4,'binding_sha256':'b','questions':[{'id':ident,'metadata_sha256':sha(pm),'logits_sha256':sha(pt)}]}));oc.write_text(json.dumps({'questions':[{'id':ident,'metadata_sha256':sha(om),'logits_sha256':sha(ot)}]}));nc.write_text(json.dumps({'questions':[{'id':ident,'metadata_sha256':sha(nm)}]}));expected={'pqr':sha(cv),'round2':sha(oc),'natural':sha(nc)}
 sys.modules['partition']=types.SimpleNamespace(partition=lambda *args:regions)
 x=read_question(root,split,ident,expected,{'id':ident,'split':split});assert x['targets']==[4] and set(x['prior_logits'])=={'211','212','221','222'}
 om.write_text('corrupt')
 try:read_question(root,split,ident,expected,{'id':ident,'split':split})
 except ValueError:pass
 else:raise AssertionError('source drift accepted')
print('LOCAL_INPUT_ADAPTER_CPU_PASS; mock partition / not production')
