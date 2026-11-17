import pathlib,sys,json,torch
R=pathlib.Path('/root/ouro-region-use-sol-20261117');P=pathlib.Path('/root/ouro-semantic-read-sol-20261117');sys.path.insert(0,str(P/'code'))
from ab1_loader_draft import load_model
def initialize(tag):
 torch.set_num_threads(4);torch.manual_seed(2026111731);torch.backends.cuda.matmul.allow_tf32=False;m,r=load_model('/root/autodl-tmp/ouro-e4-e5/model',authorized=True,device='cuda:0');m.model.total_ut_steps=2
 for p in m.parameters():p.requires_grad_(False)
 r['total_ut_steps']=2;assert r==json.loads((P/'SOURCE_RUNTIME.json').read_text());(R/('SOURCE_RUNTIME_'+str(tag)+'.json')).write_text(json.dumps(r,indent=2));return m
