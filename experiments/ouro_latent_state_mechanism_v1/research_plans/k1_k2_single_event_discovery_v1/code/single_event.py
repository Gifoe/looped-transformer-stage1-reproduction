"""One K2 layer of the exact parent ideal pair-message implementation."""
import pathlib,sys,json,hashlib,torch
PARENT=pathlib.Path('/root/ouro-semantic-read-sol-20261117');ROOT=pathlib.Path('/root/ouro-binding-interface-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-binding-interface-sol-20261117/raw');sys.path.insert(0,str(PARENT/'code'))
import record_operator_v2r1 as B
from ab1_loader_draft import load_model

def initialize(tag):
 torch.set_num_threads(4);torch.manual_seed(2026111731);torch.backends.cuda.matmul.allow_tf32=False;m,r=load_model('/root/autodl-tmp/ouro-e4-e5/model',authorized=True,device='cuda:0');m.model.total_ut_steps=2
 for p in m.parameters():p.requires_grad_(False)
 r['total_ut_steps']=2;previous=json.loads((PARENT/'SOURCE_RUNTIME.json').read_text());assert r==previous;(ROOT/('SOURCE_RUNTIME_'+str(tag)+'.json')).write_text(json.dumps(r,indent=2));return m

def run(m,q,layer,kind='semantic',prefix=None,capture=False,witness=False,targets=None):
 assert layer in range(9,17);previous=B.LAYERS;B.LAYERS=(layer-1,)
 try:
  z,heads,pref,w=B.run(m,q,kind=kind,arm=2,native_prefix=prefix,capture_prefix=capture,witness=witness,target_doses=targets)
  expected=[] if kind in ['native','noop'] else [[2,layer]];assert [[e['loop'],e['layer']] for e in z['audit']['events']]==expected;assert z['audit']['actual_original_SDPA_invocations']==48
  return z,heads,pref,w
 finally:B.LAYERS=previous
