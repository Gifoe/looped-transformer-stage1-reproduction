"""Memory adaptation of warm-up cost pilot: batch16, micro4, official CPU FP64 fit.

No original source math is rewritten. CPU fit uses align.py's existing SciPy path.
The generated own-only initialization is for cost measurement, not claimed as the
unreleased author's original mixture or checkpoint. Save step0 to avoid repeating
the expensive fit if profiling needs another memory adjustment.
"""
import json
import os
import runpy
import sys
import time
from linux_runtime import ROOT,configure,save

configure()
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
import align
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.38,0)
align.LA='cpu'
original=align.content_init
cache=ROOT/'training_pilots/warm_step0_cost_only'


def record_initialization(pair,models,toks,replies,*args,**kwargs):
    start=time.time();summary=original(pair,models,toks,replies,*args,**kwargs)
    cache.mkdir(parents=True,exist_ok=False)
    for x,model in models.items():model.save_pretrained(cache/x,save_embedding_layers=False)
    ck=pair.checkpoint(scope='cost_pilot_step0_not_author_checkpoint')
    ck['bridge']={k:v.detach().cpu() for k,v in ck['bridge'].items()}
    torch.save(ck,cache/'bridge.pt')
    save(cache/'PROVENANCE.json',dict(scope='Original architecture step0; generated own-only initialization for throughput measurement. Never label nl20/step900, step1320, or nl23.',
        init_seconds=time.time()-start,backend='official align.py CPU float64 SciPy path',mixture_distillation_targets_available=False))
    return summary


align.content_init=record_initialization
out=ROOT/'training_pilots/nl20_warm_halt3_micro4'
assert not out.exists()
flags=['--replies',str(ROOT/'data/own_replies/replies.json'),'--out',str(out),
    '--read_start','0.1','--content_init','qkvo','--gate_norm','--hold','100000','--payload_w','3',
    '--warmup_steps','10','--relay_first','100000','--relay_games',
    'relay_word,relay_number,relay_phrase,relay_digits,merge_sum,merge_fact,lookup',
    '--steps','900','--bs','16','--micro','4','--coread','0','--anchor','0','--dialogue','0','--comp','0',
    '--send_cost','0','--lr','1e-4','--lr_bridge','3e-4','--lr_alpha','3e-3',
    '--eval_every','100','--first_eval','20','--quick_n','4','--payload_eval','16','--save_every','900','--ckpt','none','--halt','3']
start=time.time();state=dict(status='running',started_unix=start,argv=flags,
    scope='Cost pilot with explicit memory adaptation; not full warm-up or QA reproduction',
    differences_from_warm_chain=['micro4 instead of1, global batch remains16','existing official CPU FP64 alignment fit backend','own-only generated init texts instead of unavailable original full-mixture distillation texts'],
    allocator_limit_fraction=.38)
save(ROOT/'logs/WARM_TRAINING_SMOKE_V2.json',state)
try:
    sys.argv=[str(ROOT/'upstream/pairing/train.py')]+flags
    runpy.run_path(sys.argv[0],run_name='__main__')
    state.update(status='passed',peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
except BaseException as exc:
    state.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
finally:
    state.update(finished_unix=time.time(),gpu_worker_seconds=time.time()-start);save(ROOT/'logs/WARM_TRAINING_SMOKE_V2.json',state)
