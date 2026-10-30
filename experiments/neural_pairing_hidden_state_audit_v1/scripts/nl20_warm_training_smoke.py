"""Three-step throughput/memory measurement of the ORIGINAL nl20 warm-up recipe.

The schedule remains 900 steps; --halt 3 is an explicit pilot. Its checkpoint is
never substituted for step900/1320 and never called a reproduced QA checkpoint.
"""
import json
import os
import runpy
import sys
import time
from linux_runtime import ROOT,configure,save

configure()
import torch
torch.set_num_threads(4)
assert (ROOT/'data/own_replies/replies.json').exists()
assert json.loads((ROOT/'logs/TINY_TESTS_EXIT.json').read_text())['exit_code']==0
torch.cuda.set_per_process_memory_fraction(.42,0)
out=ROOT/'training_pilots/nl20_warm_halt3'
assert not out.exists(),'never overwrite or blindly restart a training attempt'
flags=['--replies',str(ROOT/'data/own_replies/replies.json'),'--out',str(out),
    '--read_start','0.1','--content_init','qkvo','--gate_norm','--hold','100000','--payload_w','3',
    '--warmup_steps','10','--relay_first','100000','--relay_games',
    'relay_word,relay_number,relay_phrase,relay_digits,merge_sum,merge_fact,lookup',
    '--steps','900','--bs','16','--coread','0','--anchor','0','--dialogue','0','--comp','0',
    '--send_cost','0','--lr','1e-4','--lr_bridge','3e-4','--lr_alpha','3e-3',
    '--eval_every','100','--first_eval','20','--quick_n','4','--payload_eval','16','--save_every','900','--ckpt','none',
    '--halt','3']
start=time.time();state=dict(status='running',started_unix=start,argv=flags,
    scope='Throughput pilot of exact warm-up recipe. No full warmup or QA training claimed.',
    allocator_limit_fraction=.42)
save(ROOT/'logs/WARM_TRAINING_SMOKE.json',state)
try:
    sys.argv=[str(ROOT/'upstream/pairing/train.py')]+flags
    runpy.run_path(sys.argv[0],run_name='__main__')
    state.update(status='passed',peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
except BaseException as exc:
    state.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
finally:
    state.update(finished_unix=time.time(),gpu_worker_seconds=time.time()-start);save(ROOT/'logs/WARM_TRAINING_SMOKE.json',state)
