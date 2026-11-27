#!/usr/bin/env bash
set -euo pipefail
v=/root/autodl-tmp/ouro-v5-icml-phase1-20261127
source /root/autodl-tmp/ouro-v4-cfit-20261126/handoff_codex/env.sh
cd "$v"
export PYTHONPATH="$v:$v/codex_supervision:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 7>codex_supervision/track_b_gpu1_worker.lock
flock -n 7 || exit 0
gpu=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$gpu"; do
    sleep 12
done
if ! test -f codex_supervision/PEFT_GPU1_CANARY.json; then
"$py" -u - <<'PY'
from pathlib import Path
from types import SimpleNamespace
import torch
from v4_common import load_base_runtime,save_json
from v5_matched_peft import METHODS,build,batch_logits
from generation_followup_run import cache_type,forward,prepare
torch.set_num_threads(4)
_,archive,model,tokenizer=load_base_runtime()
examples=[{"id":"v5-preflight","question":"Which object rings in a church tower?","options":["A bell","A book"]}]
rows=[SimpleNamespace(ids=tokenizer.encode("The ocean is",add_special_tokens=False)),
      SimpleNamespace(ids=tokenizer.encode("The plant requires",add_special_tokens=False))]
tokens,valid=prepare(rows,tokenizer.eos_token_id)
guess=tokenizer.encode(" water",add_special_tokens=False)[-1]
extension=torch.full((2,1),guess,device=model.device,dtype=torch.long)
record={}
for method in METHODS:
    net=build(method,model,tokenizer,26112711).to(model.device)
    with torch.enable_grad():
        logits=batch_logits(archive,model,tokenizer,examples,4,net)[0]
        loss=-logits.log_softmax(-1)[0]
        loss.backward()
    grad=sum(float(p.grad.float().norm()) for p in net.parameters() if p.grad is not None)
    assert grad>0 and torch.isfinite(logits).all()
    record[method]={"params":sum(p.numel() for p in net.parameters()),"finite_gradient_norm":grad}
    if method=="M3_SoftPrompt":
        with torch.inference_mode(),net.attach(model):
            cache=cache_type(model)(tokens.shape[1]+4)
            idx=torch.arange(tokens.shape[1],device=model.device)
            out=forward(model,tokens,valid,idx,"native",None,1.,cache,slice(None))
            uncached=forward(model,tokens,valid,idx,"native",None,1.,None,slice(None))
            assert torch.equal(out,uncached)
            valid_next=torch.cat((valid,torch.ones_like(extension,dtype=torch.bool)),dim=1)
            idx2=torch.tensor([valid_next.shape[1]-1],device=model.device)
            next_cached=forward(model,extension,valid_next,idx2,"native",None,1.,cache,slice(None))
            next_full=forward(model,torch.cat((tokens,extension),1),valid_next,
                              torch.arange(valid_next.shape[1],device=model.device),"native",None,1.,None,slice(None))
            record[method]["incremental_max_logit_diff"]=float((next_cached-next_full).abs().max())
            assert record[method]["incremental_max_logit_diff"]<=1.
    del net
    torch.cuda.empty_cache()
save_json(Path("codex_supervision/PEFT_GPU1_CANARY.json"),record)
print("V5_PEFT_CANARY_PASS",record,flush=True)
PY
fi
for source in S1 S2 S3; do
  for seed in 26112711 26112712; do
    "$py" -u codex_supervision/v5_peft_memory_recovery.py "$v" "$source" "$seed"
  done
done
echo V5_1C_GPU1_TRAINING_AND_DEV_DONE
