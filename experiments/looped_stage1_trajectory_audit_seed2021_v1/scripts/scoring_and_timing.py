"""Independent scoring check against full HF logits / masked native CE."""
import json
import time
import torch
import torch.nn.functional as F
from ouro_audit import ROOT,CFG,Budget,load,encode_pair,save,capture_raw
from transformers import AutoModelForCausalLM,AutoTokenizer

budget=Budget();reports=[]
try:
    for name in ['Ouro-1.4B','Qwen3-4B-Base']:
        budget.check();p=ROOT/'models'/name
        tok=AutoTokenizer.from_pretrained(p,local_files_only=True)
        m=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,trust_remote_code=name.startswith('Ouro'),torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        for prompt,answer in [('Question: Which planet is closest to the Sun?\nAnswer:',' Mercury'),('Question: What do green plants need?\nAnswer:',' sunlight and water')]:
            prefix,continuation=encode_pair(tok,prompt,answer)
            full=torch.tensor([prefix+continuation],device='cuda');short=full[:,:-1]
            # Independent conventional teacher-forcing reference includes the
            # final answer token and shifts logits/labels in the standard way.
            kw=dict(use_cache=False)
            if name.startswith('Ouro'):kw['exit_at_step']=3
            normal=m(input_ids=full,**kw).logits[:,len(prefix)-1:-1,:].float()
            target=full[:,len(prefix):]
            reference=-F.cross_entropy(normal.reshape(-1,normal.shape[-1]),target.reshape(-1),reduction='sum').item()
            if name.startswith('Ouro'):
                _,states,_=m.model(input_ids=short,use_cache=False)
                actual_logits=m.lm_head(states[-1][:,len(prefix)-1:,:]).float()
            else:
                actual_logits=m(input_ids=short,use_cache=False,logits_to_keep=torch.arange(len(prefix)-1,short.shape[1],device='cuda')).logits.float()
            actual=actual_logits.log_softmax(-1).gather(-1,target[...,None]).sum().item()
            error=abs(reference-actual)/len(continuation)
            assert error<=.05,(name,prompt,error)
            reports.append(dict(model=name,prompt=prompt,answer=answer,n_tokens=len(continuation),reference_sum_loglikelihood=reference,selected_sum_loglikelihood=actual,error_per_token=error))
        if name.startswith('Ouro'):
            # Isolate actual per-prefix guidance overhead; not halting speedup.
            timings=[];x=tok('Question: Why do plants need sunlight?\nAnswer:',return_tensors='pt').to('cuda')
            for method in ['baseline','logits','hidden']:
                for i in range(20):
                    budget.check();torch.cuda.synchronize();start=time.monotonic()
                    if method=='hidden':
                        with capture_raw(m,slice(-1,None)) as raw:_,states,_=m.model(**x,use_cache=False)
                        z=m.lm_head(m.model.norm(raw[-1]+.25*(raw[-1]-raw[0])))
                    else:
                        _,states,_=m.model(**x,use_cache=False);z=m.lm_head(states[-1][:,-1:,:]).float()
                        if method=='logits':z=z+.25*(z-m.lm_head(states[0][:,-1:,:]).float())
                    torch.cuda.synchronize();timings.append(dict(method=method,elapsed_s=time.monotonic()-start))
            save('results/loopcd_latency_probe.json',dict(scope='single-prefix next-token prefill; 20 repetitions; shared GPU; includes norm/head; no wall-clock halting claim',records=timings))
            from native_gate_probe import run as probe_native_gate
            probe_native_gate(tok,m,budget)
            from math_teacherforced_probe import run as probe_math
            probe_math(tok,m,budget)
        del m;torch.cuda.empty_cache()
    save('results/scoring_parity.json',dict(passed=True,cases=reports,tolerance_nll_per_token=.05))
finally:budget.close('independent_scoring_and_latency_checks')
