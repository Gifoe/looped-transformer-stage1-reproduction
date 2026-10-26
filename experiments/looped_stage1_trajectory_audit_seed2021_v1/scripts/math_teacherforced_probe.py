"""Four-depth scalar diagnostics on the same GSM8K reference prefix.

Reference-solution likelihood is auxiliary teacher forcing, separate from the
complete generated answers at each true depth. No generated-prefix claims.
"""
import json
import time
import torch
import torch.nn.functional as F
from ouro_audit import ROOT,encode_pair,distribution,save,discrepancy

def run(tok,m,budget):
    data=[json.loads(l) for l in (ROOT/'data/math_test.jsonl').read_text(encoding='utf-8').splitlines()]
    demos=[json.loads(l) for l in (ROOT/'data/math_fewshot.jsonl').read_text(encoding='utf-8').splitlines()]
    shots='\n\n'.join('Question: '+r['question']+'\nAnswer: '+r['answer'] for r in demos)
    generated={}
    for name in ['math_generation.jsonl','math_depth_and_hidden.jsonl']:
        for line in (ROOT/'trajectories'/name).read_text(encoding='utf-8').splitlines():
            row=json.loads(line);generated[row['id'],row['method']]=row
    records=[];check=None
    for row in data:
        budget.check();prompt=shots+'\n\nQuestion: '+row['question']+'\nAnswer:'
        prefix,continuation=encode_pair(tok,prompt,' '+row['answer'])
        assert len(prefix)+len(continuation)<=2048
        x=torch.tensor([prefix+continuation[:-1]],device='cuda');target=torch.tensor(continuation,device='cuda')
        pos=len(prefix)-1;torch.cuda.synchronize();start=time.monotonic()
        _,states,_=m.model(input_ids=x,use_cache=False)
        trajectory=[];previous_lp=None;previous_h=None
        for depth,state in enumerate(states,1):
            z=m.lm_head(state[:,pos:,:])[0].float();lp=z.log_softmax(-1)
            stats,token_lp=distribution(z[0]);h=state[0,pos,:].float()
            method='baseline' if depth==4 else 'depth_'+str(depth)
            gen=generated[row['id'],method]
            trajectory.append(dict(depth=depth,predicted_next_token_id=int(z[0].argmax()),predicted_next_token=tok.decode([int(z[0].argmax())]),
                generated_answer_prediction=gen['prediction'],generated_answer_correct=gen['correct'],
                reference_solution_sum_loglikelihood=float(lp.gather(-1,target[:,None]).sum()),
                reference_solution_nll_per_token=float(-lp.gather(-1,target[:,None]).mean()),
                **stats,full_vocab_kl=None if previous_lp is None else float((token_lp.exp()*(token_lp-previous_lp)).sum()),
                hidden_relative_l2=None if previous_h is None else float((h-previous_h).norm()/previous_h.norm().clamp_min(1e-8)),
                hidden_cosine=None if previous_h is None else float(F.cosine_similarity(h[None],previous_h[None]))))
            previous_lp=token_lp;previous_h=h
        if check is None:
            selection=torch.arange(pos,x.shape[1],device='cuda')
            normal=m(input_ids=x,use_cache=False,exit_at_step=3,logits_to_keep=selection).logits[0]
            check=discrepancy(normal,z);assert check['max_abs']==0,check
        torch.cuda.synchronize()
        records.append(dict(**row,protocol='gsm8k_3shot_same_reference_solution_teacher_forcing_v1',
            prompt_tokens=len(prefix),reference_tokens=len(continuation),trajectory=trajectory,elapsed_s=time.monotonic()-start,
            note='Generated answers are linked from separate true-depth autoregressive paths; scalar vocabulary/hidden metrics use the common prompt endpoint.'))
    path=ROOT/'trajectories/ouro_math_teacherforced.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    save('results/math_teacherforced_parity.json',dict(passed=True,records=len(records),full_hf_final_depth=check))
