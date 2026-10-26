"""Complete answers at each true fixed depth, plus nonzero hidden guidance.

These are separate autoregressive paths; they are not same-prefix trajectories.
The four-depth same-prefix mechanistic audit is the MC experiment.
"""
import json
import torch
from ouro_audit import ROOT,Budget,load,save,discrepancy
from math_generation import decode

def main():
    budget=Budget()
    try:
        tok,m=load()
        test=[json.loads(l) for l in (ROOT/'data/math_test.jsonl').read_text(encoding='utf-8').splitlines()]
        examples=[json.loads(l) for l in (ROOT/'data/math_fewshot.jsonl').read_text(encoding='utf-8').splitlines()]
        shots='\n\n'.join('Question: '+r['question']+'\nAnswer: '+r['answer'] for r in examples)
        dest=ROOT/'trajectories/math_depth_and_hidden.jsonl'
        old=[json.loads(l) for l in dest.read_text(encoding='utf-8').splitlines()] if dest.exists() else []
        done={(r['id'],r['method']) for r in old};checks=[]
        for label,depth,guidance,omega in [('depth_1',1,'baseline',0),('depth_2',2,'baseline',0),('depth_3',3,'baseline',0),('hidden_fixed_025',4,'hidden',.25)]:
            m.model.total_ut_steps=depth;m.config.total_ut_steps=depth
            if depth<4:
                x=tok('The Solar System contains planets and',return_tensors='pt').to('cuda')
                normal=m(**x,use_cache=False,exit_at_step=depth-1,logits_to_keep=1).logits
                _,states,_=m.model(**x,use_cache=False);custom=m.lm_head(states[-1][:,-1:,:])
                assert torch.equal(normal,custom)
                cached=m(**x,use_cache=True,exit_at_step=depth-1,logits_to_keep=1)
                assert len(cached.past_key_values.key_cache)==24*depth
                checks.append(dict(depth=depth,diff=discrepancy(normal,custom),cache_levels=24*depth))
            for row in test:
                if (row['id'],label) in done:continue
                budget.check();prompt=shots+'\n\nQuestion: '+row['question']+'\nAnswer:'
                result=dict(**row,method=label,recurrent_depth=depth,omega=omega,protocol='gsm8k_3shot_greedy_256_per_token_v1',**decode(tok,m,prompt,guidance,omega,budget))
                result['correct']=result['prediction']==row['gold'].replace(',','')
                with dest.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                done.add((row['id'],label));save('STATUS.json',dict(stage='math_depth_and_hidden',method=label,completed=len(done),total=400))
                print(json.dumps(dict(id=row['id'],method=label,done=len(done),correct=result['correct'],elapsed_s=result['elapsed_s'])),flush=True)
        m.model.total_ut_steps=4;m.config.total_ut_steps=4
        save('results/math_depth_parity.json',dict(passed=True,checks=checks,scope='separate autoregressive paths with actual truncated recurrence'))
    finally:budget.close('math_fixed_depth_and_hidden_guidance')

if __name__=='__main__':main()
