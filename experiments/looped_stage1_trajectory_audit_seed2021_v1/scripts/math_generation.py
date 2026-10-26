"""Actual per-token LoopCD with all four official recurrent cache levels."""
import json
from pathlib import Path
import re
import time
from ouro_audit import ROOT,CFG,Budget,load,capture_raw,save
import torch

def extract(text):
    hits=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',text)
    if not hits:return None
    return hits[-1].replace(',','').lstrip('+')

def decode(tok,m,prompt,method,omega,budget,max_new_tokens=256):
    encoded=tok.encode(prompt,add_special_tokens=False)
    assert len(encoded)<=2048
    input_ids=torch.tensor([encoded],device='cuda');past=None;generated=[]
    torch.cuda.synchronize();start=time.monotonic()
    for step in range(max_new_tokens):
        budget.check(reserve=10)
        if method=='hidden':
            with capture_raw(m,slice(-1,None)) as raw:
                outputs,states,_=m.model(input_ids=input_ids,past_key_values=past,use_cache=True)
            guided=raw[-1]+omega*(raw[-1]-raw[0])
            z=m.lm_head(m.model.norm(guided))[:,-1,:]
        else:
            outputs,states,_=m.model(input_ids=input_ids,past_key_values=past,use_cache=True)
            z=m.lm_head(states[-1][:,-1:,:])[:,-1,:].float()
            if method.startswith('logits'):
                early=m.lm_head(states[0][:,-1:,:])[:,-1,:].float()
                z=z+omega*(z-early)
        token=int(z.argmax(-1).item());past=outputs.past_key_values
        if token==tok.eos_token_id:break
        generated.append(token);input_ids=torch.tensor([[token]],device='cuda')
        text=tok.decode(generated,skip_special_tokens=True)
        if '\nQuestion:' in text or '\nQ:' in text:break
        # Stop on a complete numeric answer line, without cutting an unfinished integer.
        if re.search(r'####\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\n',text):break
    torch.cuda.synchronize()
    text=tok.decode(generated,skip_special_tokens=True).split('\nQuestion:')[0].split('\nQ:')[0]
    return dict(text=text,prediction=extract(text),token_ids=generated,new_tokens=len(generated),
                reached_token_limit=len(generated)==max_new_tokens,elapsed_s=time.monotonic()-start,
                peak_memory_bytes=torch.cuda.max_memory_allocated())

def main():
    budget=Budget()
    try:
        tok,m=load();selected=json.loads((ROOT/'configs/validation_selected.json').read_text())
        parity_prompt='The Solar System contains planets and'
        encoded=tok(parity_prompt,return_tensors='pt').to('cuda')
        normal=m.generate(**encoded,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
        normal_ids=normal[0,encoded.input_ids.shape[1]:].tolist()
        if tok.eos_token_id in normal_ids:normal_ids=normal_ids[:normal_ids.index(tok.eos_token_id)]
        identity_checks={}
        for method in ['baseline','logits','hidden']:
            custom=decode(tok,m,parity_prompt,method,0,budget,max_new_tokens=8)
            assert custom['token_ids']==normal_ids,(method,custom['token_ids'],normal_ids)
            identity_checks[method]=True
        save('results/generation_parity.json',dict(passed=True,tokens=normal_ids,omega_zero_matches_hf_generate=identity_checks))
        test=[json.loads(l) for l in (ROOT/'data/math_test.jsonl').read_text(encoding='utf-8').splitlines()]
        demonstrations=[json.loads(l) for l in (ROOT/'data/math_fewshot.jsonl').read_text(encoding='utf-8').splitlines()]
        shots='\n\n'.join('Question: '+r['question']+'\nAnswer: '+r['answer'] for r in demonstrations)
        cases=[('baseline',0),('logits',float(selected['logits'].split('_')[1])),('hidden',float(selected['hidden'].split('_')[1])),('logits_fixed_025',0.25)]
        dest=ROOT/'trajectories/math_generation.jsonl'
        records=[json.loads(l) for l in dest.read_text(encoding='utf-8').splitlines()] if dest.exists() else []
        done={(r['id'],r['method']):r for r in records}
        for row in test:
            prompt=shots+'\n\nQuestion: '+row['question']+'\nAnswer:'
            for method,omega in cases:
                if (row['id'],method) in done:continue
                budget.check()
                if omega==0 and method!='baseline':
                    result={**done[(row['id'],'baseline')],'method':method,'omega':0,'identity_reused':True,'elapsed_s':0}
                else:
                    result=dict(**row,method=method,omega=omega,protocol='gsm8k_3shot_greedy_256_per_token_v1',**decode(tok,m,prompt,method,omega,budget))
                    result['correct']=result['prediction']==row['gold'].replace(',','')
                with dest.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                done[(row['id'],method)]=result
                print(json.dumps(dict(id=row['id'],method=method,correct=result['correct'],elapsed_s=result['elapsed_s'])),flush=True)
                save('STATUS.json',dict(stage='math_generation',completed=len(done),total=len(cases)*len(test)))
        report={method:dict(n=len(test),exact_match=sum(done[(r['id'],method)]['correct'] for r in test)/len(test),
                           truncated=sum(done[(r['id'],method)]['reached_token_limit'] for r in test),
                           total_seconds=sum(done[(r['id'],method)]['elapsed_s'] for r in test)) for method,_ in cases}
        save('results/math_generation_metrics.json',report)
    finally:budget.close('math_generation')

if __name__=='__main__':main()
