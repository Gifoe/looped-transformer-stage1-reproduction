"""Five fixed configurations with independent paper-recipe implementation."""
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from qwen_recipe import Recipe
from ouro_audit import ROOT,CFG,Budget,save,prompt_for,encode_pair,distribution,discrepancy

def load():
    p=ROOT/'models/Qwen3-4B-Base'
    tok=AutoTokenizer.from_pretrained(p,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    assert len(model.model.layers)==36
    return tok,model,Recipe(model)

def parity(tok,m,recipe):
    reports=[]
    for text in ['Question: What is 2 + 2?\nAnswer:', 'The Earth orbits the Sun.']:
        x=tok(text,return_tensors='pt').to('cuda')
        for cache in [False,True]:
            recipe.restore();original=m(**x,use_cache=cache,logits_to_keep=1)
            cases=[dict(strategy='naive',K=1),dict(strategy='euler',K=1),dict(strategy='block_anchored',K=1,anchor_beta=.5),dict(strategy='block_anchored',K=6,anchor_beta=1)]
            for cfg in cases:
                recipe.set(cfg);patched=m(**x,use_cache=cache,logits_to_keep=1)
                diff=discrepancy(original.logits,patched.logits)
                print(json.dumps(dict(event='parity_case',config=cfg,use_cache=cache,error=diff)),flush=True)
                assert diff['max_abs']<=.125 and diff['relative_l2']<=.01 and diff['argmax_agreement']==1,diff
                if cache:
                    lengths=[patched.past_key_values.get_seq_length(i) for i in range(36)]
                    assert lengths==[x.input_ids.shape[1]]*36,lengths
                    token=patched.logits[:,-1,:].argmax(-1,keepdim=True)
                    step=m(input_ids=token,past_key_values=patched.past_key_values,use_cache=True,logits_to_keep=1)
                    recipe.restore()
                    normal_prefill=m(**x,use_cache=True,logits_to_keep=1)
                    normal_step=m(input_ids=token,past_key_values=normal_prefill.past_key_values,use_cache=True,logits_to_keep=1)
                    cdiff=discrepancy(normal_step.logits,step.logits)
                    assert cdiff['relative_l2']<=.01 and cdiff['argmax_agreement']==1,cdiff
                    assert [step.past_key_values.get_seq_length(i) for i in range(36)]==[x.input_ids.shape[1]+1]*36
                else:cdiff=None
                reports.append(dict(config=cfg,use_cache=cache,logit_error=diff,incremental_error=cdiff))
    recipe.restore()
    # Nontrivial K=6 first-cache must append exactly once in prefill and decode.
    recipe.set(CFG['qwen_methods'][-1]);x=tok('Question: What is the capital of France?\nAnswer:',return_tensors='pt').to('cuda')
    y=m(**x,use_cache=True,logits_to_keep=1)
    assert [y.past_key_values.get_seq_length(i) for i in range(36)]==[x.input_ids.shape[1]]*36
    step=m(input_ids=y.logits[:,-1,:].argmax(-1,keepdim=True),past_key_values=y.past_key_values,use_cache=True,logits_to_keep=1)
    assert [step.past_key_values.get_seq_length(i) for i in range(36)]==[x.input_ids.shape[1]+1]*36
    recipe.restore()
    report=dict(passed=True,implementation='independent paper/author-README recipe; author source unavailable (404)',cases=reports,nontrivial_cache_lengths_passed=True,
                config=m.config.to_dict(),model_revision=json.loads((ROOT/'models/Qwen3-4B-Base/snapshot_metadata.json').read_text())['sha'])
    save('results/qwen_parity.json',report)

def score(tok,m,recipe,row):
    prompt,answers=prompt_for(row);values=[];lengths=[];chars=[];stats=None;logp=None;window=None
    torch.cuda.synchronize();start=time.monotonic()
    for option,answer in enumerate(answers):
        prefix,continuation=encode_pair(tok,prompt,answer)
        ids=torch.tensor([prefix+continuation[:-1]],device='cuda');recipe.position=len(prefix)-1
        selection=torch.arange(len(prefix)-1,ids.shape[1],device='cuda')
        output=m(input_ids=ids,use_cache=False,logits_to_keep=selection)
        z=output.logits[0].float();target=torch.tensor(continuation,device='cuda')
        values.append(z.log_softmax(-1).gather(-1,target[:,None]).sum().item());lengths.append(len(continuation));chars.append(len(answer.lstrip(' ')))
        if option==0:
            stats,logp=distribution(z[0]);window=recipe.record if recipe.config is not None else None
    scores=np.asarray(values)/np.maximum(chars,1);p=np.exp(scores-scores.max());p/=p.sum();pred=int(scores.argmax())
    torch.cuda.synchronize()
    return dict(**row,protocol='zero_shot_complementary_audit_v1',implementation='independent_published_recipe_v1',
                prediction=pred,correct=pred==row['gold'],scores=scores.tolist(),sum_loglikelihood=values,choice_probabilities=p.tolist(),
                gold_choice_logp=float(np.log(p[row['gold']])),gold_nll_per_token=-values[row['gold']]/lengths[row['gold']],
                full_vocab_stats=stats,window=window,prompt_tokens=len(prefix),elapsed_s=time.monotonic()-start,
                peak_memory_bytes=torch.cuda.max_memory_allocated()),logp

def main():
    budget=Budget()
    try:
        tok,m,recipe=load()
        parity_path=ROOT/'results/qwen_parity.json'
        if not parity_path.exists() or not json.loads(parity_path.read_text())['passed']:
            parity(tok,m,recipe)
        validation=[json.loads(l) for l in (ROOT/'data/validation.jsonl').read_text(encoding='utf-8').splitlines()]
        test=[json.loads(l) for l in (ROOT/'data/qwen_test.jsonl').read_text(encoding='utf-8').splitlines()]
        baseline={};timings=[]
        for split,rows in [('pilot',validation[:50]),('test',test)]:
            for cfg in CFG['qwen_methods']:
                recipe.set(cfg);torch.cuda.reset_peak_memory_stats()
                dest=ROOT/'results'/('qwen_'+cfg['id']+'_'+split+'.jsonl')
                done={json.loads(l)['id'] for l in dest.read_text(encoding='utf-8').splitlines()} if dest.exists() else set()
                with dest.open('a',encoding='utf-8') as f:
                    for row in rows:
                        if row['id'] in done:continue
                        budget.check();result,logp=score(tok,m,recipe,row)
                        result['configuration']=cfg
                        if cfg['id']=='A0':baseline[row['id']]=logp.detach().cpu()
                        elif row['id'] in baseline:
                            base=baseline[row['id']].to('cuda')
                            result['full_vocab_kl_vs_original']=(logp.exp()*(logp-base)).sum().item()
                        f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();done.add(row['id'])
                        if len(done)%10==0 or len(done)==len(rows):
                            save('logs/qwen_progress.json',dict(configuration=cfg['id'],split=split,completed=len(done),total=len(rows)))
                        if len(done)%25==0:print(json.dumps(dict(method=cfg['id'],split=split,done=len(done))),flush=True)
                        if split=='pilot':timings.append(result['elapsed_s'])
                        if split=='pilot' and len(done)==20:
                            estimate=float(np.mean(timings))*650*5/3600
                            assert estimate<40,'Qwen budget estimate too large; reduce protocol before continuing'
                            save('results/qwen_performance_preflight.json',dict(seconds_per_sample=float(np.mean(timings)),estimated_total_gpu_hours=estimate,peak_memory_bytes=torch.cuda.max_memory_allocated(),passed=True))
        recipe.restore()
    finally:budget.close('qwen_independent_recipe')

if __name__=='__main__':main()
