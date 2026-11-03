"""Tiny TRAIN-only diagnostic: exact legacy model call vs explicit metadata/scorer.

Does not edit the scorer or model. Run under supervise.py; writes engineering
evidence only. No validation/test files. This is NOT an accuracy improvement trial.
"""
import argparse
import hashlib
import inspect
import json
import time

from runtime import ROOT, load_model, encode_pair, outcome, save, check_budget
from ouro_cached_scoring import OuroCachedScorer


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=3)
    ap.add_argument('--source-study',default=None,
                    help='Optional engineering-only low-margin sampling; default is first n fixed discovery rows')
    ap.add_argument('--output',default='original_call_parity')
    a=ap.parse_args()
    assert 1<=a.n<=8 and a.output.replace('_','').isalnum()
    assert a.source_study is None or a.source_study.replace('_','').isalnum()
    rows=[json.loads(line) for line in (ROOT/'data/discovery.jsonl').read_text().splitlines()]
    by_id={r['id']:r for r in rows}
    ids=[r['id'] for r in rows[:a.n]]
    source_hash=None
    if a.source_study is not None:
        source_path=ROOT/'results'/a.source_study/'scores.jsonl'
        prior=[json.loads(line) for line in source_path.read_text().splitlines()]
        assert all(r['scope']=='official_train_discovery_only' for r in prior)
        ranking=sorted(prior,key=lambda r:(min(abs(r['methods'][f'native_k{k}']['margin']) for k in (4,5)),r['id']))
        ids=[rows[0]['id']]
        for r in ranking:
            if r['id'] not in ids:ids.append(r['id'])
            if len(ids)>=a.n:break
        ids=ids[:a.n];source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert len(ids)==a.n and all('/train/' in uid for uid in ids)
    out=ROOT/'results'/a.output
    out.mkdir(exist_ok=False)  # preserve every previous diagnostic without overwrite
    save(str((out/'INPUTS.json').relative_to(ROOT)),dict(ids=ids,
        selection='first n fixed discovery rows' if a.source_study is None else 'engineering-only first case plus low-margin discovery rows',
        source_study=a.source_study,source_sha256=source_hash))
    tok,model=load_model();scorer=OuroCachedScorer(model,6)
    import torch,transformers
    import transformers.masking_utils as masking
    import transformers.integrations.sdpa_attention as sdpa
    sources={name:dict(file=inspect.getfile(module),sha256=hashlib.sha256(inspect.getsource(module).encode()).hexdigest())
             for name,module in [('masking_utils',masking),('sdpa_attention',sdpa)]}
    save(str((out/'ENVIRONMENT.json').relative_to(ROOT)),dict(torch=torch.__version__,transformers=transformers.__version__,
        sources=sources,rope_scaling=model.config.rope_scaling,attention_implementation=model.config._attn_implementation,
        total_ut_steps=model.model.total_ut_steps))
    @torch.no_grad()
    def score_direct(prefix,target,metadata=False):
        x=torch.tensor([prefix+target[:-1]],device='cuda')
        kwargs=dict(input_ids=x,use_cache=False)
        if metadata:
            kwargs.update(past_key_values=None,attention_mask=torch.ones_like(x),
                cache_position=torch.arange(x.shape[1],device=x.device),
                position_ids=torch.arange(x.shape[1],device=x.device).unsqueeze(0))
        _,states,_=model.model(**kwargs)
        t=torch.tensor(target,device='cuda');selection=slice(len(prefix)-1,None)
        sums,tokens,selected={},{},[]
        for k,state in enumerate(states,1):
            z=model.lm_head(state[:,selection,:])[0].float()
            lp=z.log_softmax(-1).gather(-1,t[:,None]).squeeze(-1)
            sums[k]=float(lp.sum());tokens[k]=lp.cpu().tolist()
            selected.append(state[:,selection,:].detach().cpu())
        return dict(sum_loglikelihood=sums,token_loglikelihood=tokens,normalized_states=selected)
    def with_mask_capture(fn):
        seen=[]
        def pre(module,args,kwargs):
            if seen:return
            mask=kwargs.get('attention_mask');pos=kwargs.get('position_ids')
            seen.append(dict(mask_none=mask is None,mask_shape=None if mask is None else list(mask.shape),
                mask_dtype=None if mask is None else str(mask.dtype),
                position_shape=None if pos is None else list(pos.shape),
                position_first=None if pos is None else int(pos[0,0]),position_last=None if pos is None else int(pos[0,-1])))
        handle=model.model.layers[0].self_attn.register_forward_pre_hook(pre,with_kwargs=True)
        try:
            scorer.invalidate()
            result=fn()
        finally:
            handle.remove();scorer.invalidate()
        return result,seen
    cases=[];depth_checks=[];started=time.time()
    for uid in ids:
        row=by_id[uid];pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
        raw={name:[] for name in ['original','explicit_metadata','current_scorer']}
        differences=[];masks=[];repeat_errors=[]
        for j,(prefix,target) in enumerate(pairs):
            check_budget(90)
            base,bmask=with_mask_capture(lambda:score_direct(prefix,target,False))
            explicit,emask=with_mask_capture(lambda:score_direct(prefix,target,True))
            current,cmask=with_mask_capture(lambda:scorer.score_uncached([(prefix,target)],return_states=True,loopcd=False)[0])
            raw['original'].append(base);raw['explicit_metadata'].append(explicit);raw['current_scorer'].append(current)
            masks.append(dict(candidate=j,original=bmask,explicit_metadata=emask,current_scorer=cmask))
            for name,other in [('explicit_metadata',explicit),('current_scorer',current)]:
                for k in range(1,7):
                    errors=[abs(x-y) for x,y in zip(base['token_loglikelihood'][k],other['token_loglikelihood'][k])]
                    dsum=other['sum_loglikelihood'][k]-base['sum_loglikelihood'][k]
                    differences.append(dict(candidate=j,k=k,path=name,max_token_lp_abs_error=max(errors),
                        sum_delta=dsum,char_normalized_delta=dsum/max(row['choice_char_lengths'][j],1),
                        hidden_max_abs_error=float((base['normalized_states'][k-1].float()-other['normalized_states'][k-1].float()).abs().max())))
            if j==0:
                repeated,_=with_mask_capture(lambda:score_direct(prefix,target,False))
                repeat_errors=[abs(repeated['sum_loglikelihood'][k]-base['sum_loglikelihood'][k]) for k in range(1,7)]
        outcomes={name:{f'k{k}':outcome([r['sum_loglikelihood'][k] for r in values],row['choice_char_lengths'],row['gold'])
                        for k in range(1,7)} for name,values in raw.items()}
        case=dict(id=uid,scope='official_train_discovery_only',masks=masks,differences=differences,
            outcomes=outcomes,original_repeat_sum_errors=repeat_errors,
            exact_original_vs_scorer=all(d['sum_delta']==0 and d['max_token_lp_abs_error']==0 and d['hidden_max_abs_error']==0
                                        for d in differences if d['path']=='current_scorer'),
            prediction_flips=[k for k in range(1,7) if outcomes['original'][f'k{k}']['prediction']!=outcomes['current_scorer'][f'k{k}']['prediction']])
        check_budget(90)
        depth_checks.append(dict(id=uid,**scorer.check_depth_invariance(pairs,depths=(3,4,5),atol=0.0)))
        cases.append(case);save(str((out/'PROGRESS.json').relative_to(ROOT)),dict(completed=len(cases),target=len(ids),cases=cases,depth_checks=depth_checks))
    result=dict(status='complete',cases=cases,elapsed_seconds=time.time()-started,
        exact_original_path_parity=all(c['exact_original_vs_scorer'] for c in cases),
        original_repeat_exact=all(not any(c['original_repeat_sum_errors']) for c in cases),
        any_prediction_flip=any(c['prediction_flips'] for c in cases),depth_checks=depth_checks,
        exact_intermediate_depth_parity=all(c['passed'] for c in depth_checks),
        instruction='No automatic acceptance tolerance. Different masks/backend disprove identical execution path even if predictions agree. Preserve this engineering audit.')
    save(str((out/'COMPLETE.json').relative_to(ROOT)),result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2),flush=True)
    assert result['exact_original_path_parity'] and result['original_repeat_exact'] and result['exact_intermediate_depth_parity'], 'Strict zero-error original-call/depth readiness failed; no formal studies admitted'


if __name__=='__main__':main()
