"""Read-only original-model trajectories -> full-candidate token-logit recipes.

Existing discovery states are reused; newly collected states use flat slcd_
files in existing results/. Never load edited-weight bundles or mutate weights.
"""
import argparse,json,time,hashlib
from slcd_core import ROOT,REVISION,save,rows,read,sha,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['discovery','selection','validation','test'],required=True)
    ap.add_argument('--job',required=True);ap.add_argument('--recipes',default='configs/slcd_TOKEN_INITIAL.json')
    ap.add_argument('--reuse-states',action='store_true');a=ap.parse_args()
    assert a.job.replace('_','').isalnum()
    if a.split=='validation':assert (ROOT/'configs/slcd_VALIDATION_LOCK.json').exists()
    if a.split=='test':assert (ROOT/'FINAL_METHOD_LOCK.json').exists()
    manifest='data/slcd_'+a.split+'.jsonl';data=rows(manifest)
    assert len(data)=={'discovery':384,'selection':192,'validation':299,'test':1172}[a.split]
    recipes=read(a.recipes)['recipes'];assert 1<=len(recipes)<=64
    implementation={f:sha('scripts/'+f) for f in ['slcd_token_worker.py','slcd_logit_methods.py','runtime.py','ouro_cached_scoring.py']}
    provenance=dict(model_revision=REVISION,data_sha256=sha(manifest),recipes_sha256=sha(a.recipes),implementation=implementation)
    signature=digest(provenance);output='results/slcd_tokens_'+a.split+'_'+a.job+'.jsonl';path=ROOT/output
    existing=rows(output) if path.exists() else [];done={}
    for r in existing:
        assert r['implementation_signature']==signature and r['id'] not in done;done[r['id']]=r
    reference_path='results/slcd_original_'+a.split+'.jsonl'
    reference={r['id']:r for r in rows(reference_path)} if (ROOT/reference_path).exists() else {}
    from runtime import configure,load_model,encode_pair,check_budget,outcome
    configure()
    import torch
    import torch.nn.functional as F
    from slcd_logit_methods import score_logit_recipes
    from ouro_cached_scoring import OuroCachedScorer
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.manual_seed(2021)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # A cached-state job needs only the checkpoint's unchanged vocabulary head.
    model=scorer=tok=None
    if a.reuse_states:
        from safetensors import safe_open
        with safe_open(ROOT/'models/Ouro-1.4B/model.safetensors',framework='pt',device='cpu') as st:
            head=st.get_tensor('lm_head.weight').to('cuda')
        assert head.dtype==torch.bfloat16 and list(head.shape)==[49152,2048]
    else:
        tok,model=load_model();scorer=OuroCachedScorer(model,6);head=model.lm_head.weight
        assert not any(p.requires_grad for p in model.parameters())
    from weight_edits import tensor_sha256
    head_sha=tensor_sha256(head);started=time.time();maximum_error=0.;state_index=[]
    for row in data:
        if row['id'] in done:continue
        check_budget(120);begin=time.time();stem=row['id'].rsplit('/',1)[-1]
        state_path=ROOT/('results/slcd_state_'+a.split+'_'+stem+'.pt')
        if a.split=='discovery' and a.reuse_states:
            source=reference[row['id']];state_path=ROOT/'results/training_statistics'/source['statistics_file']
        if state_path.exists():
            payload=torch.load(state_path,map_location='cpu',weights_only=True)
            assert payload['id']==row['id'];states=payload['candidate_states']
            if state_path.name.startswith('slcd_'):
                assert payload['data_sha256']==provenance['data_sha256'] and payload['model_revision']==REVISION
                assert payload['prompt_sha256']==row['prompt_sha256'] and payload['original_weights'] is True
            else:
                assert a.split=='discovery' and payload['implementation_sha256']==reference[row['id']]['implementation_sha256']
        else:
            assert not a.reuse_states,'Missing saved states: '+str(state_path)
            pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
            values=scorer.score_uncached(pairs,return_states=True)
            states=[dict(candidate=i,target_ids=t,normalized=v['normalized_states']) for i,((_,t),v) in enumerate(zip(pairs,values))]
            torch.save(dict(id=row['id'],candidate_states=states,data_sha256=provenance['data_sha256'],
                model_revision=REVISION,prompt_sha256=row['prompt_sha256'],original_weights=True),state_path)
        assert len(states)==len(row['choice_char_lengths'])
        totals={r['id']:[] for r in recipes};native={k:[] for k in range(1,7)};counts=[]
        for i,state in enumerate(states):
            assert state['candidate']==i and len(state['normalized'])==6
            target=state['target_ids'];counts.append(len(target));logits={}
            for k,h in enumerate(state['normalized'],1):
                assert h.dtype==torch.bfloat16 and list(h.shape)==[1,len(target),2048]
                z=F.linear(h.to('cuda'),head)[0].float();logits[k]=z
                t=torch.as_tensor(target,dtype=torch.long,device='cuda')
                native[k].append(float(z.log_softmax(-1).gather(-1,t[:,None]).sum()))
            sums=score_logit_recipes(logits,target,recipes)
            for key,value in sums.items():totals[key].append(value)
            del logits
        methods={key:outcome(value,row['choice_char_lengths'],row['gold']) for key,value in totals.items()}
        originals={f'native_k{k}':outcome(v,row['choice_char_lengths'],row['gold']) for k,v in native.items()}
        if 'tf_fixed' in methods:originals['loopcd_k4_fixed']=methods['tf_fixed']
        if 'tf_adaptive' in methods:originals['loopcd_k4_adaptive']=methods['tf_adaptive']
        errors={}
        if row['id'] in reference:
            ref=reference[row['id']]
            assert ref['gold']==row['gold'] and ref['prompt_sha256']==row['prompt_sha256']
            assert ref['choice_char_lengths']==row['choice_char_lengths'] and ref['choice_token_counts']==counts
            for key,value in originals.items():
                e=max(abs(x-y) for x,y in zip(value['sum_scores'],ref['methods'][key]['sum_scores']));errors[key]=e
            assert not errors or max(errors.values())==0.,('Saved-state/native head parity',row['id'],errors)
            maximum_error=max(maximum_error,max(errors.values(),default=0.))
        state_index.append(dict(id=row['id'],file=state_path.relative_to(ROOT).as_posix(),bytes=state_path.stat().st_size))
        result=dict(id=row['id'],gold=row['gold'],scope=a.split,prompt_sha256=row['prompt_sha256'],
            choice_char_lengths=row['choice_char_lengths'],choice_token_counts=counts,methods=methods,
            original_methods=originals,implementation_signature=signature,data_sha256=provenance['data_sha256'],
            native_parity_errors=errors,original_head_sha256=head_sha,weights_modified=False,
            state_file=state_path.relative_to(ROOT).as_posix(),elapsed_seconds=time.time()-begin)
        with path.open('a',encoding='utf8') as f:f.write(json.dumps(result)+'\n')
        done[row['id']]=result
        if len(done)%16==0 or len(done)==len(data):
            progress=dict(split=a.split,job=a.job,completed=len(done),target=len(data),elapsed_seconds=time.time()-started)
            save('results/slcd_TOKEN_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    assert len(done)==len(data) and tensor_sha256(head)==head_sha
    save('results/slcd_tokens_'+a.split+'_'+a.job+'_COMPLETE.json',dict(status='complete',n=len(data),
        provenance=provenance,implementation_signature=signature,score_file=output,score_file_sha256=sha(output),
        max_baseline_sum_score_error=maximum_error,all_baseline_parity_exact=True,
        original_head_sha256=head_sha,model_weights_modified=False,elapsed_seconds=time.time()-started,
        states=state_index,required_logical_depth=max(max(r.get('final',4),r.get('late',4) if r.get('gamma',0)>0 else 4) for r in recipes)))
    print('SLCD_TOKEN_JOB_COMPLETE',a.split,a.job,flush=True)


if __name__=='__main__':main()
