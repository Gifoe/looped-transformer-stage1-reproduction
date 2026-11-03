"""R2 original-weight full-vocabulary readouts; cached TRAIN or one locked test."""
import argparse,hashlib,json,time,sys,math
from pathlib import Path
sys.dont_write_bytecode=True
from slcd_core import ROOT,REVISION,digest,read,rows,sha,save
from slcd_token_worker_v2 import MODEL_SHA,file_sha,inside,parity_errors,index_rows

SIZES={'discovery':384,'selection':192,'confirmation':517,'test':1172}
SOURCES=['slcd_r3_worker.py','slcd_r2_logit_methods.py','slcd_logit_methods.py','slcd_core.py','runtime.py','ouro_cached_scoring.py','weight_edits.py','slcd_token_worker_v2.py']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=SIZES,required=True);ap.add_argument('--job',required=True);ap.add_argument('--recipes',required=True);a=ap.parse_args();started=time.time()
    assert a.job.replace('_','').isalnum()
    manifest=f'data/slcd_{a.split}.jsonl';data_sha=sha(manifest);recipe_sha=sha(a.recipes);implementation={n:sha('scripts/'+n) for n in SOURCES}
    lock_sha=None
    if a.split=='test':
        lock=read('configs/slcd_R3_FINAL_LOCK.json');lock_sha=sha('configs/slcd_R3_FINAL_LOCK.json')
        assert lock['test_access_authorized'] and lock['dataset_sha256']['test']==data_sha and recipe_sha in lock['allowed_recipe_sha256']
        assert implementation==lock['inference_source_sha256']
    else:
        lock=read('configs/slcd_R2_DEVELOPMENT_LOCK.json')
        assert lock['train_data_sha256'][a.split]==data_sha and recipe_sha==lock['recipes_sha256'] and implementation==lock['inference_source_sha256']
        lock_sha=sha('configs/slcd_R2_DEVELOPMENT_LOCK.json')
    data=rows(manifest);expected=index_rows(data,SIZES[a.split],'R2 input');assert all(q.startswith('arc_challenge/'+('test' if a.split=='test' else 'train')+'/') for q in expected)
    recipes=read(a.recipes)['recipes'];assert len(recipes)==len({r['id'] for r in recipes}) and len(recipes)<=8
    assert {'tf_fixed','tf_adaptive'}<={r['id'] for r in recipes}
    reference_path=f'results/slcd_original_{a.split}.jsonl'
    if (ROOT/reference_path).exists():references=index_rows(rows(reference_path),len(data),'R2 independent baseline');reference_sha=sha(reference_path);reference_kind='independent_original_scores'
    else:
        assert a.split=='confirmation'
        reference_path='results/slcd_tokens_confirmation_confirmation.jsonl';receipt=read(reference_path.replace('.jsonl','_COMPLETE.json'));assert sha(reference_path)==receipt['score_file_sha256']
        references=index_rows(rows(reference_path),len(data),'R2 prior original baseline');reference_sha=sha(reference_path);reference_kind='prior_same_forward_original_scores_not_independent'
        references={q:dict(**{k:v for k,v in r.items() if k!='methods'},methods=r['original_methods']) for q,r in references.items()}
    assert set(references)==set(expected)
    if a.split!='test':assert lock['reference_sha256'][a.split]==reference_sha
    authority={};authority_sha=None
    if a.split!='test':
        authority_path='results/slcd_SOURCE_AUDIT.json' if a.split in ['discovery','selection'] else 'results/slcd_tokens_confirmation_confirmation_COMPLETE.json'
        authority_sha=sha(authority_path);receipt=read(authority_path)
        if a.split in ['discovery','selection']:
            assert receipt['status']=='passed' and receipt['split_audits'][a.split]['data_sha256']==data_sha
            assert receipt['split_audits'][a.split]['reference_sha256']==reference_sha
        else:assert receipt['status']=='complete' and receipt['provenance']['data_sha256']==data_sha and receipt['provenance']['model_identity']['safetensors_sha256']==MODEL_SHA
        assert authority_sha==lock['authority_sha256'][a.split]
        authority={v['id']:v for v in receipt['states'] if v.get('split',a.split)==a.split}
        assert set(authority)==set(expected)
    model_dir=ROOT/'models/Ouro-1.4B'
    assert file_sha(model_dir/'model.safetensors')==MODEL_SHA and read('models/Ouro-1.4B/snapshot_metadata.json')['sha']==REVISION
    metadata={p.name:file_sha(p) for p in sorted(model_dir.iterdir()) if p.is_file() and p.name!='model.safetensors' and (p.suffix in ('.json','.py','.txt','.model') or 'tokenizer' in p.name)}
    model_identity=dict(revision=REVISION,safetensors_sha256=MODEL_SHA,metadata_and_tokenizer_sha256=metadata)
    assert model_identity==lock['model_identity']
    provenance=dict(data_sha256=data_sha,recipes_sha256=recipe_sha,inference_source_sha256=implementation,model_revision=REVISION,model_sha256=MODEL_SHA,
        reference_sha256=reference_sha,reference_kind=reference_kind,state_authority_sha256=authority_sha,model_identity=model_identity,authorization_lock_sha256=lock_sha,final_lock_sha256=lock_sha if a.split=='test' else None)
    signature=digest(provenance);output=f'results/slcd_r3_{a.split}_{a.job}.jsonl';completion=output.replace('.jsonl','_COMPLETE.json')
    recipe_ids={r['id'] for r in recipes}
    def validate_saved(r):
        qid=r['id'];assert qid in expected and r['signature']==signature
        row=expected[qid];ref=references[qid]
        assert r['gold']==row['gold'] and r['prompt_sha256']==row['prompt_sha256'] and r['choice_char_lengths']==row['choice_char_lengths'] and r['choice_token_counts']==ref['choice_token_counts']
        assert set(r['methods'])==recipe_ids
        errors=parity_errors(r['original_methods'],ref,len(row['choice_char_lengths']))
        assert r['baseline_parity_errors']==errors and len(errors)==8
        for v in list(r['methods'].values())+list(r['original_methods'].values()):
            sums=v['sum_scores'];assert len(sums)==len(row['choice_char_lengths']) and all(math.isfinite(x) for x in sums)
            normalized=[s/max(n,1) for s,n in zip(sums,row['choice_char_lengths'])];pred=max(range(len(sums)),key=lambda i:normalized[i])
            assert normalized==v['normalized_scores'] and pred==v['prediction'] and v['correct']==(pred==row['gold'])
    if (ROOT/completion).exists():
        done=read(completion);saved=index_rows(rows(output),len(data),'R2 completed scores')
        assert set(saved)==set(expected) and done['signature']==signature and sha(output)==done['score_file_sha256'] and done['n']==len(data)
        for r in saved.values():validate_saved(r)
        assert done['independent_parity_coverage']==(0 if a.split=='confirmation' else len(data));print('R2_ALREADY_COMPLETE',a.split);return
    done={}
    if (ROOT/output).exists():
        values=rows(output);done={v['id']:v for v in values};assert len(done)==len(values) and set(done)<=set(expected)
        for r in done.values():validate_saved(r)
    from runtime import configure,load_model,encode_pair,outcome,check_budget
    configure()
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from weight_edits import tensor_sha256
    from ouro_cached_scoring import OuroCachedScorer
    from slcd_r2_logit_methods import score_logit_recipes
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
    if a.split=='test':tok,model=load_model();scorer=OuroCachedScorer(model,6);head=model.lm_head.weight
    else:
        tok=AutoTokenizer.from_pretrained(ROOT/'models/Ouro-1.4B',local_files_only=True)
        with safe_open(ROOT/'models/Ouro-1.4B/model.safetensors',framework='pt',device='cpu') as st:head=st.get_tensor('lm_head.weight').to('cuda')
    head_sha=tensor_sha256(head);assert head_sha==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    for row in data:
        if row['id'] in done:continue
        check_budget(120);begin=time.time();qid=row['id'];ref=references[qid]
        assert row['prompt_sha256']==hashlib.sha256(row['prompt'].encode()).hexdigest()==ref['prompt_sha256'] and row['gold']==ref['gold'] and row['choice_char_lengths']==ref['choice_char_lengths']
        assert all(p[0]==row['prompt'] for p in row['request_pairs'])
        pairs=[encode_pair(tok,*p) for p in row['request_pairs']];assert [len(t) for _,t in pairs]==ref['choice_token_counts']
        if a.split=='test':
            fresh=scorer.score_uncached(pairs,return_states=True)
            states=[dict(candidate=i,target_ids=t,normalized=v['normalized_states']) for i,((_,t),v) in enumerate(zip(pairs,fresh))]
            state_info=None
        else:
            entry=authority[qid];p=inside(entry['file']);assert file_sha(p)==entry['sha256']
            payload=torch.load(p,map_location='cpu',weights_only=True);assert payload['id']==qid
            states=payload['candidate_states'];state_info=dict(file=entry['file'],sha256=entry['sha256'])
        assert len(states)==len(pairs)
        totals={r['id']:[] for r in recipes};native={k:[] for k in range(1,7)}
        for i,(state,(_,target)) in enumerate(zip(states,pairs)):
            assert state['candidate']==i and state['target_ids']==target and len(state['normalized'])==6
            logits={};t=torch.tensor(target,device='cuda',dtype=torch.long)
            for k,h in enumerate(state['normalized'],1):
                assert h.dtype==torch.bfloat16 and list(h.shape)==[1,len(target),2048]
                z=F.linear(h.to('cuda'),head)[0].float();logits[k]=z
                native[k].append(float(z.log_softmax(-1).gather(-1,t[:,None]).sum()))
            scores=score_logit_recipes(logits,target,recipes)
            for name,value in scores.items():totals[name].append(value)
            del logits
        methods={name:outcome(sums,row['choice_char_lengths'],row['gold']) for name,sums in totals.items()}
        originals={f'native_k{k}':outcome(sums,row['choice_char_lengths'],row['gold']) for k,sums in native.items()}
        originals.update(loopcd_k4_fixed=methods['tf_fixed'],loopcd_k4_adaptive=methods['tf_adaptive'])
        errors=parity_errors(originals,ref,len(pairs));assert set(errors)=={f'native_k{k}' for k in range(1,7)}|{'loopcd_k4_fixed','loopcd_k4_adaptive'}
        r=dict(id=qid,gold=row['gold'],prompt_sha256=row['prompt_sha256'],choice_char_lengths=row['choice_char_lengths'],choice_token_counts=[len(t) for _,t in pairs],methods=methods,original_methods=originals,
            signature=signature,baseline_parity_errors=errors,baseline_reference_kind=reference_kind,state_info=state_info,weights_modified=False,elapsed_seconds=time.time()-begin)
        with (ROOT/output).open('a',encoding='utf8') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
        done[qid]=r
        if len(done)%100==0 or len(done)==len(data):
            progress=dict(split=a.split,job=a.job,completed=len(done),target=len(data),elapsed_seconds=time.time()-started);save('results/slcd_R3_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    assert set(done)==set(expected) and tensor_sha256(head)==head_sha
    assert sha(manifest)==data_sha and sha(a.recipes)==recipe_sha and sha(reference_path)==reference_sha
    for r in done.values():validate_saved(r)
    save(completion,dict(status='complete',split=a.split,n=len(data),signature=signature,provenance=provenance,score_file=output,score_file_sha256=sha(output),baseline_parity_coverage=len(done),independent_parity_coverage=0 if a.split=='confirmation' else len(done),independent_parity_status='not_available' if a.split=='confirmation' else 'exact',baseline_reference_kind=reference_kind,max_baseline_error=0.,model_weights_modified=False,elapsed_seconds=time.time()-started,actual_shared_depth=6))
    print('R2_COMPLETE',a.split,flush=True)

if __name__=='__main__':main()
