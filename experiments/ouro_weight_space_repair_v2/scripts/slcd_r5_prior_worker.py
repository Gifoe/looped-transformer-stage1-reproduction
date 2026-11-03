"""Label-free Answer: prior likelihoods; one GPU ledger, exact input memoization."""
import argparse,hashlib,json,time,sys,math
sys.dont_write_bytecode=True
from slcd_core import ROOT,read,rows,sha,digest,save
from slcd_token_worker_v2 import MODEL_SHA,file_sha

SOURCES=['slcd_r5_prior_worker.py','slcd_core.py','runtime.py','ouro_cached_scoring.py','slcd_token_worker_v2.py','weight_edits.py']
PREFIX='Answer:'
BASE={f'native_k{k}' for k in range(1,7)}|{'loopcd_k4_fixed','loopcd_k4_adaptive'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','test'],required=True);a=ap.parse_args();started=time.time()
    lp='configs/slcd_R5_DEVELOPMENT_LOCK.json' if a.split=='train' else 'configs/slcd_R5_FINAL_LOCK.json'
    lock=read(lp);implementation={n:sha('scripts/'+n) for n in SOURCES};assert implementation==lock['inference_source_sha256'] and lock['prior_prefix']==PREFIX
    if a.split=='test':assert lock['test_access_authorized']
    splits=['discovery','selection','confirmation'] if a.split=='train' else ['test']
    source_sha={s:sha('data/slcd_'+s+'.jsonl') for s in splits};assert source_sha==lock['data_sha256']
    data=[r for s in splits for r in rows('data/slcd_'+s+'.jsonl')];expected={r['id']:r for r in data};n=1093 if a.split=='train' else 1172;assert len(expected)==len(data)==n
    model_dir=ROOT/'models/Ouro-1.4B';assert file_sha(model_dir/'model.safetensors')==MODEL_SHA
    metadata={p.name:file_sha(p) for p in sorted(model_dir.iterdir()) if p.is_file() and p.name!='model.safetensors' and (p.suffix in ('.json','.py','.txt','.model') or 'tokenizer' in p.name)}
    identity=lock['model_identity'];assert identity['safetensors_sha256']==MODEL_SHA and identity['metadata_and_tokenizer_sha256']==metadata
    global_signature=digest(dict(model_identity=identity,implementation=implementation,prefix=PREFIX,seed=2021,scoring='full target LL raw sums; original BF16 and FP32 logsoftmax'))
    provenance=dict(model_identity=identity,inference_source_sha256=implementation,prior_prefix=PREFIX,data_sha256=source_sha,authorization_lock_sha256=sha(lp),global_cache_signature=global_signature)
    signature=digest(provenance);output=f'results/slcd_r5_priors_{a.split}.jsonl';completion=output.replace('.jsonl','_COMPLETE.json')
    from transformers import AutoTokenizer
    from runtime import encode_pair
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    cache_path=ROOT/'results/slcd_r5_prior_cache.jsonl';cache={}
    if cache_path.exists():
        for entry in [json.loads(l) for l in cache_path.read_text().splitlines()]:
            assert entry['signature']==global_signature and entry['key'] not in cache and set(entry['sum_scores'])==BASE and all(math.isfinite(v) for v in entry['sum_scores'].values())
            assert entry['prefix']==PREFIX and encode_pair(tok,PREFIX,entry['continuation'])==(entry['context_ids'],entry['target_ids'])
            assert entry['key']==digest(dict(global_signature=global_signature,prefix=PREFIX,continuation=entry['continuation'],context_ids=entry['context_ids'],target_ids=entry['target_ids']))
            cache[entry['key']]=entry
    done={}
    if (ROOT/output).exists():
        values=rows(output);done={v['id']:v for v in values};assert len(done)==len(values) and set(done)<=set(expected)
        for r in done.values():
            e=expected[r['id']];assert r['signature']==signature and r['gold']==e['gold'] and r['prompt_sha256']==e['prompt_sha256'] and r['choice_char_lengths']==e['choice_char_lengths'] and set(r['prior_methods'])==BASE
            assert len(r['cache_keys'])==len(r['cache_entry_sha256'])==len(e['request_pairs'])
            assert all(k in cache and digest(cache[k])==h for k,h in zip(r['cache_keys'],r['cache_entry_sha256']))
            for i,k in enumerate(r['cache_keys']):
                assert cache[k]['prefix']==PREFIX and cache[k]['continuation']==e['request_pairs'][i][1] and len(cache[k]['target_ids'])==r['choice_token_counts'][i]
                assert encode_pair(tok,*e['request_pairs'][i])[1]==cache[k]['target_ids']
            for name,value in r['prior_methods'].items():
                sums=[cache[k]['sum_scores'][name] for k in r['cache_keys']];norm=[s/max(n,1) for s,n in zip(sums,e['choice_char_lengths'])];pred=max(range(len(norm)),key=lambda i:norm[i])
                assert sums==value['sum_scores'] and norm==value['normalized_scores'] and pred==value['prediction'] and value['correct']==(pred==e['gold'])
    if (ROOT/completion).exists():
        receipt=read(completion);assert receipt['signature']==signature and receipt['n']==len(done)==n and sha(output)==receipt['score_file_sha256'];print('R5_PRIORS_ALREADY_COMPLETE',a.split);return
    from runtime import load_model,encode_pair,outcome,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    tok,model=load_model();scorer=OuroCachedScorer(model,6);head_sha=tensor_sha256(model.lm_head.weight);assert head_sha==read('results/slcd_SOURCE_AUDIT.json')['original_head_raw_tensor_sha256']
    misses=hits=0
    for row in data:
        if row['id'] in done:continue
        check_budget(120);begin=time.time();keys=[];entry_shas=[];totals={name:[] for name in BASE};target_counts=[]
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        for original_prompt,continuation in row['request_pairs']:
            assert original_prompt==row['prompt']
            context,target=encode_pair(tok,PREFIX,continuation)
            assert encode_pair(tok,original_prompt,continuation)[1]==target,'Prior changed continuation token boundary'
            key=digest(dict(global_signature=global_signature,prefix=PREFIX,continuation=continuation,context_ids=context,target_ids=target))
            if key not in cache:
                v=scorer.score_uncached([(context,target)],return_states=False)[0]
                sums={f'native_k{k}':float(v['sum_loglikelihood'][k]) for k in range(1,7)}
                sums.update({f'loopcd_k4_{name}':float(v['loopcd'][name]) for name in ['fixed','adaptive']})
                assert all(math.isfinite(x) for x in sums.values())
                entry=dict(key=key,signature=global_signature,prefix=PREFIX,continuation=continuation,context_ids=context,target_ids=target,sum_scores=sums)
                with cache_path.open('a',encoding='utf8') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
                cache[key]=entry;misses+=1
            else:hits+=1
            entry=cache[key];assert entry['context_ids']==context and entry['target_ids']==target and entry['continuation']==continuation and entry['prefix']==PREFIX
            keys.append(key);entry_shas.append(digest(entry));target_counts.append(len(target))
            for name in BASE:totals[name].append(entry['sum_scores'][name])
        methods={name:outcome(sums,row['choice_char_lengths'],row['gold']) for name,sums in totals.items()}
        r=dict(id=row['id'],gold=row['gold'],prompt_sha256=row['prompt_sha256'],choice_char_lengths=row['choice_char_lengths'],choice_token_counts=target_counts,
            prior_methods=methods,cache_keys=keys,cache_entry_sha256=entry_shas,signature=signature,weights_modified=False,elapsed_seconds=time.time()-begin)
        with (ROOT/output).open('a',encoding='utf8') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
        done[row['id']]=r
        if len(done)%100==0 or len(done)==n:
            progress=dict(split=a.split,completed=len(done),target=n,new_candidate_forwards=misses,cache_hits=hits,elapsed_seconds=time.time()-started);save('results/slcd_R5_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    assert set(done)==set(expected) and tensor_sha256(model.lm_head.weight)==head_sha
    assert implementation=={n:sha('scripts/'+n) for n in SOURCES} and sha(lp)==provenance['authorization_lock_sha256']
    assert {s:sha('data/slcd_'+s+'.jsonl') for s in splits}==source_sha
    save(completion,dict(status='complete',split=a.split,n=n,signature=signature,provenance=provenance,score_file=output,score_file_sha256=sha(output),new_candidate_forwards=misses,cache_hits=hits,
        independent_baseline_parity_coverage=0,independent_parity_status='No historical independent prior reference; uses previously audited unchanged original scorer',weights_modified=False,actual_shared_depth=6,elapsed_seconds=time.time()-started))
    print('R5_PRIORS_COMPLETE',a.split,flush=True)

if __name__=='__main__':main()
