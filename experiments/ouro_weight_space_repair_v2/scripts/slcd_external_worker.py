"""Frozen zero-shot official harness tasks, permitted only after complete ARC gate."""
import argparse,hashlib,json,os,time,urllib.request,importlib.metadata
from pathlib import Path
import numpy as np
from slcd_core import ROOT,read,sha,digest,save,scores_for,summarize,writecsv

def download(repo,relative,out):
    # Resolve once, store exact repository commit and byte SHA in the manifest receipt.
    p=ROOT/out;source_receipt=p.with_name(p.stem+'_SOURCE.json')
    assert p.parent.is_dir()
    if source_receipt.exists():
        source=json.loads(source_receipt.read_text(encoding='utf8'))
        assert source['repository']==repo and p.is_file() and source['sha256']==sha(out)
        return source
    assert not p.exists(),'Raw download without a source receipt must be inspected, not relabeled at a new revision'
    info=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo,timeout=60));revision=info['sha']
    if repo=='allenai/openbookqa':
        available={x['rfilename'] for x in info['siblings']}
        relative=next(p for p in ['main/test-00000-of-00001.parquet','main/test/0000.parquet','main/openbookqa-test.parquet'] if p in available)
    url='https://huggingface.co/datasets/'+repo+'/resolve/'+revision+'/'+relative
    if not p.exists():
        tmp=p.with_suffix('.download');assert not tmp.exists()
        with urllib.request.urlopen(url,timeout=120) as src,tmp.open('xb') as dst:
            import shutil
            shutil.copyfileobj(src,dst)
        tmp.replace(p)
    source=dict(repository=repo,revision=revision,path=relative,url=url,sha256=sha(out))
    save(source_receipt.relative_to(ROOT),source)
    return source

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['hellaswag','openbookqa'],required=True);a=ap.parse_args();started=time.time()
    gate=read('results/slcd_ARC_TEST_COMPLETE.json');lock=read('FINAL_METHOD_LOCK.json')
    assert gate['status']=='complete' and gate['n']==1172 and gate['primary_correct']>717 and gate['other_benchmarks_authorized']
    assert gate['final_lock_sha256']==sha('FINAL_METHOD_LOCK.json') and gate['primary']==lock['primary']
    for name,h in lock['inference_source_sha256'].items():assert sha('scripts/'+name)==h
    assert sha('configs/slcd_TOKEN_TEST.json') in lock['allowed_recipe_sha256']
    for name,h in lock['tokenizer_and_model_metadata_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==h
    assert importlib.metadata.version('lm_eval')=='0.4.9'
    from runtime import check_budget
    check_budget(120)
    expected={'hellaswag':10042,'openbookqa':500}[a.task]
    prefix='slcd_'+a.task;manifest=f'data/{prefix}.jsonl';receipt=f'configs/{prefix}_PROTOCOL.json'
    if not (ROOT/manifest).exists():
        repo,relative={'hellaswag':('Rowan/hellaswag','data/validation-00000-of-00001.parquet'),'openbookqa':('allenai/openbookqa','main/test-00000-of-00001.parquet')}[a.task]
        source=download(repo,relative,f'data/{prefix}_raw.parquet')
        import pyarrow.parquet as pq
        from datasets import Dataset,DatasetDict
        from lm_eval.api.task import ConfigurableTask
        from lm_eval.utils import load_yaml_config
        import lm_eval
        folder=Path(lm_eval.__file__).parent/'tasks'/a.task
        config=load_yaml_config(str(folder/(a.task+'.yaml')))
        evaluation_split='validation' if a.task=='hellaswag' else 'test'
        config['custom_dataset']=lambda **kw:DatasetDict({evaluation_split:Dataset.from_list(pq.read_table(ROOT/f'data/{prefix}_raw.parquet').to_pylist())})
        config['dataset_kwargs']={};config['num_fewshot']=0;config['training_split']=None
        # ConfigurableTask initializes its few-shot sampler even at zero shots.
        # Point that sampler at the available split without drawing examples.
        config['fewshot_split']=evaluation_split
        task=ConfigurableTask(config=config);task.set_fewshot_seed(2021)
        docs=list(task.validation_docs() if a.task=='hellaswag' else task.test_docs());assert len(docs)==expected
        with (ROOT/manifest).open('x',encoding='utf8') as f:
            for i,doc in enumerate(docs):
                context=task.fewshot_context(doc,0);pairs=[list(x.args) for x in task.construct_requests(doc,context,metadata=(a.task,i,1))]
                choices=task.doc_to_choice(doc);gold=int(task.doc_to_target(doc));assert len(pairs)==len(choices)==4
                f.write(json.dumps(dict(id=f'{a.task}/'+('validation' if a.task=='hellaswag' else 'test')+f'/{i}',gold=gold,prompt=context,prompt_sha256=hashlib.sha256(context.encode()).hexdigest(),request_pairs=pairs,choice_char_lengths=[len(c) for c in choices]),ensure_ascii=False)+'\n')
        save(receipt,dict(status='frozen_before_scoring',source=source,n=expected,num_fewshot=0,seed=2021,metric='acc_norm',denominator='complete answer character count',
            harness_version='0.4.9',official_task_config_sha256=hashlib.sha256((folder/(a.task+'.yaml')).read_bytes()).hexdigest(),
            task_utils_sha256=hashlib.sha256((folder/'utils.py').read_bytes()).hexdigest() if (folder/'utils.py').exists() else None,
            manifest_sha256=sha(manifest),final_method_lock_sha256=sha('FINAL_METHOD_LOCK.json'),worker_source_sha256=sha('scripts/slcd_external_worker.py')))
    protocol=read(receipt);assert protocol['manifest_sha256']==sha(manifest) and protocol['final_method_lock_sha256']==sha('FINAL_METHOD_LOCK.json') and protocol['worker_source_sha256']==sha('scripts/slcd_external_worker.py')
    data=[json.loads(l) for l in (ROOT/manifest).read_text().splitlines()];assert len(data)==expected
    from runtime import load_model,encode_pair,outcome,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from slcd_logit_methods import score_logit_recipes
    from slcd_token_worker_v2 import file_sha,MODEL_SHA
    assert file_sha(ROOT/'models/Ouro-1.4B/model.safetensors')==MODEL_SHA
    import torch
    import torch.nn.functional as F
    tok,model=load_model();scorer=OuroCachedScorer(model,6)
    reg=lock['method_registry'];primary=lock['primary'];need=any(s['kind']=='token' for s in reg.values());recipes=read('configs/slcd_TOKEN_TEST.json')['recipes']
    output=f'results/{prefix}_scores.jsonl';done={}
    if (ROOT/output).exists():
        prior=[json.loads(l) for l in (ROOT/output).read_text().splitlines()]
        done={r['id']:r for r in prior};expected_rows={r['id']:r for r in data}
        assert len(done)==len(prior) and set(done)<=set(expected_rows)
        for old in done.values():
            assert old['protocol_sha256']==sha(receipt) and old['primary']==primary
            assert all(old[key]==expected_rows[old['id']][key] for key in ['gold','prompt_sha256','request_pairs','choice_char_lengths'])
    for row in data:
        if row['id'] in done:continue
        check_budget(120);begin=time.time();pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
        values=scorer.score_uncached(pairs,return_states=need)
        original={f'native_k{k}':outcome([v['sum_loglikelihood'][k] for v in values],row['choice_char_lengths'],row['gold']) for k in range(1,7)}
        original.update({f'loopcd_k4_{name}':outcome([v['loopcd'][name] for v in values],row['choice_char_lengths'],row['gold']) for name in ['fixed','adaptive']})
        tokens=None
        if need:
            sums={r['id']:[] for r in recipes}
            for v,(_,target) in zip(values,pairs):
                logits={k:F.linear(h.to('cuda'),model.lm_head.weight)[0].float() for k,h in enumerate(v['normalized_states'],1)}
                vals=score_logit_recipes(logits,target,recipes)
                for key,value in vals.items():sums[key].append(value)
            methods={key:outcome(value,row['choice_char_lengths'],row['gold']) for key,value in sums.items()}
            for key,name in [('tf_fixed','loopcd_k4_fixed'),('tf_adaptive','loopcd_k4_adaptive')]:assert methods[key]['sum_scores']==original[name]['sum_scores']
            tokens={row['id']:dict(methods=methods)}
        base=dict(id=row['id'],gold=row['gold'],methods={k:np.asarray(v['normalized_scores'],dtype=float) for k,v in original.items()})
        selected=scores_for(base,reg[primary],reg,tokens)
        prediction=int(selected.argmax());r=dict(**row,original_methods=original,primary=primary,selected_scores=selected.tolist(),prediction=prediction,correct=prediction==row['gold'],
            protocol_sha256=sha(receipt),weights_modified=False,actual_trajectory_depth=6,elapsed_seconds=time.time()-begin)
        with (ROOT/output).open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        done[row['id']]=r
        if len(done)%100==0 or len(done)==expected:
            progress=dict(task=a.task,completed=len(done),target=expected,elapsed_seconds=time.time()-started)
            save('results/slcd_EXTERNAL_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    assert set(done)=={r['id'] for r in data}
    rng=np.random.default_rng(2021);results=[];transitions=[]
    for mid,key in [('native4','native_k4'),('fixed','loopcd_k4_fixed'),('adaptive','loopcd_k4_adaptive'),(primary,None)]:
        rescues=damage=correct=0
        for row in data:
            r=done[row['id']];bp=r['original_methods']['loopcd_k4_fixed']['prediction'];p=r['prediction'] if key is None else r['original_methods'][key]['prediction']
            good=p==r['gold'];old=bp==r['gold'];correct+=good;rescues+=good and not old;damage+=old and not good
            transitions.append(dict(method=mid,id=r['id'],gold=r['gold'],fixed_prediction=bp,prediction=p,correct=good,base_correct=old))
        draws=rng.multinomial(expected,[rescues/expected,damage/expected,1-(rescues+damage)/expected],size=50000);delta=100*(draws[:,0]-draws[:,1])/expected
        from scipy.stats import binomtest
        results.append(dict(method=mid,n=expected,correct=correct,accuracy=correct/expected,corrections=rescues,regressions=damage,net_correct=rescues-damage,delta_accuracy_pp=100*(rescues-damage)/expected,
            exact_mcnemar_p=1. if rescues+damage==0 else float(binomtest(rescues,rescues+damage,.5).pvalue),paired_bootstrap_95ci_low_pp=float(np.quantile(delta,.025)),paired_bootstrap_95ci_high_pp=float(np.quantile(delta,.975))))
    save(f'results/{prefix}_COMPLETE.json',dict(status='complete',task=a.task,n=expected,results=results,score_file=output,score_file_sha256=sha(output),protocol_sha256=sha(receipt),elapsed_seconds=time.time()-started,
        independent_baseline_parity_coverage=0,same_forward_control_parity_exact=need,actual_shared_depth=6,logical_primary_depth=lock['logical_loop_depth'],no_runtime_speed_claim=True))
    writecsv(f'results/{prefix}_results.csv',results);writecsv(f'results/{prefix}_paired_transitions.csv',transitions)
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
