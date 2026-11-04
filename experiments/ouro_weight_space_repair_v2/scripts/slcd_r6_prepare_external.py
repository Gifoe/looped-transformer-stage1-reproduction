"""Render official zero-shot inputs after full ARC success; no model inference."""
import hashlib,json
from pathlib import Path
from slcd_core import ROOT,read,rows,sha,save

def main():
    acceptance=read('results/slcd_R6_TEST_ACCEPTANCE.json');arc=read('results/slcd_R6_ARC_COMPLETE.json')
    assert acceptance['status']=='passed' and acceptance['n']==arc['n']==1172 and acceptance['correct']==arc['correct']>=733
    assert acceptance['final_lock_sha256']==arc['final_lock_sha256']==sha('configs/slcd_R6_FINAL_LOCK.json')
    import lm_eval
    import importlib.metadata
    assert importlib.metadata.version('lm_eval')=='0.4.9'
    from datasets import Dataset,DatasetDict
    import pyarrow.parquet as pq
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.utils import load_yaml_config
    task_root=Path(lm_eval.__file__).parent/'tasks';tasks={}
    for task,n,split in [('hellaswag',10042,'validation'),('openbookqa',500,'test')]:
        raw=f'data/slcd_{task}_raw.parquet';source=read(f'data/slcd_{task}_raw_SOURCE.json');assert sha(raw)==source['sha256']
        folder=task_root/task;yaml=folder/(task+'.yaml');manifest=f'data/slcd_{task}.jsonl'
        if not (ROOT/manifest).exists():
            docs=Dataset.from_list(pq.read_table(ROOT/raw).to_pylist());assert len(docs)==n
            collection=DatasetDict({split:docs});cfg=load_yaml_config(str(yaml))
            cfg.update(custom_dataset=lambda **kw:collection,dataset_kwargs={},num_fewshot=0,training_split=None,fewshot_split=split)
            engine=ConfigurableTask(config=cfg);engine.set_fewshot_seed(2021)
            processed=list(engine.validation_docs() if split=='validation' else engine.test_docs());assert len(processed)==n
            with (ROOT/manifest).open('x',encoding='utf8') as f:
                for i,doc in enumerate(processed):
                    context=engine.fewshot_context(doc,0);pairs=[list(x.args) for x in engine.construct_requests(doc,context,metadata=(task,i,1))]
                    choices=engine.doc_to_choice(doc);gold=int(engine.doc_to_target(doc));assert len(pairs)==len(choices)==4 and 0<=gold<4
                    f.write(json.dumps(dict(id=f'{task}/{split}/{i}',gold=gold,prompt=context,prompt_sha256=hashlib.sha256(context.encode()).hexdigest(),request_pairs=pairs,choice_char_lengths=[len(c) for c in choices]),ensure_ascii=False)+'\n')
        data=rows(manifest);assert len(data)==len({r['id'] for r in data})==n
        for r in data:
            assert r['id'].startswith(f'{task}/{split}/') and hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
            assert len(r['request_pairs'])==len(r['choice_char_lengths'])==4 and all(p[0]==r['prompt'] for p in r['request_pairs'])
        old_path=f'results/slcd_{task}_scores.jsonl';old_protocol=f'configs/slcd_{task}_PROTOCOL.json'
        meta=dict(n=n,split=split,manifest_path=manifest,manifest_sha256=sha(manifest),source=source,num_fewshot=0,seed=2021,
            official_yaml_sha256=hashlib.sha256(yaml.read_bytes()).hexdigest(),official_utils_sha256=hashlib.sha256((folder/'utils.py').read_bytes()).hexdigest() if (folder/'utils.py').exists() else None,
            old_main_path=old_path if (ROOT/old_path).exists() else None,old_main_sha256=sha(old_path) if (ROOT/old_path).exists() else None,
            old_protocol_path=old_protocol if (ROOT/old_protocol).exists() else None,old_protocol_sha256=sha(old_protocol) if (ROOT/old_protocol).exists() else None)
        if meta['old_main_path']:
            protocol=read(old_protocol);assert protocol['manifest_sha256']==sha(manifest) and protocol['num_fewshot']==0 and protocol['n']==n
            assert protocol['worker_source_sha256']==sha('scripts/slcd_external_worker.py')
            meta.update(old_worker_source_sha256=sha('scripts/slcd_external_worker.py'),old_first_final_lock_sha256=sha('FINAL_METHOD_LOCK.json'),old_main_count=sum(1 for line in (ROOT/old_path).open()))
        tasks[task]=meta
    save('configs/slcd_R6_EXTERNAL_DATA.json',dict(status='prepared_after_full_ARC_target',arc_result_sha256=sha('results/slcd_R6_ARC_COMPLETE.json'),
        acceptance_sha256=sha('results/slcd_R6_TEST_ACCEPTANCE.json'),tasks=tasks,preparation_source_sha256=sha('scripts/slcd_r6_prepare_external.py'),metric='acc_norm',denominator='same complete answer character count'))
    print(json.dumps(dict(status='external_data_prepared',tasks={k:dict(n=v['n'],old_main_count=v.get('old_main_count',0)) for k,v in tasks.items()})))

if __name__=='__main__':main()
