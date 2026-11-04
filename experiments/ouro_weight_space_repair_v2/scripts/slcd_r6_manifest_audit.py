"""Independently rerender official external inputs; reads no model outcomes.

Closes the distinction between binding a saved manifest hash and reproducing
that manifest from pinned raw parquet and installed official task definitions.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads((ROOT/path).read_text(encoding='utf8'))


def main():
    assert importlib.metadata.version('lm_eval')=='0.4.9'
    from datasets import Dataset,DatasetDict
    import pyarrow.parquet as pq
    import lm_eval
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.utils import load_yaml_config
    lock=read('configs/slcd_R6_EXTERNAL_LOCK.json');lock_sha=sha(ROOT/'configs/slcd_R6_EXTERNAL_LOCK.json')
    taskroot=Path(lm_eval.__file__).parent/'tasks';results={}
    for task,n,split in [('hellaswag',10042,'validation'),('openbookqa',500,'test')]:
        spec=lock['tasks'][task];raw=ROOT/f'data/slcd_{task}_raw.parquet'
        assert sha(raw)==spec['source']['sha256']
        manifest=ROOT/spec['manifest_path'];assert sha(manifest)==spec['manifest_sha256']
        folder=taskroot/task;yaml=folder/(task+'.yaml');utils=folder/'utils.py'
        assert sha(yaml)==spec['official_yaml_sha256']
        if utils.exists():assert sha(utils)==spec['official_utils_sha256']
        docs=Dataset.from_list(pq.read_table(raw).to_pylist());assert len(docs)==n
        cfg=load_yaml_config(str(yaml));collection=DatasetDict({split:docs})
        cfg.update(custom_dataset=lambda **kw:collection,dataset_kwargs={},num_fewshot=0,training_split=None,fewshot_split=split)
        engine=ConfigurableTask(config=cfg);engine.set_fewshot_seed(2021)
        processed=engine.validation_docs() if split=='validation' else engine.test_docs()
        count=0
        with manifest.open(encoding='utf8') as stream:
            for i,doc in enumerate(processed):
                saved=json.loads(next(stream));context=engine.fewshot_context(doc,0)
                pairs=[list(req.args) for req in engine.construct_requests(doc,context,metadata=(task,i,1))]
                choices=engine.doc_to_choice(doc);gold=int(engine.doc_to_target(doc))
                expected=dict(id=f'{task}/{split}/{i}',gold=gold,prompt=context,
                              prompt_sha256=hashlib.sha256(context.encode()).hexdigest(),request_pairs=pairs,
                              choice_char_lengths=[len(c) for c in choices])
                assert saved==expected,(task,i,[k for k in expected if saved.get(k)!=expected[k]])
                count+=1
            assert not stream.read().strip(),'Extra saved manifest rows'
        assert count==spec['n']==n
        assert sha(manifest)==spec['manifest_sha256'] and sha(raw)==spec['source']['sha256']
        results[task]=dict(status='passed',n=n,exact_full_row_equality=True,
                           manifest_sha256=spec['manifest_sha256'],raw_parquet_sha256=sha(raw),
                           official_yaml_sha256=sha(yaml),official_utils_sha256=sha(utils) if utils.exists() else None,
                           num_fewshot=0,seed=2021)
    assert sha(ROOT/'configs/slcd_R6_EXTERNAL_LOCK.json')==lock_sha
    output=dict(status='passed',scope='Complete official input rendering; no model scores or partial outcomes read',tasks=results,
                external_lock_sha256=lock_sha,source_sha256=sha(Path(__file__)),
                versions={n:importlib.metadata.version(n) for n in ['lm_eval','datasets','pyarrow']},model_forwards=0)
    (ROOT/'results/slcd_R6_MANIFEST_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
