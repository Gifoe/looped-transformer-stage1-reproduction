"""ARC-gated official zero-shot rendering; field rules predate ARC testing."""
import copy,hashlib,time
from pathlib import Path
from r7_core import ROOT,read,save,sha,jsonlines


def main():
    from runtime import configure
    configure()
    arc=read('results/r7_ARC_COMPLETE.json');accept=read('results/r7_TEST_ACCEPTANCE.json')
    assert arc['status']=='complete' and arc['n']==1172 and arc['correct']>736 and accept['status']=='passed'
    final=read('configs/r7_FINAL_METHOD_LOCK.json');rules=read('configs/r7_EXTERNAL_REFERENCE_RULES.json')
    assert rules['created_unix']<final['created_unix'] and arc['final_method_lock_sha256']==sha('configs/r7_FINAL_METHOD_LOCK.json')
    assert accept['n']==1172 and accept['max_R6_candidate_score_abs_error']==0. and accept['N0_prior_sums_exact']
    assert accept['final_method_lock_sha256']==arc['final_method_lock_sha256']==sha('configs/r7_FINAL_METHOD_LOCK.json')
    assert accept['new_score_file_sha256']==arc['score_file_sha256']==sha('results/r7_test_final.jsonl')
    assert arc['primary']==final['primary']
    assert sha('configs/r7_EXTERNAL_REFERENCE_RULES.json')==final['external_reference_rules_sha256']
    oldaudit=read('results/slcd_R6_EXTERNAL_ACCEPTANCE.json');oldrelease=read('artifacts/slcd_R6_RELEASE_INDEX.json')
    assert oldaudit['status']=='passed' and oldaudit['scope']=='combined'
    assert sha('results/slcd_R6_EXTERNAL_ACCEPTANCE.json')==oldrelease['external_acceptance_sha256']
    old=read('configs/slcd_R6_EXTERNAL_DATA.json');tasks={}
    import lm_eval
    import importlib.metadata
    from datasets import Dataset,DatasetDict
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.utils import load_yaml_config
    import pyarrow.parquet as pq
    assert importlib.metadata.version('lm_eval')=='0.4.9'
    for task,n,split in [('hellaswag',10042,'validation'),('openbookqa',500,'test')]:
        rule=rules['tasks'][task];spec=old['tasks'][task]
        p=Path(lm_eval.__file__).parent/'tasks'/task/(task+'.yaml')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==rule['official_yaml_sha256']==spec['official_yaml_sha256']
        raw=f'data/slcd_{task}_raw.parquet';assert sha(raw)==spec['source']['sha256']
        dataset=Dataset.from_list(pq.read_table(ROOT/raw).to_pylist());collection=DatasetDict({split:dataset})
        cfg=load_yaml_config(str(p));cfg.update(custom_dataset=lambda **kw:collection,dataset_kwargs={},num_fewshot=0,training_split=None,fewshot_split=split)
        engine=ConfigurableTask(config=cfg);engine.set_fewshot_seed(2021)
        docs=list(engine.validation_docs() if split=='validation' else engine.test_docs())
        manifest=spec['manifest_path'];assert sha(manifest)==spec['manifest_sha256']
        values=list(jsonlines(manifest));assert len(docs)==len(values)==n
        for doc,row in zip(docs,values):
            assert engine.fewshot_context(doc,0)==row['prompt']
            for name,placeholder in [('N1','N/A'),('N2','Unknown')]:
                reference=copy.deepcopy(doc);reference[rule['current_problem_field']]=placeholder
                assert engine.fewshot_context(reference,0)==rule[name]
        oldpath=f'results/slcd_r6_{task}_scores.jsonl';oldreceipt=read(f'results/slcd_r6_{task}_COMPLETE.json')
        assert oldreceipt['status']=='complete' and oldreceipt['n']==n and sha(oldpath)==oldreceipt['score_file_sha256']
        assert sha(f'results/slcd_r6_{task}_COMPLETE.json')==oldaudit['tasks'][task]['complete_sha256']
        assert oldreceipt['score_file_sha256']==oldaudit['tasks'][task]['score_sha256']
        tasks[task]=dict(n=n,split=split,manifest_path=manifest,manifest_sha256=sha(manifest),
            old_scores_path=oldpath,old_scores_sha256=sha(oldpath),old_receipt_sha256=sha(f'results/slcd_r6_{task}_COMPLETE.json'),
            references={key:rule[key] for key in ['N0','N1','N2']},num_fewshot=0,current_problem_field=rule['current_problem_field'],
            official_yaml_sha256=rule['official_yaml_sha256'],reference_rendering_verified_questions=n)
    from r7_worker import SOURCES
    names=SOURCES+['r7_external_prepare.py','r7_external_worker.py']
    save('configs/r7_EXTERNAL_LOCK.json',dict(status='frozen_after_ARC_gate',created_unix=time.time(),primary=final['primary'],
        final_method_lock_sha256=sha('configs/r7_FINAL_METHOD_LOCK.json'),arc_complete_sha256=sha('results/r7_ARC_COMPLETE.json'),
        acceptance_sha256=sha('results/r7_TEST_ACCEPTANCE.json'),pretest_reference_rules_sha256=sha('configs/r7_EXTERNAL_REFERENCE_RULES.json'),
        old_external_acceptance_sha256=sha('results/slcd_R6_EXTERNAL_ACCEPTANCE.json'),old_release_index_sha256=sha('artifacts/slcd_R6_RELEASE_INDEX.json'),
        inference_source_sha256={name:sha('scripts/'+name) for name in names},tasks=tasks,metric='acc_norm',
        package_versions={name:importlib.metadata.version(name) for name in ['torch','transformers','tokenizers','lm_eval','safetensors']},
        model_identity=final['model_identity'],formula_unchanged=True,external_outcomes_not_used_for_selection=True))
    print('R7_EXTERNAL_PREPARED',flush=True)


if __name__=='__main__':main()
