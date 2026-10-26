"""Independent validation-only selection and complete artifact evidence audit."""
import argparse
from decimal import Decimal,InvalidOperation
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
from common import ROOT,BASE,CFG,PROTOCOL_SHA,read,save
from policies import METRICS,exit_index
from audit_actual_halting import audit_model as audit_actual_model

CATEGORIES=['arc_challenge','openbookqa','mmlu']
category=lambda r:'mmlu' if r['task'].startswith('mmlu_') else r['task']

def fingerprint(script):
    files=[ROOT/'scripts/common.py',ROOT/'scripts/methods.py',BASE/'scripts/qwen_recipe.py',ROOT/'scripts'/script]
    return hashlib.sha256(''.join(p.name+hashlib.sha256(p.read_bytes()).hexdigest() for p in files).encode()).hexdigest()

def records(path,script,expected,key='id'):
    rows=read(path);assert len(rows)==expected,(path,len(rows),expected)
    assert len({r[key] for r in rows})==len(rows),path
    expected_sha=fingerprint(script)
    assert all(r['protocol_sha256']==PROTOCOL_SHA and r['implementation_sha256']==expected_sha for r in rows),path
    return rows

def check_prediction(prediction,gold,lengths=None,metric=None):
    raw=np.array(prediction['sum_scores']);normalized=np.array(prediction['normalized_scores'])
    assert np.isfinite(raw).all() and np.isfinite(normalized).all()
    if lengths is not None:assert np.allclose(normalized,raw/np.asarray(lengths),rtol=1e-12,atol=1e-10)
    if metric is not None:assert prediction['primary_metric']==metric
    assert int(raw.argmax())==prediction['raw_prediction'] and int(normalized.argmax())==prediction['normalized_prediction']
    scores=raw if prediction['primary_metric']=='acc' else normalized
    assert int(scores.argmax())==prediction['prediction'] and (int(scores.argmax())==gold)==prediction['correct']

def check_generated_payload(row,tokenizer,vocab):
    tokens=row['token_ids'];assert all(type(t) is int and 0<=t<vocab for t in tokens)
    decoded=tokenizer.decode(tokens,skip_special_tokens=True).split('\nQuestion:')[0].split('\nQ:')[0]
    assert row['text']==decoded
    matches=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',decoded)
    prediction=matches[-1].replace(',','').lstrip('+') if matches else None
    assert row['prediction']==prediction and row['correct']==(prediction==row['gold'].replace(',',''))
    numeric=False
    if prediction is not None:
        try:numeric=Decimal(prediction)==Decimal(row['gold'].replace(',',''))
        except InvalidOperation:pass
    assert row['numeric_correct']==numeric
    assert row['reached_token_limit']==(len(tokens)==row['max_new_tokens'])
    assert math.isfinite(row['elapsed_s']) and row['elapsed_s']>=0 and row['peak_memory_bytes']>0
    if not row.get('identity_reused_from'):
        assert row['elapsed_s']>0
        assert row['baseline_audit_extra_heads']==(row['method']=='baseline' and row['condition']['depth']==4)
    lo,hi=row['actual_strength_min'],row['actual_strength_max'];case=row['condition']
    if case['kind']=='adaptive':assert 0<=lo<=hi<=case['cap']+1e-7
    else:assert lo==hi==case['omega']
    trace=row['token_trajectory']
    if not row['baseline_audit_extra_heads']:assert not trace;return
    assert len(trace) in (len(tokens),len(tokens)+1)
    assert [s['position'] for s in trace]==list(range(len(trace)))
    assert [s['chosen_token_id'] for s in trace[:len(tokens)]]==tokens
    if len(trace)>len(tokens):assert trace[-1]['chosen_token_id']==tokenizer.eos_token_id
    for step in trace:
        states=step['by_depth'];assert [s['depth'] for s in states]==list(range(1,case['depth']+1))
        assert step['chosen_token_id']==states[-1]['prediction']
        for depth,state in enumerate(states):
            assert 0<=state['prediction']<vocab
            for key in ['confidence','entropy','probability_margin','logit_margin']:
                assert math.isfinite(state[key]) and state[key]>=-1e-5
            assert state['confidence']<=1.00001 and state['probability_margin']<=1.00001
            for key in ['full_vocab_kl','hidden_relative_l2','hidden_absolute_l2','hidden_cosine']:
                value=state[key]
                if depth==0:assert value is None
                else:
                    assert math.isfinite(value)
                    assert abs(value)<=1.0001 if key=='hidden_cosine' else value>=-1e-4

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['mc','all'],default='all');args=parser.parse_args()
    original=json.loads((BASE/'ARTIFACT_MANIFEST.json').read_text())
    for name,entry in original.items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==entry['sha256'],name
    for source in json.loads((ROOT/'sources/source_manifest.json').read_text()):
        if source.get('status')==200 and source.get('file') and source.get('sha256'):
            assert hashlib.sha256((ROOT/'sources'/source['file']).read_bytes()).hexdigest()==source['sha256'],source['file']
    supplemental=json.loads((ROOT/'sources/supplemental_source_manifest.json').read_text())
    license=supplemental['harness_license'];payload=(ROOT/'sources'/license['file']).read_bytes()
    assert hashlib.sha256(payload).hexdigest()==license['sha256']
    assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==license['git_blob_sha1']
    for source in supplemental['mirror_tree_checks']:
        assert hashlib.sha256((ROOT/'sources'/source['snapshot']).read_bytes()).hexdigest()==source['snapshot_sha256']
    choices=json.loads((ROOT/'configs/mc_selected.json').read_text());halting=json.loads((ROOT/'configs/halting_selected.json').read_text())
    file_count=0;prediction_count=0
    mc_docs={r['id']:r for r in read('data/validation.jsonl')+read('data/qwen_test.jsonl')}
    def check_mc_row(row):
        doc=mc_docs[row['id']]
        assert row['gold']==doc['gold'] and row['prompt_sha256']==doc['prompt_sha256'] and row['task']==doc['task']
        return doc
    expected_revisions={'Qwen3-4B-Base':'906bfd4b4dc7f14ee4320094d8b41684abff8539','Ouro-1.4B':'574fa66cb8bf5abdc979642d01cf2b79b16bfab1','Ouro-2.6B':CFG['ouro26_revision']}
    for model in CFG['models']:
        for prefix in ['parity_','model_load_']:
            entry=json.loads((ROOT/'results'/(prefix+model+'.json')).read_text())
            if prefix=='parity_':assert entry['passed']
            else:
                assert entry['revision']==expected_revisions[model] and entry['dtype']=='torch.bfloat16'
                assert not any(entry['loading_info'].get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        if model.startswith('Qwen'):
            vv={};tt={}
            for method in CFG['qwen_methods']:
                vv[method]=records(f'trajectories/{model}_{method}_validation.jsonl','score_mc.py',250)
                tt[method]=records(f'trajectories/{model}_{method}_test.jsonl','score_mc.py',600)
                for row in vv[method]+tt[method]:
                    doc=check_mc_row(row);check_prediction(row['result'],doc['gold'],doc['choice_char_lengths'],doc['primary_metric']);prediction_count+=1
                assert {r['id'] for r in tt[method]}=={r['id'] for r in read('data/qwen_test.jsonl')}
                file_count+=2
            for task in CATEGORIES:
                best=min(['dola_low','dola_high'],key=lambda m:(-np.mean([r['result']['correct'] for r in vv[m] if category(r)==task]),['dola_low','dola_high'].index(m)))
                assert choices['qwen_dola'][task]==best
        else:
            vv=records(f'trajectories/{model}_validation.jsonl','score_mc.py',250);tt=records(f'trajectories/{model}_test.jsonl','score_mc.py',300);file_count+=2
            assert {r['id'] for r in tt}=={r['id'] for r in read('data/ouro_test.jsonl')}
            for task in CATEGORIES:
                part=[r for r in vv if category(r)==task]
                for space in ['logits','hidden']:
                    best=min(CFG['loopcd_strength_grid'],key=lambda w:(-np.mean([r['predictions'][f'{space}_{w:g}']['correct'] for r in part]),w))
                    assert choices['ouro'][model][space][task]==best
                best=min(CFG['temperature_grid'],key=lambda t:(-np.mean([r['predictions'][f'temperature_{t:g}']['correct'] for r in part]),abs(t-1)))
                assert choices['ouro'][model]['temperature'][task]==best
            for row in vv+tt:
                doc=check_mc_row(row)
                assert len(row['predictions'])==23 and len(row['trajectory'])==4
                for prediction in row['predictions'].values():check_prediction(prediction,doc['gold'],doc['choice_char_lengths'],doc['primary_metric']);prediction_count+=1
                for zero in ['logits_0','hidden_0','temperature_1']:
                    assert np.allclose(row['predictions'][zero]['sum_scores'],row['predictions']['r4']['sum_scores'],atol=.02,rtol=0)
            for metric in METRICS:
                for task in CATEGORIES:
                    validation=[r for r in vv if category(r)==task]
                    values=[s[metric] for r in validation for s in r['trajectory'][:-1] if s.get(metric) is not None]
                    grid=np.linspace(.01,1,100) if metric=='native_gate_cdf' else np.unique(np.quantile(values,np.linspace(.005,.995,51)))
                    options=[]
                    for threshold in grid:
                        exits_at=[exit_index(r['trajectory'],metric,float(threshold)) for r in validation]
                        options.append(dict(threshold=float(threshold),depth=float(np.mean(exits_at)+1),accuracy=float(np.mean([r['trajectory'][k]['correct'] for r,k in zip(validation,exits_at)]))))
                    for target in CFG['halting']['validation_budget_targets']:
                        feasible=[r for r in options if r['depth']<=target+1e-9]
                        stored=halting[model][metric][task]
                        if not feasible:assert str(target) not in stored;continue
                        best=min(feasible,key=lambda r:(-r['accuracy'],r['depth'],r['threshold']))
                        assert stored[str(target)]==best['threshold'],(model,task,metric,target)
    if args.stage=='mc':
        save('results/mc_completion_audit.json',dict(passed=True,scope='MC data/source identity, parity and validation selection only; generation/actual stopping/final delivery are NOT covered',formal_files=file_count,mc_scored_readouts=prediction_count,unique_mc_question_ids=len({r['id'] for r in read('data/validation.jsonl')+read('data/qwen_test.jsonl')}),validation_choices_independently_recomputed=True,source_limitations=CFG['source_gate']))
        print(json.dumps(dict(passed=True,scope='mc only',files=file_count,readouts=prediction_count)),flush=True);return
    math_selected=json.loads((ROOT/'configs/math_selected.json').read_text());execution=json.loads((ROOT/'configs/math_execution.json').read_text());n=execution['test_count']
    assert 50<=n<=CFG['math']['test_count']
    tokenizers=json.loads((ROOT/'sources/tokenizer_export_manifest.json').read_text());assert tokenizers['passed']
    assert {entry['model'] for entry in tokenizers['exports']}=={'Ouro-1.4B','Ouro-2.6B'}
    for entry in tokenizers['exports']:
        assert entry['revision']==json.loads((ROOT/f"results/model_load_{entry['model']}.json").read_text())['revision']
        for file in entry['files']:
            payload=(ROOT/'sources/tokenizers'/entry['model']/file['file']).read_bytes()
            assert len(payload)==file['bytes'] and hashlib.sha256(payload).hexdigest()==file['sha256']
    actual_audits={}
    for model in ['Ouro-1.4B','Ouro-2.6B']:
        assert json.loads((ROOT/f'results/generation_parity_{model}.json').read_text())['passed']
        vv=records(f'trajectories/math_{model}_validation.jsonl','generate_math.py',900,'record_key')
        tt=records(f'trajectories/math_{model}_test.jsonl','generate_math.py',8*n,'record_key');file_count+=2
        validation_docs={r['id']:r for r in read('data/math_validation.jsonl')}
        test_docs={r['id']:r for r in read('data/math_test.jsonl')[:n]}
        for measured,docs,methods in [(vv,validation_docs,CFG['math']['validation_methods']),(tt,test_docs,CFG['math']['test_methods'])]:
            assert {r['record_key'] for r in measured}=={uid+'|'+method for uid in docs for method in methods}
            assert all(r['record_key']==r['id']+'|'+r['method'] and r['gold']==docs[r['id']]['gold'] for r in measured)
        for space in ['logits','hidden']:
            grid=[(0,'baseline')]+[(w,f'{space}_{w:g}') for w in CFG['loopcd_strength_grid'] if w]
            omega,method=min(grid,key=lambda v:(-np.mean([r['correct'] for r in vv if r['method']==v[1]]),v[0]))
            assert math_selected[model][space]['omega']==omega
        ids={r['id'] for r in read('data/math_test.jsonl')[:n]}
        for method in CFG['math']['test_methods']:assert {r['id'] for r in tt if r['method']==method}==ids
        by_key={r['record_key']:r for r in tt}
        model_config=json.loads((ROOT/f'results/model_load_{model}.json').read_text())['config'];layers=model_config['num_hidden_layers']
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'sources/tokenizers'/model,local_files_only=True)
        for row in vv+tt:
            check_generated_payload(row,tokenizer,model_config['vocab_size'])
            extracted=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',row['text'])
            prediction=extracted[-1].replace(',','').lstrip('+') if extracted else None
            assert row['prediction']==prediction
            assert row['correct']==(row['prediction']==row['gold'].replace(',',''))
            depth=int(row['method'].split('_')[-1]) if row['method'].startswith('depth_') else 4
            assert row['condition']['depth']==depth and row['cache_layers']==layers*depth
            assert len(row['token_ids'])==row['new_tokens'] and len(row['token_ids'])<=CFG['math']['max_new_tokens'] and row['max_new_tokens']==CFG['math']['max_new_tokens']
            if row.get('identity_reused_from') and row['record_key'] in by_key:
                source=by_key[row['id']+'|'+row['identity_reused_from']]
                assert row['token_ids']==source['token_ids'] and row['elapsed_s']==0
                def equivalent_case(case):
                    kind=case['kind'];omega=case['omega'];cap=case['cap']
                    if kind in ['logits','hidden'] and omega==0 or kind=='adaptive' and cap==0:kind='baseline'
                    return case['depth'],kind,omega if kind!='baseline' else 0,cap if kind=='adaptive' else 0
                assert equivalent_case(row['condition'])==equivalent_case(source['condition'])
                for key in ['text','prediction','correct','numeric_correct','token_trajectory','baseline_audit_extra_heads','cache_layers','actual_strength_min','actual_strength_max']:
                    assert row[key]==source[key],(row['record_key'],key)
            if row['method'].endswith('_validation_selected'):
                space=row['method'].split('_')[0]
                assert row['condition']['kind']==space and row['condition']['omega']==math_selected[model][space]['omega'] and row['condition']['depth']==4
        actual=records(f'trajectories/actual_halting_{model}.jsonl','actual_halting.py',750,'record_key');file_count+=1
        actual_audits[model]=audit_actual_model(model)
        assert actual_audits[model]['timed_records']==len(actual) and actual_audits[model]['layer_count_parity_cases']==150
    budget=json.loads((ROOT/'logs/gpu_budget.json').read_text());assert budget['used_seconds']<50*3600
    assert abs(budget['used_seconds']-budget['stage1_used_seconds']-sum(r['seconds'] for r in budget['runs']))<1e-4
    recovery_path=ROOT/'logs/recovery_validation1p4_20261026/RECOVERY_SNAPSHOT.json'
    if recovery_path.exists():
        snapshot=json.loads(recovery_path.read_text());assert snapshot['passed']
        lineage=json.loads((ROOT/'results/recovery_lineage_check.json').read_text());assert lineage['passed']
        entry=snapshot['preserved_files']['trajectories/math_Ouro-1.4B_validation.jsonl']
        payload=(ROOT/'trajectories/math_Ouro-1.4B_validation.jsonl').read_bytes()
        assert hashlib.sha256(payload[:entry['bytes']]).hexdigest()==entry['sha256']
        probe=json.loads((ROOT/'results/recovery_next_case_probe.json').read_text())
        formal=next(r for r in read('trajectories/math_Ouro-1.4B_validation.jsonl') if r['record_key']==probe['record_key'])
        for key in ['token_ids','text','prediction','correct','numeric_correct','condition']:
            assert formal[key]==probe[key]
        assert json.loads((ROOT/'logs/recovery_probe_exit_attempt2.json').read_text())['exit_code']==0
        assert any(r.get('pid')==30772 and r['state']=='native_abort_0xc0000409' for r in budget['runs'])
    save('results/completion_audit.json',dict(passed=True,original_stage1_files_unchanged=len(original),audited_formal_files=file_count,audited_mc_predictions=prediction_count,validation_selection_independently_recomputed=True,math_test_count=n,audited_actual_timed_records=sum(a['timed_records'] for a in actual_audits.values()),audited_actual_layer_parity_cases=sum(a['layer_count_parity_cases'] for a in actual_audits.values()),cumulative_gpu_hours=budget['used_seconds']/3600,scope='All configured MC, generation and actual halting outputs; source/protocol fingerprints; validation-only selection; exact actual-exit ID/method/repeat coverage, observed parity-layer counts and independently recomputed exit/scores/quality'))
    print(json.dumps(dict(passed=True,files=file_count)),flush=True)

if __name__=='__main__':main()
