"""GPU integrity probe after a terminal native crash; never writes formal JSONL."""
import hashlib
import json
import os
import time
from common import ROOT,CFG,PROTOCOL_SHA,Budget,load,read,save,append
from generate_math import decode,condition,numeric_equal


def main():
    model_name='Ouro-2.6B'
    snapshot=json.loads((ROOT/'logs/recovery_validation2p6_20261026/RECOVERY_SNAPSHOT.json').read_text())
    path=ROOT/f'trajectories/math_{model_name}_validation.jsonl'
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    assert snapshot['passed'] and before==snapshot['preserved_files']['trajectories/math_Ouro-2.6B_validation.jsonl']['sha256']
    records=read(path);assert len(records)==207
    docs=read('data/math_validation.jsonl');by_id={r['id']:r for r in docs}
    examples=read('data/math_fewshot.jsonl')
    shots='\n\n'.join('Question: '+r['question']+'\nAnswer: '+r['answer'] for r in examples)
    baselines=[r for r in records if r['method']=='baseline']
    chosen={r['record_key']:r for r in [baselines[0],baselines[-1],records[-1]]}
    checks=[]
    with Budget('recovery_boundary_replay_Ouro-2.6B') as budget:
        tok,model=load(model_name)
        for key,old in chosen.items():
            prompt=shots+'\n\nQuestion: '+by_id[old['id']]['question']+'\nAnswer:'
            outcome=decode(tok,model,prompt,old['condition'],budget,audit=old['baseline_audit_extra_heads'])
            assert outcome['token_ids']==old['token_ids'],('Restart token mismatch',key)
            assert outcome['text']==old['text'] and outcome['prediction']==old['prediction']
            if old['baseline_audit_extra_heads']:
                old_predictions=[[s['prediction'] for s in step['by_depth']] for step in old['token_trajectory']]
                new_predictions=[[s['prediction'] for s in step['by_depth']] for step in outcome['token_trajectory']]
                assert old_predictions==new_predictions,('Restart depth-readout mismatch',key)
            check=dict(record_key=key,token_ids_exact=True,text_exact=True,prediction_exact=True,
                       all_depth_argmax_exact=bool(old['baseline_audit_extra_heads']),new_tokens=outcome['new_tokens'])
            checks.append(check);print(json.dumps(check),flush=True)
        methods=CFG['math']['validation_methods']
        row=docs[len(records)//len(methods)];method=methods[len(records)%len(methods)]
        case=condition(method,model_name)
        prompt=shots+'\n\nQuestion: '+row['question']+'\nAnswer:'
        next_outcome=decode(tok,model,prompt,case,budget,audit=method=='baseline')
        probe=dict(id=row['id'],method=method,record_key=row['id']+'|'+method,condition=case,**next_outcome,
                   scope='Recovery diagnostic only; not appended to formal validation and not used to select parameters')
        probe.update(model=model_name,task='gsm8k',phase='recovery_diagnostic',gold=row['gold'],max_new_tokens=CFG['math']['max_new_tokens'])
        probe['correct']=probe['prediction']==row['gold'].replace(',','')
        probe['numeric_correct']=numeric_equal(probe['prediction'],row['gold'])
        from audit_suite import check_generated_payload
        check_generated_payload(probe,tok,model.config.vocab_size)
        destination='logs/recovery_validation2p6_20261026/next_case_append_probe.jsonl'
        assert not (ROOT/destination).exists(),'Diagnostic already exists; inspect before rerun'
        append(destination,probe)
        persisted=read(destination)[0]
        assert persisted['token_ids']==probe['token_ids'] and persisted['text']==probe['text'] and persisted['numeric_correct']==probe['numeric_correct']
        save('results/native26_recovery_next_case_probe.json',probe)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        save('results/native26_recovery_boundary_replay.json',dict(passed=True,model=model_name,protocol_sha256=PROTOCOL_SHA,probe_pid=os.getpid(),completed_unix=time.time(),
             existing_replays=checks,next_case_key=probe['record_key'],next_case_completed=True,
             next_case_metadata_numeric_and_append_checked=True,
             formal_prefix_records=len(records),formal_prefix_sha256=before,formal_jsonl_unchanged=True,
             scope='Exact token/text and baseline depth-argmax replay of three completed boundary cases plus successful retry of the previously unsaved next case; not a replay of all207 cases'))
        print(json.dumps(dict(passed=True,existing_cases=len(checks),next_case=probe['record_key'],next_tokens=probe['new_tokens'])),flush=True)


if __name__=='__main__':main()
