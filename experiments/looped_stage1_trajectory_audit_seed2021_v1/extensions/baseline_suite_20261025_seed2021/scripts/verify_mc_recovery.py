"""Recover only CPU postprocessing after all GPU MC workers passed."""
import json
from common import ROOT,CFG,read,save
from audit_suite import records,check_prediction

def main():
    recovery=json.loads((ROOT/'logs/mc_analysis_recovery_1_exit.json').read_text(encoding='utf-8-sig'));assert recovery['exit_code']==0
    original=json.loads((ROOT/'logs/mc_supervisor_exit.json').read_text(encoding='utf-8-sig'));assert original['exit_code']!=0
    files=[]
    for model in CFG['models']:
        assert json.loads((ROOT/f'results/parity_{model}.json').read_text())['passed']
        if model.startswith('Qwen'):
            for method in CFG['qwen_methods']:
                for split,count in [('validation',250),('test',600)]:
                    path=f'trajectories/{model}_{method}_{split}.jsonl';rows=records(path,'score_mc.py',count)
                    expected={r['id']:r for r in read('data/'+('validation' if split=='validation' else 'qwen_test')+'.jsonl')}
                    assert {r['id'] for r in rows}==set(expected)
                    for row in rows:
                        assert row['prompt_sha256']==expected[row['id']]['prompt_sha256'];check_prediction(row['result'],row['gold'])
                    files.append(path)
        else:
            for split,count in [('validation',250),('test',300)]:
                path=f'trajectories/{model}_{split}.jsonl';rows=records(path,'score_mc.py',count)
                expected={r['id']:r for r in read('data/'+('validation' if split=='validation' else 'ouro_test')+'.jsonl')}
                assert {r['id'] for r in rows}==set(expected)
                for row in rows:
                    assert row['prompt_sha256']==expected[row['id']]['prompt_sha256'] and len(row['trajectory'])==4 and len(row['predictions'])==23
                    for prediction in row['predictions'].values():check_prediction(prediction,row['gold'])
                files.append(path)
    assert (ROOT/'configs/mc_selected.json').exists() and (ROOT/'configs/halting_selected.json').exists()
    save('results/mc_recovery_gate.json',dict(passed=True,formal_files_checked=len(files),all_gpu_outputs_preserved=True,original_supervisor_exit=original,cpu_recovery_exit=recovery,root_cause='Initial CPU analysis terminated with native status 0xC0000409; unchanged calculations succeeded after adding diagnostic progress/faulthandler. Root cause not established; original log/exit retained.'))
    print(json.dumps(dict(passed=True,files=len(files))),flush=True)

if __name__=='__main__':main()
