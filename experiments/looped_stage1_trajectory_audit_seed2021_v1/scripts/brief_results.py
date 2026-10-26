import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def table(name,columns,group='all'):
    path=ROOT/'results'/name
    if not path.exists():return None
    frame=pd.read_csv(path)
    if 'task' in frame and group is not None:frame=frame[frame.task==group]
    return json.loads(frame[[c for c in columns if c in frame]].to_json(orient='records'))
report=dict(
    ouro=table('ouro_metrics.csv',['method','accuracy','wrong_to_correct','correct_to_wrong','delta','ci_low','ci_high']),
    patterns=table('trajectory_patterns.csv',['pattern','n','count','rate','ci_low','ci_high']),
    qwen=table('qwen_metrics.csv',['method','accuracy','wrong_to_correct','correct_to_wrong','delta','ci_low','ci_high','mean_gold_nll_per_token','mean_full_vocab_entropy']),
    halting=table('halting_selected.csv',['method','target_validation_depth','average_depth','accuracy','missed_final_corrections','avoided_final_degradations','ci_low','ci_high'],group=None))
for filename in ['validation_selected.json']:
    p=ROOT/'configs'/filename
    if p.exists():report[filename]=json.loads(p.read_text())
for filename in ['math_generation_metrics.json','oracle.json']:
    p=ROOT/'results'/filename
    if p.exists():report[filename]=json.loads(p.read_text())
print(json.dumps(report,indent=2),flush=True)
