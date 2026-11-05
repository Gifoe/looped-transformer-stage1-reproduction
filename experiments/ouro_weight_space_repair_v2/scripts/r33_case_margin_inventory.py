"""Postrun complete24 descriptive margins/cases, no estimator or new label."""
from pathlib import Path
import json,math,hashlib
from fractions import Fraction
from r28_prefix_task_inventory import selected
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
 a=read('results/r33_ANALYSIS.json');s=read('results/r33_SCORES.json');c=read('results/r33_COMPLETE.json');i=read('results/r33_LABEL_FREE_INTEGRITY.json');assert a['status']=='complete_analyzed_once' and i['controls_matched'] and c['status']=='complete' and c['scores_sha256']==sha('results/r33_SCORES.json')
 requests={r['id']:r for r in read('configs/r28_SANITIZED_REQUESTS.json')['rows']};labels={r['id']:r for r in a['predictions']};wanted=set(requests);assert len(wanted)==len(labels)==len(s['rows'])==24;history={}
 for split in ['discovery','selection']:
  with (ROOT/('results/r7_'+split+'_development.jsonl')).open(encoding='utf8') as f:
   for line in f:
    qid=selected(line,{'id'})['id']
    if qid in wanted:history[qid]=selected(line,{'baseline_scores','choice_char_lengths','prompt_sha256'})
 complete=[];methods=list(s['rows'][0]['candidates'][0]['scores'])+['R6_historical','Fixed_historical'];comparators=['R6_historical','R6_matched','A1','reference_consensus_matched'];changes={b:[] for b in comparators};paired={b:dict(rescue=0,damage=0,net=0) for b in comparators};failures=[]
 for row in s['rows']:
  qid=row['id'];request=requests[qid];label=labels[qid];y=label['gold'];h=history[qid];assert h['choice_char_lengths']==[v['characters'] for v in request['candidates']] and h['prompt_sha256']==hashlib.sha256(request['candidates'][0]['request_pair'][0].encode()).hexdigest()
  vectors={m:[v['scores'][m] for v in row['candidates']] for m in methods if m not in ['R6_historical','Fixed_historical']};vectors.update(R6_historical=h['baseline_scores']['M0_R6'],Fixed_historical=h['baseline_scores']['fixed']);pred={};margins={}
  for name,values in vectors.items():
   assert len(values)==len(row['candidates'])==4 and all(math.isfinite(v) for v in values);p=max(range(len(values)),key=lambda j:values[j]);rival=max((j for j in range(len(values)) if j!=y),key=lambda j:values[j]);assert p==label['predictions'][name];pred[name]=p;exact=Fraction.from_float(values[y])-Fraction.from_float(values[rival]);margins[name]=dict(best_wrong_candidate=rival,gold_minus_best_wrong=values[y]-values[rival],exact_stored_score_difference=str(exact))
  harmed=[]
  for b in comparators:
   rescue=pred['primary']==y and pred[b]!=y;damage=pred['primary']!=y and pred[b]==y;paired[b]['rescue']+=int(rescue);paired[b]['damage']+=int(damage);changes[b].append(margins['primary']['gold_minus_best_wrong']-margins[b]['gold_minus_best_wrong'])
   if damage:harmed.append(b)
  candidates=[]
  for j,(v,source) in enumerate(zip(row['candidates'],request['candidates'])):
   candidates.append(dict(candidate=j,continuation=source['request_pair'][1],characters=v['characters'],tokens=v['tokens'],native_late=v['late'],all19_original_position_LP_pieces=v['pieces'],stored_scores={name:values[j] for name,values in vectors.items()}))
  record=dict(id=qid,split=row['split'],gold=y,original_Q_prompt=request['candidates'][0]['request_pair'][0],prompt_sha256=h['prompt_sha256'],predictions=pred,gold_rival_margins=margins,primary_damage_relative_to=harmed,candidates=candidates);complete.append(record)
  if harmed:failures.append(record)
 for b,p in paired.items():p['net']=p['rescue']-p['damage']
 summary={b:dict(increased=sum(v>0 for v in vs),unchanged=sum(v==0 for v in vs),decreased=sum(v<0 for v in vs),mean=math.fsum(vs)/len(vs),minimum=min(vs),maximum=max(vs)) for b,vs in changes.items()}
 assert paired['R6_historical']==a['pairs']['R6_historical']['primary'] and paired['R6_matched']==a['pairs']['R6_matched']['primary']
 output=dict(status='complete_postrun_all24_margin_and_failure_inventory',n=24,candidates=96,postrun_descriptive=True,model_loaded=False,solver_run=False,new_GPU_seconds=0,paired_primary_vs_comparators=paired,all24_gold_margin_change_summary=summary,all24_rows=complete,failure_cases_union=failures,historical_R6_LP_scope='R6 is an already-released candidate-level score, not a fabricated per-token normalized distribution. All19 actual token LP primitives are retained separately.',label_scope='Only labels and predictions already released by the completed frozen R33 analyzer. Selective historical lookup reads only baseline scores of these same24, no other gold or heldout row.',interpretation='Arithmetic observations only, not a new selector or causal mechanism attribution; all24 retained, no tuning or selected-case estimator.',input_sha256={p:sha(p) for p in ['results/r33_SCORES.json','results/r33_ANALYSIS.json','results/r33_COMPLETE.json','configs/r28_SANITIZED_REQUESTS.json']},source_sha256=sha('scripts/r33_case_margin_inventory.py'))
 with (ROOT/'results/r33_ALL24_MARGIN_FAILURE_INVENTORY.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(pairs=paired,margin_changes=summary,all_damage_case_ids=[v['id'] for v in failures])))
if __name__=='__main__':main()
