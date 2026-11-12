import json,pathlib,statistics,hashlib
R=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113/results/analysis_train64');src=R/'rows.jsonl';rows=[json.loads(x) for x in src.read_text().splitlines()];assert len(rows)==64;out={}
for group in ('all64','rescue','damage','stable_correct','stable_wrong','rescue_no_tie'):
 rs=[r for r in rows if group=='all64' or (r['actual_group_4090']=='rescue' and r['natural_no_tie'] if group=='rescue_no_tie' else r['actual_group_4090']==group)];out[group]={'n':len(rs),'branches':{}}
 for b in ('a','b'):
  details=[]
  for r in rs:
   c=r['conditions'];p=c['P'];padd=c[f'P_plus_{b}_main'];n=c['N'];nminus=c[f'N_minus_{b}_main'];d=r['contrasts'][b];details.append({'id':r['id'],'P_to_Pplus':('rescue' if not p['correct'] and padd['correct'] else 'damage' if p['correct'] and not padd['correct'] else 'unchanged_correct' if p['correct'] else 'unchanged_wrong'),'N_to_Nminus':('rescue' if not n['correct'] and nminus['correct'] else 'damage' if n['correct'] and not nminus['correct'] else 'unchanged_correct' if n['correct'] else 'unchanged_wrong'),'gap_restoration_positive':d['rest']['allwrong']>0,'gap_loss_positive':d['loss']['allwrong']>0})
  out[group]['branches'][b]={'same_id_both_positive_gaps':[d['id'] for d in details if d['gap_restoration_positive'] and d['gap_loss_positive']],'same_id_decision_restore_and_loss':[d['id'] for d in details if d['P_to_Pplus']=='rescue' and d['N_to_Nminus']=='damage'],'P_to_Pplus_counts':{k:sum(d['P_to_Pplus']==k for d in details) for k in ('rescue','damage','unchanged_correct','unchanged_wrong')},'N_to_Nminus_counts':{k:sum(d['N_to_Nminus']==k for d in details) for k in ('rescue','damage','unchanged_correct','unchanged_wrong')},'per_question':details}
p=R/'paired_decisions.json';assert not p.exists();p.write_text(json.dumps({'rows_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'groups':out},indent=2)+'\n');print(hashlib.sha256(p.read_bytes()).hexdigest())
