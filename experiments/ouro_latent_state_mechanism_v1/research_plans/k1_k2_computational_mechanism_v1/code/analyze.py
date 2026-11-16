"""CPU-only analysis; base-pair clustered synthetic inference, question-paired ARC inference."""
import pathlib,json,collections,hashlib,sys,gzip
import numpy as np
import os
R=pathlib.Path(os.environ.get('OURO_COMPUTE_ROOT','/root/ouro-computation-sol-20261116'));OUT=R/'analysis';OUT.mkdir(exist_ok=True)
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2))
def ci(a,alpha=.05,clusters=None):
 a=np.asarray(a,dtype=np.float64)
 if not len(a):return None
 if clusters is not None:
  gs=sorted(set(clusters));a=np.array([np.mean(a[np.array(clusters)==g]) for g in gs])
 gen=np.random.default_rng(20261116);samples=[]
 for k in range(0,5000,250):
  index=gen.integers(0,len(a),(min(250,5000-k),len(a)));samples.extend(a[index].mean(axis=1).tolist())
 return {'n_units':len(clusters) if clusters is not None else len(a),'n_resampling_units':len(a),'mean':float(a.mean()),'interval':np.quantile(samples,[alpha/2,1-alpha/2]).tolist(),'confidence':1-alpha,'replicates':5000,'seed':20261116}
def margin(r,g):v=r['candidate_logits'];return v[g]-max(x for i,x in enumerate(v) if i!=g)
def group(k1,k2,g):return 'rescue' if k1!=g and k2==g else 'damage' if k1==g and k2!=g else 'stable_correct' if k1==g else 'stable_wrong'
def analyzeA():
 labels={(x['split'],x['id']):x['gold'] for x in map(json.loads,(R/'prior/analysis_inputs/analysis_labels.jsonl').read_text().splitlines())};rows=[]
 for p in sorted((R/'data/A').glob('*.json')):
  x=json.loads(p.read_text());z={r['condition']:r for r in x['records']};g=labels[x['split'],x['id']];rows.append({**x,'by_condition':z,'gold':g,'group':group(z['K1']['prediction'],z['native']['prediction'],g)})
 fs=json.loads((R/'A_FREEZE.json').read_text());summaries={}
 for cohort in ['development64','train1030','train1094','validation299','arc_structural8']:
  ids=set(fs['development64']) if cohort=='development64' else set(fs['confirmation_train1030_order']) if cohort=='train1030' else set(x['id'] for x in rows if x['split']=='train') if cohort=='train1094' else set(fs['validation299_order']) if cohort=='validation299' else set(x['id'] for x in json.loads((R/'ARC_STRUCTURE_FREEZE.json').read_text())['cases'])
  selected=[x for x in rows if x['id'] in ids];bygroup={}
  for label in ['all','rescue','damage','stable_correct','stable_wrong']:
   s=selected if label=='all' else [x for x in selected if x['group']==label];cond={}
   for c in ['K1','native',*fs['conditions']]:
    if not s:continue
    correct=[int(x['by_condition'][c]['prediction']==x['gold']) for x in s];nativecorrect=[int(x['by_condition']['native']['prediction']==x['gold']) for x in s]
    cond[c]={'n':len(s),'correct':sum(correct),'accuracy':ci(correct),'native_agreement':sum(x['by_condition'][c]['prediction']==x['by_condition']['native']['prediction'] for x in s),'native_correct_lost':sum(a and not b for a,b in zip(nativecorrect,correct)),'native_wrong_rescued':sum(b and not a for a,b in zip(nativecorrect,correct)),'ties':sum(len(x['by_condition'][c]['ties'])>1 for x in s),'mean_gold_margin':float(np.mean([margin(x['by_condition'][c],x['gold']) for x in s])),'paired_accuracy_vs_native':ci(np.array(correct)-np.array(nativecorrect)),'paired_margin_vs_native':ci([margin(x['by_condition'][c],x['gold'])-margin(x['by_condition']['native'],x['gold']) for x in s],alpha=.05/3 if c.startswith('freeze_') else .05)}
    if c.startswith('freeze_'):
     region=c.split('_',1)[1];cond[c]['margin_vs_norm_control']=ci([margin(x['by_condition'][c],x['gold'])-margin(x['by_condition']['norm_'+region],x['gold']) for x in s]);cond[c]['margin_vs_dose_control']=ci([margin(x['by_condition'][c],x['gold'])-margin(x['by_condition']['dose_'+region],x['gold']) for x in s])
   bygroup[label]={'n':len(s),'ids':[x['id'] for x in s],'conditions':cond}
  summaries[cohort]={'expected_n':len(ids),'observed_n':len(selected),'complete':len(selected)==len(ids),'groups':bygroup}
 dump('A_SUMMARY.json',summaries)
 with gzip.open(OUT/'A_QUESTION_RESULTS.jsonl.gz','wt') as f:
  for x in rows:
   x.pop('by_condition');
   for z in x['records']:z['captured_K2_stage_R_sha256']=z.pop('stage_R_sha256')
   f.write(json.dumps(x)+'\n')
 return summaries

def analyzeB():
 qs={q['id']:q for q in map(json.loads,(R/'B_INPUTS.jsonl').read_text().splitlines())};units=[]
 for p in sorted((R/'data/B').glob('*.json')):
  for u in json.loads(p.read_text())['units']:
   u['by_condition']={r['condition']:r for r in u['records']};units.append(u)
 native_predictions={u['receiver']:u['by_condition']['native']['prediction'] for u in units}
 summaries={};controls=['same_variable','unrelated','dose','readout_copy'];conditions=['K1','native','text_edit_K1','text_edit_K2','source_boundary',*controls,'descendant_edit']
 for cohort in ['development','heldout','heldout_template0','heldout_template1','heldout_original_direction','heldout_donor_direction','heldout_labels_normal','heldout_labels_reverse']:
  selected=[u for u in units if u['split']==('development' if cohort=='development' else 'heldout') and (not cohort.startswith('heldout_template') or u['template']==int(cohort[-1])) and (not cohort.endswith('_direction') or u['receiver'].endswith('_'+cohort.split('_')[1])) and (not cohort.startswith('heldout_labels') or u['reverse_labels']==cohort.endswith('reverse'))];clusters=[u['pair_id'] for u in selected];cond={}
  for c in conditions:
   if not selected:continue
   crossed=[u['by_condition'][c]['prediction']==u['predicted_cross_index'] for u in selected];correct=[u['by_condition'][c]['prediction']==qs[u['text_edit'] if c.startswith('text_edit') else u['receiver']]['correct_index'] for u in selected];margins=[margin(u['by_condition'][c],u['predicted_cross_index']) for u in selected];classes=collections.Counter('crossed' if u['by_condition'][c]['prediction']==u['predicted_cross_index'] else 'original' if u['by_condition'][c]['prediction']==u['original_index'] else 'donor' if u['by_condition'][c]['prediction']==u['donor_index'] else 'other' for u in selected)
   cond[c]={'n':len(selected),'crossed':sum(crossed),'outcome_categories':dict(classes),'cross_accuracy':ci(crossed,clusters=clusters),'semantic_native_or_text_correct':sum(correct),'cross_margin':ci(margins,clusters=clusters),'ties':sum(len(u['by_condition'][c]['ties'])>1 for u in selected)}
  for name in conditions:
   if name not in cond:continue
   novel=[u['by_condition'][name]['prediction']==u['predicted_cross_index'] and native_predictions[u['receiver']]!=u['predicted_cross_index'] and native_predictions[u['donor']]!=u['predicted_cross_index'] for u in selected]
   cond[name]['novel_cross_neither_native_output']=sum(novel);cond[name]['novel_cross_rate']=ci(novel,clusters=clusters)
  specificity={c:ci([margin(u['by_condition']['source_boundary'],u['predicted_cross_index'])-margin(u['by_condition'][c],u['predicted_cross_index']) for u in selected],alpha=.05/4,clusters=clusters) for c in controls} if selected else {}
  native_gain=ci([int(u['by_condition']['native']['prediction']==qs[u['receiver']]['correct_index'])-int(u['by_condition']['K1']['prediction']==qs[u['receiver']]['correct_index']) for u in selected],clusters=clusters) if selected else None
  summaries[cohort]={'n_units':len(selected),'base_pairs':len(set(clusters)),'conditions':cond,'source_margin_specificity':specificity,'native_K2_minus_K1_accuracy':native_gain}
 if len([u for u in units if u['split']=='heldout'])==512:
  strata=['heldout_template0','heldout_template1','heldout_original_direction','heldout_donor_direction','heldout_labels_normal','heldout_labels_reverse'];stratified=all(summaries[s]['conditions']['source_boundary']['cross_accuracy']['mean']>=.5 and summaries[s]['conditions']['source_boundary']['cross_margin']['mean']>0 for s in strata);specific=all(v['interval'][0]>0 for v in summaries['heldout']['source_margin_specificity'].values());summaries['frozen_gate']={'stratified_majority_and_margin':stratified,'all4_control_specificity':specific,'PASS':stratified and specific}
 dump('B_SUMMARY.json',summaries)
 examples={'positive':[],'negative':[]}
 for u in units:
  target='positive' if u['by_condition']['source_boundary']['prediction']==u['predicted_cross_index'] else 'negative'
  if len(examples[target])<8:examples[target].append({k:v for k,v in u.items() if k!='by_condition'})
 dump('B_EXAMPLES.json',examples)
 with gzip.open(OUT/'B_QUESTION_RESULTS.jsonl.gz','wt') as f:
  for u in units:
   u.pop('by_condition')
   for z in u['records']:z['captured_K2_stage_R_sha256']=z.pop('stage_R_sha256')
   f.write(json.dumps(u)+'\n')
 return summaries
if __name__=='__main__':
 which=sys.argv[1] if len(sys.argv)>1 else 'both'
 if which in ('A','both'):a=analyzeA();print('A',[(k,x['observed_n']) for k,x in a.items()])
 if which in ('B','both'):b=analyzeB();print('B',[(k,x.get('n_units')) for k,x in b.items()])
