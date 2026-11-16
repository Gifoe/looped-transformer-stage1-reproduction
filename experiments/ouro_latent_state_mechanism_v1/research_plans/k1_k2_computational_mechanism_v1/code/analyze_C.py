import analyze as base
import json,pathlib,gzip,collections
R=base.R;OUT=base.OUT;rows=[json.loads(p.read_text()) for p in sorted((R/'data/C').glob('*.json'))];rows=[x for x in rows if x['phase']=='heldout'];summary={}
for cohort in ['all','source_cross_correct','source_cross_wrong']:
 s=[x for x in rows if cohort=='all' or (x['source_reference']['prediction']==x['predicted_cross_index'])==(cohort=='source_cross_correct')];clusters=[x['pair_id'] for x in s];conditions={}
 for name in ['restore_attention','attention_norm_control','attention_dose_control','restore_early_R','restore_L18_R']:
  values=[next(r for r in x['records'] if r['condition']==name) for x in s];m=[base.margin(v,x['predicted_cross_index']) for x,v in zip(s,values)];loss=[base.margin(x['source_reference'],x['predicted_cross_index'])-a for x,a in zip(s,m)];cross=[v['prediction']==x['predicted_cross_index'] for x,v in zip(s,values)]
  conditions[name]={'n':len(s),'crossed':sum(cross),'cross_accuracy':base.ci(cross,clusters=clusters),'margin':base.ci(m,clusters=clusters),'source_minus_treatment_margin':base.ci(loss,alpha=.05/3 if name.startswith('restore_') else .05,clusters=clusters),'categories':dict(collections.Counter('crossed' if v['prediction']==x['predicted_cross_index'] else 'original' if v['prediction']==x['original_index'] else 'donor' if v['prediction']==x['donor_index'] else 'other' for x,v in zip(s,values)))}
 contrasts={}
 for name in ['attention_norm_control','attention_dose_control']:
  vals=[]
  for x in s:
   by={r['condition']:r for r in x['records']};vals.append(base.margin(by[name],x['predicted_cross_index'])-base.margin(by['restore_attention'],x['predicted_cross_index']))
  contrasts[name+'_minus_restore_attention_margin']=base.ci(vals,alpha=.05/2,clusters=clusters)
 summary[cohort]={'n':len(s),'base_pairs':len(set(clusters)),'conditions':conditions,'raw_restoration_specificity':contrasts,'subset_scope':'all512 is primary; selecting source correct/wrong is post-intervention descriptive, equal-basepair-weighted bootstrap when cluster sizes differ'}
base.dump('C_SUMMARY.json',summary)
with gzip.open(OUT/'C_QUESTION_RESULTS.jsonl.gz','wt') as f:
 for x in rows:
  for v in [x['source_reference'],x['native_reference'],*x['records']]:
   if 'stage_R_sha256' in v:v['captured_K2_stage_R_sha256']=v.pop('stage_R_sha256')
  f.write(json.dumps(x)+'\n')
print('C_ANALYSIS',len(rows))
