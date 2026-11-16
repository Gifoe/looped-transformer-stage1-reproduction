"""Gold used only after frozen forwards; full Train1094 paired analysis."""
import pathlib,json,datetime,hashlib,collections,gzip
import numpy as np
import os
from E6A_METHODS import infer,fixed_specs
R=pathlib.Path(os.environ.get('CATC_ROOT','/root/ouro-catc-sol-20261116'));f=json.loads((R/'FREEZE.json').read_text());labels={q['id']:q['gold'] for q in map(json.loads,(R/'ANALYSIS_LABELS.jsonl' if (R/'ANALYSIS_LABELS.jsonl').exists() else pathlib.Path('/root/ouro-computation-sol-20261116/prior/analysis_inputs/analysis_labels.jsonl')).read_text().splitlines()) if q['split']=='train'};rows=[json.loads((R/'results'/f'{id}.json').read_text()) for id in f['ids']];assert len(rows)==1094
rng=np.random.default_rng(20261116);indices=rng.integers(0,1094,(5000,1094),dtype=np.int32)
def ci(v,alpha=.05):
 v=np.array(v,dtype=np.float64);s=v[indices].mean(1);return {'mean':float(v.mean()),'ci95':np.quantile(s,[.025,.975]).tolist(),'lower_bonferroni30':float(np.quantile(s,.05/30)),'n':len(v)}
def gap(s,g):v=s['candidate_logits'];return v[g]-max(x for i,x in enumerate(v) if i!=g)
allmethods={};allmargins={};summary={};clip={};cost={}
for method in f['methods']:
 name=method['name'];rr=[next(v for v in x['records'] if v['name']==name) for x in rows];clip[name]={'sites':sum(len(x['norms']) for x in rr),'clipped':sum(v[2]<1 for x in rr for v in x['norms']),'max_realized_relative_norm':max((v[3]/v[4] for x in rr for v in x['norms']),default=0),'mean_realized_relative_norm':float(np.mean([v[3]/v[4] for x in rr for v in x['norms']])) if name!='T0' else 0};cost[name]={'sum_forward_seconds':sum(x['seconds'] for x in rr),'mean_forward_seconds':float(np.mean([x['seconds'] for x in rr])),'max_allocated_bytes':max(x['peak_allocated_bytes'] for x in rr)}
 for depth in ['2','4']:
  n=name+'_K'+depth;allmethods[n]=np.array([x['scores'][depth]['prediction']==labels[q['id']] for q,x in zip(rows,rr)],dtype=int);allmargins[n]=np.array([gap(x['scores'][depth],labels[q['id']]) for q,x in zip(rows,rr)])
for depth in ['1','3']:
 n='T0_K'+depth;allmethods[n]=np.array([q['records'][0]['scores'][depth]['prediction']==labels[q['id']] for q in rows],dtype=int);allmargins[n]=np.array([gap(q['records'][0]['scores'][depth],labels[q['id']]) for q in rows])
for name in rows[0]['records'][0]['output_baselines']:
 allmethods[name]=np.array([q['records'][0]['output_baselines'][name]['prediction']==labels[q['id']] for q in rows],dtype=int);allmargins[name]=np.array([gap(q['records'][0]['output_baselines'][name],labels[q['id']]) for q in rows])
for name,spec in fixed_specs().items():
 n='E6_'+name;pred=[infer([q['records'][0]['scores'][str(k)]['candidate_logits'] for k in range(1,5)],spec) for q in rows];allmethods[n]=np.array([v['prediction']==labels[q['id']] for q,v in zip(rows,pred)],dtype=int);allmargins[n]=np.array([gap({'candidate_logits':v['scores']},labels[q['id']]) for q,v in zip(rows,pred)])
base=allmethods['T0_K4'];best=max([k for k in allmethods if not k.startswith('T') or k.startswith('T0')],key=lambda k:int(allmethods[k].sum()))
for name,v in allmethods.items():summary[name]={'correct':int(v.sum()),'n':1094,'accuracy':float(v.mean()),'pp_change_vs_K4':float((v-base).mean()*100),'rescue':int(((v==1)&(base==0)).sum()),'damage':int(((v==0)&(base==1)).sum()),'paired_accuracy_vs_K4':ci(v-base),'paired_margin_vs_K4':ci(allmargins[name]-allmargins['T0_K4']),'vs_strongest_fixed_output':ci(v-allmethods[best])}
for name,v in allmethods.items():
 ref='T0_K2' if name.endswith('_K2') else 'T0_K4';baseline=allmethods[ref];summary[name]['same_depth_native_reference']=ref;summary[name]['same_depth_rescue']=int(((v==1)&(baseline==0)).sum());summary[name]['same_depth_damage']=int(((v==0)&(baseline==1)).sum());summary[name]['paired_accuracy_vs_same_depth']=ci(v-baseline)
gates={}
for i in range(1,7):
 name=f'T{i}_structured_K4';controls=[f'T{i}_{c}_K4' for c in ['unrelated','random','prefix','scaling']];accuracy={c:ci(allmethods[name]-allmethods[c]) for c in ['T0_K4',*controls]};margins={c:ci(allmargins[name]-allmargins[c]) for c in ['T0_K4',*controls]};gain=allmethods[name].sum()-base.sum();primary=all(v['lower_bonferroni30']>0 for v in accuracy.values()) and allmethods[name].sum()>allmethods[best].sum();secondary=all(v['lower_bonferroni30']>0 for v in margins.values()) and gain>=0;gates[name]={'accuracy_contrasts':accuracy,'margin_contrasts':margins,'primary_pass':bool(primary),'mechanism_specific_secondary_pass':bool(secondary),'gate_to_C':bool(primary or secondary)}
status={'phaseB':'COMPLETE','gate_to_C':any(x['gate_to_C'] for x in gates.values()),'scope':'fullTrain1094 development, no Test, no independent generalization claim','outcome':'PROMISING_DIRECTION_REQUIRES_NEXT_PHASE' if any(x['gate_to_C'] for x in gates.values()) else 'C_NEGATIVE_STOP_BRANCH','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name,obj in [('SUMMARY.json',summary),('GATES.json',gates),('RUNTIME_AND_MEMORY.json',cost),('CORRECTION_NORMS.json',clip),('STATUS.json',status)]: (R/name).write_text(json.dumps(obj,indent=2))
with (R/'TRAINING_FREE_ALL_ROWS.jsonl').open('w') as out:
 for x in rows:x['gold']=labels[x['id']];out.write(json.dumps(x)+'\n')
print(json.dumps(status),flush=True)
