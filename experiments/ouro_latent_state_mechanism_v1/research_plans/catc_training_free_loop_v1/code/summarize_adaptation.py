"""CPU-only full-population statistics, no fitting or dataset selection."""
import pathlib,json,numpy as np,os,gzip,datetime
R=pathlib.Path(os.environ.get('CATC_ADAPT_ROOT') or '/root/ouro-catc-sol-20261116/phaseC')
def paired(a,b):
 d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64);rng=np.random.default_rng(20261116);sample=[]
 for k in range(0,5000,250):sample.extend(d[rng.integers(0,len(d),(min(250,5000-k),len(d)))].mean(1).tolist())
 return {'mean':float(d.mean()),'ci95':np.quantile(sample,[.025,.975]).tolist(),'lower_bonferroni6':float(np.quantile(sample,.05/6)),'n':len(d)}
def extract(q):
 r={};native=q['methods']['native'];
 for d,z in native['scores'].items():r['native_K'+d]=z
 for name,v in q['methods'].items():
  if name=='native':continue
  for d in ['2','4']:r[name+'_K'+d]=v['scores'][d]
 for n,v in native.get('output_baselines',{}).items():r['output_'+n]=v
 for n,v in q['output_only'].items():r['E6_'+n]={'prediction':v['prediction'],'candidate_logits':v['scores'],'ties':[i for i,x in enumerate(v['scores']) if x==max(v['scores'])]}
 v1=np.array(native['scores']['1']['candidate_logits'],dtype=np.float32);v4=np.array(native['scores']['4']['candidate_logits'],dtype=np.float32);v=v4+.5*(v4-v1);r['LoopCD_fixed_0.5']={'prediction':int(v.argmax()),'candidate_logits':v.tolist(),'ties':[i for i,x in enumerate(v) if x==v.max()]};return r
summary={};datasets={'validation':json.loads((R/'FINAL_VALIDATION_ROWS.json').read_text()),'sciq':[json.loads(p.read_text()) for p in sorted((R/'sciq_results').glob('*.json'))]}
assert len(datasets['validation'])==299 and len(datasets['sciq'])==json.loads((R/'SCIQ_DATA_AUDIT.json').read_text())['evaluated_n']
for dataset,rows in datasets.items():
 frames=[extract(q) for q in rows];gold=np.array([q['gold'] for q in rows]);acc={n:np.array([r[n]['prediction']==g for r,g in zip(frames,gold)],dtype=int) for n in frames[0]};native=acc['native_K4'];methods={}
 for name,v in acc.items():
  methods[name]={'correct':int(v.sum()),'n':len(v),'accuracy':float(v.mean()),'pp_vs_nativeK4':float((v-native).mean()*100),'rescue':int(((v==1)&(native==0)).sum()),'damage':int(((v==0)&(native==1)).sum()),'paired_vs_nativeK4':paired(v,native),'new_correct_all_native_loops_wrong':int(((v==1)&np.logical_and.reduce([acc['native_K'+str(d)]==0 for d in range(1,5)])).sum())}
 contrasts={}
 for name in ['catc14_K4','validation_selected_structured_K4']:
  contrasts[name]={ref:paired(acc[name],acc[ref]) for ref in ['native_K4','E6_conservative','unstructured14_K4']}
 summary[dataset]={'methods':methods,'primary_and_secondary_contrasts':contrasts,'independent_benefit_gate':{n:all(v['lower_bonferroni6']>0 for v in vals.values()) for n,vals in contrasts.items()},'scope':'Validation usedforselectionandexposed' if dataset=='validation' else 'IndependentSciQvalidation,fulloutcome-independentfilteredpopulation; noSciQselection'}
(R/'ADAPTATION_SUMMARY.json').write_text(json.dumps(summary,indent=2))
with gzip.open(R/'ADAPTATION_QUESTION_ROWS.jsonl.gz','wt') as f:
 for dataset,rows in datasets.items():
  for q in rows:f.write(json.dumps({'dataset':dataset,**q})+'\n')
print('ADAPTATION_STATS',[(n,[(m,v['correct']) for m,v in s['methods'].items() if m in ['native_K4','fixedVplus_K4','catc14_K4','routing7_K4','validation_selected_structured_K4','E6_conservative']]) for n,s in summary.items()])
