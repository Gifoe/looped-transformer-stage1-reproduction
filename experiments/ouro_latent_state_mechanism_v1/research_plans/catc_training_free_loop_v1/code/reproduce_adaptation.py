"""CPU-only replay of independent/validation/FAS summaries and selected coefficients."""
import pathlib,argparse,json,gzip,tempfile,shutil,subprocess,os,sys,numpy as np
p=argparse.ArgumentParser();p.add_argument('--release',default='.');a=p.parse_args();pub=pathlib.Path(a.release).resolve()
if (pub/'SHARDED_ARTIFACTS.json').exists():
 from assemble_release import assemble
 assemble(pub)
P=pub/'phaseC'
with tempfile.TemporaryDirectory(prefix='catc-adaptation-replay-') as tmp:
 r=pathlib.Path(tmp);(r/'sciq_results').mkdir();validation=[]
 with gzip.open(P/'ADAPTATION_QUESTION_ROWS.jsonl.gz','rt') as f:
  for line in f:
   q=json.loads(line);dataset=q.pop('dataset')
   if dataset=='validation':validation.append(q)
   else:(r/'sciq_results'/f'{q["id"]}.json').write_text(json.dumps(q))
 (r/'FINAL_VALIDATION_ROWS.json').write_text(json.dumps(validation));shutil.copy2(P/'SCIQ_DATA_AUDIT.json',r/'SCIQ_DATA_AUDIT.json');env=os.environ.copy();env['CATC_ADAPT_ROOT']=str(r);subprocess.run([sys.executable,str(pub/'code/summarize_adaptation.py')],env=env,check=True);assert json.loads((r/'ADAPTATION_SUMMARY.json').read_text())==json.loads((P/'ADAPTATION_SUMMARY.json').read_text());print('ADAPTATION_SUMMARY_EXACT_PASS')
with gzip.open(P/'FULL_ANSWER_QUESTION_ROWS.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
assert len(rows)==299;gold=np.array([q['gold'] for q in rows]);rng=np.random.default_rng(20261116);index=rng.integers(0,299,(5000,299));acc={n:np.array([q['methods'][n]['prediction']==g for q,g in zip(rows,gold)],dtype=int) for n in rows[0]['methods']};base=acc['nativeK4'];summary={}
for n,v in acc.items():
 d=v-base;summary[n]={'n':299,'correct':int(v.sum()),'accuracy':float(v.mean()),'pp_vs_nativeK4':float(d.mean()*100),'rescue':int(((v==1)&(base==0)).sum()),'damage':int(((v==0)&(base==1)).sum()),'paired_ci95':np.quantile(d[index].mean(1),[.025,.975]).tolist()}
assert summary==json.loads((P/'FULL_ANSWER_SUMMARY.json').read_text());print('FULL_ANSWER_SUMMARY_EXACT_PASS')
for q in rows:
 for n,v in q['methods'].items():
  assert v['prediction']==int(np.argmax(v['scores'])) and v['ties']==[i for i,x in enumerate(v['scores']) if x==max(v['scores'])]
  for i,z in enumerate(v['options']):
   error=abs(sum(z['token_logprobs'])-z['sum']);bound=4*np.finfo(np.float32).eps*max(1,len(z['token_logprobs']))*max(1,sum(abs(t) for t in z['token_logprobs']));assert error<=bound;assert z['sum']/q['answer_char_lengths'][i]==z['per_character']==v['scores'][i]
selected=json.loads((P/'VALIDATION_MODEL_SELECTION.json').read_text());models={n:json.loads((P/n/'COMPLETE.json').read_text()) for n in ['catc14','value7','routing7']};best=min(models,key=lambda n:(models[n]['validation_CE'],models[n]['parameters'],models[n]['best_step'],n));assert selected['selected_name']==best and selected['selected']==models[best]
for n in ['catc14','unstructured14','value7','routing7']:
 v=json.loads((P/n/'COMPLETE.json').read_text());hist=json.loads((P/n/'history.json').read_text());assert len(hist)==v['steps'] and hist[-1]['evals']==v['training_forward_evaluations'];alpha=[0.]*14 if v['best_step']==0 else next(x['alpha_after'] for x in hist if x['step']==v['best_step']);assert alpha==v['alpha'];assert max(abs(x) for x in alpha)<=.25
print('SELECTED_PARAMETERS_AND_FAS_ARITHMETIC_PASS')
