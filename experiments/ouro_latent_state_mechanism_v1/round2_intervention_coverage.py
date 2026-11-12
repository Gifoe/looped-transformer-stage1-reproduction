"""Independent complete ROUND 2 coverage and same-device full-logit replay."""
import argparse,hashlib,json,sys,subprocess,os
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python forbidden')
import torch,pyarrow.parquet as pq
from safetensors.torch import load_file
from cohort_runner import bound_hashes,sha,split_data_path
from ledger_guard import require_production_ledger
from round2_intervention_core import CORE,NORM
from round2_intervention_runner import validate_envelope,write_once
P=Path(__file__).resolve();ROOT=P.parents[2].parent;ORIG=ROOT/'orchestration_runs/run-20261107T053919Z-6d01138e';RESUME=ROOT/'orchestration_runs/resume-test-20261111';ANCHOR=ROOT/'orchestration_runs/round2-anchor-20261111'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=('train','validation','test'));ap.add_argument('--shards',required=True);ap.add_argument('--report',required=True);a=ap.parse_args();d=Path(a.shards).resolve();d.relative_to(ROOT.resolve());out=Path(a.report).resolve();out.relative_to(ROOT.resolve());assert not out.exists()
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
 receipt=json.loads((d/'_receipt.json').read_text());assert receipt['status']=='SPLIT_COMPLETE_PENDING_COVERAGE' and receipt['split']==a.split
 cv=(RESUME if a.split=='test' else ORIG)/f'coverage_{a.split}.json';coverage=json.loads(cv.read_text());require_production_ledger(coverage);ledger={x['id']:x for x in coverage['questions']};rows=pq.read_table(split_data_path(a.split)).to_pylist();assert len(rows)==len(ledger)==coverage['counts']['expected_questions'] and set(ledger)=={r['id'] for r in rows}
 current_bound=bound_hashes(a.split);source={str(p):sha(p) for p in (P.with_name('round2_intervention_runner.py'),P.with_name('round2_intervention_core.py'),P.with_name('round2_full_sequence.py'),P.with_name('ledger_guard.py'),P.with_name('cohort_runner.py'),P.with_name('round2_intervention_coverage.py'))};commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parents[2],text=True).strip()
 base={'commit':commit,'source_sha256':source,'bound_hashes':current_bound,'coverage_sha256':sha(cv),'anchor_sha256':sha(ANCHOR/'anchor.json'),'anchor_post_audit_sha256':sha(ANCHOR/'anchor_post_audit.json'),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda};assert receipt['source']==base and receipt['questions']==receipt['expected']==len(rows)
 frozen=sorted((P.parent/'environment_pip_freeze.txt').read_text().splitlines());assert sorted(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,timeout=60).splitlines())==frozen
 demo=json.loads((P.parents[2]/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text());demos=set(demo['ids']);keys={hashlib.sha256(r['id'].encode()).hexdigest() for r in rows};expected_names={'_receipt.json'}|{k+'.json' for k in keys}|{k+'.safetensors' for k in keys};assert {p.name for p in d.iterdir()}==expected_names
 done=0;replay_count=0;ledger_entries=[]
 for row in rows:
  ident=row['id'];q=ledger[ident];assert q['status']=='complete';key=hashlib.sha256(ident.encode()).hexdigest();mp=ORIG/'shards'/a.split/(key+'.json');tp=mp.with_suffix('.safetensors');mh=sha(mp);th=sha(tp);assert mh==q['metadata_sha256'] and th==q['tensor_sha256'];meta=json.loads(mp.read_text());assert meta['id']==ident and meta['sha256']==current_bound
  records=sorted(meta['records'],key=lambda z:z['candidate']);ids=records[0]['encoding']['input_ids'];idhash=hashlib.sha256(json.dumps(ids).encode()).hexdigest();assert all(x['input_ids_sha256']==idhash for x in records);targets=[r['encoding']['target_ids'][0] for r in records]
  identity={**base,'split':a.split,'id':ident,'metadata_path':str(mp),'metadata_sha256':mh,'tensor_path':str(tp),'tensor_sha256':th,'input_ids_sha256':idhash,'is_demonstration':ident in demos};path=d/(key+'.json');j=validate_envelope(path,identity,meta['gold_index'],targets,records);result=j['payload']['result'];z=load_file(str(path.with_suffix('.safetensors')))
  mapping={'baseline_K1':result['baseline_K1']['scores'],'baseline_K2':result['baseline_K2']['scores'],'negative_last_C':result['baseline_K2']['scores']};mapping.update({'core_'+k:result['conditions'][k]['scores'] for k in CORE});mapping.update({'norm_'+k:result['norm_conditions'][k]['scores'] for k in NORM})
  for k,expected in mapping.items():
   logits=z[k].to('cuda');actual=[]
   for target in targets:
    t=torch.tensor([target],device='cuda',dtype=torch.long);actual.append(float(logits.float().log_softmax(-1).gather(-1,t[:,None]).sum()))
   if actual!=expected:raise RuntimeError('zero-tolerance replay mismatch '+ident+' '+k)
   replay_count+=1
  ledger_entries.append({'id':ident,'metadata_sha256':sha(path),'logits_sha256':sha(path.with_suffix('.safetensors')),'status':'complete'});done+=1
  if done%50==0:print('AUDITED',a.split,done,'/',len(rows),flush=True)
 assert bound_hashes(a.split)==current_bound and sha(cv)==base['coverage_sha256'] and all(sha(Path(p))==h for p,h in source.items());assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parents[2],text=True).strip()==commit
 report={'status':'COMPLETE_CUDA_REPLAY_ZERO_TOLERANCE','split':a.split,'expected_questions':len(rows),'complete_questions':done,'replayed_condition_logits':replay_count,'conditions_per_question':11,'questions':ledger_entries,'source':base,'coverage_script_sha256':sha(P),'receipt_sha256':sha(d/'_receipt.json'),'note':'coverage is not a causal mechanism conclusion'}
 write_once(out,report);print('COVERAGE_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
