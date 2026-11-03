"""Independent cross-artifact integrity audit after COMPLETE frozen evaluation."""
import argparse,collections,csv,hashlib,json,math,struct
from runtime import ROOT,save

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text(encoding='utf8').splitlines()]
def csvrows(p):
 with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def qhash(q):return hashlib.sha256(' '.join(q.casefold().split()).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--verify-model',action='store_true');a=ap.parse_args()
 # Receipt gates precede access to any held-out prediction/label data.
 assert read('logs/FROZEN_EVALUATION_PIPELINE.json')['status']=='complete'
 decision=read('configs/STAGE_C_DECISION.json');recipe=read('configs/frozen_baseline_recipe.json');final=read('configs/FINAL_LOCK.json');candidate=read('configs/CANDIDATE_LOCK.json')
 assert decision['halt_weight_editing'] and decision['primary_weight_edit'] is None
 recipe_sha=digest(ROOT/'configs/frozen_baseline_recipe.json');decision_sha=digest(ROOT/'configs/STAGE_C_DECISION.json')
 assert final['test_evaluation_authorized'] and final['recipe_sha256']==candidate['recipe_sha256']==recipe_sha
 assert final['decision_sha256']==candidate['decision_sha256']==decision_sha
 assert decision['review_sha256']==digest(ROOT/decision['review_path'])
 assert read('data/VALIDATION_MANIFEST_AUDIT.json')['lock_sha256']==digest(ROOT/'configs/CANDIDATE_LOCK.json')
 assert read('data/TEST_MANIFEST_AUDIT.json')['lock_sha256']==digest(ROOT/'configs/FINAL_LOCK.json')
 assert final['frozen_unix']<=read('data/TEST_DOWNLOAD.json')['downloaded_unix']
 completions={s:read(f'results/frozen_{s}/COMPLETE.json') for s in ['selection','validation','test']}
 for s,n in [('selection',192),('validation',299),('test',1172)]:assert completions[s]['status']=='complete' and completions[s]['n']==n
 raw_metadata=read('sources/arc_lfs_metadata.json')
 for split in ['train','validation','test']:
  expected=next(x['lfs']['oid'] for x in raw_metadata if x['path']==f'ARC-Challenge/{split}-00000-of-00001.parquet')
  assert digest(ROOT/f'data/raw/{split}.parquet')==expected
 gate=read('results/original_call_parity/COMPLETE.json')
 assert all(gate[k] for k in ['exact_original_path_parity','original_repeat_exact','exact_intermediate_depth_parity'])
 assert read('results/layer_screen_original/NOOP_GATE_SUMMARY.json')['status']=='passed'
 assert read('results/layer_screen_original/BOUNDARY_READOUT_IDENTITY.json')['passed']
 assert read('results/shared_confirmation_original/NOOP_PARITY.json')['passed']
 demos=read('data/fixed_demonstrations.json');assert len(demos['documents'])==len(set(demos['ids']))==25
 discover=rows('data/discovery.jsonl');select=rows('data/selection.jsonl')
 dq={qhash(r['question']) for r in discover};sq={qhash(r['question']) for r in select};fq={qhash(d['question']) for d in demos['documents']}
 assert len(dq)==len(sq)==192 and not dq&sq and not (dq|sq)&fq
 train_audit=read('data/TRAIN_MANIFEST_AUDIT.json')
 for s in ['discovery','selection']:assert digest(ROOT/f'data/{s}.jsonl')==train_audit['cohorts'][s]['sha256']
 codehash=hashlib.sha256(''.join(digest(ROOT/'scripts'/n) for n in ['runtime.py','ouro_cached_scoring.py','run_frozen_eval.py']).encode()).hexdigest()
 discovery_codehash=hashlib.sha256(''.join(digest(ROOT/'scripts'/n) for n in ['runtime.py','ouro_cached_scoring.py','run_discovery.py']).encode()).hexdigest()
 discovery_receipt=read('results/discovery_original/COMPLETE_192.json')
 assert discovery_receipt['status']=='complete' and discovery_receipt['n']==192 and discovery_receipt['implementation_sha256']==discovery_codehash
 by_split={};choice_counts={}
 for split,n in [('discovery',192),('selection',192),('validation',299),('test',1172)]:
  manifest=discover if split=='discovery' else rows(f'data/{split}.jsonl')
  path='results/discovery_original/scores.jsonl' if split=='discovery' else f'results/frozen_{split}/scores.jsonl'
  manifest_sha=digest(ROOT/f'data/{split}.jsonl')
  scored=rows(path);assert len(manifest)==len(scored)==n and len({r['id'] for r in scored})==n
  assert [r['id'] for r in scored]==[r['id'] for r in manifest]
  if split in ['validation','test']:assert [r['source_index'] for r in manifest]==list(range(n))
  for m,r in zip(manifest,scored):
   assert r['prompt_sha256']==m['prompt_sha256'] and r['gold']==m['gold'] and r['cached'] is False
   assert hashlib.sha256(m['prompt'].encode()).hexdigest()==m['prompt_sha256']
   assert len(m['request_pairs'])==len(m['choices'])==len(m['choice_char_lengths'])
   assert all(pair[0]==m['prompt'] for pair in m['request_pairs'])
   assert [len(choice) for choice in m['choices']]==m['choice_char_lengths']
   assert m['fewshot_count']==25 and m['primary_metric']=='acc_norm'
   assert r['choice_char_lengths']==m['choice_char_lengths'] and r['fewshot']==25
   if split=='discovery':assert r['implementation_sha256']==discovery_codehash
   if split!='discovery':
    assert r['implementation_sha256']==completions[split]['implementation_sha256']==codehash
    assert r['recipe_sha256']==recipe_sha and r['data_sha256']==completions[split]['data_sha256']==manifest_sha
    assert set(r['methods'])==set(recipe['methods'])
   for name,value in r['methods'].items():
    assert len(value['sum_scores'])==len(value['normalized_scores'])==len(m['choices'])
    norm=[v/max(length,1) for v,length in zip(value['sum_scores'],m['choice_char_lengths'])]
    assert len(norm)==len(m['choices']) and all(math.isfinite(x) for x in norm)
    assert all(abs(x-y)<1e-12 for x,y in zip(norm,value['normalized_scores']))
    pred=max(range(len(norm)),key=norm.__getitem__);gold=m['gold']
    assert pred==value['prediction'] and value['correct']==(pred==gold)
    assert abs(norm[gold]-max(v for j,v in enumerate(norm) if j!=gold)-value['margin'])<1e-12
  by_split[split]={r['id']:r for r in scored};choice_counts[split]=dict(collections.Counter(len(r['choices']) for r in manifest))
 summary=csvrows('evaluation_summary.csv')
 expected_summary={(split,method) for split in by_split for method in recipe['methods']}
 assert len(summary)==len(expected_summary)==44 and {(r['split'],r['method']) for r in summary}==expected_summary
 for r in summary:
  part=list(by_split[r['split']].values());correct=sum(x['methods'][r['method']]['correct'] for x in part)
  assert int(r['n'])==len(part) and int(r['correct'])==correct and abs(float(r['accuracy'])-correct/len(part))<1e-12
  assert abs(float(r['mean_gold_margin'])-sum(x['methods'][r['method']]['margin'] for x in part)/len(part))<1e-12
 statistics=csvrows('paired_statistics.csv');assert len(statistics)==32
 expected_comparisons=[('native_k3','native_k4'),('native_k4','native_k5'),('native_k5','native_k6'),('native_k4','loopcd_k4_fixed'),('native_k4','loopcd_k4_adaptive')]+[('native_k5',f'normalized_chord_{x:g}') for x in recipe['readout_chord_coefficients']]
 expected_stat_keys={(split,before,after) for split in by_split for before,after in expected_comparisons}
 assert {(c['split'],c['before'],c['after']) for c in statistics}==expected_stat_keys and len(expected_stat_keys)==32
 from scipy.stats import binomtest
 for c in statistics:
  part=list(by_split[c['split']].values());before=c['before'];after=c['after']
  rescues=sum(not r['methods'][before]['correct'] and r['methods'][after]['correct'] for r in part)
  damage=sum(r['methods'][before]['correct'] and not r['methods'][after]['correct'] for r in part)
  assert (rescues,damage)==(int(c['rescues']),int(c['regressions'])) and int(c['net_correct'])==rescues-damage
  assert int(c['n'])==len(part) and int(c['discordant_pairs'])==rescues+damage
  assert abs(float(c['delta_accuracy_pp'])-100*(rescues-damage)/len(part))<1e-12
  expected_p=1. if rescues+damage==0 else float(binomtest(rescues,rescues+damage,.5).pvalue)
  assert abs(float(c['exact_mcnemar_p'])-expected_p)<1e-12
  assert int(c['bootstrap_replicates'])==10000 and int(c['bootstrap_seed'])==2021 and c['bootstrap_unit']=='question'
  assert -100<=float(c['bootstrap_accuracy_pp_low'])<=float(c['bootstrap_accuracy_pp_high'])<=100
 for split in by_split:
  family=sorted([c for c in statistics if c['split']==split],key=lambda c:float(c['exact_mcnemar_p']));running=0.
  for i,c in enumerate(family):
   running=max(running,min(1.,float(c['exact_mcnemar_p'])*(len(family)-i)))
   assert int(c['holm_family_size'])==len(family)==8 and abs(float(c['holm_adjusted_p_within_split'])-running)<1e-12
 paired=csvrows('paired_transitions.csv');assert len(paired)==8*(192+192+299+1172)
 expected_pair_keys={(split,uid,before,after) for split,items in by_split.items() for uid in items for before,after in expected_comparisons}
 assert len(expected_pair_keys)==len(paired) and {(c['split'],c['id'],c['before'],c['after']) for c in paired}==expected_pair_keys
 for c in paired:
  r=by_split[c['split']][c['id']];b=r['methods'][c['before']];d=r['methods'][c['after']]
  assert int(c['prediction_before'])==b['prediction'] and int(c['prediction_after'])==d['prediction']
  assert (c['correct_before']=='True')==b['correct'] and (c['correct_after']=='True')==d['correct']
  assert c['prompt_sha256']==r['prompt_sha256']
  assert int(c['gold'])==r['gold'] and json.loads(c['normalized_scores_before'])==b['normalized_scores'] and json.loads(c['normalized_scores_after'])==d['normalized_scores']
  assert abs(float(c['delta_margin'])-(d['margin']-b['margin']))<1e-12
 assert len(csvrows('failure_mechanisms.csv'))==576
 layer=csvrows('layer_intervention_results.csv');assert len(layer)==1288
 assert len({(c['stage'],c['id'],c['condition'],c['k']) for c in layer})==len(layer)
 assert collections.Counter(c['stage'] for c in layer)=={'layer_screen':1224,'shared_confirmation':64}
 expected_layer={}
 for r in rows('results/layer_screen_original/scores.jsonl'):
  for k in [4,5,6]:expected_layer[('layer_screen',r['id'],r['condition'],str(k))]=r['methods'][f'native_k{k}']
 for r in rows('results/shared_confirmation_original/scores.jsonl'):
  if r['evidence']['kind']=='reuse_completed_screen':continue
  for k in [3,4,5,6]:
   condition=f"damp_{r['scope']}_{r['branch']}_layer{r['layer']:02d}"
   expected_layer[('shared_confirmation',r['id'],condition,str(k))]=r['methods'][f'native_k{k}']
 assert set(expected_layer)=={(c['stage'],c['id'],c['condition'],c['k']) for c in layer}
 for c in layer:
  assert '/train/' in c['id']
  assert abs(float(c['margin'])-float(c['native_margin'])-float(c['delta_margin']))<1e-12
  source=expected_layer[(c['stage'],c['id'],c['condition'],c['k'])]
  assert json.loads(c['candidate_normalized_scores'])==source['normalized_scores']
  assert json.loads(c['candidate_sum_scores'])==source['sum_scores']
  assert int(c['prediction'])==source['prediction'] and (c['correct']=='True')==source['correct']
  assert abs(float(c['margin'])-source['margin'])<1e-12
 manifest=read('weight_edit_manifest.json')
 assert manifest['actual_changed_parameters']==0 and manifest['edits']==[] and manifest['primary_weight_edit'] is None
 model=ROOT/'models/Ouro-1.4B/model.safetensors'
 with model.open('rb') as f:header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
 params=sum(math.prod(v['shape']) for k,v in header.items() if k!='__metadata__');assert params==manifest['total_unique_parameters']==1434652673
 if a.verify_model:
  model_sha=digest(model);assert model_sha==manifest['base_model_sha256']
  for item in read('sources/ouro_hub.json')['siblings']:
   p=ROOT/'models/Ouro-1.4B'/item['rfilename']
   if not p.exists():continue
   assert p.stat().st_size==item['size']
   if item.get('lfs'):
    assert (model_sha if p==model else digest(p))==item['lfs']['sha256']
   elif item.get('blobId'):
    b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item['blobId']
 budget=read('logs/gpu_budget.json');assert budget['used_seconds']<=28800
 assert abs(sum(r['gpu_worker_seconds'] for r in budget['runs'])-budget['used_seconds'])<1e-6
 outputs=['REPORT_ZH.md','failure_mechanisms.csv','layer_intervention_results.csv','weight_edit_manifest.json','evaluation_summary.csv','paired_transitions.csv','paired_statistics.csv','failure_cases.jsonl']
 hashes={p:digest(ROOT/p) for p in outputs}
 assert read('results/REPORT_ASSEMBLY.json')['report_sha256']==hashes['REPORT_ZH.md']
 script_hashes={p.name:digest(p) for p in sorted((ROOT/'scripts').glob('*.py'))}
 save('ARTIFACT_AUDIT.json',dict(status='passed',official_test_n=1172,choice_count_distribution=choice_counts,summary_rows=len(summary),statistics_rows=len(statistics),paired_rows=len(paired),layer_rows=len(layer),demo_train_discovery_selection_disjoint=True,prompt_text_hashes_and_exported_score_arithmetic_recomputed=True,model_forwards_recomputed_in_this_audit=False,bootstrap_samples_independently_recomputed=False,exact_mcnemar_and_holm_recomputed=True,strict_original_call_parity=True,zero_weight_edits=True,total_unique_parameters=params,base_model_sha256_reverified=a.verify_model,gpu_worker_seconds=budget['used_seconds'],gpu_budget_seconds=28800,output_sha256=hashes,script_sha256=script_hashes))
 print('ARTIFACT_AUDIT_PASSED',json.dumps(dict(test_n=1172,paired_rows=len(paired),layer_rows=len(layer),choice_counts=choice_counts['test'],gpu_worker_hours=budget['used_seconds']/3600)))
if __name__=='__main__':main()
