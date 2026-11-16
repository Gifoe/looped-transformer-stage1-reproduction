"""Full saved-vector and scalar self-audit. Does not invoke the neural model."""
import pathlib,json,hashlib,numpy as np,torch,datetime,sys
torch.set_num_threads(4)
from freeze_guard import verify,exact_ids,sha
R=pathlib.Path('/root/ouro-semantic-read-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-semantic-read-sol-20261117/raw');fsha=verify();phase=sys.argv[1];qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines()) if q['split']==phase};expected=set(qs);seen_native=[];seen_science=[];vectors=0;maxlp=0.;events=0;prefix=0;globalrecon=0.;scores_for_arithmetic=[]
def digest(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def endpoint(raw,score):
 global vectors,maxlp
 assert raw.numel()==49152;assert digest(raw)==score['full49152_sha256'];x=raw.flatten().double().numpy();off=x[[330,389,340,422]];assert off.tolist()==score['candidate_logits'];ties=np.flatnonzero(off==off.max()).tolist();assert ties==score['ties'] and ties[0]==score['prediction'];m=float(x.max());lp=off-(m+np.log(np.exp(x-m).sum()));err=float(np.max(np.abs(lp-score['candidate_logprobs'])));maxlp=max(err,maxlp);assert err<2e-5;vectors+=1
for p in sorted((R/'native_results').glob('SR*.json')):
 z=json.loads(p.read_text())
 if z['split']!=phase:continue
 assert z['freeze_sha256']==fsha;exact_ids(z['rows'],[q['id'] for q in qs.values() if q['pair_id']==z['pair_id']]);assert len(z['canonical_native_prefixes'])==8;rawfile=RAW/(z['pair_id']+'_native.pt');assert sha(rawfile)==z['raw_sha256'];raw=torch.load(rawfile,map_location='cpu',weights_only=False);assert set(raw)=={u['id'] for u in z['rows']}
 for u in z['rows']:
  id=u['id'];q=qs[id];seen_native.append(id);assert u['prefix_key']==q['prefix_ids_sha256'];assert len(z['canonical_native_prefixes'][u['prefix_key']]['hashes'])==194
  for loop in [1,2]:endpoint(raw[id][loop],u['native']['scores'][str(loop)])
for p in sorted((R/'results').glob('SR*.json')):
 z=json.loads(p.read_text())
 if z['split']!=phase:continue
 assert z['freeze_sha256']==fsha;exact_ids(z['rows'],[q['id'] for q in qs.values() if q['pair_id']==z['pair_id']]);rawfile=RAW/(z['pair_id']+'_science.pt');assert sha(rawfile)==z['raw_sha256'];raw=torch.load(rawfile,map_location='cpu',weights_only=False);assert set(raw)=={u['id'] for u in z['rows']};native=json.loads((R/'native_results'/p.name).read_text());natives={u['id']:u['native'] for u in native['rows']}
 for u in z['rows']:
  id=u['id'];q=qs[id];seen_science.append(id);assert u['conditions']['native']==natives[id];assert set(raw[id])=={'semanticK1','semanticK2','randomK1','randomK2'}
  for kind,byloop in raw[id].items():
   rec=u['conditions'][kind];arm=int(kind[-1]);a=rec['audit'];assert a['actual_original_SDPA_invocations']==48 and a['explicit_FP64_attention_calls']==8 and a['prefix_identity_sites']==194 and a['unchanged_nonH_checks']==8;assert a['receiver_row_writebacks']==8*(q['R']+1-q['H_start']);assert [[e['loop'],e['layer']] for e in a['events']]==[[arm,l] for l in range(9,17)];events+=8;prefix+=194
   for loop in [1,2]:endpoint(byloop[loop],rec['scores'][str(loop)])
   if arm==2:assert rec['scores']['1']['full49152_sha256']==natives[id]['scores']['1']['full49152_sha256']
   for e in a['events']:
    assert e['record_spans']==q['record_spans'] and e['receiver_span']==[q['H_start'],q['R']+1] and e['same_layer_current_QKV'] and e['all_heads'];assert len(e['actual_delta_norms'])==len(e['projected_actual_delta_norms'])==q['R']+1-q['H_start'];assert all(np.isfinite(e['actual_delta_norms'])) and all(np.isfinite(e['projected_actual_delta_norms']));globalrecon=max(globalrecon,e['FP64_native_reconstruction_relative_l2']);assert e['FP64_native_reconstruction_relative_l2']<=.01
    if kind.startswith('random'):
     sem=u['conditions']['semanticK'+str(arm)]['audit']['events'][e['layer']-9];d=e['dose'];assert d['target']==sem['actual_delta_norms'];assert np.allclose(d['actual'],e['actual_delta_norms'],atol=1e-11,rtol=1e-11)
  scores_for_arithmetic.append(u)
assert len(seen_native)==len(set(seen_native))==len(expected) and set(seen_native)==expected
if seen_science:assert len(seen_science)==len(set(seen_science))==len(expected) and set(seen_science)==expected
# Separate arithmetic of every primary mean from the seven-contrast bootstrap implementation.
direct={}
def y(u,name,loop):
 x=u['conditions'][name]['scores'][str(loop)]['candidate_logits'];return x[u['opposite_index']]-(x[u['decoy_indices'][0]]+x[u['decoy_indices'][1]])/2
def g(u,name,loop):
 x=u['conditions'][name]['scores'][str(loop)]['candidate_logits'];return x[u['opposite_index']]-x[u['correct_index']]
if scores_for_arithmetic:
 s=json.loads((R/(phase+'_SEMANTIC_SUMMARY.json')).read_text())
 for f in [0,1]:
  for e in [0,1]:direct[f'specific_F{f}_E{e}']=float(np.mean([y(u,'semanticK2',2)-y(u,'randomK2',2) for u in scores_for_arithmetic if u['firsthop_factor']==f and u['endpoint_factor']==e]))
 direct['Y_semantic_vs_native']=float(np.mean([y(u,'semanticK2',2)-y(u,'native',2) for u in scores_for_arithmetic]));direct['G_semantic_vs_native']=float(np.mean([g(u,'semanticK2',2)-g(u,'native',2) for u in scores_for_arithmetic]));direct['round_specificity_difference_K2_minus_K1']=float(np.mean([y(u,'semanticK2',2)-y(u,'randomK2',2)-y(u,'semanticK1',1)+y(u,'randomK1',1) for u in scores_for_arithmetic]));assert all(v==s['primary_contrasts'][k]['mean'] for k,v in direct.items())
out={'status':'PASS','phase':phase,'worlds':len(expected),'complete_saved_full49152_endpoints':vectors,'full_logprob_reference_max_error':maxlp,'intervention_events':events,'prefix_bit_identity_assertions_covered':prefix,'max_current_trajectory_native_reconstruction_relative_global_H':globalrecon,'note_reconstruction_scope':'Global H norm recorded allscienceevents. Perheadrow bound separately on4oldtechnicalworlds/160sites; nofullscienceperheadrawtensorclaim.','direct_primary_means':direct,'freeze_sha256':fsha,'executor':'Sol CPU vector/scalar self-audit, notindependent agent GPUreplication','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/(phase+'_FULL_VECTOR_CPU_AUDIT.json')).write_text(json.dumps(out,indent=2));print('SCIENCE_CPU_AUDIT_PASS',phase,vectors,events)
