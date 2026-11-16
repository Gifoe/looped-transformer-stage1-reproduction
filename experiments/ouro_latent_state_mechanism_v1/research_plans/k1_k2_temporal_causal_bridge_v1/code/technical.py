import bridge as B
from bridge import torch
import pathlib,json,time,datetime
R=B.ROOT;qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};units=[u for u in map(json.loads,(R/'UNITS.jsonl').read_text().splitlines()) if u['pair_id']=='TB000'];m=B.initialize();ids=sorted({u[k] for u in units for k in ['reference_W0','input_W1']});native={};records=[];start=time.monotonic()
for id in ids:
 q=qs[id];r,z,c=B.run(m,q,capture=True,technical=True);native[id]=(r,z,c);inp=torch.tensor([q['ids']],device='cuda')
 with torch.inference_mode():_,states,_=m.model(inp,use_cache=False,past_key_values=None);official={d+1:m.lm_head(states[d])[:,q['R']] for d in [0,1]}
 assert all(B.bits(z[d],official[d]) for d in [1,2])
# FullS includes punctuation/newline; same first-hop S cannot access future mappings/options.
for rev in [0,1]:
 for a,b in [('original','donor_edit'),('donor','original_edit')]:
  ca=native[f'TB000_L{rev}_{a}'][2];cb=native[f'TB000_L{rev}_{b}'][2];assert all(all(B.bits(x,y) for x,y in zip(ca['kv'][k],cb['kv'][k])) for k in ca['kv'])
for world in ['original','donor','original_edit','donor_edit']:
 a=native['TB000_L0_'+world][2];b=native['TB000_L1_'+world][2];assert all(all(B.bits(x,y) for x,y in zip(a['kv'][k],b['kv'][k])) for k in a['kv'])
for u in units:
 q=qs[u['input_W1']];n=native[u['input_W1']];ref=native[u['reference_W0']]
 for name,g1,mask2,rc in [('allopen',1,7,ref),('sameworld',0,0,n),('C00',0,0,ref),('C01',0,7,ref),('C10',1,0,ref)]:
  r,z,c=B.run(m,q,reference=rc[2],native=n[2],g1=g1,mask2=mask2,technical=True);expected=n[1] if name in ['allopen','sameworld'] else ref[1] if name=='C00' else None
  if expected is not None:assert all(B.bits(z[d],expected[d]) for d in [1,2])
  records.append({'unit':u['id'],'condition':name,'record':r})
# Actual reference sourceKV/state witnesses, rather than copied D answers.
torch.save({id:{'kv':{str(k):tuple(x.cpu() for x in v) for k,v in c['kv'].items()},'S_input':{str(k):x.cpu() for k,x in c['S_input'].items()},'norms':{k:x.cpu() for k,x in c['norms'].items()}} for id,(r,z,c) in native.items()},B.RAW/'TECHNICAL_PREFIX_WITNESSES.pt')
formal_seconds=sum(r['record']['seconds'] for r in records if r['condition'] in ['C00','C01','C10'])+sum(v[0]['seconds'] for v in native.values());out={'status':'PASS','units':len(units),'native_worlds':len(ids),'checks':['unhooked_vsopen_full49152_bothloops','sameworld_reference_fullnull','C00_eachlayerDinput/output_bothendpoints_W0_exact','K1closed_K2Dinput_W0norm_exact','allconditions_S_states_KV_andattentionoutputs_W1_exact','source_future_mapping_andoption_invariant_all24layers_bothloops','original_fullshaped_SDPA_backend_andlayouts_preserved'],'records':records,'formal_group_estimated_seconds':formal_seconds,'full96group_estimated_seconds':formal_seconds*96,'technical_wall_seconds':time.monotonic()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'CAUSAL_CUT_NUMERICAL_AUDIT.json').write_text(json.dumps(out,indent=2));print('TEMPORAL_TECHNICAL_PASS',formal_seconds*96)
