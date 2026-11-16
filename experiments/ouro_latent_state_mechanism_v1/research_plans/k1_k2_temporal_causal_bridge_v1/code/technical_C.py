import bridge as B
from bridge import torch
import json,datetime,time
R=B.ROOT;g=json.loads((R/'B_CONFIRMATION_GATE.json').read_text());assert g['pass'] and g['mask']==1;bg=g['bg'];qs={q['id']:q for q in map(json.loads,(R/'INPUTS.jsonl').read_text().splitlines())};units=[u for u in map(json.loads,(R/'UNITS.jsonl').read_text().splitlines()) if u['pair_id']=='TB000'];m=B.initialize();native={};start=time.monotonic();records=[];raw={}
for id in sorted({u[k] for u in units for k in ['reference_W0','input_W1']}):
 r,z,c=B.run(m,qs[id],capture=True,extra_span=tuple(qs[id]['regions']['map']));native[id]=(r,z,c)
for u in units:
 q=qs[u['input_W1']];w0=native[u['reference_W0']];w1=native[u['input_W1']];span=tuple(q['regions']['map']);Z,zheads,zc=B.run(m,q,reference=w0[2],native=w1[2],g1=bg,mask2=0,capture=True,extra_span=span);T,thead,tc=B.run(m,q,reference=w0[2],native=w1[2],g1=bg,mask2=1,capture=True,extra_span=span,path={'span':span,'Z':zc,'observe_only':True});previous=json.loads((R/'B_results'/(u['pair_id']+'.json')).read_text());old=next(x for x in previous['units'] if x['id']==u['id']);assert all(T['scores'][d]['full49152_sha256']==old['masks']['1']['scores'][d]['full49152_sha256'] and Z['scores'][d]['full49152_sha256']==old['masks']['0']['scores'][d]['full49152_sha256'] for d in ['1','2']);assert all(all(B.bits(a,b) for a,b in zip(kv,w1[2]['extra_kv'][k])) for k,kv in tc['extra_kv'].items());targets={e['layer']-1:e['counterfactual_T_writeback_norms'] for e in T['cut']['path_edits']}
 conditions={'block':[],'full_restore':list(range(24)),'restore_early':list(range(8)),'restore_mid':list(range(8,16)),'restore_late':list(range(16,24)),'reverse_early':list(range(8,24)),'reverse_mid':list(range(8))+list(range(16,24)),'reverse_late':list(range(16)),'random':[]}
 for name,layers in conditions.items():
  path={'span':span,'Z':zc,'T':tc,'restore_layers':layers}
  if name=='random':path.update(random=True,unit_id=u['id'],targets=targets)
  r,z,c=B.run(m,q,reference=w0[2],native=w1[2],g1=bg,mask2=1,path=path)
  if name=='block':assert all(B.bits(z[d],zheads[d]) for d in [1,2])
  if name=='full_restore':assert all(B.bits(z[d],thead[d]) for d in [1,2])
  records.append({'unit':u['id'],'condition':name,'record':r})
 if u==units[0]:raw={'Z_G_actual_postRoPE_KV':{str(k):tuple(x.cpu() for x in kv) for k,kv in zc['extra_kv'].items()},'T_G_actual_postRoPE_KV':{str(k):tuple(x.cpu() for x in kv) for k,kv in tc['extra_kv'].items()},'Z_normed_states':{k:x.cpu() for k,x in zc['norms'].items()},'T_normed_states':{k:x.cpu() for k,x in tc['norms'].items()}}
torch.save(raw,B.RAW/'C_TECHNICAL_G_WITNESSES.pt');out={'status':'PASS','units':len(units),'records':records,'checks':['newCengineZ/Tfullendpoints matcholdA/B','GpostRoPEKVidenticalT_andnativeW1','fullGpathblockR_full49152==Z_bothloops','fullrestorationR_full49152==T_bothloops','Sallstates/KV/attentionoutputs remainnativeW1','Gneveroverwritten_Honly_SDPArowsplice','actualsame_layerinputG_notnewlywrittenG','dosematchingtargetsactualTrowdifference'],'per_unit_estimated_C_seconds':sum(q['record']['seconds'] for q in records if q['condition']!='full_restore')/len(units)+.15,'wall_seconds':time.monotonic()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'C_TECHNICAL_PASS.json').write_text(json.dumps(out,indent=2));print('C_TECHNICAL_PASS_ESTIMATE_SECONDS',out['per_unit_estimated_C_seconds']*768)
