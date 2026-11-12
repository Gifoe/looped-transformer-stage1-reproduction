"""Natural cohort R geometry, descriptive only; run on Linux project CPU."""
import os,json,hashlib,math,traceback
from pathlib import Path
import numpy as np,torch,pyarrow.parquet as pq
from safetensors import safe_open
from geometry import compute,candidate_basis
ROOT=Path('/home/luansiyu/loop-latent');ORIG=ROOT/'orchestration_runs/run-20261107T053919Z-6d01138e';RESUME=ROOT/'orchestration_runs/resume-test-20261111';OLD=ROOT/'orchestration_runs/round2-full-20261111';SMOKE=int(os.environ.get('NATURAL_R_SMOKE','0'));OUT=ROOT/'research_plans'/('natural_r_geometry_smoke_20261111' if SMOKE else 'natural_r_geometry_20261111');torch.set_num_threads(1)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def status(**obj):
 tmp=OUT/'status.tmp';tmp.write_text(json.dumps(obj,sort_keys=True));os.replace(tmp,OUT/'status.json')
def main():
 OUT.mkdir(exist_ok=False)
 try:
  plan=ROOT/'research_plans/NATURAL_K1_K2_R_GEOMETRY_PLAN_20261111.md';assert sha(plan)=='abac3de3856ae20cc74fb5d848b77920b720c822f977734276dc90767c3adddf'
  model=ROOT/'model/model.safetensors';config=ROOT/'model/config.json';conf=json.loads(config.read_text());assert conf['hidden_size']==2048 and conf['rms_norm_eps']==1e-6 and not conf['tie_word_embeddings']
  status(stage='checkpoint_hash');model_hash=sha(model)
  with safe_open(str(model),framework='pt',device='cpu') as f:
   assert 'lm_head.bias' not in f.keys();W=f.get_tensor('lm_head.weight').float().numpy().astype(np.float64);gamma=f.get_tensor('model.norm.weight').float().numpy().astype(np.float64)
  assert W.shape==(49152,2048) and gamma.shape==(2048,)
  source={'model_sha256':model_hash,'config_sha256':sha(config),'plan_sha256':sha(plan),'script_sha256':sha(Path(__file__)),'geometry_sha256':sha(Path(__file__).with_name('geometry.py')),'semantics':'selected R, layer output pre-loop-norm; float64 readout lens, not GPU production scores'}
  for split,n in [('train',1119),('validation',299),('test',1172)]:
   if SMOKE and split!='train':break
   cv=(RESUME if split=='test' else ORIG)/('coverage_'+split+'.json');c=json.loads(cv.read_text());co=c['counts'];assert co['complete_questions']==co['expected_questions']==n and co['failed_questions']==co['missing_questions']==0 and c['score_replay']['status']=='passed' and c['score_replay']['tolerance']==0
   data=ROOT/'data'/f'{split}.parquet';dataset={r['id']:r for r in pq.read_table(data).to_pylist()};assert len(dataset)==n and set(dataset)=={x['id'] for x in c['questions']}
   oldc=OLD/('coverage_'+split+'.json');old={x['id']:x for x in json.loads(oldc.read_text())['questions']};assert set(old)==set(dataset)
   output=OUT/(split+'_rows.jsonl');status(stage='running',split=split,done=0,total=n)
   with output.open('x') as out:
    for i,q in enumerate(c['questions'][:SMOKE] if SMOKE else c['questions']):
     ident=q['id'];key=hashlib.sha256(ident.encode()).hexdigest();mp=ORIG/'shards'/split/(key+'.json');tp=mp.with_suffix('.safetensors');assert sha(mp)==q['metadata_sha256'] and sha(tp)==q['tensor_sha256']
     m=json.loads(mp.read_text());d=dataset[ident];gold=d['choices']['label'].index(d['answerKey']);assert m['id']==ident and m['split']==split and m['gold_index']==gold and m['sha256']['model/model.safetensors']==model_hash and m['sha256']['model/config.json']==sha(config)
     rec=sorted(m['records'],key=lambda x:x['candidate']);assert [x['candidate'] for x in rec]==list(range(len(d['choices']['text']))) and len(rec)>1
     ids=rec[0]['encoding']['input_ids'];pos=rec[0]['captured_positions'];R=rec[0]['encoding']['predecessor_positions'];assert len(R)==1 and R[0] in pos and len(pos)==len(set(pos));idx=pos.index(R[0]);ih=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
     assert all(x['input_ids_sha256']==ih and x['encoding']['input_ids']==ids and x['captured_positions']==pos and x['encoding']['predecessor_positions']==R and len(x['encoding']['target_ids'])==1 for x in rec)
     targets=[x['encoding']['target_ids'][0] for x in rec];scores=[[x['scores'][k] for x in rec] for k in (0,1)];pred=[max(range(len(rec)),key=lambda j:(z[j],-j)) for z in scores];assert pred==m['predictions'][:2];ties=[sum(x==z[j] for x in z)>1 for z,j in zip(scores,pred)]
     wrong=max((j for j in range(len(rec)) if j!=gold),key=lambda j:(scores[0][j],-j));dyn=[max((j for j in range(len(rec)) if j!=gold),key=lambda j:(z[j],-j)) for z in scores];fixed=[z[gold]-z[wrong] for z in scores];allgap=[sum(z[gold]-z[j] for j in range(len(rec)) if j!=gold)/(len(rec)-1) for z in scores];dynamic=[z[gold]-z[w] for z,w in zip(scores,dyn)]
     op=OLD/split/(key+'.json');assert sha(op)==old[ident]['metadata_sha256'];o=json.loads(op.read_text())['payload'];assert o['identity']['metadata_sha256']==sha(mp) and o['identity']['input_ids_sha256']==ih and o['result']['target_ids']==targets and o['result']['gold']==gold
     group={(False,True):'rescue',(True,False):'damage',(True,True):'stable_correct',(False,False):'stable_wrong'}[(pred[0]==gold,pred[1]==gold)]
     sv,tol,rank,basis=candidate_basis(W,targets)
     with safe_open(str(tp),framework='pt',device='cpu') as f:
      actual=[];canonical_actual=[]
      for loop in (1,2):
       vals=[]
       for j,v in enumerate(targets):
        sl=f.get_slice(f'candidate{j}_loop{loop}_lm_head_logits');assert sl.get_shape()==[1,49152] and sl.get_dtype()=='BF16';vals.append(sl[0,v:v+1].float().item())
       ref=f.get_slice(f'candidate0_loop{loop}_lm_head_logits');assert ref.get_shape()==[1,49152] and ref.get_dtype()=='BF16'
       canonical_actual.append([ref[0,v:v+1].float().item() for v in targets]);actual.append(vals)
      states=[]
      for layer in range(24):
       h=[]
       for loop in (1,2):
        v=f.get_slice(f'candidate0_loop{loop}_layer{layer:02d}');assert v.get_shape()==[len(pos),2048] and v.get_dtype()=='BF16';h.append(v[idx:idx+1,:].float().numpy()[0].astype(np.float64))
       item,u,z=compute(h[0],h[1],W,targets,gamma,conf['rms_norm_eps'],gold,wrong,basis,rank,sv,tol);item['layer']=layer
       wrong_indices=[j for j in range(len(targets)) if j!=gold];gap1=[float(z[0][gold]-z[0][j]) for j in wrong_indices];gap2=[float(z[1][gold]-z[1][j]) for j in wrong_indices];advantage=[b-a for a,b in zip(gap1,gap2)]
       item['wrong_indices']=wrong_indices;item['proxy_gap1']=gap1;item['proxy_gap2']=gap2;item['proxy_advantage']=advantage
       if states:
        prev=states[-1];item['adjacent_gap1_increment']=[x-y for x,y in zip(gap1,prev['proxy_gap1'])];item['adjacent_gap2_increment']=[x-y for x,y in zip(gap2,prev['proxy_gap2'])];item['adjacent_advantage_increment']=[x-y for x,y in zip(advantage,prev['proxy_advantage'])]
       else:item['adjacent_gap1_increment']=item['adjacent_gap2_increment']=item['adjacent_advantage_increment']=None
       if layer==23:
        ns=[]
        for k in (1,2):
         sl=f.get_slice(f'candidate0_loop{k}_post_norm');assert sl.get_shape()==[len(pos),2048] and sl.get_dtype()=='BF16';ns.append(sl[idx:idx+1,:].float().numpy()[0].astype(np.float64))
        post_z=[W[targets]@v for v in ns]
        item['calibration']={'lens_minus_fp64_savednorm_head':[(z[k]-post_z[k]).tolist() for k in (0,1)],'fp64_savednorm_head_minus_candidate0_native':[(post_z[k]-np.asarray(canonical_actual[k])).tolist() for k in (0,1)],'offline_norm_vs_saved_postnorm_maxabs':[float(np.max(np.abs(u[k]-ns[k]))) for k in (0,1)],'offline_proxy_minus_candidate0_target_logits':[(z[k]-np.asarray(canonical_actual[k])).tolist() for k in (0,1)],'request_minus_candidate0_target_logits':[(np.asarray(actual[k])-np.asarray(canonical_actual[k])).tolist() for k in (0,1)]}
       states.append(item)
     row={'id':ident,'split':split,'gold':gold,'targets':targets,'scores':scores,'actual_target_logits':actual,'candidate0_target_logits':canonical_actual,'predictions':pred,'top_ties':ties,'fixed_wrong':wrong,'dynamic_wrong':dyn,'fixed_gap':fixed,'all_wrong_gap':allgap,'dynamic_gap':dynamic,'group':group,'demo':o['identity']['is_demonstration'],'natural_no_top_tie':not any(ties),'R_position':R[0],'captured_positions_count':len(pos),'input_ids_sha256':ih,'metadata_sha256':q['metadata_sha256'],'tensor_sha256':q['tensor_sha256'],'layers':states}
     out.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
     if (i+1)%25==0:out.flush();status(stage='running',split=split,done=i+1,total=n)
   source[split]={'coverage_sha256':sha(cv),'old_coverage_sha256':sha(oldc),'data_sha256':sha(data),'rows_sha256':sha(output),'questions':min(SMOKE,n) if SMOKE else n};status(stage='split_complete',split=split,done=min(SMOKE,n) if SMOKE else n,total=n)
  (OUT/'source.json').write_text(json.dumps(source,sort_keys=True,indent=2)+'\n');status(stage='smoke_rows_complete' if SMOKE else 'rows_complete',source_sha256=sha(OUT/'source.json'))
 except BaseException as e:
  status(stage='failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
