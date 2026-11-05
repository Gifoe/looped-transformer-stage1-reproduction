"""Complete numerical validation first, then fixed question-level descriptive analysis."""
from core import ROOT,OUT,H,read,save,gate
import json,csv,itertools,math
from fractions import Fraction
import numpy as np
LP_REPLAY_ABS_TOL=5e-5
NORM64_REPLAY_ABS_TOL=1e-10

def numeric_integrity():
 spec=gate();s=read(OUT/'SUPERVISOR.json');b=read(OUT/'gpu_budget.json');d=read(OUT/'COMPLETE.json');plan=read(OUT/'INFERENCE_TASK_PLAN.json');assert s['status']=='complete' and s['exit_code']==0 and not s['timed_out'] and not s['overrun'] and s['old_ledgers_unchanged'] and 0<s['gpu_worker_seconds']<=680
 assert b['used_seconds']==s['gpu_worker_seconds'] and b['total_used_seconds']==20889.326696651056+b['used_seconds']<=21600
 assert d['parameter_hashes_before']==d['parameter_hashes_after']==read(ROOT/'recurrent_decision_audit_v1/a4_COMPLETE.json')['parameter_hashes_after'] and len(d['parameter_hashes_before'])==269 and d['cleanup_exact']
 assert d['protocol_sha256']==s['protocol_sha256']==H((OUT/'EXECUTION_LOCK.json').read_bytes()) and d['plan_sha256']==H((OUT/'INFERENCE_TASK_PLAN.json').read_bytes())
 for f,key in [('SCORES.jsonl','SCORES_sha256'),('VECTORS.npz','VECTORS_sha256'),('VECTOR_INDEX.json','VECTOR_INDEX_sha256')]:assert H((OUT/f).read_bytes())==d[key]
 full=d['status']=='complete_all218_tasks';assert full or d['status']=='P0_only_runtime_not_admitted_no_effects';n=218 if full else 24;tasks=plan['tasks'][:n];raw=[json.loads(x) for x in (OUT/'SCORES.jsonl').read_text(encoding='utf8').splitlines()];assert len(raw)==n==d['tasks'] and d['native_heads']==4*n and d['head_prediction_positions']==4*sum(len(t['positions']) for t in tasks)
 assert [r['task_key'] for r in raw]==[t['task_key'] for t in tasks];idx=read(OUT/'VECTOR_INDEX.json');z=np.load(OUT/'VECTORS.npz',allow_pickle=False);assert set(z.files)==set(idx);cache={};maxlp=maxnorm=maxsum=0.;bycid={};used=set();parity_count=0
 def expand(key,kind,shape):
  if key not in cache:
   a=z[key];meta=idx[key];assert a.dtype==np.uint16 and list(a.shape)==meta['shape'] and meta['kind']==kind and a.shape==shape and H((kind+str(a.shape)).encode()+a.tobytes())==key
   v=(a.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(v).all();cache[key]=v
  used.add(key);return cache[key]
 for r,t in zip(raw,tasks):
  length=len(t['positions']);assert len(r['state_keys'])==len(r['head_keys'])==len(r['fullV_log_normalizer_FP64'])==4;assert set(r['scores'])=={c['consumer_id'] for c in t['consumers']}
  for k in range(4):
   expand(r['state_keys'][k],'normalized_state',(1,length,2048));v=expand(r['head_keys'][k],'head_logits',(length,49152));vd=v.astype(np.float64);m=vd.max(1);norm=m+np.log(np.exp(vd-m[:,None]).sum(1));recorded=np.asarray(r['fullV_log_normalizer_FP64'][k]);err=float(np.max(np.abs(norm-recorded)));maxnorm=max(maxnorm,err);assert err<=NORM64_REPLAY_ABS_TOL
   for c in t['consumers']:
    sc=r['scores'][c['consumer_id']][k];lp=np.asarray(sc['token_logprobs'],dtype=float);rt=np.asarray(sc['raw_target_logits'],dtype=float);target=c['target_ids'];assert len(lp)==len(rt)==len(target)==length and np.isfinite(lp).all() and np.all(lp<=0)
    assert np.array_equal(rt,v[np.arange(length),target]) and np.array_equal(lp,lp.astype(np.float32).astype(float));sval=sc['sum_logprob'];assert np.isfinite(sval) and float(np.float32(sval))==sval and sc['char_denominator']==c['char_denominator']>0 and sc['char_normalized']==sval/c['char_denominator']
    exact=sum((Fraction.from_float(float(x)) for x in lp),Fraction());u=Fraction(1,2**24);gamma=(max(0,length-1)*u)/(1-max(0,length-1)*u);bound=gamma*sum(abs(Fraction.from_float(float(x))) for x in lp)+length*Fraction(1,2**149);assert abs(Fraction.from_float(sval)-exact)<=bound
    maxsum=max(maxsum,float(abs(Fraction.from_float(sval)-exact)));error=float(np.max(np.abs(lp-(rt-norm))));maxlp=max(maxlp,error);assert error<=LP_REPLAY_ABS_TOL
  for c in t['consumers']:
   cid=c['consumer_id'];assert cid not in bycid;bycid[cid]=dict(scores=r['scores'][cid],normalizers=r['fullV_log_normalizer_FP64'],head_keys=r['head_keys'])
   if cid in plan['historical_endpoint_references']:
    ref=plan['historical_endpoint_references'][cid];assert bycid[cid]['scores'][0]['sum_logprob']==ref['K1'] and bycid[cid]['scores'][3]['sum_logprob']==ref['K4'];parity_count+=1
 assert used==set(idx) and parity_count==(192 if full else 96);parity=read(OUT/'PARITY.json');assert parity['status']=='all24_P0_native1_native4_Fixed_exact_before_interventions' and len(parity['checks'])==96
 assert len(d['historical_parity_checks'])==(192 if full else 96) and all(all(p[k] is True for k in ['K1_exact','K4_exact','Fixed_exact']) for p in d['historical_parity_checks'])
 validation=dict(status='passed_all_numeric_before_labels',full_intervention_matrix=full,tasks=n,native_heads=4*n,consumers=len(bycid),vectors=len(idx),FP64_normalizer_max_abs_replay_error=maxnorm,FP32_token_LP_vs_FP64_reference_max_abs_error=maxlp,FP32_token_LP_reference_abs_tolerance=LP_REPLAY_ABS_TOL,normalizer64_tolerance=NORM64_REPLAY_ABS_TOL,FP32_native_sum_vs_exact_token_sum_max_abs_error=maxsum,sum_validation='Fraction exact sum of stored FP32 tokens; conservative FP32 gamma(n-1) bound plus subnormal term. Not NumPy sum called GPUexact.',decomposition_scope='FullV FP64 normalizer is diagnostic; primaryS uses actual original FP32 log_softmax/gather/sum, so logit-minus-FP64norm is not claimed bitexact.',labels_read=False)
 save('numerical_validation.json',validation);return full,bycid,validation,b

def csvout(name,rows,empty_fields=None):
 with (OUT/name).open('w',newline='',encoding='utf8') as f:
  fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else (empty_fields or ['status']);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def factorial_contrasts(a,b,c,d):
 a,b,c,d=[np.asarray(x,float) for x in [a,b,c,d]];assert a.ndim==1 and a.shape==b.shape==c.shape==d.shape
 return dict(demo_effect_letter_target=c-a,demo_effect_full_target=d-b,target_effect_letter_demo=b-a,target_effect_full_demo=d-c,accuracy_factor_interaction=(d-b)-(c-a))

def main():
 assert not (OUT/'ANALYSIS.json').exists();full,bycid,validation,budget=numeric_integrity()
 if not full:
  for name in ['permutation_transitions.csv','distractor_effects.csv','content_letter_readout.csv','question_level_statistics.csv','native_predictions.csv','native_loop_transitions.csv']:csvout(name,[],['status'])
  save('ANALYSIS.json',dict(status='P0_only_no_intervention_effects_unsealed',numerical_validation=validation));return
 # Sidecar and benchmark labels are first deserialized only after every task/vector/LP/lifecycle check passed.
 side=read(OUT/'CONSTRUCTOR_WITH_LABELS.json');protocol=read(OUT/'intervention_protocol.json');cohort=side['cohort'];groups=side['groups'];assert len(cohort)==24 and len(groups)==196;gm={(g['id'],g['condition']):g for g in groups};ids=[r['id'] for r in cohort];first4=ids[:4];stats=[];boot={}
 def stat(metric,condition,K,values,qids):
  if not values:stats.append(dict(metric=metric,condition=condition,K=K,n=0,mean=None,ci95_low=None,ci95_high=None,ids='[]'));return
  x=np.asarray(values,float);n=len(x)
  if n not in boot:boot[n]=np.random.default_rng(2021).integers(0,n,(20000,n))
  ci=np.quantile(x[boot[n]].mean(1),[.025,.975]);stats.append(dict(metric=metric,condition=condition,K=K,n=n,mean=float(x.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),ids=json.dumps(qids),scope='question-level paired seed2021/20k; n4 bounded descriptive only' if n<=4 else 'question-level paired seed2021/20k'))
 def vector(g,k,field='sum_logprob'):
  return np.asarray([bycid[c['consumer_id']]['scores'][k-1][field] for c in g['consumers']],float)
 def prediction(g,k,field='sum_logprob'):
  v=vector(g,k,field);display=int(v.argmax());content=g['order_to_original'][display];return display,content,v
 native=[]
 for g in groups:
  for k in range(1,5):
   p,content,v=prediction(g,k);pn,cn,vn=prediction(g,k,'char_normalized');gold=g['expected_original_content_index'];assert g['expected_display_index']==g['order_to_original'].index(gold);native.append(dict(id=g['id'],condition=g['condition'],K=k,demo_format=g['demo_format'],target_format=g['target_format'],display_prediction=p,original_content_prediction=content,expected_original_content_index=gold,expected_display_index=g['expected_display_index'],target_match=content==gold,raw_sum_scores=json.dumps(v.tolist()),char_normalized_scores=json.dumps(vn.tolist()),char_normalized_prediction=pn,raw_gold_margin=float(v[g['expected_display_index']]-np.delete(v,g['expected_display_index']).max()),char_normalized_gold_margin=float(vn[g['expected_display_index']]-np.delete(vn,g['expected_display_index']).max()),tie_count=int((v==v.max()).sum())))
 for condition in sorted({g['condition'] for g in groups}):
  qids=[q for q in ids if (q,condition) in gm]
  for field,readout in [('sum_logprob','raw'),('char_normalized','char_norm')]:
   acc={k:[prediction(gm[q,condition],k,field)[1]==gm[q,condition]['expected_original_content_index'] for q in qids] for k in range(1,5)}
   for k in range(1,5):stat('target_match_'+readout,condition,k,acc[k],qids)
   for k in [1,2,3]:stat('adjacent_target_match_delta_'+readout,condition,str(k)+'to'+str(k+1),[int(a)-int(b) for a,b in zip(acc[k+1],acc[k])],qids)
   stat('K4_minus_K1_target_match_'+readout,condition,'1to4',[int(a)-int(b) for a,b in zip(acc[4],acc[1])],qids)
 # B5 uses the same fixed four questions in every cell; P0n24 is never its statistical baseline.
 b5cells=[('B5_L_letter','P0'),('B5_L_full','B5_L_full'),('B5_F_letter','B5_F_letter'),('B5_F_full','B5_F_full')]
 for field,readout in [('sum_logprob','raw'),('char_normalized','char_norm')]:
  cellvalues={}
  for alias,condition in b5cells:
   cellvalues[alias]={}
   for k in range(1,5):
    values=[int(prediction(gm[q,condition],k,field)[1]==gm[q,condition]['expected_original_content_index']) for q in first4];cellvalues[alias][k]=np.asarray(values,float);stat('B5_target_match_'+readout,alias,k,values,first4)
   stat('B5_K4_minus_K1_target_match_'+readout,alias,'1to4',(cellvalues[alias][4]-cellvalues[alias][1]).tolist(),first4)
  contrasts={k:factorial_contrasts(*(cellvalues[a][k] for a,c in b5cells)) for k in range(1,5)}
  for k,v in contrasts.items():
   for name,values in v.items():stat('B5_'+name+'_'+readout,'paired_same4_factorial',k,values.tolist(),first4)
  for name in contrasts[1]:stat('B5_K4_minus_K1_'+name+'_'+readout,'paired_same4_factorial','1to4',(contrasts[4][name]-contrasts[1][name]).tolist(),first4)
 permutation=[]
 for q in ids:
  for condition in ['P1','P2','P3']:
   base=gm[q,'P0'];g=gm[q,condition];order=g['order_to_original']
   for k in range(1,5):
    bp,bc,bv=prediction(base,k);p,c,v=prediction(g,k);moved=order.index(bc)!=bp;permutation.append(dict(id=q,condition=condition,K=k,P0_display_prediction=bp,P0_content_prediction=bc,new_display_prediction=p,new_content_prediction=c,content_equivariant=c==bc,display_letter_invariant=p==bp,P0_predicted_content_moved=moved,correct=c==g['expected_original_content_index'],P0_correct=bc==base['expected_original_content_index'],order_to_original=json.dumps(order),tie_count=int((v==v.max()).sum()),gold_margin=float(v[g['expected_display_index']]-np.delete(v,g['expected_display_index']).max())))
 for condition in ['P1','P2','P3']:
  for metric in ['content_equivariant','display_letter_invariant']:
   byk={}
   for k in range(1,5):
    rs=[r for r in permutation if r['condition']==condition and r['K']==k];byk[k]=[int(r[metric]) for r in rs];stat(metric,condition,k,byk[k],[r['id'] for r in rs]);moved=[r for r in rs if r['P0_predicted_content_moved']];stat(metric+'_moved_P0_content_subset',condition,k,[int(r[metric]) for r in moved],[r['id'] for r in moved])
   for k in [1,2,3]:stat('adjacent_'+metric+'_delta',condition,str(k)+'to'+str(k+1),[a-b for a,b in zip(byk[k+1],byk[k])],ids)
   stat('K4_minus_K1_'+metric,condition,'1to4',[a-b for a,b in zip(byk[4],byk[1])],ids)
 effects=[];metric_vectors={}
 def addmetric(name,variant,k,q,value):metric_vectors.setdefault((name,variant,k),{})[q]=float(value)
 for variant in ['placeholder','matched']:
  qids=[q for q in ids if (q,'Q_'+variant) in gm]
  for q in qids:
   j=next(r['wrong_slot'] for r in cohort if r['id']==q);n=gm[q,'P0']['nchoices'];keep=[i for i in range(n) if i!=j];pairs=list(itertools.combinations(keep,2));gold=gm[q,'P0']['expected_original_content_index'];allI={};allRaw={};allNorm={}
   for family,baseline,modified in [('Q','P0','Q_'+variant),('N','N_original','N_'+variant)]:
    for k in range(1,5):
     a=gm[q,baseline];c=gm[q,modified];I=vector(a,k)-vector(c,k);allI[family,k]=I;raw=[];norm=[]
     for ca,cc in zip(a['consumers'],c['consumers']):
      aa=bycid[ca['consumer_id']];bb=bycid[cc['consumer_id']];raw.append(aa['scores'][k-1]['raw_target_logits'][0]-bb['scores'][k-1]['raw_target_logits'][0]);norm.append(aa['normalizers'][k-1][0]-bb['normalizers'][k-1][0])
     allRaw[family,k]=raw;allNorm[family,k]=norm;J=np.asarray([I[i]-I[l] for i,l in pairs]);addmetric('meanabs_J_'+family,variant,k,q,np.abs(J).mean());addmetric('RMS_J_'+family,variant,k,q,np.sqrt(np.mean(J**2)));addmetric('mean_I_'+family,variant,k,q,I[keep].mean());addmetric('gold_relative_margin_response_'+family,variant,k,q,np.mean([I[gold]-I[i] for i in keep if i!=gold]))
   for k in range(1,5):
    IQ,IN=allI['Q',k],allI['N',k];did=IQ-IN;JD=np.asarray([did[i]-did[l] for i,l in pairs]);addmetric('meanabs_J_QminusN',variant,k,q,np.abs(JD).mean());addmetric('RMS_J_QminusN',variant,k,q,np.sqrt(np.mean(JD**2)));addmetric('gold_relative_margin_response_QminusN',variant,k,q,np.mean([did[gold]-did[i] for i in keep if i!=gold]))
    for i in keep:
     effects.append(dict(id=q,variant=variant,record_type='unchanged_option',K=k,changed_wrong_index=j,option_i=i,option_l='',option_i_is_original_gold=i==gold,I_Q=float(IQ[i]),I_N=float(IN[i]),I_QminusN=float(did[i]),delta_to_next_I_Q=float(allI['Q',k+1][i]-IQ[i]) if k<4 else '',delta_to_next_I_N=float(allI['N',k+1][i]-IN[i]) if k<4 else '',delta_to_next_I_QminusN=float((allI['Q',k+1][i]-allI['N',k+1][i])-did[i]) if k<4 else '',raw_target_logit_response_Q=allRaw['Q',k][i],raw_target_logit_response_N=allRaw['N',k][i],fullV_normalizer_response_Q_FP64=allNorm['Q',k][i],fullV_normalizer_response_N_FP64=allNorm['N',k][i],FP32S_minus_FP64decomposition_residual_Q=float(IQ[i]-(allRaw['Q',k][i]-allNorm['Q',k][i])),FP32S_minus_FP64decomposition_residual_N=float(IN[i]-(allRaw['N',k][i]-allNorm['N',k][i]))))
    for i,l in pairs:
     jq=float(IQ[i]-IQ[l]);jn=float(IN[i]-IN[l]);effects.append(dict(id=q,variant=variant,record_type='unchanged_pair',K=k,changed_wrong_index=j,option_i=i,option_l=l,J_Q=jq,J_N=jn,J_QminusN=jq-jn,raw_logit_J_Q=allRaw['Q',k][i]-allRaw['Q',k][l],raw_logit_J_N=allRaw['N',k][i]-allRaw['N',k][l],delta_to_next_J_Q=float((allI['Q',k+1][i]-allI['Q',k+1][l])-jq) if k<4 else '',delta_to_next_J_N=float((allI['N',k+1][i]-allI['N',k+1][l])-jn) if k<4 else '',delta_to_next_J_QminusN=float((allI['Q',k+1][i]-allI['Q',k+1][l])-(allI['N',k+1][i]-allI['N',k+1][l])-(jq-jn)) if k<4 else ''))
   for family,baseline,modified in [('Q','P0','Q_'+variant),('N','N_original','N_'+variant)]:
    a=gm[q,baseline];c=gm[q,modified];pa=prediction(a,4)[0];pc=prediction(c,4)[0];addmetric('K4_display_switch_'+family,variant,'final',q,pa!=pc);addmetric('K4_content_text_switch_'+family,variant,'final',q,a['choices'][pa]!=c['choices'][pc])
 for (name,variant,k),values in metric_vectors.items():
  qids=[q for q in ids if q in values];stat(name,variant,k,[values[q] for q in qids],qids)
  if isinstance(k,int) and k<4 and (name,variant,k+1) in metric_vectors:
   nxt=metric_vectors[name,variant,k+1];stat('adjacent_delta_'+name,variant,str(k)+'to'+str(k+1),[nxt[q]-values[q] for q in qids],qids)
 readouts=[]
 for q in first4:
  for label,condition in [('B5_L_letter','P0'),('B5_L_full','B5_L_full'),('B5_F_letter','B5_F_letter'),('B5_F_full','B5_F_full'),('B6_lookup','B6_lookup'),('B6_copy','B6_copy')]:
   g=gm[q,condition]
   for k in range(1,5):
    pr,cr,vr=prediction(g,k);pn,cn,vn=prediction(g,k,'char_normalized')
    for i,c in enumerate(g['consumers']):
     sc=bycid[c['consumer_id']]['scores'][k-1];readouts.append(dict(id=q,condition=label,K=k,demo_format=g['demo_format'],target_format=g['target_format'],candidate_display_index=i,expected_display_index=g['expected_display_index'],raw_sum=sc['sum_logprob'],char_denominator=sc['char_denominator'],char_normalized=sc['char_normalized'],raw_prediction=pr,char_normalized_prediction=pn,raw_target_match=pr==g['expected_display_index'],char_normalized_target_match=pn==g['expected_display_index'],token_count=len(sc['token_logprobs'])))
 looprows=[]
 for g in groups:
  for field,label in [('sum_logprob','raw'),('char_normalized','char_norm')]:
   for k in [1,2,3]:
    pa,ca,va=prediction(g,k,field);pb,cb,vb=prediction(g,k+1,field);gold=g['expected_original_content_index'];j=g['expected_display_index'];looprows.append(dict(id=g['id'],condition=g['condition'],readout=label,K_from=k,K_to=k+1,display_from=pa,display_to=pb,content_from=ca,content_to=cb,rescue=ca!=gold and cb==gold,damage=ca==gold and cb!=gold,content_changed=ca!=cb,gold_margin_change=float((vb[j]-np.delete(vb,j).max())-(va[j]-np.delete(va,j).max()))))
 csvout('native_loop_transitions.csv',looprows)
 csvout('native_predictions.csv',native);csvout('permutation_transitions.csv',permutation);csvout('distractor_effects.csv',effects);csvout('content_letter_readout.csv',readouts);csvout('question_level_statistics.csv',stats)
 save('ANALYSIS.json',dict(status='complete24_all_conditions_numeric_before_labels',pilot_questions=24,bounded_secondary_questions=4,B5_B6_questions=4,groups=196,tasks=218,stats_rows=len(stats),source_protocol_sha256=H((OUT/'EXECUTION_LOCK.json').read_bytes()),raw_sha256=H((OUT/'SCORES.jsonl').read_bytes()),labels_unsealed_after_numeric=True,numerical_validation=validation,bootstrap_seed=2021,bootstrap_samples=20000,scope='Historical train/dev preregistered24; n4 secondary/readout/simpletask evidence bounded, all metrics/stages reported; no trained selector or new decoder.'))
 print('B_ANALYSIS_COMPLETE',len(stats),len(effects),flush=True)
if __name__=='__main__':main()
