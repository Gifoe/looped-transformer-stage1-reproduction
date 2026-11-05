"""CPU-only preregistered input intervention/task builder. No model predictions."""
from pathlib import Path
import sys,json,hashlib,itertools,collections
import numpy as np
from tokenizers import Tokenizer
from construct_semantics import ROOT,OUT,read,rows,wrong_index,render,field_counts,H,tokenizer_path
sys.path.insert(0,str(ROOT/'scripts'))
from r10_core import question_prompt,joint_prefix
from r7_core import digest

def encode_pair(tok,context,target):
 spaces=len(context)-len(context.rstrip())
 if spaces:target=context[-spaces:]+target;context=context[:-spaces]
 pre=tok.encode(context,add_special_tokens=False).ids;tar=tok.encode(context+target,add_special_tokens=False).ids[len(pre):]
 assert pre and tar and len(pre)+len(tar)<=32768;return pre,tar

def main():
 data=rows();selection=read(OUT/'COHORT_PREREGISTRATION.json');assert [r['id'] for r in data[:64]]==[x['id'] for x in selection['ordered64']]
 semantic=read(OUT/'ASTRA_SEMANTIC_REVIEW.json');assert semantic['constructor_sha256']==H((OUT/'SEMANTIC_CONSTRUCTOR_REVIEW.json').read_bytes());approved={r['id']:r['approved_text'] for r in semantic['secondary_approved']}
 tp=tokenizer_path();assert H(tp.read_bytes())=='fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d';tok=Tokenizer.from_file(str(tp));demos=read(ROOT/'data/slcd_fixed_demonstrations.json');letter_prefix=joint_prefix(demos)
 full_prefix='\n\n'.join(question_prompt(d['question'],d['choices']['text'])+' '+d['choices']['text'][d['choices']['label'].index(d['answerKey'])] for d in demos['documents'])+'\n\n';assert H(letter_prefix.encode())==read(ROOT/'configs/r10_PROTOCOL.json')['joint_demonstrations_sha256']
 history_receipt=read(ROOT/'recurrent_decision_audit_v1/recovered/results/r10_development_initial_COMPLETE.json');assert history_receipt['status']=='complete' and H((ROOT/'recurrent_decision_audit_v1/recovered/results/r10_development_initial.jsonl').read_bytes())==history_receipt['score_file_sha256']
 history={r['id']:r for r in (json.loads(s) for s in (ROOT/'recurrent_decision_audit_v1/recovered/results/r10_development_initial.jsonl').read_text(encoding='utf8').splitlines())};assert len(history)==576
 tasks={};groups=[];constructor=[];audit=[];request_texts={};baseline_refs={};p0keys=[];matched_checks=[]
 def add(row,condition,question,choices,order=None,demo='letter',target='letter',expected=None,changed=None):
  prefix=letter_prefix if demo=='letter' else full_prefix;context,fields=render(prefix,question,choices);n=len(choices);order=list(range(n)) if order is None else order
  assert sorted(order)==list(range(n));gid=digest(dict(question_id=row['id'],condition=condition,demo=demo,target=target));consumers=[]
  for i,choice in enumerate(choices):
   text=' '+chr(65+i) if target=='letter' else ' '+choice;pre,tar=encode_pair(tok,context,text);ids=pre+tar[:-1];positions=list(range(len(pre)-1,len(pre)+len(tar)-1));key=digest(dict(ids=ids,positions=positions,depth=4,head_shape=[1,len(tar),2048],primitive='original_uncached_no_explicit_mask_BF16_native_depth4',model_revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'))
   cid=digest(dict(group=gid,candidate=i));request_texts[cid]=dict(context=context,continuation=text);consumer=dict(consumer_id=cid,target_ids=tar,char_denominator=len(choice) if target=='full' else 1)
   if key not in tasks:tasks[key]=dict(task_key=key,input_ids=ids,positions=positions,consumers=[])
   tasks[key]['consumers'].append(consumer);consumers.append(dict(consumer_id=cid,task_key=key,display_index=i))
   audit.append(dict(consumer_id=cid,group_id=gid,condition=condition,id=row['id'],candidate_display_index=i,target_text=text,target_ids=tar,target_ids_sha256=digest(tar),prefix_ids_sha256=digest(pre),input_ids_sha256=digest(ids),context_sha256=H(context.encode()),request_sha256=digest((context,text)),target_token_count=len(tar),model_input_token_count=len(ids),char_denominator=consumer['char_denominator'],positions=positions))
  group=dict(group_id=gid,id=row['id'],condition=condition,demo_format=demo,target_format=target,order_to_original=order,nchoices=n,consumers=consumers,question_text=question,choices=choices,expected_original_content_index=row['gold'] if expected is None else expected,expected_display_index=order.index(row['gold'] if expected is None else expected),changed_wrong_index=changed,prompt_text=context,prompt_sha256=H(context.encode()),source_fields=fields)
  groups.append(group)
  if condition in ['P0','P1']:
   shift=0 if condition=='P0' else 1;old=history[row['id']];rot=old['cyclic_rotations'][shift];assert rot['order']==order
   actual_requests=[digest((context,' '+chr(65+i))) for i in range(n)];known=old['auxiliary_request_sha256'] if shift==0 else rot['request_sha256'];assert actual_requests==known
   for i,c in enumerate(consumers):
    # Historical arrays already mapped to original content order; native baseline check uses displayed target order.
    baseline_refs[c['consumer_id']]=dict(K1=rot['sum_scores']['native_k1'][order[i]],K4=rot['sum_scores']['native_k4'][order[i]],Fixed=rot['sum_scores']['loopcd_k4_fixed'][order[i]])
   if condition=='P0':p0keys.extend(c['task_key'] for c in consumers)
  return group
 for rank,row in enumerate(data[:24]):
  n=len(row['choices']);assert len(set(row['choices']))==n;gold=row['gold'];j=wrong_index(row);identity=list(range(n));cycle=[(i+1)%n for i in range(n)];swap=identity.copy();swap[gold],swap[j]=swap[j],swap[gold];cyclics={tuple((i+s)%n for i in range(n)) for s in range(n)};priority=[list(x) for x in itertools.permutations(range(n)) if x not in cyclics];noncyclic=next(p for p in priority if p!=swap)
  p0=add(row,'P0',row['question'],row['choices']);add(row,'P1',row['question'],[row['choices'][i] for i in cycle],cycle);add(row,'P2',row['question'],[row['choices'][i] for i in swap],swap);add(row,'P3',row['question'],[row['choices'][i] for i in noncyclic],noncyclic)
  modified=list(row['choices']);modified[j]='Not a valid answer';assert modified[gold]==row['choices'][gold];add(row,'Q_placeholder',row['question'],modified,changed=j);add(row,'N_original','N/A',row['choices']);add(row,'N_placeholder','N/A',modified,changed=j)
  secondary=None
  if rank<4:
   replacement=approved[row['id']];mc=list(row['choices']);mc[j]=replacement;checks=[]
   for qtext in [row['question'],'N/A']:
    a,fa=render(letter_prefix,qtext,row['choices']);b,fb=render(letter_prefix,qtext,mc);ea=tok.encode(a,add_special_tokens=False);eb=tok.encode(b,add_special_tokens=False);ca=field_counts(tok,a,fa,j);cb=field_counts(tok,b,fb,j);pa,pb=ca['positions'],cb['positions'];passed=bool(pa and pb and pa==pb and len(ea.ids)==len(eb.ids) and ea.ids[:pa[0]]==eb.ids[:pb[0]] and ea.ids[pa[-1]+1:]==eb.ids[pb[-1]+1:]);checks.append(dict(question_present=qtext!='N/A',passed=passed,original_tokens=len(ea.ids),modified_tokens=len(eb.ids),original_field_positions=pa,modified_field_positions=pb,prefix_ids_exact=ea.ids[:pa[0]]==eb.ids[:pb[0]],suffix_ids_positions_exact=ea.ids[pa[-1]+1:]==eb.ids[pb[-1]+1:],boundary_tokens=[k for k in pa if ea.offsets[k][0]<fa['options'][j]['start'] or ea.offsets[k][1]>fa['options'][j]['end']]))
   eligible=all(x['passed'] for x in checks);matched_checks.append(dict(id=row['id'],rank=rank,replacement=replacement,eligible=eligible,checks=checks))
   if eligible:add(row,'Q_matched',row['question'],mc,changed=j);add(row,'N_matched','N/A',mc,changed=j);secondary=replacement
  if rank<4:
   add(row,'B5_L_full',row['question'],row['choices'],demo='letter',target='full');add(row,'B5_F_letter',row['question'],row['choices'],demo='full',target='letter');add(row,'B5_F_full',row['question'],row['choices'],demo='full',target='full')
   colors=['red','blue','green','yellow','white'][:n];target_index=rank%n
   add(row,'B6_lookup','Which option contains the word "'+colors[target_index]+'"?',colors,expected=target_index);add(row,'B6_copy','For this instruction, select option '+chr(65+target_index)+'.',colors,expected=target_index)
  constructor.append(dict(original_joint_display_labels=[chr(65+i) for i in range(n)],original_dataset_source_labels=row['source_doc']['choices']['label'],rank=rank,id=row['id'],question=row['question'],choices=row['choices'],gold=gold,wrong_slot=j,P0_order=identity,P1_order=cycle,P2_order=swap,P3_order=noncyclic,P3_skipped=[p for p in priority[:priority.index(noncyclic)]],P2_new_gold_display=swap.index(gold),secondary=secondary,secondary_scope='first4_only' if rank<4 else 'scope_not_attempted'))
 ordered_p0=list(dict.fromkeys(p0keys));ordered=ordered_p0+[k for k in tasks if k not in set(ordered_p0)];tasklist=[tasks[k] for k in ordered]
 # Inference receives only numerical input IDs, positions, target IDs/denominators, opaque consumers and baseline numerical parity references.
 text_bytes=(json.dumps(request_texts,separators=(',',':'),ensure_ascii=True)+'\n').encode('utf8');(OUT/'INFERENCE_REQUEST_TEXTS.json').write_bytes(text_bytes)
 plan=dict(request_texts_sha256=H(text_bytes),status='CPU_constructed_no_GPU_authorization',tasks=tasklist,baseline_task_keys=ordered_p0,historical_endpoint_references=baseline_refs,model_identity=read(ROOT/'configs/r11_PROTOCOL.json')['model_identity'],primitive='native_depth4_original_uncached_BF16_no_explicit_mask_complete_target_FP32',inference_inputs_sha256={p:H((ROOT/p).read_bytes()) for p in ['scripts/runtime.py','scripts/ouro_cached_scoring.py','scripts/weight_edits.py','scripts/r7_inference.py','recurrent_decision_audit_v1/collect_a4.py','configs/r11_PROTOCOL.json']})
 sidecar=dict(status='pre_inference_constructor_labels_not_read_by_worker',selection=selection,cohort=constructor,groups=groups,demo_ids=demos['ids'],demo_order_unchanged=True,letter_demo_prefix_sha256=H(letter_prefix.encode()),fullanswer_demo_prefix_sha256=H(full_prefix.encode()),secondary_audit=matched_checks,semantic_review_sha256=H((OUT/'ASTRA_SEMANTIC_REVIEW.json').read_bytes()))
 summaries=dict(logical_groups=len(groups),logical_candidates=sum(len(g['consumers']) for g in groups),unique_body_forwards=len(tasklist),baseline_body_forwards=len(ordered_p0),intervention_body_forwards=len(tasklist)-len(ordered_p0),native_head_calls=4*len(tasklist),native_head_prediction_positions=4*sum(len(t['positions']) for t in tasklist),input_tokens=sum(len(t['input_ids']) for t in tasklist),squared_input_tokens=sum(len(t['input_ids'])**2 for t in tasklist),min_input_tokens=min(len(t['input_ids']) for t in tasklist),max_input_tokens=max(len(t['input_ids']) for t in tasklist),target_token_lengths=dict(collections.Counter(len(c['target_ids']) for t in tasklist for c in t['consumers'])),groups_by_condition=dict(collections.Counter(g['condition'] for g in groups)),secondary_eligible=sum(x['eligible'] for x in matched_checks),secondary_scope_not_attempted=20,pilot_nchoices=dict(collections.Counter(len(r['choices']) for r in data[:24])),raw_all_heads_BF16_upper_bytes=4*sum(len(t['positions']) for t in tasklist)*49152*2,raw_native_states_BF16_upper_bytes=4*sum(len(t['positions']) for t in tasklist)*2048*2)
 timing=read(ROOT/'recurrent_decision_audit_v1/a4_TIMINGS.json');warm=[t for t in timing if t['purpose']=='parity_native4'][1:];rate=max(t['seconds']/t['length']**2 for t in warm);bodyforecast=1.5*rate*summaries['squared_input_tokens'];extra_positions=sum(max(0,len(t['positions'])-1) for t in tasklist);reserve=20.+.02*4*extra_positions
 cost=dict(policy='predeclared A7warm max(seconds/L²)*1.5shared-load safety*sum L² +startup/cold60 +final60 +head-target reserve20+0.02sec per additional native head prediction position',A_warm_records=warm,A_warm_timing_sha256=H((ROOT/'recurrent_decision_audit_v1/a4_TIMINGS.json').read_bytes()),rate_seconds_per_L2=rate,body_forecast_seconds=bodyforecast,startup_cold_reserve_seconds=60.,final_hash_IO_reserve_seconds=60.,head_target_reserve_seconds=reserve,forecast_total_seconds=bodyforecast+120.+reserve,proposed_hard_seconds=680,remaining_global_seconds=710.6733033489436,admission_not_a_physical_time_guarantee=True)
 protocol=dict(status='CPU_prepared_pending_source_review_no_GPU',pilot_questions=24,selection_algorithm=selection['algorithm'],B5_B6_bounded_questions=4,secondary_bounded_questions=4,secondary_semantic_review=semantic['status'],primary_scoring='nativeK1..4 complete target FP32 LP sum; letters unnormalized; full-answer both raw and original-body-char-normalized, charnorm designated full-answer primary and raw diagnostic',H1='All unchanged option I; every adjacent deltaI; all unchanged pairs J and meanabs/RMS; Q-vs-N/A DID; gold-relative direction separately, no sign or stage selection',H2='Displayed argmax then original-content mapping; letter-invariance separately; moved-P0-content denominator reported without excluding other24',H3='B6first4 distinct lookup/copy tasks; B5first4 complete2x2, ceiling and format confounds explicit',summary=summaries,cost=cost,constructor_source_sha256=H(Path(__file__).read_bytes()))
 for n,v in [('INFERENCE_TASK_PLAN.json',plan),('CONSTRUCTOR_WITH_LABELS.json',sidecar),('prompt_and_target_audit.json',dict(status='complete_CPU_token_and_mapping_audit',records=audit,matched_pair_boundary_checks=matched_checks)),('intervention_protocol.json',protocol)]:
  (OUT/n).write_text(json.dumps(v,separators=(',',':'),ensure_ascii=True)+'\n',encoding='utf8')
 print(json.dumps(dict(summary=summaries,cost=cost),indent=2))
if __name__=='__main__':main()
