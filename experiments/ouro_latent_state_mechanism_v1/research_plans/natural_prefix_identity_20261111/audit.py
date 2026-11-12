"""Natural-cohort inputs only. No PQR/ROUND_3 effect data read."""
import json,hashlib,pathlib,datetime
P=pathlib.Path('/home/luansiyu/loop-latent');O=pathlib.Path(__file__).parent;N=P/'orchestration_runs/run-20261107T053919Z-6d01138e';R=P/'orchestration_runs/resume-test-20261111';L=P/'research_plans/prompt_boundary_audit_20261111/boundary_ledger.jsonl';S=P/'research_plans/prefix_question_partition_draft_20261111/saved_offset_coverage.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=lambda v:hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest();saved=json.loads(S.read_text());assert saved['status']=='ALL_SAVED_OFFSET_PARTITIONS_PASS_INPUTS_ONLY' and saved['boundary_ledger_sha256']==sha(L);ledger={}
for x in L.open():
 r=json.loads(x);key=(r['split'],r['id']);assert key not in ledger;ledger[key]=r
reference=None;reference_hash=None;stats={};missing_fields={};first_problem=None
with (O/'candidate_ledger.jsonl').open('x') as out:
 for split,expected_questions,expected_candidates in [('train',1119,4476),('validation',299,1194),('test',1172,4687)]:
  cp=(R if split=='test' else N)/('coverage_'+split+'.json');before=sha(cp);coverage=json.loads(cp.read_text());qs=coverage['questions'];assert len(qs)==expected_questions and len({q['id'] for q in qs})==expected_questions and saved['stats'][split]['natural_coverage_sha256']==before
  counts={'questions':0,'candidates':0,'matching_prefix_ids':0,'matching_prefix_positions':0,'mask_omitted_batch_one':0,'first_issue':None,'candidate_errors':0}
  for q in qs:
   ident=q['id'];row=ledger[(split,ident)];path=N/'shards'/split/(hashlib.sha256(ident.encode()).hexdigest()+'.json');assert sha(path)==q['metadata_sha256'];m=json.loads(path.read_text());assert m['id']==ident and m['split']==split and q['status']=='complete';pidx=row['groups']['prefix'];assert pidx==list(range(1839)) and not row['groups']['crossing'] and not row['groups']['zero_offset'];records=sorted(m['records'],key=lambda r:r['candidate']);assert [r['candidate'] for r in records]==list(range(len(records)));counts['questions']+=1
   for rec in records:
    counts['candidates']+=1;issues=[];enc=rec['encoding'];ids=enc['input_ids'];prefix=[ids[i] for i in pidx];prefix_sha=digest(prefix)
    if reference is None:reference=prefix;reference_hash=prefix_sha
    first_diff=next((i for i,(a,b) in enumerate(zip(prefix,reference)) if a!=b),None)
    if len(prefix)!=len(reference):issues.append('prefix length');first_diff=min(len(prefix),len(reference))
    if first_diff is None and prefix==reference:counts['matching_prefix_ids']+=1
    else:issues.append('prefix token mismatch')
    if 'effective_position_ids' not in rec:missing_fields['effective_position_ids']=missing_fields.get('effective_position_ids',0)+1;positions_ok=False
    else:
     pos=rec['effective_position_ids'];positions_ok=len(pos)==len(ids) and [pos[i] for i in pidx]==pidx
     if positions_ok:counts['matching_prefix_positions']+=1
     else:issues.append('effective position mismatch')
    mask_ok=('attention_mask_argument' in rec and rec['attention_mask_argument'] is None and rec.get('batch_size')==1 and rec.get('mask_semantics')=='argument omitted (native unpadded batch of one)')
    if mask_ok:counts['mask_omitted_batch_one']+=1
    else:issues.append('mask metadata missing or different')
    if 'past_key_values_argument' not in rec:missing_fields['past_key_values_argument']=missing_fields.get('past_key_values_argument',0)+1
    if 'effective_attention_mask' not in rec:missing_fields['effective_attention_mask']=missing_fields.get('effective_attention_mask',0)+1
    if 'rope_cos_sin' not in rec:missing_fields['rope_cos_sin']=missing_fields.get('rope_cos_sin',0)+1
    if m.get('runtime',{}).get('use_cache') is not False:issues.append('use_cache metadata mismatch')
    if issues:
     counts['candidate_errors']+=1
     if counts['first_issue'] is None:counts['first_issue']={'id':ident,'candidate':rec['candidate'],'first_token_difference':first_diff,'issues':issues}
     if first_problem is None:first_problem={'split':split,'id':ident,'candidate':rec['candidate'],'first_token_difference':first_diff,'issues':issues}
    out.write(json.dumps({'split':split,'id':ident,'candidate':rec['candidate'],'natural_metadata_sha256':q['metadata_sha256'],'prefix_token_count':len(prefix),'prefix_ids_sha256':prefix_sha,'prefix_positions_sha256':digest([rec['effective_position_ids'][i] for i in pidx]) if positions_ok else None,'positions_match':positions_ok,'mask_omitted_unpadded_batch_one':mask_ok,'first_token_difference':first_diff,'issues':issues},sort_keys=True,separators=(',',':'))+'\n')
  assert counts['questions']==expected_questions and counts['candidates']==expected_candidates and sha(cp)==before;stats[split]=counts
assert reference is not None
with (O/'reference_prefix_ids.json').open('x') as f:json.dump({'input_ids':reference,'sha256':reference_hash,'length':len(reference)},f,sort_keys=True);f.write('\n')
config=P/'model/config.json';model=P/'model/modeling_ouro.py';cohort=P/'repo/experiments/ouro_latent_state_mechanism_v1/cohort_runner.py';cfg=json.loads(config.read_text());report={'status':'NATURAL_PREFIX_INPUT_IDENTITY_AUDITED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stats':stats,'first_issue':first_problem,'missing_metadata_fields':missing_fields,'reference_prefix_sha256':reference_hash,'reference_token_count':len(reference),'reference_prefix_ids_file_sha256':sha(O/'reference_prefix_ids.json'),'candidate_ledger_sha256':sha(O/'candidate_ledger.jsonl'),'boundary_ledger_sha256':sha(L),'saved_offset_coverage_sha256':sha(S),'model_config_sha256':sha(config),'model_source_sha256':sha(model),'cohort_source_sha256':sha(cohort),'audit_source_sha256':sha(pathlib.Path(__file__)),'config_rope_scaling':cfg.get('rope_scaling'),'config_rope_theta':cfg.get('rope_theta'),'config_layer_types':sorted(set(cfg.get('layer_types',[]))),'limits':'Actual effective attention-mask tensor, actual past_key_values argument and computed RoPE cos/sin were not stored in metadata. No hidden states are compared. Causal prefix invariance is conditional on actual strict causal no-cache execution, identical prefix positions and identical effective RoPE; dynamic RoPE needs separate consideration.'}
with (O/'summary.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'stats':stats,'missing_fields':missing_fields,'first_issue':first_problem,'reference_prefix_sha256':reference_hash,'ledger_sha256':report['candidate_ledger_sha256'],'summary_sha256':sha(O/'summary.json'),'audit_sha256':report['audit_source_sha256'],'rope_scaling':report['config_rope_scaling']},sort_keys=True))
