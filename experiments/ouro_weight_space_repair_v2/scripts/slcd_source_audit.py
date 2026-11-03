"""Fail-closed CPU posthoc provenance audit of the two INITIAL training jobs.

No validation/test access, no model instantiation, no GPU, no mkdir. This adds
evidence; it does not retroactively enlarge the worker's original signature.
Run only after both initial COMPLETE receipts exist. Existing output is immutable.
"""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
REVISION='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
MODEL_SHA='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
BASE_METHODS={f'native_k{k}' for k in range(1,7)}|{'loopcd_k4_fixed','loopcd_k4_adaptive'}


def require(ok,message):
    if not ok:raise ValueError(message)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def jsonl(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line]


def indexed(records,n,label):
    require(len(records)==n,label+' count mismatch')
    result={r['id']:r for r in records}
    require(len(result)==n,label+' duplicate IDs')
    require(all(q.startswith('arc_challenge/train/') for q in result),label+' contains a non-training ID')
    return result


def tensor_sha(tensor):
    import torch
    value=tensor.detach().cpu().contiguous()
    flat=torch.empty(value.numel(),dtype=value.dtype,device='cpu')
    flat.copy_(value.reshape(-1))
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def verify_method(method,row,label):
    lengths=row['choice_char_lengths'];s=method['sum_scores'];z=method['normalized_scores'];gold=row['gold']
    require(len(s)==len(z)==len(lengths)>=2,label+' candidate count mismatch')
    require(all(math.isfinite(x) for x in s+z),label+' nonfinite score')
    require(all(a==b/max(n,1) for a,b,n in zip(z,s,lengths)),label+' character normalization mismatch')
    prediction=max(range(len(z)),key=lambda i:z[i])
    margin=z[gold]-max(x for i,x in enumerate(z) if i!=gold)
    require(method['prediction']==prediction and method['correct']==(prediction==gold),label+' outcome mismatch')
    require(method['margin']==margin,label+' margin mismatch')


def main():
    # These gates execute before importing tokenizer/torch or reading any data.
    complete_paths={s:ROOT/f'results/slcd_tokens_{s}_initial_COMPLETE.json' for s in ('discovery','selection')}
    for split,path in complete_paths.items():
        if not path.is_file():raise FileNotFoundError('Audit stops immediately: missing '+str(path))
    output=ROOT/'results/slcd_SOURCE_AUDIT.json'
    require(output.parent.is_dir(),'Existing results directory required')
    if output.exists():raise FileExistsError('Preserve existing audit receipt: '+str(output))
    complete={s:read(p) for s,p in complete_paths.items()}
    for split,n in [('discovery',384),('selection',192)]:
        require(complete[split]['status']=='complete' and complete[split]['n']==n,'Incomplete '+split)
    os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from lm_eval.api.model import TemplateLM
    import importlib.metadata
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    model_dir=ROOT/'models/Ouro-1.4B'
    model_path=model_dir/'model.safetensors'
    require(file_sha(model_path)==MODEL_SHA,'Original model safetensors SHA mismatch')
    require(read(model_dir/'snapshot_metadata.json')['sha']==REVISION,'Snapshot revision mismatch')
    with safe_open(model_path,framework='pt',device='cpu') as tensors:
        head=tensors.get_tensor('lm_head.weight')
        require(head.device.type=='cpu' and head.dtype==torch.bfloat16 and list(head.shape)==[49152,2048],'Unexpected original head')
        head_sha=tensor_sha(head)
    del head
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    coder=SimpleNamespace(tok_encode=lambda text,**kwargs:tokenizer.encode(text,add_special_tokens=False))
    pair_source=inspect.getsource(TemplateLM._encode_pair)
    old_provenance_path=ROOT/'edited_weights/statistics/provenance.json'
    old_provenance=read(old_provenance_path)
    require(old_provenance['checkpoint_file_sha256']==MODEL_SHA,'Discovery provenance checkpoint mismatch')
    old_states=indexed(old_provenance['statistics_files'],384,'Old discovery state provenance')
    recipe_path=ROOT/'configs/slcd_TOKEN_INITIAL.json';recipe_document=read(recipe_path)
    recipes=recipe_document['recipes'];recipe_by_id={r['id']:r for r in recipes}
    require(len(recipe_by_id)==len(recipes),'Duplicate recipe ID')
    for name,kind,omega in [('tf_fixed','fixed',.5),('tf_adaptive','adaptive',1.),('A2_g0p0','two_sided_logit',.5)]:
        spec=recipe_by_id[name]
        require(spec['kind']==kind and spec.get('omega',1. if kind=='adaptive' else .5)==omega,'Baseline recipe definition changed')
        require(spec.get('reference',1)==1 and spec.get('final',4)==4 and spec.get('gamma',0)==0,'Baseline depth/gamma changed')
    code_names={p.name for p in (ROOT/'scripts').glob('slcd_*.py')}
    code_names.update(['runtime.py','ouro_cached_scoring.py','weight_edits.py','collect_statistics.py'])
    code_hashes={name:file_sha(ROOT/'scripts'/name) for name in sorted(code_names)}
    tokenizer_hashes={p.name:file_sha(p) for p in sorted(model_dir.iterdir()) if p.is_file() and
                     p.name!='model.safetensors' and (p.suffix in ('.json','.model','.txt','.py') or 'tokenizer' in p.name)}
    all_states=[];split_audits={};split_id_sets={}
    for split,n,original_manifest in [('discovery',384,'discovery_expanded.jsonl'),('selection',192,'selection.jsonl')]:
        data_path=ROOT/f'data/slcd_{split}.jsonl'
        original_data_path=ROOT/'data'/original_manifest
        ref_path=ROOT/f'results/slcd_original_{split}.jsonl'
        scores_path=ROOT/f'results/slcd_tokens_{split}_initial.jsonl'
        data_sha=file_sha(data_path)
        require(data_sha==file_sha(original_data_path),'SLCD input differs from original training manifest: '+split)
        data=indexed(jsonl(data_path),n,split+' input')
        reference=indexed(jsonl(ref_path),n,split+' reference')
        scored=indexed(jsonl(scores_path),n,split+' token scores')
        require(set(data)==set(reference)==set(scored),split+' ID sets differ')
        split_id_sets[split]=set(data)
        if split=='discovery':require(set(data)==set(old_states),'Discovery state provenance IDs differ')
        receipt=complete[split];provenance=receipt['provenance']
        require(receipt['score_file']==scores_path.relative_to(ROOT).as_posix(),'Score path mismatch')
        require(receipt['score_file_sha256']==file_sha(scores_path),'Score file digest mismatch')
        require(provenance['model_revision']==REVISION and provenance['data_sha256']==data_sha,'Worker input provenance mismatch')
        require(provenance['recipes_sha256']==file_sha(recipe_path),'Worker recipe hash mismatch')
        require(set(provenance['implementation'])=={'slcd_token_worker.py','slcd_logit_methods.py','runtime.py','ouro_cached_scoring.py'},'Unexpected initial worker source set')
        for name,sha in provenance['implementation'].items():require(code_hashes[name]==sha,'Frozen worker source changed: '+name)
        signature=digest(provenance)
        require(signature==receipt['implementation_signature'],'Worker signature mismatch')
        require(receipt['original_head_sha256']==head_sha and receipt['model_weights_modified'] is False,'Worker head identity mismatch')
        current_states=[];parity_rechecks=0;target_tokens=0
        for qid,row in data.items():
            ref=reference[qid];score=scored[qid]
            require(isinstance(row['gold'],int) and 0<=row['gold']<len(row['choice_char_lengths']),'Invalid gold index')
            require(row['prompt_sha256']==hashlib.sha256(row['prompt'].encode()).hexdigest(),'Manifest prompt hash mismatch')
            require(len(row['request_pairs'])==len(row['choice_char_lengths']),'Request/choice count mismatch')
            require(all(pair[0]==row['prompt'] for pair in row['request_pairs']),'Candidate context mismatch')
            for item,label in [(ref,'reference'),(score,'token')]:
                require(item['gold']==row['gold'] and item['prompt_sha256']==row['prompt_sha256'],label+' input mismatch')
                require(item['choice_char_lengths']==row['choice_char_lengths'],label+' normalization lengths mismatch')
            require(score['scope']==split and score['implementation_signature']==signature and score['data_sha256']==data_sha,'Row provenance mismatch')
            require(score['original_head_sha256']==head_sha and score['weights_modified'] is False,'Row head/weight mismatch')
            errors=score['native_parity_errors']
            require(set(errors)==BASE_METHODS and all(isinstance(v,(int,float)) and v==0 for v in errors.values()),'Require all eight exact parity errors: '+qid)
            require(set(score['original_methods'])==BASE_METHODS,'Missing recomputed baseline method')
            require(set(score['methods'])==set(recipe_by_id),'Token recipe set mismatch')
            for name in BASE_METHODS:
                verify_method(ref['methods'][name],row,'reference '+name)
                verify_method(score['original_methods'][name],row,'recomputed '+name)
                require(score['original_methods'][name]['sum_scores']==ref['methods'][name]['sum_scores'],'Nonzero rechecked baseline error: '+qid+' '+name)
                parity_rechecks+=1
            for name,value in score['methods'].items():verify_method(value,row,'token '+name)
            for name,baseline in [('tf_fixed','loopcd_k4_fixed'),('tf_adaptive','loopcd_k4_adaptive'),('A2_g0p0','loopcd_k4_fixed')]:
                require(score['methods'][name]['sum_scores']==ref['methods'][baseline]['sum_scores'],'Token/baseline exact identity failed: '+name)
            stem=qid.rsplit('/',1)[-1]
            expected_file=('results/training_statistics/'+old_states[qid]['file'] if split=='discovery'
                           else 'results/slcd_state_selection_'+stem+'.pt')
            require(score['state_file']==expected_file,'Unexpected state path')
            state_path=(ROOT/expected_file).resolve()
            require(state_path.is_relative_to(ROOT.resolve()) and state_path.is_file(),'State path escapes V2 or is missing')
            state_sha=file_sha(state_path)
            if split=='discovery':require(state_sha==old_states[qid]['sha256'],'Old discovery state content mismatch: '+qid)
            payload=torch.load(state_path,map_location='cpu',weights_only=True)
            require(payload['id']==qid,'State ID mismatch')
            if split=='discovery':
                require(payload['loops']==[3,4,5,6] and payload['implementation_sha256']==ref['implementation_sha256'],'Old state implementation mismatch')
            else:
                require(payload['data_sha256']==data_sha and payload['model_revision']==REVISION and
                        payload['prompt_sha256']==row['prompt_sha256'] and payload['original_weights'] is True,'Selection state metadata mismatch')
            states=payload['candidate_states'];require(len(states)==len(row['request_pairs']),'State choice count mismatch')
            target_hashes=[];counts=[]
            for i,(state,pair) in enumerate(zip(states,row['request_pairs'])):
                require(state['candidate']==i and len(state['normalized'])==6,'Candidate/depth mismatch')
                prefix,target=TemplateLM._encode_pair(coder,*pair)
                require(prefix and target and state['target_ids']==target,'State targets differ from exact original tokenizer encoding')
                require(all(isinstance(t,int) and 0<=t<49152 for t in target),'Target outside vocabulary')
                counts.append(len(target));target_tokens+=len(target);target_hashes.append(digest(target))
                for h in state['normalized']:
                    require(h.device.type=='cpu' and h.dtype==torch.bfloat16 and list(h.shape)==[1,len(target),2048], 'State tensor shape/dtype mismatch')
                    require(bool(torch.isfinite(h).all()),'Nonfinite normalized state')
            require(counts==score['choice_token_counts']==ref['choice_token_counts'],'Reference/token/state token counts differ')
            item=dict(id=qid,split=split,file=expected_file,bytes=state_path.stat().st_size,sha256=state_sha,
                      candidate_count=len(states),choice_token_counts=counts,target_id_sha256=target_hashes,
                      content_binding='verified_against_preexisting_discovery_provenance' if split=='discovery' else 'posthoc_snapshot_after_initial_selection',
                      native_methods_rechecked=8,baseline_max_sum_score_error=0.)
            current_states.append(item);all_states.append(item)
            del payload,states
        require(len({x['file'] for x in current_states})==n,'Duplicate state files')
        split_audits[split]=dict(n=n,ids_sha256=digest(sorted(data)),data_sha256=data_sha,
            original_input_manifest=original_data_path.relative_to(ROOT).as_posix(),original_input_manifest_sha256=file_sha(original_data_path),
            reference_sha256=file_sha(ref_path),score_file_sha256=file_sha(scores_path),
            complete_receipt_sha256=file_sha(complete_paths[split]),preexisting_worker_signature=signature,
            preexisting_signature_fields=sorted(provenance),preexisting_source_hashes=provenance['implementation'],
            exact_baseline_comparisons_rechecked=parity_rechecks,all_eight_baselines_per_question_exact=True,
            fixed_adaptive_gamma0_token_identity_exact=True,target_token_ids_reencoded=True,total_target_tokens=target_tokens,
            state_count=n,complete_receipt_state_index_count=len(receipt.get('states',[])),
            state_index_reconstructed_from_all_score_rows=True)
    require(not split_id_sets['discovery']&split_id_sets['selection'],'Discovery/selection overlap')
    result=dict(status='passed',scope='train discovery384 and selection192 initial jobs only',
        no_validation_or_test_access=True,cpu_only=True,no_model_instantiation=True,model_revision=REVISION,
        model_safetensors_sha256=MODEL_SHA,original_head_raw_tensor_sha256=head_sha,
        code_sha256=code_hashes,tokenizer_and_model_metadata_sha256=tokenizer_hashes,
        encode_pair_source_sha256=hashlib.sha256(pair_source.encode()).hexdigest(),
        package_versions={k:importlib.metadata.version(k) for k in ['torch','transformers','tokenizers','lm_eval','safetensors']},
        old_discovery_provenance_sha256=file_sha(old_provenance_path),initial_recipes_sha256=file_sha(recipe_path),
        split_audits=split_audits,states=all_states,state_index_sha256=digest(all_states),
        posthoc_binding_disclosure=dict(reference_files=True,selection_state_content=True,additional_helpers=True,
            original_worker_signature_not_retroactively_expanded=True,
            limitation='Selection payloads did not record their generator source hash. This audit records current frozen worker/scorer/helper hashes, exact original-score agreement and a complete state snapshot; it cannot create a missing contemporaneous payload field.'),
        all576_state_files_hashed=True,all_target_sequences_for576_questions_verified_with_original_tokenizer=True)
    temporary=output.with_suffix('.json.tmp')
    require(not temporary.exists(),'Preserve partial audit output')
    with temporary.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    temporary.replace(output)
    print(json.dumps(dict(status='passed',questions=576,states=576,output=output.relative_to(ROOT).as_posix(),sha256=file_sha(output))),flush=True)


if __name__=='__main__':main()
