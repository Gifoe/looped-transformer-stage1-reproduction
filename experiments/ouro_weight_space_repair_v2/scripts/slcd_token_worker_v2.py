"""Original-weight token recipes with strict authorization and cache receipts.

Preserves the initial worker. No mkdir, edited bundle or parameter modification.
New states are flat slcd_state_{split}_{job}_{stem}.pt files. Cache reuse requires
SOURCE_AUDIT or an explicit completed v2 generation receipt (--state-receipt).
Validation/test authorization is checked before parsing their manifest or scores.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
from slcd_core import ROOT,REVISION,digest

MODEL_SHA='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
SIZES={'discovery':384,'selection':192,'confirmation':517,'validation':299,'test':1172}
BASE_METHODS={f'native_k{k}' for k in range(1,7)}|{'loopcd_k4_fixed','loopcd_k4_adaptive'}
LOCK_SOURCES={'slcd_token_worker_v2.py','slcd_logit_methods.py','runtime.py','ouro_cached_scoring.py','slcd_core.py'}


def require(ok,message):
    if not ok:raise ValueError(message)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x]


def inside(relative):
    p=(ROOT/relative).resolve()
    require(p.is_relative_to(ROOT.resolve()),'Path outside V2')
    return p


def save_flat(relative,value,immutable=False):
    p=inside(relative)
    require(p.parent.is_dir() and p.name.startswith('slcd_'),'Only slcd_ files in existing parents may be written')
    if immutable and p.exists():
        require(read(p)==value,'Immutable receipt changed: '+str(p));return
    tmp=p.with_suffix(p.suffix+'.tmp')
    require(not tmp.exists(),'Preserve partial write: '+str(tmp))
    with tmp.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
    tmp.replace(p)


def authorize(split,lock,recipe_sha,data_sha,implementation):
    """Pure authorization check; caller checks lock existence before data parsing."""
    if split not in ('validation','test'):return
    require(recipe_sha in lock['allowed_recipe_sha256'],'Recipe file is not frozen/authorized')
    require(lock['dataset_sha256'][split]==data_sha,'Held-out manifest is not frozen/authorized')
    allowed=lock['inference_source_sha256']
    require(LOCK_SOURCES<=set(allowed),'Lock omits required inference source hashes')
    for name,expected in allowed.items():
        require(name in implementation and implementation[name]==expected,'Locked inference source differs: '+name)


def index_rows(values,n,label):
    require(len(values)==n,label+' count mismatch')
    result={r['id']:r for r in values}
    require(len(result)==n,label+' duplicate IDs')
    return result


def parity_errors(originals,ref,nchoices):
    require(set(originals)==BASE_METHODS,'Missing recomputed baseline')
    require(BASE_METHODS<=set(ref['methods']),'Reference omits baseline method')
    errors={}
    for name in sorted(BASE_METHODS):
        a=originals[name]['sum_scores'];b=ref['methods'][name]['sum_scores']
        require(len(a)==len(b)==nchoices,'Baseline candidate count mismatch')
        require(all(math.isfinite(x) for x in a+b),'Nonfinite baseline score')
        errors[name]=max(abs(x-y) for x,y in zip(a,b))
    require(all(v==0 for v in errors.values()),'Independent baseline parity failed: '+str(errors))
    return errors


def validate_recipes(recipes):
    by_id={r['id']:r for r in recipes}
    require(1<=len(recipes)==len(by_id)<=64,'Invalid recipe count or duplicate ID')
    for name,kind,omega in [('tf_fixed','fixed',.5),('tf_adaptive','adaptive',1.)]:
        r=by_id[name]
        require(r['kind']==kind and r.get('omega',1. if kind=='adaptive' else .5)==omega,'Wrong mandatory baseline recipe')
        require(r.get('reference',1)==1 and r.get('final',4)==4 and r.get('gamma',0)==0,'Wrong mandatory baseline depths/gamma')
    return by_id


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=list(SIZES),required=True)
    ap.add_argument('--job',required=True)
    ap.add_argument('--recipes',default='configs/slcd_TOKEN_INITIAL.json')
    ap.add_argument('--reuse-states',action='store_true')
    ap.add_argument('--state-receipt',action='append',default=[],help='Completed v2 generation receipt; repeat if needed. Training SOURCE_AUDIT is automatic.')
    a=ap.parse_args();started=time.time()
    require(a.job.replace('_','').isalnum(),'Invalid job name')
    for relative in ['results','configs','data','logs','tmp','cache/hf','cache/xdg','cache/pycache','cache/torch','cache/torchinductor','cache/triton','cache/cuda']:
        require((ROOT/relative).is_dir(),'Required existing directory missing: '+relative)
    implementation={name:file_sha(ROOT/'scripts'/name) for name in sorted(LOCK_SOURCES|{'weight_edits.py'})}
    lock_path=None;lock=None
    if a.split in ('validation','test'):
        lock_path=ROOT/('configs/slcd_VALIDATION_LOCK.json' if a.split=='validation' else 'FINAL_METHOD_LOCK.json')
        require(lock_path.is_file(),'Held-out authorization lock missing')
        lock=read(lock_path)
    manifest=ROOT/f'data/slcd_{a.split}.jsonl';recipe_path=inside(a.recipes)
    data_sha=file_sha(manifest);recipe_sha=file_sha(recipe_path)
    authorize(a.split,lock,recipe_sha,data_sha,implementation)
    # No held-out data or reference is parsed before the authorization above.
    data=rows(manifest);expected=index_rows(data,SIZES[a.split],'Input')
    prefix='arc_challenge/'+('train' if a.split in ('discovery','selection','confirmation') else a.split)+'/'
    require(all(q.startswith(prefix) for q in expected),'Input split namespace mismatch')
    recipe_document=read(recipe_path);recipes=recipe_document['recipes'];recipe_by_id=validate_recipes(recipes)
    reference_path=ROOT/f'results/slcd_original_{a.split}.jsonl'
    reference=None
    if reference_path.exists():
        reference=index_rows(rows(reference_path),len(data),'Independent reference')
        require(set(reference)==set(expected),'Reference must cover the entire input ID set')
        for qid,row in expected.items():
            ref=reference[qid]
            require(ref['gold']==row['gold'] and ref['prompt_sha256']==row['prompt_sha256'] and
                    ref['choice_char_lengths']==row['choice_char_lengths'],'Reference input binding mismatch')
            require(BASE_METHODS<=set(ref['methods']),'Reference omits one of eight baselines')
    elif a.split in ('discovery','selection'):
        raise FileNotFoundError('Audited development reference is required: '+str(reference_path))
    model_dir=ROOT/'models/Ouro-1.4B';model_path=model_dir/'model.safetensors'
    require(file_sha(model_path)==MODEL_SHA,'Original checkpoint SHA mismatch')
    require(read(model_dir/'snapshot_metadata.json')['sha']==REVISION,'Checkpoint revision mismatch')
    model_metadata={p.name:file_sha(p) for p in sorted(model_dir.iterdir()) if p.is_file() and
                    p.name!='model.safetensors' and (p.suffix in ('.json','.py','.txt','.model') or 'tokenizer' in p.name)}
    model_identity=dict(revision=REVISION,safetensors_sha256=MODEL_SHA,metadata_and_tokenizer_sha256=model_metadata)
    # Cache authorities are completed snapshots, not arbitrary existing .pt files.
    authorities={};receipt_hashes={}
    authority_paths=[inside(p) for p in a.state_receipt]
    audit_path=ROOT/'results/slcd_SOURCE_AUDIT.json'
    if a.split in ('discovery','selection') and audit_path.exists():authority_paths.insert(0,audit_path)
    for path in authority_paths:
        require(path.is_file(),'Cache authority missing: '+str(path))
        r=read(path);rel=path.relative_to(ROOT).as_posix();receipt_hashes[rel]=file_sha(path)
        is_audit=path==audit_path
        if is_audit:
            require(r['status']=='passed' and r['model_safetensors_sha256']==MODEL_SHA,'Invalid SOURCE_AUDIT')
            require(r['split_audits'][a.split]['data_sha256']==data_sha,'SOURCE_AUDIT dataset mismatch')
            for name in ('runtime.py','ouro_cached_scoring.py'):
                require(r['code_sha256'][name]==implementation[name],'Audited state inference source changed')
        else:
            require(r['status']=='complete' and r.get('worker_format')=='slcd_token_worker_v2','Only complete v2 generation receipts authorize new caches')
            require(r['split']==a.split and r['n']==len(data),'Cache generation split/count mismatch')
            require(r['provenance']['model_identity']==model_identity and r['provenance']['data_sha256']==data_sha,'Cache generation model/data mismatch')
            for name in ('runtime.py','ouro_cached_scoring.py'):
                require(r['provenance']['implementation'][name]==implementation[name],'Cache inference source changed')
            require(file_sha(inside(r['score_file']))==r['score_file_sha256'],'Cache generation score receipt changed')
            require(len(r['states'])==len(data) and {s['id'] for s in r['states']}==set(expected),'Incomplete generation state index')
        for item in r['states']:
            if item.get('split',a.split)!=a.split:continue
            qid=item['id'];require(qid in expected,'Cache index unexpected ID')
            entry=dict(item);entry['_authority']=rel;entry['_kind']='source_audit' if is_audit else 'generation_complete'
            entry['_generator_provenance']=None if is_audit else r['provenance']
            if qid in authorities:
                require((authorities[qid]['file'],authorities[qid]['sha256'])==(entry['file'],entry['sha256']),'Conflicting cache authorities')
            authorities[qid]=entry
    if a.reuse_states:require(set(authorities)==set(expected),'--reuse-states requires complete authorized state coverage')
    provenance=dict(model_revision=REVISION,model_identity=model_identity,data_sha256=data_sha,
        recipes_sha256=recipe_sha,reference_sha256=file_sha(reference_path) if reference is not None else None,
        implementation=implementation,state_authority_sha256=receipt_hashes,
        authorization_lock_sha256=file_sha(lock_path) if lock_path else None)
    signature=digest(provenance)
    output=f'results/slcd_tokens_{a.split}_{a.job}.jsonl';path=ROOT/output
    completion=f'results/slcd_tokens_{a.split}_{a.job}_COMPLETE.json'
    if (ROOT/completion).exists():
        old=read(ROOT/completion)
        require(old['implementation_signature']==signature and old['score_file_sha256']==file_sha(path),'Existing completion changed')
        require(old['status']=='complete' and old['n']==len(data),'Incomplete prior completion')
        saved=index_rows(rows(path),len(data),'Completed scores')
        require(set(saved)==set(expected),'Completed IDs differ')
        rebuilt=[];coverage=0
        for row in data:
            r=saved[row['id']]
            require(r['implementation_signature']==signature and r['scope']==a.split and r['data_sha256']==data_sha,'Completed row provenance mismatch')
            require(r['gold']==row['gold'] and r['prompt_sha256']==row['prompt_sha256'] and r['choice_char_lengths']==row['choice_char_lengths'],'Completed row input mismatch')
            require(r['original_head_sha256']==old['original_head_sha256'] and r['weights_modified'] is False,'Completed row head mismatch')
            require(set(r['methods'])==set(recipe_by_id),'Completed recipe set mismatch')
            if reference is not None:
                require(r['choice_token_counts']==reference[row['id']]['choice_token_counts'],'Completed reference token-count mismatch')
                require(r['native_parity_errors']==parity_errors(r['original_methods'],reference[row['id']],len(row['choice_char_lengths'])),'Completed parity mismatch');coverage+=1
            else:require(r['native_parity_errors']=={} and r['independent_parity_status']=='independent_parity_not_available','False completed parity claim')
            s=r['state_info']
            require(s['id']==row['id'] and s['file']==r['state_file'] and s['sha256']==r['state_sha256'],'Completed row state binding mismatch')
            require(file_sha(inside(s['file']))==s['sha256'],'Completed state changed');rebuilt.append(s)
        require(rebuilt==old['states'] and digest(rebuilt)==old['state_index_sha256'],'Completed full state index differs from rows')
        require(old['independent_parity_coverage']==coverage and old['all_baseline_parity_exact']==(coverage==len(data)),'Completed parity coverage mismatch')
        print('SLCD_V2_ALREADY_COMPLETE_VERIFIED',a.split,a.job,flush=True);return
    done={}
    for r in rows(path) if path.exists() else []:
        require(r['id'] in expected and r['id'] not in done,'Unexpected/duplicate resumed ID')
        require(r['implementation_signature']==signature and r['scope']==a.split,'Resumed provenance mismatch')
        require(r['data_sha256']==data_sha and r['reference_sha256']==provenance['reference_sha256'],'Resumed input/reference mismatch')
        require(r['weights_modified'] is False,'Resumed row is not original-weight inference')
        require(set(r['methods'])==set(recipe_by_id),'Resumed recipe set mismatch')
        require(file_sha(inside(r['state_file']))==r['state_sha256'],'Resumed state changed')
        done[r['id']]=r
    from runtime import configure,load_model,encode_pair,check_budget,outcome
    configure()  # All directories it could create were required to exist above.
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from slcd_logit_methods import score_logit_recipes
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.manual_seed(2021)
    torch.backends.cuda.matmul.allow_tf32=False
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'BF16 GPU required for execution')
    model=scorer=None
    if a.reuse_states:
        tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        with safe_open(model_path,framework='pt',device='cpu') as st:head=st.get_tensor('lm_head.weight').to('cuda')
    else:
        tok,model=load_model();scorer=OuroCachedScorer(model,6);head=model.lm_head.weight
        require(not any(p.requires_grad for p in model.parameters()),'Autograd must be disabled')
    require(head.dtype==torch.bfloat16 and list(head.shape)==[49152,2048],'Unexpected original head')
    head_sha=tensor_sha256(head)
    if audit_path in authority_paths:
        require(read(audit_path)['original_head_raw_tensor_sha256']==head_sha,'Audited original head mismatch')
    for row in data:
        if row['id'] in done:continue
        check_budget(120);begin=time.time();qid=row['id'];stem=qid.rsplit('/',1)[-1]
        require(stem and '/' not in stem and '\\' not in stem and '..' not in stem,'Unsafe state filename stem')
        require(row['prompt_sha256']==hashlib.sha256(row['prompt'].encode()).hexdigest(),'Input prompt digest mismatch')
        require(all(p[0]==row['prompt'] for p in row['request_pairs']),'Candidate prompt mismatch')
        pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
        target_hashes=[digest(t) for _,t in pairs]
        generated_values=None;authority=authorities.get(qid)
        if authority is not None:
            state_path=inside(authority['file'])
            require(file_sha(state_path)==authority['sha256'],'Authorized cache content changed')
            payload=torch.load(state_path,map_location='cpu',weights_only=True)
            require(payload['id']==qid,'Cached state ID mismatch')
            if authority['_kind']=='generation_complete' and payload.get('format')=='slcd_original_states_v2':
                require(payload['generation_source_sha256']==authority['generation_source_sha256'],'Cached generation source binding mismatch')
                for name in ('runtime.py','ouro_cached_scoring.py'):
                    require(payload['generation_source_sha256'][name]==implementation[name],'Cached state generating inference source changed')
                require(payload['model_identity']==model_identity and payload['data_sha256']==data_sha,'Cached model/data identity mismatch')
                require(payload['prompt_sha256']==row['prompt_sha256'] and payload['original_weights'] is True,'Cached prompt/weights mismatch')
                require(payload['target_id_sha256']==target_hashes,'Cached target digest mismatch')
                require(payload['request_pairs_sha256']==digest(row['request_pairs']) and payload['original_head_sha256']==head_sha,'Cached requests/head mismatch')
            elif authority['_kind']=='generation_complete':
                # A v2 receipt may forward an original SOURCE_AUDIT cache; it must
                # retain that immutable authority rather than relabel generation.
                require(authority.get('state_origin')=='source_audit_cache','Unknown legacy payload origin')
            states=payload['candidate_states'];origin=authority.get('state_origin','source_audit_cache')
        else:
            require(not a.reuse_states,'Missing authorized cache')
            state_path=ROOT/f'results/slcd_state_{a.split}_{a.job}_{stem}.pt'
            require(state_path.parent.is_dir() and not state_path.exists(),'Unreceipted state exists; preserve it rather than silently reusing/overwriting')
            generated_values=scorer.score_uncached(pairs,return_states=True)
            states=[dict(candidate=i,target_ids=t,normalized=v['normalized_states']) for i,((_,t),v) in enumerate(zip(pairs,generated_values))]
            payload=dict(format='slcd_original_states_v2',id=qid,split=a.split,candidate_states=states,
                model_revision=REVISION,model_identity=model_identity,data_sha256=data_sha,
                prompt_sha256=row['prompt_sha256'],request_pairs_sha256=digest(row['request_pairs']),
                target_id_sha256=target_hashes,generation_source_sha256=implementation,
                generation_signature=signature,original_head_sha256=head_sha,original_weights=True)
            tmp=state_path.with_suffix('.pt.tmp');require(not tmp.exists(),'Preserve partial state file')
            torch.save(payload,tmp);tmp.replace(state_path);origin='v2_original_generation'
        require(len(states)==len(pairs)==len(row['choice_char_lengths']),'Candidate count mismatch')
        totals={key:[] for key in recipe_by_id};native={k:[] for k in range(1,7)};counts=[]
        for i,(state,(_,target)) in enumerate(zip(states,pairs)):
            require(state['candidate']==i and state['target_ids']==target and len(state['normalized'])==6,'Candidate targets/depth mismatch')
            counts.append(len(target));logits={}
            t=torch.tensor(target,dtype=torch.long,device='cuda')
            for k,h in enumerate(state['normalized'],1):
                require(h.dtype==torch.bfloat16 and list(h.shape)==[1,len(target),2048] and bool(torch.isfinite(h).all()),'State shape/dtype/value mismatch')
                z=F.linear(h.to('cuda'),head)[0].float();logits[k]=z
                native[k].append(float(z.log_softmax(-1).gather(-1,t[:,None]).squeeze(-1).sum()))
            sums=score_logit_recipes(logits,target,recipes)
            for key,value in sums.items():totals[key].append(value)
            del logits
        methods={key:outcome(value,row['choice_char_lengths'],row['gold']) for key,value in totals.items()}
        originals={f'native_k{k}':outcome(v,row['choice_char_lengths'],row['gold']) for k,v in native.items()}
        originals.update(loopcd_k4_fixed=methods['tf_fixed'],loopcd_k4_adaptive=methods['tf_adaptive'])
        same_forward_errors={}
        if generated_values is not None:
            direct={f'native_k{k}':dict(sum_scores=[v['sum_loglikelihood'][k] for v in generated_values]) for k in range(1,7)}
            direct.update({f'loopcd_k4_{kind}':dict(sum_scores=[v['loopcd'][kind] for v in generated_values]) for kind in ('fixed','adaptive')})
            same_forward_errors=parity_errors(originals,dict(methods=direct),len(pairs))
        errors={}
        if reference is not None:
            ref=reference[qid]
            require(ref['gold']==row['gold'] and ref['prompt_sha256']==row['prompt_sha256'],'Reference input mismatch')
            require(ref['choice_char_lengths']==row['choice_char_lengths'] and ref['choice_token_counts']==counts,'Reference lengths mismatch')
            errors=parity_errors(originals,ref,len(pairs))
        state_info=dict(id=qid,split=a.split,file=state_path.relative_to(ROOT).as_posix(),sha256=file_sha(state_path),
            bytes=state_path.stat().st_size,candidate_count=len(pairs),choice_token_counts=counts,
            target_id_sha256=target_hashes,state_origin=origin,
            generation_source_sha256=payload.get('generation_source_sha256'),
            cache_authority=None if authority is None else authority['_authority'])
        result=dict(id=qid,gold=row['gold'],scope=a.split,prompt_sha256=row['prompt_sha256'],
            choice_char_lengths=row['choice_char_lengths'],choice_token_counts=counts,methods=methods,original_methods=originals,
            implementation_signature=signature,data_sha256=data_sha,reference_sha256=provenance['reference_sha256'],
            native_parity_errors=errors,independent_parity_status='exact' if reference is not None else 'independent_parity_not_available',
            same_forward_head_replay_errors=same_forward_errors,original_head_sha256=head_sha,weights_modified=False,
            state_file=state_info['file'],state_sha256=state_info['sha256'],state_info=state_info,elapsed_seconds=time.time()-begin)
        with path.open('a',encoding='utf8') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
        done[qid]=result
        if len(done)%16==0 or len(done)==len(data):
            progress=dict(split=a.split,job=a.job,completed=len(done),target=len(data),elapsed_seconds=time.time()-started)
            save_flat('results/slcd_TOKEN_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
        del payload,states,generated_values
    require(set(done)==set(expected) and tensor_sha256(head)==head_sha,'Coverage/head changed')
    state_index=[];coverage=0
    for row in data:
        r=done[row['id']]
        require(r['gold']==row['gold'] and r['prompt_sha256']==row['prompt_sha256'] and r['choice_char_lengths']==row['choice_char_lengths'],'Final resumed-row input mismatch')
        require(r['original_head_sha256']==head_sha,'Resumed head mismatch')
        require(set(r['original_methods'])==BASE_METHODS,'Resumed baseline set mismatch')
        if reference is not None:
            require(r['choice_token_counts']==reference[row['id']]['choice_token_counts'],'Resumed reference token counts differ')
            fresh=parity_errors(r['original_methods'],reference[row['id']],len(row['choice_char_lengths']))
            require(r['native_parity_errors']==fresh and r['independent_parity_status']=='exact','Resumed parity receipt mismatch');coverage+=1
        else:require(r['native_parity_errors']=={} and r['independent_parity_status']=='independent_parity_not_available','False independent-parity claim')
        state=r['state_info']
        require(state['id']==row['id'] and state['file']==r['state_file'] and state['sha256']==r['state_sha256'],'State-index binding mismatch')
        require(file_sha(inside(state['file']))==state['sha256'],'Final state SHA mismatch')
        state_index.append(state)
    require(file_sha(manifest)==data_sha and file_sha(recipe_path)==recipe_sha,'Input changed during execution')
    if reference is not None:require(file_sha(reference_path)==provenance['reference_sha256'],'Reference changed during execution')
    save_flat(completion,dict(status='complete',worker_format='slcd_token_worker_v2',split=a.split,n=len(data),
        provenance=provenance,implementation_signature=signature,score_file=output,score_file_sha256=file_sha(path),
        independent_parity_coverage=coverage,independent_parity_status='exact' if coverage==len(data) else 'independent_parity_not_available',
        max_baseline_sum_score_error=0. if coverage else None,all_baseline_parity_exact=coverage==len(data),
        original_head_sha256=head_sha,model_weights_modified=False,elapsed_seconds=time.time()-started,
        states=state_index,state_index_sha256=digest(state_index),state_index_complete_on_resume=True,
        actual_executed_trajectory_depth=6,required_logical_depth=max(max(r.get('final',4),r.get('late',4) if r.get('gamma',0)>0 else 4) for r in recipes)),immutable=True)
    print('SLCD_TOKEN_V2_JOB_COMPLETE',a.split,a.job,flush=True)


if __name__=='__main__':main()
