"""Post-selection correctness gate; immutable edit, training examples only.

This does not select, tune, or modify the frozen method. Supervisor charges GPU.
"""
import json
from runtime import ROOT,load_model,encode_pair,save,check_budget
from ouro_cached_scoring import OuroCachedScorer
from weight_edits import load_edit_bundle,applied_edits,tensor_sha256,file_sha256
from run_edit_pipeline import own_logits_check,methods


def main():
    complete=json.loads((ROOT/'results/PIPELINE_COMPLETE.json').read_text());assert complete['status']=='complete'
    lock=json.loads((ROOT/'configs/FINAL_LOCK.json').read_text())
    cid=lock['selected_candidate_id'];bundle=ROOT/'edited_weights/selected'
    assert file_sha256(bundle/'manifest.json')==lock['bundle_manifest_sha256']
    tok,model=load_model();scorer=OuroCachedScorer(model,6)
    params=dict(model.named_parameters());original={n:tensor_sha256(p) for n,p in params.items()}
    edits,manifest=load_edit_bundle(bundle,model);changed_names={e.name for e in edits}
    row=json.loads((ROOT/'data/discovery_screen.jsonl').read_text().splitlines()[0])
    expected=next(json.loads(l) for l in (ROOT/'results/screen'/cid/'scores.jsonl').read_text().splitlines() if json.loads(l)['id']==row['id'])
    pairs=[encode_pair(tok,*p) for p in row['request_pairs']];calls={n:[] for n in changed_names}
    check_budget(90)
    with applied_edits(model,edits,invalidate=scorer.invalidate):
        current={n:tensor_sha256(p) for n,p in params.items()}
        assert {n for n in params if original[n]!=current[n]}==changed_names
        assert all(current[e.name]==tensor_sha256(e.edited) for e in edits)
        handles=[]
        for name in changed_names:
            module=dict(model.named_modules())[name.rsplit('.',1)[0]]
            def record(mod,args,name=name):
                p=mod.weight;calls[name].append(dict(object_id=id(p),data_ptr=p.data_ptr(),version=p._version))
            handles.append(module.register_forward_pre_hook(record))
        scorer.invalidate()  # Observer registration changes the scorer's hook fingerprint.
        try:values=scorer.score_uncached(pairs,return_states=True)
        finally:
            for h in handles:h.remove()
            scorer.invalidate()
        for name,records in calls.items():
            assert len(records)==6*len(pairs)
            assert len({(r['object_id'],r['data_ptr'],r['version']) for r in records})==1
        scored=methods(values,row)
        errors={m:max(abs(a-b) for a,b in zip(v['sum_scores'],expected['methods'][m]['sum_scores'])) for m,v in scored.items()}
        assert max(errors.values())==0,errors
        own=own_logits_check(model,values[0],pairs[0][1])
        depth=scorer.check_depth_invariance(pairs[:1],depths=(3,4,5),atol=0)
        assert depth['passed']
    assert all(tensor_sha256(params[n])==h for n,h in original.items())
    # Load the portable exact patch again, rather than redo its floating GEMM.
    edits2,_=load_edit_bundle(bundle,model)
    with applied_edits(model,edits2,invalidate=scorer.invalidate):
        replay=scorer.score_uncached(pairs[:1])[0]
        assert replay['sum_loglikelihood']==values[0]['sum_loglikelihood'] and replay['loopcd']==values[0]['loopcd']
    assert all(tensor_sha256(params[n])==h for n,h in original.items())
    save('results/SELECTED_CHECKPOINT_AUDIT.json',dict(status='passed',selected_candidate_id=cid,
        bundle_manifest_sha256=lock['bundle_manifest_sha256'],training_question_id=row['id'],
        edited_tensor_names=sorted(changed_names),parameter_tensor_count=len(params),
        only_declared_physical_parameters_changed=True,all_unedited_tensor_hashes_unchanged=True,
        exact_original_restoration_all_tensors=True,exact_bf16_patch_replay=True,
        original_parameter_tensor_sha256=original,selected_parameter_tensor_sha256={n:current[n] for n in changed_names},
        exact_prior_training_scores_errors=errors,own_edited_loopcd=own,separate_depth_parity=depth,
        physical_shared_weight_observations=calls,actual_changed_parameters=manifest['actual_changed_parameters']))
    print('SELECTED_CHECKPOINT_AUDIT_PASSED',cid,flush=True)


if __name__=='__main__':main()
