"""Final artifact/data/checkpoint/preservation audit; CPU, after all evaluations."""
import csv
import hashlib
import json
from pathlib import Path
from runtime import ROOT,save


def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text(encoding='utf8').splitlines() if l]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def csvcount(name):
    with (ROOT/name).open(encoding='utf8',newline='') as f:return sum(1 for _ in csv.DictReader(f))
def fp(q):return hashlib.sha256(' '.join(q.casefold().split()).encode()).hexdigest()


def main():
    complete=read('results/PIPELINE_COMPLETE.json');assert complete['status']=='complete'
    gate=read('results/SELECTED_CHECKPOINT_AUDIT.json');assert gate['status']=='passed'
    analysis=read('results/FINAL_ANALYSIS.json');assert analysis['status']=='complete'
    final=read('configs/FINAL_LOCK.json');selected=read('selected_edit_manifest.json')
    cid=final['selected_candidate_id'];assert cid==selected['selected_candidate_id']==gate['selected_candidate_id']
    assert sha(ROOT/'selected_edit_manifest.json')==final['selected_manifest_sha256']
    bundle_sha=sha(ROOT/'edited_weights/selected/manifest.json')
    assert bundle_sha==final['bundle_manifest_sha256']==gate['bundle_manifest_sha256']
    assert selected['actual_changed_parameters']>0 and selected['total_unique_parameters']==1434652673
    assert selected['training'] is False and selected['all_recurrent_invocations_share_edited_parameters']
    assert selected['embeddings_tokenizer_head_and_norm_weights_unchanged']
    test=rows('results/final_test/'+cid+'/scores.jsonl');assert len(test)==len({r['id'] for r in test})==1172
    assert all(r['bundle_manifest_sha256']==bundle_sha and r['loopcd_source']=='this edited checkpoint own K1/K4 logits' for r in test)
    assert final['plain_depths']==[3,4,5,6] and final['primary_combination']['omega']==.5 and final['secondary_combination']['cap']==1.
    counts={name:csvcount(name) for name in ['edit_search_results.csv','candidate_edit_audit.csv','final_comparison.csv',
        'final_comparison_long.csv','paired_transitions.csv','paired_statistics.csv','generalization_by_split.csv']}
    assert counts=={'edit_search_results.csv':22,'candidate_edit_audit.csv':15,'final_comparison.csv':6,
        'final_comparison_long.csv':12,'paired_transitions.csv':7032,'paired_statistics.csv':6,'generalization_by_split.csv':12}
    with (ROOT/'accuracy_by_loop.pdf').open('rb') as f:assert f.read(5)==b'%PDF-'
    discovery=rows('data/discovery_expanded.jsonl');selection=rows('data/selection.jsonl')
    val=rows('data/validation.jsonl');test_data=rows('data/test.jsonl')
    demos=read('data/fixed_demonstrations.json')['documents']
    cohorts={'discovery':discovery,'selection':selection,'validation':val,'test':test_data,'demos':demos}
    assert {k:len(v) for k,v in cohorts.items()}=={'discovery':384,'selection':192,'validation':299,'test':1172,'demos':25}
    fingerprints={k:{fp(r['question']) for r in rr} for k,rr in cohorts.items()}
    overlaps={a+'__'+b:len(fingerprints[a]&fingerprints[b]) for a in cohorts for b in cohorts if a<b}
    assert overlaps['discovery__selection']==overlaps['demos__discovery']==overlaps['demos__selection']==0
    # Keep official full validation/test even if their source dataset has text duplicates.
    save('results/SPLIT_OVERLAP_AUDIT.json',dict(normalization='casefold + collapse whitespace',
        overlaps=overlaps,official_sets_not_filtered=True,all1172_retained=True))
    for name,lock_name in [('VALIDATION','VALIDATION_LOCK'),('TEST','FINAL_LOCK')]:
        receipt=ROOT/'logs'/(name+'_STAGING.json');locked=ROOT/'configs'/(lock_name+'.json')
        assert receipt.stat().st_mtime_ns>=locked.stat().st_mtime_ns
        staging=read('logs/'+name+'_STAGING.json');assert staging['prerequisite_sha256']==sha(locked)
    preservation=read('logs/V1_PRESERVATION_BASELINE.json');v1=ROOT.with_name('ouro_weight_space_repair_v1')
    now={}
    for folder in ['results','scripts','configs','sources','data']:
        for p in (v1/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                s=p.stat();now[p.relative_to(v1).as_posix()]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    for p in v1.glob('*.md'):
        s=p.stat();now[p.name]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    modified=[p for p in preservation if p not in now or now[p]!=preservation[p]]
    added=sorted(set(now)-set(preservation));assert not modified and not added,(modified,added)
    base_sha=sha(ROOT/'models/Ouro-1.4B/model.safetensors')
    assert base_sha=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
    provenance=read('edited_weights/statistics/provenance.json')
    assert provenance['checkpoint_file_sha256']==base_sha
    budget=read('logs/gpu_budget.json');assert budget['limit_seconds']==21600 and budget['used_seconds']<=21600
    assert all(r['status'] in ['complete','failed'] for r in budget['runs'])
    assert abs(sum(r['gpu_worker_seconds'] for r in budget['runs'])-budget['used_seconds'])<1e-6
    save('ARTIFACT_AUDIT.json',dict(status='passed',selected_candidate_id=cid,
        csv_row_counts=counts,n_test=1172,actual_changed_parameters=selected['actual_changed_parameters'],
        selected_bundle_manifest_sha256=bundle_sha,only_one_selected_checkpoint=True,
        both_required_edit_families_executed=True,exact_same_weights_plain_and_loopcd=True,
        test_opened_after_final_lock=True,baseline_pairs_same_prompts_lengths_token_counts=True,
        v1_tracked_files=len(preservation),v1_size_mtime_unchanged=True,v1_added_tracked_files=added,
        original_model_safetensors_sha256=base_sha,model_revision=selected['model_revision'],
        independent_gpu_worker_seconds=budget['used_seconds'],gpu_budget_seconds=21600,
        failed_charged_runs=[r['name'] for r in budget['runs'] if r['status']=='failed'],
        normalized_question_overlap=overlaps,selected_train_gate_passed=True,
        observation='Audit confirms execution/integrity, not scientific benefit or a new mechanism.'))
    print('V2_ARTIFACT_AUDIT_PASSED',cid,flush=True)


if __name__=='__main__':main()
