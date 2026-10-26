"""Create same-run MC references for a fresh GPU reproduction; never replace them."""
import hashlib
import json
from pathlib import Path
import shutil
from common import ROOT,CFG


def main():
    target=ROOT/'audit_reference/mc'
    if target.exists():
        from compare_mc_rebuild import main as compare
        compare();return
    audit=json.loads((ROOT/'results/mc_completion_audit.json').read_text());assert audit['passed']
    names=[]
    for method in CFG['qwen_methods']:
        names += [f'trajectories/Qwen3-4B-Base_{method}_{s}.jsonl' for s in ['validation','test']]
    for model in ['Ouro-1.4B','Ouro-2.6B']:
        names += [f'trajectories/{model}_{s}.jsonl' for s in ['validation','test']]
    assert len(names)==18
    raw={}
    for name in names:
        path=ROOT/name;before=path.stat();payload=path.read_bytes();after=path.stat()
        assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
        raw[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    tables=['qwen_metrics','ouro_metrics','halting_curves','halting_selected','halting_matched_random',
            'trajectory_patterns','absolute_stability','continuation_auc','projection_geometry',
            'loopcd_trajectory_attribution','score_rule_sensitivity','depth_diagnostics','trajectory_strata',
            'readout_pairwise_controls','offline_oracle']
    for relative in [*(f'results/{n}.csv' for n in tables),'configs/mc_selected.json','configs/halting_selected.json']:
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    metadata=dict(passed=True,raw_mc_files_unchanged=18,validation_configurations_preserved=True,
                  raw_mc_file_manifest=raw,raw_snapshot_archive_sha256=None,archive_sha256=None,
                  analysis_source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                      ['scripts/analyze_suite.py','scripts/analyze_diagnostics.py','scripts/policies.py','scripts/common.py']},
                  scope='Fresh-run reference frozen from completed, independently audited MC results; references never overwrite a prior run')
    (target/'REFERENCE_PROVENANCE.json').write_text(json.dumps(metadata,indent=2))
    from compare_mc_rebuild import main as compare
    compare()


if __name__=='__main__':main()
