"""Post-package local public-record replay; restore canonical bytes afterward.

Exercises the public numeric verifier and report/figure regeneration with no
raw HellaSwag inputs/cache/model available in this experiment directory.
The archive's canonical server artifacts are restored byte-for-byte afterward.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    unavailable=['results/slcd_r6_hellaswag_scores.jsonl','results/slcd_hellaswag_scores.jsonl',
                 'results/slcd_r6_external_prior_cache.jsonl','data/slcd_hellaswag.jsonl',
                 'models/Ouro-1.4B/model.safetensors']
    assert not any((ROOT/p).exists() for p in unavailable),'Run in the public-record copy, not the full server snapshot'
    outputs=['results/slcd_R6_PUBLIC_NUMERIC_ACCEPTANCE.json','results/slcd_R6_compute_costs.csv',
             'results/slcd_R6_all_followup_development.csv','results/slcd_R6_all_benchmark_results.csv',
             'results/slcd_accuracy_comparison.pdf','results/slcd_accuracy_comparison.png',
             'results/slcd_rescue_regression_analysis.pdf','results/slcd_rescue_regression_analysis.png',
             'results/slcd_REPORT_ZH.md','slcd_README.md','slcd_FOLLOWUP_STATUS.md','results/slcd_R6_DELIVERABLES.json']
    canonical={p:(ROOT/p).read_bytes() for p in outputs};steps=[]
    try:
        for arguments in [['scripts/slcd_r6_public_scores_portable.py'],['scripts/slcd_r6_make_deliverables.py','--public-records']]:
            done=subprocess.run([sys.executable,'-B',*arguments],cwd=ROOT,capture_output=True,text=True)
            steps.append(dict(arguments=arguments,exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr))
            if done.returncode:raise RuntimeError(done.stdout+'\n'+done.stderr)
        def normalize(raw):return raw.decode('utf8').replace('\r\n','\n')
        for p in ['results/slcd_REPORT_ZH.md','slcd_README.md','slcd_FOLLOWUP_STATUS.md']:
            assert normalize((ROOT/p).read_bytes())==normalize(canonical[p]),'Public report text differs: '+p
        for p in ['results/slcd_R6_compute_costs.csv','results/slcd_R6_all_followup_development.csv','results/slcd_R6_all_benchmark_results.csv']:
            assert (ROOT/p).read_bytes()==canonical[p],'Public numeric table differs: '+p
    finally:
        # Platform-dependent PDF timestamps/font metadata must not replace the
        # canonical bytes already bound by the final server archive/index.
        for p,raw in canonical.items():(ROOT/p).write_bytes(raw)
    contents=json.loads((ROOT/'artifacts/slcd_R6_RECORDS_CONTENTS.json').read_text())
    for record in contents['files']:
        assert sha(ROOT/record['path'])==record['sha256'],'Canonical record changed: '+record['path']
    result=dict(status='passed',stage='post_package_local_public_reproduction',model_forwards=0,
                unavailable_raw_inputs=unavailable,steps=steps,report_text_equal_after_newline_normalization=True,
                numeric_csv_bytes_equal=True,canonical_archive_records_restored_and_sha_checked=contents['file_count'],
                archive_sha256=json.loads((ROOT/'artifacts/slcd_R6_RELEASE_INDEX.json').read_text())['archive']['sha256'],
                source_sha256=sha(Path(__file__)),
                note='This post-package receipt and checker are published separately; they do not mutate the 329 canonical archive records.')
    (ROOT/'results/slcd_R6_PUBLIC_REPRODUCTION_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='steps'},indent=2))


if __name__=='__main__':main()
