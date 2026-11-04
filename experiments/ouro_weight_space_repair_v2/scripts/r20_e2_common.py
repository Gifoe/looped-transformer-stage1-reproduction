"""R20 gates; writes exclusively in new namespace."""
import json
from r7_core import ROOT,read,sha,digest,jsonlines
LOCK='configs/r20_e2_VALUE_ABLATION_PILOT_LOCK.json'
def save(path,value):
    p=ROOT/path;assert p.name.startswith('r20_e2_') and p.parent.is_dir()
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def gate():
    spec=read(LOCK);assert spec['status']=='frozen_bounded_development_pilot' and len(spec['cohort'])==24
    assert not spec['heldout_access_authorized'] and not spec['external_benchmarks_authorized']
    for p,h in spec['source_sha256'].items():assert sha('scripts/'+p)==h,p
    for p,h in spec['inherited_source_sha256'].items():assert sha('scripts/'+p)==h,p
    for p,h in spec['input_sha256'].items():assert sha(p)==h,p
    receipt=read('logs/r20_e2_PUBLIC_DEFINITION_RECEIPT.json');assert receipt['readback_verified'] and receipt['protocol_sha256']==sha(LOCK)
    assert read('results/r20_CPU_ADMISSION.json')['R6_unique_vs_Fixed']>=1
    return spec

