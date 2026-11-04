"""CPU/static E2 equivalence and strict comparator negatives; no data access."""
import copy,json,ast
from pathlib import Path
from r20_admission import ROOT,save,sha
from r20_e2_overlap import compare_records

def main():
    old=(ROOT/'scripts/r20_e1_worker.py').read_text(encoding='utf-8-sig');new=(ROOT/'scripts/r20_e2_worker.py').read_text(encoding='utf-8-sig')
    restored=new.replace('r20_e2_','r20_e1_');assert restored==old
    assert 'from r20_e1_hooks import ValueAblation' in new and 'from r20_math import' in new
    a=[dict(id='synthetic',candidates=[dict(sums={'primary':1.25},metrics={'alpha':[0.,.5],'norm':[1.,2.],'unmatched':[False,False]},request_sha256='abc',positions=[1,2])])]
    assert compare_records(a,copy.deepcopy(a))['status']=='passed_strict_zero'
    variants=[]
    b=copy.deepcopy(a);b[0]['candidates'][0]['metrics']['alpha'][1]+=.0000001;variants.append(b)
    b=copy.deepcopy(a);del b[0]['candidates'][0]['sums'];variants.append(b)
    b=copy.deepcopy(a);b[0]['extra']=1;variants.append(b)
    b=copy.deepcopy(a);b[0]['candidates'][0]['positions'].append(3);variants.append(b)
    b=copy.deepcopy(a);b[0]['candidates'][0]['sums']['primary']=1;variants.append(b)
    b=copy.deepcopy(a);b[0]['candidates'][0]['request_sha256']='different';variants.append(b)
    b=copy.deepcopy(a);b[0]['candidates'][0]['metrics']['unmatched'][0]=True;variants.append(b)
    b=copy.deepcopy(a);b[0]['candidates'][0]['sums']['primary']=float('nan');variants.append(b)
    assert all(compare_records(a,b)['status']=='failed_stop_no_effects' for b in variants)
    fs=sorted((ROOT/'scripts').glob('r20_e2_*.py'))
    for p in fs:ast.parse(p.read_text(encoding='utf-8-sig'))
    out=dict(status='passed_CPU_static_and_synthetic',worker_namespace_only_exact=True,direct_E1_hook_reference=True,direct_R20_math_reference=True,unchanged_E1_CPU_hook_checks_sha256=sha('results/r20_e1_CPU_CHECKS.json'),comparator_exact_pass=True,comparator_negative_cases=len(variants),source_sha256={p.name:sha(str(p.relative_to(ROOT))) for p in fs},scope='Static worker source equivalence and synthetic comparator only; no model, GPU, data/partial contents, labels or outcomes')
    save('results/r20_e2_CPU_CHECKS.json',out);print(json.dumps(out))
if __name__=='__main__':main()
