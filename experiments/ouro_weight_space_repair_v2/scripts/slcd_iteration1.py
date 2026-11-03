"""Initial train-only reconstruction and required A1 correction grid."""
import json
from slcd_core import *


def main():
    assert not (ROOT/'configs/slcd_METHODS.json').exists()
    registry={}
    for mid,source in [('native4','native_k4'),('fixed','loopcd_k4_fixed'),('adaptive','loopcd_k4_adaptive')]:
        register(registry,dict(id=mid,kind='baseline',method=source,required_depth=4,complexity=0))
    ids=[]
    for gamma in [0.,.25,.5,.75,1.]:
        mid='A1_g'+str(gamma).replace('.','p');ids.append(mid)
        register(registry,dict(id=mid,kind='score_late',gamma=gamma,late=5,required_depth=4 if gamma==0 else 5,complexity=1))
    result=evaluate_registry(['native4','fixed','adaptive']+ids,registry)
    for split in ['discovery','selection']:
        data=load_development(split)
        zero=registry['A1_g0p0']
        assert all(np.array_equal(scores_for(r,zero,registry),r['methods']['loopcd_k4_fixed']) for r in data)
    save('configs/slcd_METHODS.json',registry)
    save('results/slcd_iteration1.json',dict(status='complete',train_only=True,results=result,
        score_gamma0_exact_fixed=True,code_sha256=sha('scripts/slcd_core.py')))
    writecsv('results/slcd_development_results.csv',result)
    # Decisions are written by the researcher after inspecting these real numbers.
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
