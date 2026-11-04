"""CPU E1 exact hook equivalence and negative/cleanup tests; no data/model."""
import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from types import SimpleNamespace
from r20_hooks import ValueAblation as Old
from r20_e1_hooks import ValueAblation as New
from r20_admission import save,sha
class Scorer:
    def __init__(self):self.calls=0
    def invalidate(self):self.calls+=1
def fixture():
    modules=[torch.nn.Identity(),torch.nn.Identity()];m=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(v_proj=x)) for x in modules]));return m,modules,Scorer()
def run(cls,span,loops,active):
    m,modules,s=fixture();original=modules[0].register_forward_hook(lambda m,a,o:o)
    x=torch.arange(30,dtype=torch.bfloat16).reshape(1,5,6);out=[]
    with cls(m,s) as h:
        h.configure(span,active,5)
        for _ in range(loops):
            for mod in modules:
                y=mod(x);assert active or y is x;out.append(y)
        h.verify(loops)
        if cls is New:assert len(h.flags)==(len(modules)*loops if active else 0)
    assert len(modules[0]._forward_hooks)==1 and not modules[1]._forward_hooks;original.remove()
    return out

def main():
    for span in [[0],[1,2],[0,3,4]]:
      for loops in [1,4]:
       for active in [False,True]:assert all(torch.equal(a,b) for a,b in zip(run(Old,span,loops,active),run(New,span,loops,active)))
    m,mods,s=fixture()
    with New(m,s) as h:
        h.configure([1],True,5);mods[0](torch.ones(1,5,3,dtype=torch.bfloat16))
        try:h.configure([2],True,5)
        except AssertionError:pass
        else:raise AssertionError('Unverified configure allowed')
        try:h.verify(1)
        except AssertionError:pass
        else:raise AssertionError('Missing layer count accepted')
        mods[1](torch.ones(1,5,3,dtype=torch.bfloat16));h.flags[0]=torch.tensor(False)
        try:h.verify(1)
        except AssertionError:pass
        else:raise AssertionError('False check accepted')
    assert all(not x._forward_hooks for x in mods)
    m,mods,s=fixture()
    try:
        with New(m,s) as h:
            h.configure([1],True,5);mods[0](torch.ones(1,4,3,dtype=torch.bfloat16))
    except AssertionError:pass
    else:raise AssertionError('Bad sequence accepted')
    assert all(not x._forward_hooks for x in mods)
    m,mods,s=fixture();real=mods[1].register_forward_hook
    def fail(*a,**k):raise RuntimeError('deliberate registration failure')
    mods[1].register_forward_hook=fail
    try:
        with New(m,s):pass
    except RuntimeError as e:assert str(e)=='deliberate registration failure'
    else:raise AssertionError('Registration failure lost')
    assert all(not x._forward_hooks for x in mods)
    out=dict(status='passed_CPU_old_new_exact',spans=3,loop_counts=[1,4],off_tensor_identity=True,all_flags_count=True,false_flag_rejected=True,unverified_configure_rejected=True,forward_exception_cleanup=True,partial_registration_cleanup=True,original_hooks_restored=True,source_sha256={p:sha('scripts/'+p) for p in ['r20_hooks.py','r20_e1_hooks.py','r20_e1_cpu_checks.py','r20_math.py']},scope='CPU synthetic only, no model/GPU/data/effects')
    save('results/r20_e1_CPU_CHECKS.json',out);print(json.dumps(out))
if __name__=='__main__':main()
